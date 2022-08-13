"""
normal_modes.py: Compute Cartesian Hessian and perform vibrational analysis

Copyright 2016-2020 Regents of the University of California and the Authors

Authors: Lee-Ping Wang, Chenchen Song

Contributors: Yudong Qiu, Daniel G. A. Smith, Sebastian Lee, Chaya Stern, Qiming Sun,
Alberto Gobbi, Josh Horton

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import division
from __future__ import print_function
import os, shutil
import numpy as np
from .errors import FrequencyError
from .molecule import Molecule, PeriodicTable
from .nifty import logger, kb, kb_si, hbar, au2kj, au2kcal, ang2bohr, bohr2ang, c_lightspeed, avogadro, cm2au, amu2au, ambervel2au, wq_wait, getWorkQueue, commadash, bak

def calc_cartesian_hessian(coords, molecule, engine, dirname, read_data=True, verbose=0):
    """ 
    Calculate the Cartesian Hessian using finite difference, and/or read data from disk. 
    Data is stored in a folder <prefix>.tmp/hessian, with gradient calculations found in
    <prefix>.tmp/hessian/displace/001p.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of Cartesian coordinates in atomic units
    molecule : Molecule
        Molecule object
    engine : Engine
        Object containing methods for calculating energy and gradient
    dirname : str
        Directory name for files to be written, i.e. <prefix>.tmp
    read_data : bool
        Read Hessian data from disk if valid
        
    Returns
    -------
        Hx : np.ndarray
        (Nx3)x(Nx3) array containing Cartesian Hessian.
        Files are also written to <prefix.tmp>/hessian.
    """
    nc = len(coords)
    # Attempt to read existing Hessian data if it exists.
    # Read from files hessian.txt/coords.xyz, hessian_1.txt/coords_1.xyz, etc.
    counter = 0
    while read_data:
        if counter > 0:
            hesstxt = os.path.join(dirname, "hessian", "hessian_%i.txt" % counter)
            hessxyz = os.path.join(dirname, "hessian", "coords_%i.xyz" % counter)
        else:
            hesstxt = os.path.join(dirname, "hessian", "hessian.txt")
            hessxyz = os.path.join(dirname, "hessian", "coords.xyz")
        if os.path.exists(hesstxt) and os.path.exists(hessxyz):
            Hx = np.loadtxt(hesstxt)
            if Hx.shape[0] == nc:
                hess_mol = Molecule(hessxyz)
                if np.allclose(coords.reshape(-1, 3)*bohr2ang, hess_mol.xyzs[0], atol=1e-6):
                    logger.info("Using Hessian matrix read from file: %s\n" % hesstxt)
                    return Hx
        elif counter >= 1:
            logger.info("Valid Hessian data not found, calculating from scratch.\n")
            break
        counter += 1

    # Compute hessian from scratch.
    hesstxt = os.path.join(dirname, "hessian", "hessian.txt")
    hessxyz = os.path.join(dirname, "hessian", "coords.xyz")
    # First back up any existing Hessian data in a way that a future calculation could read it.
    if os.path.exists(hessxyz) and os.path.exists(hesstxt):
        bak(hessxyz)
        bak(hesstxt)

    oldxyz = molecule.xyzs[0].copy()
    molecule.xyzs[0] = coords.reshape(-1, 3)*bohr2ang
    if not os.path.exists(os.path.join(dirname, "hessian")):
        os.makedirs(os.path.join(dirname, "hessian"))
        
    molecule[0].write(hessxyz)
    if not read_data: 
        if os.path.exists(os.path.join(dirname, "hessian", "displace")):
            shutil.rmtree(os.path.join(dirname, "hessian", "displace"))
    # Calculate Hessian using finite difference
    # Finite difference step
    h = 1.0e-3
    wq = getWorkQueue()
    Hx = np.zeros((nc, nc), dtype=float)
    logger.info("Calculating Cartesian Hessian using finite difference on Cartesian gradients (%i grads total)\n" % (2*nc))
    if wq:
        for i in range(nc):
            if verbose >= 2: logger.info(" Submitting gradient calculation for coordinate %i/%i\n" % (i+1, nc))
            coords[i] += h
            dirname_d = os.path.join(dirname, "hessian/displace/%03ip" % (i+1))
            engine.calc_wq(coords, dirname_d, read_data=read_data, copydir=dirname)
            coords[i] -= 2*h
            dirname_d = os.path.join(dirname, "hessian/displace/%03im" % (i+1))
            engine.calc_wq(coords, dirname_d, read_data=read_data, copydir=dirname)
            coords[i] += h
        wq_wait(wq, print_time=600)
        for i in range(nc):
            if verbose >= 2: logger.info(" Reading gradient results for coordinate %i/%i\n" % (i+1, nc))
            coords[i] += h
            dirname_d = os.path.join(dirname, "hessian/displace/%03ip" % (i+1))
            gfwd = engine.read_wq(coords, dirname_d)['gradient']
            coords[i] -= 2*h
            dirname_d = os.path.join(dirname, "hessian/displace/%03im" % (i+1))
            gbak = engine.read_wq(coords, dirname_d)['gradient']
            coords[i] += h
            Hx[i] = (gfwd-gbak)/(2*h)
    else:
        # First calculate a gradient at the central point, for linking scratch files.
        engine.calc(coords, dirname, read_data=read_data)
        for i in range(nc):
            if verbose >= 1: logger.info(" Running gradient calculation for coordinate %i/%i\n" % (i+1, nc))
            elif i%5 == 0: logger.info("%i / %i gradient calculations complete\n" % (i*2, nc*2))
            coords[i] += h
            dirname_d = os.path.join(dirname, "hessian/displace/%03ip" % (i+1))
            gfwd = engine.calc(coords, dirname_d, read_data=read_data, copydir=dirname)['gradient']
            coords[i] -= 2*h
            dirname_d = os.path.join(dirname, "hessian/displace/%03im" % (i+1))
            gbak = engine.calc(coords, dirname_d, read_data=read_data, copydir=dirname)['gradient']
            coords[i] += h
            Hx[i] = (gfwd-gbak)/(2*h)
            if i == (nc-1) : logger.info("%i / %i gradient calculations complete\n" % (nc*2, nc*2))
    # Save Hessian to text file
    oldxyz = molecule.xyzs[0].copy()
    molecule.xyzs[0] = coords.reshape(-1, 3)*bohr2ang
    molecule[0].write(hessxyz)
    molecule.xyzs[0] = oldxyz
    np.savetxt(hesstxt, Hx)
    # Delete displacement calcs because they take up too much space
    keep_displace = False
    if not keep_displace:
        if os.path.exists(os.path.join(dirname, "hessian", "displace")):
            shutil.rmtree(os.path.join(dirname, "hessian", "displace"))
    return Hx

def frequency_analysis(coords, Hessian, elem=None, mass=None, energy=0.0, temperature=300.0, pressure=1.0, verbose=0, outfnm=None, note=None, wigner=None, ignore=0, normalized=True):
    """
    Parameters
    ----------
    coords : np.array
        n_atoms*3 length array containing coordinates in bohr
    Hessian : np.array
        (n_atoms*3)*(n_atoms*3) length array containing Hessian elements in au
        (i.e. Hessian/bohr^2), the typical units output by QM calculations
    elem : list
        n_atoms length list containing atomic symbols. 
        Used in printing displacements and for looking up masses if mass = None.
    mass : list or np.array
        n_atoms length list or 1D array containing atomic masses in amu.
        If provided, masses will not be looked up using elem.
        If neither mass nor elem will be provided, will assume masses are all unity.
    energy : float
        Electronic energy passed to the harmonic free energy module
    temperature : float
        Temperature passed to the harmonic free energy module
    pressure : float
        Pressure passed to the harmonic free energy module
    verbose : int
        Print debugging info
    outfnm : str
        If provided, write vibrational data to a ForceBalance-parsable vdata.txt file
    note : str
        If provided, write a note into the comment line of the xyz structure in vdata.txt
    wigner : tuple
        If provided, should be a 2-tuple containing (nSamples, dirname)
        containing the output folder and number of samples and the output folder
        to which samples should be written
    ignore : int
        Ignore the free energy contributions from the lowest N vibrational modes
        (including negative force constants if there are any). 
    normalized : bool
        If True, normalize the un-mass-weighted Cartesian displacements of each normal mode (default)
        If False, return the un-normalized vectors (necessary for IR and Raman intensities)

    Returns
    -------
    freqs_wavenumber : np.array
        n_vibmodes length array containing vibrational frequencies in wavenumber
        (imaginary frequencies are reported as negative)
    normal_modes_cart : np.array
        n_vibmodes*n_atoms length array containing un-mass-weighted Cartesian displacements 
        of each normal mode
    """
    # Create a copy of coords and reshape into a 2D array
    coords = coords.copy().reshape(-1, 3)
    na = coords.shape[0]
    if mass is not None:
        mass = np.array(mass)
        assert len(mass) == na
    elif elem:
        mass = np.array([PeriodicTable[j] for j in elem])
        assert len(elem) == na
    else:
        logger.warning("neither elem nor mass is provided; assuming all masses unity")
        mass = np.ones(na)
    assert coords.shape == (na, 3)
    assert Hessian.shape == (3*na, 3*na)

    # Convert Hessian eigenvalues into wavenumbers:
    # 
    # omega = sqrt(k/m)
    # 
    # X hartree bohr^-2 amu^-1 * 2625.5 (kJ mol^-1 / hartree) * (1/0.0529 bohr/nm)^2
    # --> omega^2 = 938211*X ps^-2
    # 
    # Frequencies in units of inverse ps:
    # nu = sqrt(938211*X)/2*pi
    # 
    # Convert to inverse wavelength in units of cm^-1:
    # 
    # 1/lambda = nu/c = (sqrt(938211*X)/2*pi) / (2.998e8 m/s) * (m/100cm) * 10^12ps/s

    bohr2nm = bohr2ang / 10
    mwHess_wavenumber = 1e10*np.sqrt(au2kj / bohr2nm**2)/(2*np.pi*c_lightspeed)

    TotDOF = 3*na
    # Compute the mass weighted Hessian matrix
    # Each element is H[i, j] / sqrt(m[i]) / sqrt(m[j])
    invsqrtm3 = 1.0/np.sqrt(np.repeat(mass, 3))
    wHessian = Hessian.copy() * np.outer(invsqrtm3, invsqrtm3)

    if verbose >= 2:
        # Eigenvalues before projection of translation and rotation
        logger.info("Eigenvalues before projection of translation and rotation\n")
        w_eigvals = np.linalg.eigvalsh(wHessian)
        for i in range(TotDOF):
            val = mwHess_wavenumber*np.sqrt(abs(w_eigvals[i]))
            logger.info("%5i % 10.3f\n" % (i, val))

    #=============================================#
    #| Remove rotational and translational modes |#
    #=============================================#

    # Compute the center of mass
    cxyz = np.sum(coords * mass[:, np.newaxis], axis=0)/np.sum(mass)

    # Coordinates in the center-of-mass frame
    xcm = coords - cxyz[np.newaxis, :]
    
    # Moment of inertia tensor
    I = np.sum([mass[i] * (np.eye(3)*(np.dot(xcm[i], xcm[i])) - np.outer(xcm[i], xcm[i])) for i in range(na)], axis=0)

    # Principal moments
    Ivals, Ivecs = np.linalg.eigh(I)
    # Eigenvectors are in the rows after transpose
    Ivecs = Ivecs.T 

    # Obtain the number of rotational degrees of freedom
    RotDOF = 0
    for i in range(3):
        if abs(Ivals[i]) > 1.0e-10:
            RotDOF += 1
    TR_DOF = 3 + RotDOF
    if TR_DOF not in (5, 6):
        raise FrequencyError("Unexpected number of trans+rot DOF: %i not in (5, 6)" % TR_DOF)

    if verbose >= 2:
        logger.info("Center of mass: % .12f % .12f % .12f\n" % (cxyz[0], cxyz[1], cxyz[2]))
        logger.info("Moment of inertia tensor:\n")
        for i in range(3):
            logger.info("   % .12f % .12f % .12f\n" % (I[i, 0], I[i, 1], I[i, 2]))
        logger.info("Principal moments of inertia:\n")
        for i in range(3):
            logger.info("Eigenvalue = %.12f   Eigenvector = % .12f % .12f % .12f\n" % (Ivals[i], Ivecs[i, 0], Ivecs[i, 1], Ivecs[i, 2]))
        logger.info("Translational-Rotational degrees of freedom: %i\n" % TR_DOF)

    # Internal coordinates of the Eckart frame
    ic_eckart=np.zeros((6, TotDOF))
    for i in range(na):
        # The dot product of (the coordinates of the atoms with respect to the center of mass) and 
        # the corresponding row of the matrix used to diagonalize the moment of inertia tensor
        p_vec = np.dot(Ivecs, xcm[i])
        smass = np.sqrt(mass[i]) 
        ic_eckart[0,3*i  ] = smass 
        ic_eckart[1,3*i+1] = smass 
        ic_eckart[2,3*i+2] = smass 
        for ix in range(3):
            ic_eckart[3,3*i+ix] = smass*(Ivecs[2,ix]*p_vec[1] - Ivecs[1,ix]*p_vec[2])
            ic_eckart[4,3*i+ix] = smass*(Ivecs[2,ix]*p_vec[0] - Ivecs[0,ix]*p_vec[2])
            ic_eckart[5,3*i+ix] = smass*(Ivecs[0,ix]*p_vec[1] - Ivecs[1,ix]*p_vec[0])
    
    if verbose >= 2:
        logger.info("Coordinates in Eckart frame:\n")
        for i in range(ic_eckart.shape[0]):
            for j in range(ic_eckart.shape[1]):
                logger.info(" % .12f " % ic_eckart[i, j])
            logger.info("\n")

    # Sort the rotation ICs by their norm in descending order, then normalize them
    ic_eckart_norm = np.sqrt(np.sum(ic_eckart**2, axis=1))
    # If the norm is equal to zero, then do not scale.
    ic_eckart_norm += (ic_eckart_norm == 0.0)
    sortidx = np.concatenate((np.array([0,1,2]), 3+np.argsort(ic_eckart_norm[3:])[::-1]))
    ic_eckart1 = ic_eckart[sortidx, :]
    ic_eckart1 /= ic_eckart_norm[sortidx, np.newaxis]
    ic_eckart = ic_eckart1.copy()

    if verbose >= 2:
        logger.info("Eckart frame basis vectors:\n")
        for i in range(ic_eckart.shape[0]):
            for j in range(ic_eckart.shape[1]):
                logger.info(" % .12f " % ic_eckart[i, j])
            logger.info("\n")

    # Using Gram-Schmidt orthogonalization, create a basis where translation 
    # and rotation is projected out of Cartesian coordinates
    proj_basis = np.identity(TotDOF)
    maxIt = 100
    for iteration in range(maxIt):
        max_overlap = 0.0
        for i in range(TotDOF):
            for n in range(TR_DOF):
                proj_basis[i] -= np.dot(ic_eckart[n], proj_basis[i]) * ic_eckart[n] 
            overlap = np.sum(np.dot(ic_eckart, proj_basis[i]))
            max_overlap = max(overlap, max_overlap)        
        if verbose >= 2:
            logger.info("Gram-Schmidt Iteration %i: % .12f\n" % (iteration, overlap))
        if max_overlap < 1e-12 : break
        if iteration == maxIt - 1:
            raise FrequencyError("Gram-Schmidt orthogonalization failed after %i iterations" % maxIt)
    
    # Diagonalize the overlap matrix to create (3N-6) orthonormal basis vectors
    # constructed from translation and rotation-projected proj_basis
    proj_overlap = np.dot(proj_basis, proj_basis.T)
    if verbose >= 3:
        logger.info("Overlap matrix:\n")
        for i in range(proj_overlap.shape[0]):
            for j in range(proj_overlap.shape[1]):
                logger.info(" % .12f " % proj_overlap[i, j])
            logger.info("\n")
    proj_vals, proj_vecs = np.linalg.eigh(proj_overlap)
    proj_vecs = proj_vecs.T
    if verbose >= 3:
        logger.info("Eigenvectors of overlap matrix:\n")
        for i in range(proj_vecs.shape[0]):
            for j in range(proj_vecs.shape[1]):
                logger.info(" % .12f " % proj_vecs[i, j])
            logger.info("\n")

    # Make sure number of vanishing eigenvalues is roughly equal to TR_DOF
    numzero_upper = np.sum(abs(proj_vals) < 1.0e-8)  # Liberal counting of zeros - should be more than TR_DOF
    numzero_lower = np.sum(abs(proj_vals) < 1.0e-12) # Conservative counting of zeros - should be less than TR_DOF
    if numzero_upper == TR_DOF and numzero_lower == TR_DOF:
        if 0: logger.info("Expected number of vanishing eigenvalues: %i\n" % TR_DOF)
    elif numzero_upper < TR_DOF:
        raise FrequencyError("Not enough vanishing eigenvalues: %i < %i (detected < expected)" % (numzero_upper, TR_DOF))
    elif numzero_lower > TR_DOF:
        raise FrequencyError("Too many vanishing eigenvalues: %i > %i (detected > expected)" % (numzero_lower, TR_DOF))
    else:
        logger.warning("Eigenvalues near zero: N(<1e-12) = %i, N(1e-12-1e-8) = %i Expected = %i\n" % (numzero_lower, numzero_upper, TR_DOF))

    # Construct eigenvectors of unit length in the space of Cartesian displacements
    VibDOF = TotDOF - TR_DOF
    norm_vecs = proj_vecs[TR_DOF:] / np.sqrt(proj_vals[TR_DOF:, np.newaxis])

    if verbose >= 3:
        logger.info("Coefficients of Gram-Schmidt orthogonalized vectors:\n")
        for i in range(norm_vecs.shape[0]):
            for j in range(norm_vecs.shape[1]):
                logger.info(" % .12f " % norm_vecs[i, j])
            logger.info("\n")

    # These are the orthonormal, TR-projected internal coordinates
    ic_basis = np.dot(norm_vecs, proj_basis)
    
    #===========================================#
    #| Calculate frequencies and displacements |#
    #===========================================#

    # Calculate the internal coordinate Hessian and diagonalize
    ic_hessian = np.linalg.multi_dot((ic_basis, wHessian, ic_basis.T))
    ichess_vals, ichess_vecs = np.linalg.eigh(ic_hessian)
    ichess_vecs = ichess_vecs.T
    normal_modes = np.dot(ichess_vecs, ic_basis)
    
    # Undo mass weighting to get Cartesian displacements
    normal_modes_cart = normal_modes * invsqrtm3[np.newaxis, :]
    if normalized:
        normal_modes_cart /= np.linalg.norm(normal_modes_cart, axis=1)[:, np.newaxis]

    # Convert IC Hessian eigenvalues to wavenumbers
    freqs_wavenumber = mwHess_wavenumber * np.sqrt(np.abs(ichess_vals)) * np.sign(ichess_vals)

    if verbose:
        logger.info("\n-=# Vibrational Frequencies (wavenumber) and Cartesian displacements #=-\n\n")
        i = 0
        while True:
            j = min(i+3, VibDOF)
            for k in range(i, j):
                logger.info("  Frequency(cm^-1): % 12.6f     " % freqs_wavenumber[k])
            logger.info("\n")
            for k in range(i, j):
                logger.info("--------------------------------     ")
            logger.info("\n")
            for n in range(na):
                for k in range(i, j):
                    if elem:
                        logger.info("%-2s " % elem[n])
                    else:
                        logger.info("   ")
                    logger.info("% 9.6f " % normal_modes_cart[k, 3*n])
                    logger.info("% 9.6f " % normal_modes_cart[k, 3*n+1])
                    logger.info("% 9.6f " % normal_modes_cart[k, 3*n+2])
                    logger.info("    ")
                logger.info("\n")
            if i+3 >= VibDOF: break
            logger.info("\n")
            i += 3

    # Write results to file (can be parsed by ForceBalance)
    G_tot_au, components, out_lines = free_energy_harmonic(coords, mass, freqs_wavenumber, energy, temperature, pressure, verbose, ignore)
    if outfnm:
        write_vdata(freqs_wavenumber, normal_modes_cart, coords, elem, outfnm, out_lines, note=note)
        logger.info("Vibrational analysis written to %s\n" % outfnm)
    if wigner is not None:
        # Overwrite control. If negative number is provided, then overwrite samples.
        nSample, dirname = wigner
        overwrite = False
        if nSample < 0:
            nSample = abs(nSample)
            overwrite = True
        wigner_sample(coords, mass, elem, freqs_wavenumber, normal_modes, temperature, nSample, dirname, overwrite)
    return freqs_wavenumber, normal_modes_cart, G_tot_au

def free_energy_harmonic(coords, mass, freqs_wavenumber, energy, temperature, pressure, verbose = 0, ignore = 0):
    """
    Calculate Gibbs free energy (i.e. thermochemical analysis) of a system where
    translation / rotation / vibration degrees of freedom are approximated using
    ideal gas / rigid rotor / harmonic oscillator respectively.

    Caveat: For rotational degrees of freedom, the high-temperature limit is used and
    the symmetry is assumed to be C1.

    Parameters
    ----------
    coords : np.array
        n_atoms*3 length or (natoms, 3) shape array containing coordinates in bohr
    mass : np.array
        n_atoms length 1D array containing atomic masses in amu.
    freqs_wavenumber : np.array
        n_freqs (3*n_atoms-5 or 6) length 1D array containing frequencies in wavenumber.
    energy : float
        Ground state energy (in a.u.)
    temperature : float
        Temperature (in K) for which the free energy corrections are to be computed
    pressure : float
        Pressure (in bar) for which the free energy corrections are to be computed
    verbose : int
        Print debugging info
    ignore : int
        Ignore contributions of lowest N vibrational modes to free energy
    """
    # Create a copy of coords and reshape into a 2D array
    coords = coords.copy().reshape(-1, 3)
    na = coords.shape[0]
    mass = np.array(mass)
    assert mass.shape == (na,)
    assert coords.shape == (na, 3)
    assert freqs_wavenumber.shape in ((3*na-5,), (3*na-6,))

    E, T, P = energy, temperature, pressure
    # Total mass
    mtot = np.sum(mass)
    # Compute the center of mass
    cxyz = np.sum(coords * mass[:, np.newaxis], axis=0)/mtot
    # Coordinates in the center-of-mass frame
    xcm = coords - cxyz[np.newaxis, :]
    # Moment of inertia tensor
    I = np.sum([mass[i] * (np.eye(3)*(np.dot(xcm[i], xcm[i])) - np.outer(xcm[i], xcm[i])) for i in range(na)], axis=0)
    # Principal moments in units of amu bohr^2
    Ivals, Ivecs = np.linalg.eigh(I)
    # In descending order, and converted to SI units
    Ivals_si = np.sort(Ivals)[::-1] / (avogadro * 1000) * (bohr2ang * 1e-10)**2
    # Obtain the number of rotational degrees of freedom
    RotDOF = 0
    for i in range(3):
        if abs(Ivals[i]) > 1.0e-10:
            RotDOF +=1
    # Planck's constant in SI units
    h = 2 * np.pi * hbar
    # These values should be calculated in kcal/mol (for energy) and cal/mol/K (for entropy).
    # H is "enthalpy" which is equivalent to thermal energy,
    # except for translational degrees of freedom where it is equal to E+PV.
    # For ideal gas, H = <E> + PV = 5/2 kT
    H_trans = 2.5 * kb * T / 4.184
    # Total mass in kg
    m_kg = mtot / avogadro / 1000
    # kb_T in Si units
    kT_si = kb_si * T
    # Boltzmann's constant in units of cal mol^-1 K^-1
    kb_cal = kb * 1000 / 4.184
    # Volume per particle at the prvided temperature and pressure in m^3
    # (includes factor of 1000 for kJ/mol -> J/mol and 1/1e5 for bar -> Pa
    V = kT_si / (P * 1e5)
    # Thermal wavelength at the current temperature
    L = h/np.sqrt(2*np.pi*m_kg*kT_si)
    # Entropy for one particle from Sackur-Tetrode equation (N=1):
    # S = N * kb * ln(V/N/L**3) + 5/2
    # Convert from units of kJ mol^-1 K^-1 to cal mol^-1 K^-1
    S_trans = kb_cal * (np.log(V/L**3) + 2.5)
    # Rotational energy in classical approximation
    E_rot = kb_cal * T * RotDOF / (2 * 1000)
    # Rotational entropy 
    if RotDOF == 3:
        (Ia, Ib, Ic) = Ivals_si
        # Not dividing by the symmetry number; may implement in future
        Z_rot = np.sqrt(np.pi * Ia * Ib * Ic) * (np.sqrt(2 * kT_si) / hbar)**3
        S_rot = kb_cal * (np.log(Z_rot) + 1.5)
    elif RotDOF == 2:
        # For a linear molecule, the moment of inertia is just one number
        Ia = Ivals_si[0]
        Z_rot = 2 * Ia * kT_si / hbar**2
        S_rot = kb_cal * (np.log(Z_rot) + 1.0)
    # Vibrational energy
    ZPVE = 0.0
    E_vib = 0.0
    S_vib = 0.0
    nimag = 0
    if verbose >= 1:
        logger.info("\nMode   Freq(1/cm)     Zero-point  +  Thermal = Evib(kcal/mol) Svib(cal/mol/K) DG(ZPE+Thermal-TS)\n\n")
    imaginary_freqs = []
    for ifreq, freq in enumerate(freqs_wavenumber):
        if ifreq < ignore:
            e_vib1 = 0.0
            s_vib1 = 0.0
            zpve1 = 0.0
        elif freq < 0:
            imaginary_freqs.append(freq)
            nimag += 1
            e_vib1 = 0.0
            s_vib1 = 0.0
            zpve1 = 0.0
        else:
            # Energy quantum of harmonic oscillator
            hv_si = h * (freq * 100 * c_lightspeed)
            # In units of kcal/mol
            hv = hv_si * avogadro / 1000 / 4.184
            # Zero point vibrational energy
            zpve1 = hv/2
            # <E> = hv [1/2 + 1/(exp(beta*hv) - 1)]
            expfac = np.exp(hv_si/kT_si)
            e_vib1 = hv * 1.0/(expfac-1.0)
            # Vibrational partition function
            z_vib1 = np.exp(-hv_si/(kT_si*2))/(1-np.exp(-hv_si/kT_si))
            # F = -kT ln Z
            f_vib1 = -kb_cal * T * np.log(z_vib1) / 1000
            # F = E - TS -> S = (E-F)/T
            s_vib1 = 1000*(zpve1+e_vib1-f_vib1)/T
        if verbose >= 1:
            logger.info("%4i   % 10.4f       %8.4f    %8.4f         %8.4f        %8.4f        %8.4f\n" % 
                        (ifreq, freq, zpve1, e_vib1, zpve1+e_vib1, s_vib1, zpve1+e_vib1 - T*s_vib1/1000))
        ZPVE += zpve1
        E_vib += e_vib1
        S_vib += s_vib1
    H_tot = H_trans + E_rot + E_vib
    S_tot = S_trans + S_rot + S_vib
    DG_tot = ZPVE + H_tot - T*S_tot/1000
    G_tot_au = E + DG_tot/au2kcal
    components = {'Trans':{'H':H_trans, 'S':S_trans, 'G': H_trans-T*S_trans},
                  'Rot':{'E':E_rot, 'S':S_rot, 'G': E_rot-T*S_rot},
                  'Vib':{'ZPVE':ZPVE, 'E':E_vib, 'S':S_vib, 'G':ZPVE+E_vib-T*S_vib}}
    out_lines = ["\n"]
    out_lines.append("== Summary of harmonic free energy analysis ==\n")
    out_lines.append("Note: Rotational symmetry is set to 1 regardless of true symmetry\n")
    if ignore > 0:
        out_lines.append("Note: Free energy ignores contributions from %i lowest force constants\n" % ignore)
    if nimag > 0:
        out_lines.append("%i Imaginary Frequencies (cm^-1): %s\n" % (nimag, ' '.join(["%.3fi" % (-1*freq) for freq in imaginary_freqs])))
        out_lines.append("Note: Free energy does not include contribution from imaginary mode(s)\n")
    else:
        out_lines.append("No Imaginary Frequencies\n")
    out_lines.append("\n")
    out_lines.append("Free energy contributions calculated at @ %.2f K:\n" % T)
    out_lines.append("Zero-point vibrational energy:                              %12.4f kcal/mol \n" % ZPVE)
    out_lines.append("H   (Trans + Rot + Vib = Tot): %8.4f + %8.4f + %8.4f = %8.4f kcal/mol \n" % (H_trans, E_rot, E_vib, H_tot))
    out_lines.append("S   (Trans + Rot + Vib = Tot): %8.4f + %8.4f + %8.4f = %8.4f cal/mol/K\n" % (S_trans, S_rot, S_vib, S_tot))
    out_lines.append("TS  (Trans + Rot + Vib = Tot): %8.4f + %8.4f + %8.4f = %8.4f kcal/mol \n" % (T*S_trans/1000, T*S_rot/1000, T*S_vib/1000, T*S_tot/1000))
    out_lines.append("\n")
    out_lines.append("Ground State Electronic Energy    : E0                        = % 14.8f au (% 15.4f kcal/mol)\n" % (E, E*au2kcal))
    out_lines.append("Free Energy Correction (Harmonic) : ZPVE + [H-TS]_T,R,V       = % 14.8f au (% 15.4f kcal/mol)\n" % (DG_tot/au2kcal, DG_tot))
    out_lines.append("Gibbs Free Energy (Harmonic)      : E0 + ZPVE + [H-TS]_T,R,V  = % 14.8f au (% 15.4f kcal/mol)\n" % (G_tot_au, G_tot_au*au2kcal))
    out_lines.append("\n")
    logger.info(''.join(out_lines))
    return G_tot_au, components, out_lines

def write_vdata(freqs_wavenumber, normal_modes_cart, xyz, elem, outfnm, extracomms=None, note=None):
    """
    Write vibrational data to a text file readable by ForceBalance.
    
    Parameters
    ----------
    freqs_wavenumber : np.array
        n_vibmodes length array containing vibrational frequencies in wavenumber
        (imaginary frequencies should be negative)
    normal_modes_cart : np.array
        n_vibmodes*n_atoms length array containing un-mass-weighted Cartesian displacements 
        of each normal mode
    coords : np.ndarray
        Nx3 array of Cartesian coordinates in atomic units
        (note: coordinates will be written to file in Angstrom)
    elem : list
        n_atoms length list containing atomic symbols. 
    outfnm : str
        Output file name:
    extracomms : list
        List of additional lines to be printed as comments before the start of data
        (for example, the harmonic free energy components)
    note : str
        Optional note to print in comment line of xyz block
    """
    commblk = """    #==========================================#
    #|   File containing vibrational modes    |#
    #|      generated by geomeTRIC and        |#
    #|       readable by ForceBalance         |# 
    #|                                        |#
    #| Octothorpes are comments               |#
    #| This file should be formatted like so: |#
    #| (Full XYZ file for the molecule)       |#
    #| Number of atoms                        |#
    #| Comment line                           |#
    #| a1 x1 y1 z1 (xyz for atom 1)           |#
    #| a2 x2 y2 z2 (xyz for atom 2)           |#
    #|                                        |#
    #| These coords will be actually used     |#
    #|                                        |#
    #| (Followed by vibrational modes)        |#
    #| Do not use mass-weighted coordinates   |#
    #| ...                                    |#
    #| v (Eigenvalue in wavenumbers)          |#
    #| dx1 dy1 dz1 (Eigenvector for atom 1)   |#
    #| dx2 dy2 dz2 (Eigenvector for atom 2)   |#
    #| ...                                    |#
    #| (Empty line is optional)               |#
    #| v (Eigenvalue)                         |#
    #| dx1 dy1 dz1 (Eigenvector for atom 1)   |#
    #| dx2 dy2 dz2 (Eigenvector for atom 2)   |#
    #| ...                                    |#
    #| and so on                              |#
    #|                                        |#
    #| Please list freqs in increasing order  |#
    #==========================================#
    """
    with open(outfnm, 'w') as f:
        print(commblk, file=f)
        if extracomms:
            for line in extracomms:
                print("# " + line, file=f, end='')
            print("", file=f)
        print(len(elem), file=f)
        if note:
            print(note, file=f)
        else:
            print("Coordinates and vibrations generated by geomeTRIC", file=f)
        for e, i in zip(elem, xyz):
            print("%2s % 15.10f % 15.10f % 15.10f" % (e, i[0]*bohr2ang, i[1]*bohr2ang, i[2]*bohr2ang), file=f)
        for frq, mode in zip(freqs_wavenumber, normal_modes_cart):
            print(file=f)
            print("% 12.6f" % frq, file=f)
            for i in mode.reshape(-1,3):
                print("% 9.6f % 9.6f % 9.6f" % (i[0], i[1], i[2]), file=f)

def wigner_sample(coords, mass, elem, freqs_wavenumber, normal_modes, temperature, n_samples, dirname, overwrite):
    """
    Generate samples from a Wigner distribution.

    Parameters
    ----------
    coords : np.array
        1D or 2D (n_atoms x 3) array containing coordinates in a.u.
    mass : list or np.array
        Atomic masses in amu
    elem : list
        Atomic symbols
    freqs_wavenumber : np.array
        Vibrational frequencies in cm^-1 (output of frequency_analysis)
    normal_modes : np.array
        2D array (3N-6 or 3N-5 x (natoms x 3)) mass-weighted displacements of vibrational modes (output of frequency_analysis)
    temperature : float
        Desired temperature for distribution
    n_samples : int
        Desired number of samples
    dirname : str
        Output directory name where Wigner samples should be written
        Files will be written to dirname/000/, and include coords.xyz (Angstroms),
        vel.xyz (Amber units), and fms.dat (contains coordinates and momenta in a.u.)
    overwrite : bool
        If True, then overwrite any existing Wigner sample files.
        (Accessed by providing a negative number to OptParams.wigner)
    """
    mass = np.array(mass)
    nAtoms = len(mass)
    coords = coords.reshape(-1, 3)
    assert coords.shape[0] == nAtoms

    # convert frequency to a.u.
    freq_au = freqs_wavenumber* cm2au
    mass_au = np.array(mass)* amu2au
    au2joule = 1000 * 2625.4996394798254 / 6.02214076e23

    # beta = k_B*T 
    beta = 1.0 /( kb_si / au2joule * temperature);

    nmodes = len(freq_au)
    ZPE = 0.5*np.sum(freq_au)

    # Total mass
    totmass = np.sum(mass)
    # Compute the center of mass
    cxyz = np.sum(coords * mass[:, np.newaxis], axis=0)/totmass
    
    # Coordinates in the center-of-mass frame
    ctr_coors = coords - cxyz[np.newaxis, :]

    # how to test imaginary frequency

    sigma_x = []
    sigma_p = []    
    for n in range(nmodes):
        freq = freq_au[n]
        if temperature ==0:
            tanhf = 1.0
        else:
            tanhf = np.tanh(freq*0.5*beta)
        # f= 0.5* beta* hbar* omega
        # position: exp( -tanh(f)* m* omega/hbar * q^2)
        sigma_x2 = 0.5/(freq*tanhf) 
        sigma_x.append(np.sqrt(sigma_x2)) 
        
        # momentum: exp( -tanh(f)/(hbar*m*omega)* p^2)
        sigma_p2 = 0.5*(freq/tanhf)
        sigma_p.append(np.sqrt(sigma_p2))
    # print("sigma_x", sigma_x)
    # print("sigma_p", sigma_p)
    sample_data = []
    ovr_idx = []
    
    for idx in range(n_samples):
        xvec = np.zeros(nAtoms*3)
        pvec = np.zeros(nAtoms*3)

        # apply random displacement
        for n in range(nmodes):
            xlen = np.random.normal(0.0, sigma_x[n])
            plen = np.random.normal(0.0, sigma_p[n])
            xvec += xlen* normal_modes[n]
            pvec += plen* normal_modes[n]

        # undo mass-weights           
        for a in range(nAtoms):
            smass = np.sqrt(mass_au[a])
            for ix in range(3):
                xvec[a*3+ix] /= smass
                pvec[a*3+ix] *= smass

        # generate coordinates
        xvec = np.reshape(xvec, (nAtoms, 3))
        coors = xvec + ctr_coors

        # remove COM and generate velocity/momenta
        totmass = 0
        vel_com = np.zeros(3)
        for a in range(nAtoms):
            totmass += mass_au[a]
            for ix in range(3):
                vel_com[ix] += pvec[a*3+ix]  
        vel_com /= totmass

        velos = np.zeros((nAtoms, 3))
        momenta = np.zeros((nAtoms, 3))
        for a in range(nAtoms):
            smass = mass_au[a]
            for ix in range(3):
                velos[a,ix] = pvec[a*3+ix]/ smass - vel_com[ix]
                momenta[a,ix] = velos[a,ix]*mass_au[a] 

        EKin = 0.0
        for a in range(nAtoms):
            for ix in range(3):
                EKin += mass_au[a]* velos[a,ix]* velos[a,ix] 
        EKin *= 0.5

        # Write sample data to files.
        outd = os.path.join(dirname, "%03i" % idx)
        if not os.path.exists(outd) : os.makedirs(outd)
        if any([os.path.exists(os.path.join(outd, f)) for f in ['coords.xyz', 'vel.xyz', 'fms.dat']]):
            ovr_idx.append(idx)
            if not overwrite: continue
        M = Molecule()
        M.elem = elem
        M.xyzs = [coors.copy() * bohr2ang]
        M.comms = ['Randomly sampled initial positions; COM at origin']
        M.write(os.path.join(outd, 'coords.xyz'))
        M.xyzs = [velos.copy() / ambervel2au ]
        M.comms = ['Randomly sampled initial velocities; COM removed; AMBER units; KE = %.8f a.u.' % EKin]
        M.write(os.path.join(outd, 'vel.xyz'))
        with open(os.path.join(outd, 'fms.dat'), 'w') as f:
            print("UNITS=BOHR", file=f)
            print("%i" % nAtoms, file=f)
            for i in range(nAtoms):
                print("%-5s% 18.10f% 18.10f% 18.10f" % (elem[i], coors[i,0], coors[i,1], coors[i,2]), file=f)
            print("# momenta", file=f)
            for i in range(nAtoms):
                print("  % 18.10f% 18.10f% 18.10f" % (momenta[i,0], momenta[i,1], momenta[i,2]), file=f)
    logger.info("Wigner distribution sampling: %i samples using T = %.2f written to %s\n" % (n_samples, temperature, dirname))
    if ovr_idx:
        print("Wigner distribution sample generation: %s samples %s" % (commadash(ovr_idx), 'overwritten' if overwrite else 'skipped'))
            
def main():
    import logging.config, pkg_resources
    import geometric.optimize
    logIni = pkg_resources.resource_filename(geometric.optimize.__name__, 'config/logTest.ini')
    logging.config.fileConfig(logIni,disable_existing_loggers=False)
    M = Molecule("start.xyz")
    coords = M.xyzs[0] / bohr2ang
    mass = np.array([PeriodicTable[M.elem[j]] for j in range(M.na)])
    hessian = np.loadtxt("hessian.txt")
    frequencies, displacements = frequency_analysis(coords, hessian, elem=M.elem, mass=mass, verbose=True)

if __name__ == "__main__":
    main()
