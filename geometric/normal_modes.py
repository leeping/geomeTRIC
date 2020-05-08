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
import os, shutil
import numpy as np
from .errors import GramSchmidtError
from .molecule import Molecule, PeriodicTable
from .nifty import logger, au2kj, ang2bohr, bohr2ang, c_lightspeed, wq_wait, getWorkQueue

def calc_cartesian_hessian(coords, molecule, engine, dirname, readfiles=True, verbose=0):
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
    readfiles : bool
        Read Hessian data from disk if valid
        
    Returns
    -------
        Hx : np.ndarray
        (Nx3)x(Nx3) array containing Cartesian Hessian.
        Files are also written to <prefix.tmp>/hessian.
    """
    nc = len(coords)
    # Attempt to read existing Hessian data if it exists.
    hesstxt = os.path.join(dirname, "hessian", "hessian.txt")
    hessxyz = os.path.join(dirname, "hessian", "coords.xyz")
    if readfiles and os.path.exists(hesstxt) and os.path.exists(hessxyz):
        Hx = np.loadtxt(hesstxt)
        if Hx.shape[0] == nc:
            hess_mol = Molecule(hessxyz)
            if np.allclose(coords.reshape(-1, 3)*bohr2ang, hess_mol.xyzs[0], atol=1e-6):
                logger.info("Using Hessian matrix read from file\n")
                return Hx
            else:
                logger.info("Coordinates for Hessian don't match current coordinates, recalculating.\n")
                readfiles=False
        else:
            logger.info("Hessian read from file doesn't have the right shape, recalculating.\n")
            readfiles=False
    elif not os.path.exists(hessxyz):
        logger.info("Coordinates for Hessian not found, recalculating.\n")
        readfiles = False
    # Save Hessian to text file
    oldxyz = molecule.xyzs[0].copy()
    molecule.xyzs[0] = coords.reshape(-1, 3)*bohr2ang
    if not os.path.exists(os.path.join(dirname, "hessian")):
        os.makedirs(os.path.join(dirname, "hessian"))
    molecule[0].write(hessxyz)
    if not readfiles: 
        if os.path.exists(os.path.join(dirname, "hessian", "displace")):
            shutil.rmtree(os.path.join(dirname, "hessian", "displace"))
    # Calculate Hessian using finite difference
    # Finite difference step
    h = 1.0e-3
    wq = getWorkQueue()
    Hx = np.zeros((nc, nc), dtype=float)
    logger.info("Calculating Cartesian Hessian using finite difference on Cartesian gradients\n")
    if wq:
        for i in range(nc):
            if verbose >= 2: logger.info(" Submitting gradient calculation for coordinate %i/%i\n" % (i+1, nc))
            coords[i] += h
            dirname_d = os.path.join(dirname, "hessian/displace/%03ip" % (i+1))
            readfiles_d = readfiles and os.path.exists(dirname_d)
            engine.copy_scratch(dirname, dirname_d)
            engine.calc_wq(coords, dirname_d, readfiles=readfiles_d)
            coords[i] -= 2*h
            dirname_d = os.path.join(dirname, "hessian/displace/%03im" % (i+1))
            readfiles_d = readfiles and os.path.exists(dirname_d)
            engine.copy_scratch(dirname, dirname_d)
            engine.calc_wq(coords, dirname_d, readfiles=readfiles_d)
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
        engine.calc(coords, dirname, readfiles=readfiles)
        for i in range(nc):
            if verbose >= 2: logger.info(" Running gradient calculation for coordinate %i/%i\n" % (i+1, nc))
            elif verbose >= 1 and (i%5 == 0): logger.info("%i / %i gradient calculations complete\n" % (i*2, nc*2))
            coords[i] += h
            dirname_d = os.path.join(dirname, "hessian/displace/%03ip" % (i+1))
            readfiles_d = readfiles and os.path.exists(dirname_d)
            engine.copy_scratch(dirname, dirname_d)
            gfwd = engine.calc(coords, dirname_d, readfiles=readfiles_d)['gradient']
            coords[i] -= 2*h
            dirname_d = os.path.join(dirname, "hessian/displace/%03im" % (i+1))
            readfiles_d = readfiles and os.path.exists(dirname_d)
            engine.copy_scratch(dirname, dirname_d)
            gbak = engine.calc(coords, dirname_d, readfiles=readfiles_d)['gradient']
            coords[i] += h
            Hx[i] = (gfwd-gbak)/(2*h)
            if verbose == 1 and i == (nc-1) : logger.info("%i / %i gradient calculations complete\n" % (nc*2, nc*2))
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

def frequency_analysis(coords, Hessian, elem=None, mass=None, verbose=0):
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
    verbose : bool
        Print debugging info

    Returns
    -------
    freqs_wavenumber : np.array
        n_vibmodes length array containing vibrational frequencies in wavenumber
        (imaginary frequencies are reported as negative)
    normal_modes_cart : np.array
        n_vibmodes*n_atoms length array containing un-mass-weighted Cartesian displacements 
        of each normal mode
    """
    na = coords.shape[0]
    if mass:
        mass = np.array(mass)
        assert len(mass) == na
    elif elem:
        mass = np.array([PeriodicTable[elem[j]] for j in elem])
        assert len(elem) == na
    else:
        logger.warning("neither elem nor mass is provided; assuming all masses unity")
        mass = np.ones(na)
        
    assert coords.shape == (na, 3)
    assert Hessian.ndim == (3*na, 3*na)

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

    # Obtain the rotational degrees of freedom
    RotDOF = 0;
    for i in range(3):
        if abs(Ivals[i]) > 1.0e-10:
            RotDOF +=1;
    TR_DOF = 3 + RotDOF;

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
        gEckart = np.dot(Ivecs, xcm[i])
        smass = np.sqrt(mass[i]) 
        ic_eckart[0][3*i  ] = smass 
        ic_eckart[1][3*i+1] = smass 
        ic_eckart[2][3*i+2] = smass 
        for ix in range(3):
            ic_eckart[3][3*i+ix] = -Ivecs[1][ix]*smass*gEckart[2] + Ivecs[2][ix]*smass*gEckart[1];
            ic_eckart[4][3*i+ix] = -Ivecs[0][ix]*smass*gEckart[2] + Ivecs[2][ix]*smass*gEckart[0];
            ic_eckart[5][3*i+ix] =  Ivecs[0][ix]*smass*gEckart[1] - Ivecs[1][ix]*smass*gEckart[0];
    
    if verbose >= 2:
        logger.info("Coordinates in Eckart frame:\n")
        for i in range(ic_eckart.shape[0]):
            for j in range(ic_eckart.shape[1]):
                logger.info(" % .12f " % ic_eckart[i, j])
            logger.info("\n")

    # Sort the rotation ICs by their norm in descending order, then normalize them
    ic_eckart_norm = np.sqrt(np.sum(ic_eckart**2, axis=1))
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
        if verbose:
            logger.info("Gram-Schmidt Iteration %i: % .12f" % (iteration, overlap))
        if max_overlap < 1e-12 : break
        if iteration == maxIt - 1:
            raise GramSchmidtError("Gram-Schmidt orthogonalization failed after %i iterations" % maxIt)
    
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
        raise GramSchmidtError("Not enough vanishing eigenvalues: %i < %i (detected < expected)" % (numzero_upper, TR_DOF))
    elif numzero_lower > TR_DOF:
        raise GramSchmidtError("Too many vanishing eigenvalues: %i > %i (detected > expected)" % (numzero_lower, TR_DOF))
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
    
    # Calculate the internal coordinate Hessian and diagonalize
    ic_hessian = np.linalg.multi_dot((ic_basis, wHessian, ic_basis.T))
    ichess_vals, ichess_vecs = np.linalg.eigh(ic_hessian)
    ichess_vecs = ichess_vecs.T
    normal_modes = np.dot(ichess_vecs, ic_basis)
    
    # Undo mass weighting to get Cartesian displacements
    normal_modes_cart = normal_modes * invsqrtm3[np.newaxis, :]
    normal_modes_cart /= np.linalg.norm(normal_modes_cart, axis=1)[:, np.newaxis]

    # Convert IC Hessian eigenvalues to wavenumbers
    freqs_wavenumber = mwHess_wavenumber * np.sqrt(np.abs(ichess_vals)) * np.sign(ichess_vals)

    if verbose:
        logger.info("Vibrational Frequencies (wavenumber) and Cartesian displacements:\n")
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

    # for i in range(VibDOF):
    #     print(freqs_wavenumber[i])
    # Print summary
    # if verbose:
    # print("Number of Atoms", na)
    # print("Number of vibrational modes", VibDOF)
    # print("Vibrational modes in Cartesian coordinates (not mass-weighted)")
    # print(freqs_wavenumber)
    # print(normal_modes_cart)
    return freqs_wavenumber, normal_modes_cart

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
    np.savetxt('displacements.txt', displacements)

if __name__ == "__main__":
    main()
