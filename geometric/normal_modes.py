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
from .molecule import Molecule
from .nifty import logger, ang2bohr, bohr2ang, wq_wait, getWorkQueue

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
            engine.calc_wq(coords, dirname_d, readfiles=readfiles_d)['gradient']
            coords[i] -= 2*h
            dirname_d = os.path.join(dirname, "hessian/displace/%03im" % (i+1))
            readfiles_d = readfiles and os.path.exists(dirname_d)
            engine.calc_wq(coords, dirname_d, readfiles=readfiles_d)['gradient']
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
            engine.link_scratch(dirname, dirname_d)
            gfwd = engine.calc(coords, dirname_d, readfiles=readfiles_d)['gradient']
            coords[i] -= 2*h
            dirname_d = os.path.join(dirname, "hessian/displace/%03im" % (i+1))
            readfiles_d = readfiles and os.path.exists(dirname_d)
            engine.link_scratch(dirname, dirname_d)
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
