"""
ic_tools.py: Useful functions for checking or working with internal coordinates

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

import numpy as np
from .nifty import logger

def check_internal_grad(coords, molecule, IC, engine, dirname, verbose=0):
    """ Check the internal coordinate gradient using finite difference. """
    # Initial energy and gradient
    spcalc = engine.calc(coords, dirname)
    E = spcalc['energy']
    gradx = spcalc['gradient']
    # Initial internal coordinates
    q0 = IC.calculate(coords)
    Gq = IC.calcGrad(coords, gradx)
    logger.info("-=# Now checking gradient of the energy in internal coordinates vs. finite difference #=-\n")
    logger.info("%20s : %14s %14s %14s\n" % ('IC Name', 'Analytic', 'Numerical', 'Abs-Diff'))
    h = 1e-3
    Gq_f = np.zeros_like(Gq)
    for i in range(len(q0)):
        dq = np.zeros_like(q0)
        dq[i] += h
        x1 = IC.newCartesian(coords, dq, verbose)
        EPlus = engine.calc(x1, dirname)['energy']
        dq[i] -= 2*h
        x1 = IC.newCartesian(coords, dq, verbose)
        EMinus = engine.calc(x1, dirname)['energy']
        fdiff = (EPlus-EMinus)/(2*h)
        logger.info("%20s : % 14.6f % 14.6f % 14.6f\n" % (IC.Internals[i], Gq[i], fdiff, Gq[i]-fdiff))
        Gq_f[i] = fdiff
    return Gq, Gq_f

def check_internal_hess(coords, molecule, IC, engine, dirname, verbose=0):
    """ Calculate the Cartesian Hessian using finite difference,
    transform to internal coordinates, then check the internal coordinate
    Hessian using finite difference. """
    # Initial energy and gradient
    spcalc = engine.calc(coords, dirname)
    E = spcalc['energy']
    gradx = spcalc['gradient']
    # Finite difference step
    h = 1.0e-3

    # Calculate Hessian using finite difference
    nc = len(coords)
    Hx = np.zeros((nc, nc), dtype=float)
    logger.info("Calculating Cartesian Hessian using finite difference on Cartesian gradients\n")
    for i in range(nc):
        logger.info(" coordinate %i/%i\n" % (i+1, nc))
        coords[i] += h
        gplus = engine.calc(coords, dirname)['gradient']
        coords[i] -= 2*h
        gminus = engine.calc(coords, dirname)['gradient']
        coords[i] += h
        Hx[i] = (gplus-gminus)/(2*h)

    # Internal coordinate Hessian using analytic transformation
    Hq = IC.calcHess(coords, gradx, Hx)

    # Initial internal coordinates and gradient
    q0 = IC.calculate(coords)
    Gq = IC.calcGrad(coords, gradx)

    Hq_f = np.zeros((len(q0), len(q0)), dtype=float)
    logger.info("-=# Now checking Hessian of the energy in internal coordinates using finite difference on gradient #=-\n")
    logger.info("%20s %20s : %14s %14s %14s\n" % ('IC1 Name', 'IC2 Name', 'Analytic', 'Numerical', 'Abs-Diff'))
    h = 1.0e-2
    for i in range(len(q0)):
        dq = np.zeros_like(q0)
        dq[i] += h
        x1 = IC.newCartesian(coords, dq, verbose)
        qplus = IC.calculate(x1)
        gplus = engine.calc(x1, dirname)['gradient']
        gqplus = IC.calcGrad(x1, gplus)
        dq[i] -= 2*h
        x1 = IC.newCartesian(coords, dq, verbose)
        qminus = IC.calculate(x1)
        gminus = engine.calc(x1, dirname)['gradient']
        gqminus = IC.calcGrad(x1, gminus)
        fdiffg = (gqplus-gqminus)/(2*h)
        for j in range(len(q0)):
            fdiff = fdiffg[j]
            Hq_f[i, j] = fdiff
            logger.info("%20s %20s : % 14.6f % 14.6f % 14.6f\n" % (IC.Internals[i], IC.Internals[j], Hq[i, j], fdiff, Hq[i, j]-fdiff))

    Eigx = sorted(np.linalg.eigh(Hx)[0])
    logger.info("Hessian Eigenvalues (Cartesian):\n")
    for i in range(len(Eigx)):
        logger.info("% 10.5f %s" % (Eigx[i], "\n" if i%9 == 8 else ""))
    Eigq = sorted(np.linalg.eigh(Hq)[0])
    logger.info("Hessian Eigenvalues (Internal):\n")
    for i in range(len(Eigq)):
        logger.info("% 10.5f %s" % (Eigq[i], "\n" if i%9 == 8 else ""))
    return Hq, Hq_f

def write_displacements(coords, M, IC, dirname, verbose):
    """
    Write coordinate files containing animations
    of displacements along the internal coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Flat array of Cartesian coordinates in a.u.
    M : Molecule
        Molecule object allowing writing of files
    IC : InternalCoordinates
        The internal coordinate system
    dirname : str
        Directory name for files to be written
    verbose : int
        Print diagnostic messages
    """
    for i in range(len(IC.Internals)):
        x = []
        for j in np.linspace(-0.3, 0.3, 7):
            if j != 0:
                dq = np.zeros(len(IC.Internals))
                dq[i] = j
                x1 = IC.newCartesian(coords, dq, verbose=verbose)
            else:
                x1 = coords.copy()
            rms_displacement, max_displacement = calc_drms_dmax(x1, coords, align=False)
            x.append(x1.reshape(-1,3) * bohr2ang)
            logger.info("%i %.1f Displacement (rms/max) = %.5f / %.5f %s\n" % (i, j, rms_displacement, max_displacement, "(Bork)" if IC.bork else "(Good)"))
        M.xyzs = x
        M.write("%s/ic_%03i.xyz" % (dirname, i))

