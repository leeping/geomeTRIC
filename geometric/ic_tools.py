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
from copy import deepcopy
from .nifty import logger, bohr2ang
from .step import calc_drms_dmax

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

def write_displacements(coords, M, IC, dirname, ic_select="all", displace_range=(-0.3, 0.3, 7), verbose=0):
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
    ic_select : str, int, or list
        If "all", then include all the ICs.
        If a string, then select the IC 
        If integer, select the IC numerically from the list.
        If a list, select one or more ICs numerically or by name from the list.
    displace_range : tuple
        Provide three numbers which will be used to create a linearly spaced array of displacements.
    verbose : int
        Print diagnostic messages
    """
    M1 = deepcopy(M)
    ic_names = [str(i) for i in IC.Internals]
    if len(ic_names) != len(set(ic_names)): raise RuntimeError("IC names should be unique; please check __repr__ functions")
    if type(ic_select) is str and ic_select == "all": ic_select_nums = list(range(len(IC.Internals)))
    elif type(ic_select) is str: ic_select_nums = [ic_names.index(ic_select)]
    elif type(ic_select) is int: ic_select_nums = [ic_select]
    elif type(ic_select) is list:
        ic_select_nums = []
        for ic in ic_select:
            if type(ic) is str: ic_select_nums.append(ic_names.index(ic))
            elif type(ic) is int: ic_select_nums.append(ic)
            else: raise TypeError("ic_select should be int, str, or list of ints/strs")
    else: raise TypeError("ic_select should be int, str, or list of ints/strs")

    for i in ic_select_nums:
        x = []
        for j in np.linspace(*displace_range):
            if j != 0:
                dq = np.zeros(len(IC.Internals))
                dq[i] = j
                x1 = IC.newCartesian(coords, dq, verbose=verbose)
            else:
                x1 = coords.copy()
            rms_displacement, max_displacement = calc_drms_dmax(x1, coords, align=False)
            x.append(x1.reshape(-1,3) * bohr2ang)
            logger.info("%i %.1f Displacement (rms/max) = %.5f / %.5f %s\n" % (i, j, rms_displacement, max_displacement, "(Bork)" if IC.bork else "(Good)"))
        M1.xyzs = x
        M1.write("%s/ic_%03i.xyz" % (dirname, i))
    return M1
