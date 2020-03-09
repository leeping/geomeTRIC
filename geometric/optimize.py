from __future__ import print_function, division

import argparse
import itertools
import os
import shutil
import sys
import time
import traceback

import numpy as np
from numpy.linalg import multi_dot

import pkg_resources

import geometric
from .logo import printLogo
from .engine import set_tcenv, load_tcin, TeraChem, ConicalIntersection, Psi4, QChem, Gromacs, Molpro, OpenMM, QCEngineAPI
from .internal import *
from .molecule import Molecule, Elements
from .nifty import row, col, flat, invert_svd, uncommadash, isint, bohr2ang, ang2bohr, logger, bak, pvec1d, pmat2d, getWorkQueue
from .rotate import get_rot, sorted_eigh, calc_fac_dfac
from .errors import EngineError, GeomOptNotConvergedError
from enum import Enum

def RebuildHessian(IC, H0, coord_seq, grad_seq, params):
    """
    Rebuild the Hessian after making a change to the internal coordinate system.

    Parameters
    ----------
    IC : InternalCoordinates
        Object describing the internal coordinate system
    H0 : np.ndarray
        N_ic x N_ic square matrix containing the guess Hessian
    coord_seq : list
        List of N_atom x 3 Cartesian coordinates in atomic units
    grad_seq : list
        List of N_atom x 3 Cartesian gradients in atomic units
    params : OptParams object
        Uses trust, epsilon, and reset
        trust : Only recover using previous geometries within the trust radius
        epsilon : Small eigenvalue threshold
        reset : Revert to the guess Hessian if eigenvalues smaller than threshold

    Returns
    -------
    np.ndarray
        Internal coordinate Hessian updated with series of internal coordinate gradients
    """
    Na = len(coord_seq[0])/3
    history = 0
    for i in range(2, len(coord_seq)+1):
        disp = bohr2ang*(coord_seq[-i]-coord_seq[-1])
        rmsd = np.sqrt(np.sum(disp**2)/Na)
        if rmsd > params.trust: break
        history += 1
    if history < 1:
        return H0.copy()
    logger.info("Rebuilding Hessian using %i gradients\n" % history)
    y_seq = [IC.calculate(i) for i in coord_seq[-history-1:]]
    g_seq = [IC.calcGrad(i, j) for i, j in zip(coord_seq[-history-1:],grad_seq[-history-1:])]
    Yprev = y_seq[0]
    Gprev = g_seq[0]
    H = H0.copy()
    for i in range(1, len(y_seq)):
        Y = y_seq[i]
        G = g_seq[i]
        Yprev = y_seq[i-1]
        Gprev = g_seq[i-1]
        Dy   = col(Y - Yprev)
        Dg   = col(G - Gprev)
        # Mat1 = (Dg*Dg.T)/(Dg.T*Dy)[0,0]
        # Mat2 = ((H*Dy)*(H*Dy).T)/(Dy.T*H*Dy)[0,0]
        Mat1 = np.dot(Dg,Dg.T)/np.dot(Dg.T,Dy)[0,0]
        Mat2 = np.dot(np.dot(H,Dy),np.dot(H,Dy).T)/multi_dot([Dy.T,H,Dy])[0,0]
        Hstor = H.copy()
        H += Mat1-Mat2
    if np.min(np.linalg.eigh(H)[0]) < params.epsilon and params.reset:
        logger.info("Eigenvalues below %.4e (%.4e) - returning guess\n" % (params.epsilon, np.min(np.linalg.eigh(H)[0])))
        return H0.copy()
    return H

def calc_drms_dmax(Xnew, Xold, align=True):
    """
    Align and calculate the RMSD for two geometries.

    Xnew : np.ndarray
        First set of coordinates as a flat array in a.u.
    Xold : np.ndarray
        Second set of coordinates as a flat array in a.u.
    align : bool
        Align before calculating RMSD or no?

    Returns
    -------
    float, float
        RMS and maximum displacements in Angstrom
    """
    # Shift to the origin
    Xold = Xold.copy().reshape(-1, 3)
    Xold -= np.mean(Xold, axis=0)
    Xnew = Xnew.copy().reshape(-1, 3)
    Xnew -= np.mean(Xnew, axis=0)
    # Obtain the rotation
    if align:
        U = get_rot(Xnew, Xold)
        # Xrot = np.array((U*np.matrix(Xnew).T).T).flatten()
        Xrot = np.dot(U, Xnew.T).T.flatten()
        Xold = np.array(Xold).flatten()
        displacement = np.sqrt(np.sum((((Xrot-Xold)*bohr2ang).reshape(-1,3))**2, axis=1))
    else:
        displacement = np.sqrt(np.sum((((Xnew-Xold)*bohr2ang).reshape(-1,3))**2, axis=1))
    rms_displacement = np.sqrt(np.mean(displacement**2))
    max_displacement = np.max(displacement)
    return rms_displacement, max_displacement

def getCartesianNorm(X, dy, IC, enforce=0.0, verbose=0):
    """
    Get the norm of the optimization step in Cartesian coordinates.

    Parameters
    ----------
    X : np.ndarray
        Nx3 array of Cartesian coordinates in atomic units
    dy : np.ndarray
        N_ic array of internal coordinate displacements
    IC : InternalCoordinates
        Object describing the internal coordinate system
    enforce : float
        Enforce constraints in the internal coordinate system when
        all constraints are satisfied to within the provided tolerance.
        Passing a value of zero means this is not used.
    verbose : int
        Print diagnostic messages

    Returns
    -------
    float
        The RMSD between the updated and original Cartesian coordinates
    """
    # Displacement of each atom in Angstrom
    if IC.haveConstraints() and enforce > 0.0:
        Xnew = IC.newCartesian_withConstraint(X, dy, thre=enforce, verbose=verbose)
    else:
        Xnew = IC.newCartesian(X, dy, verbose=verbose)
    rmsd, maxd = calc_drms_dmax(Xnew, X)
    return rmsd

# def truncateStep(X, dy, IC, truncate, enforce=0.0, verbose=0, align=True):
#     if IC.haveConstraints() and enforce > 0.0:
#         Xnew = IC.newCartesian_withConstraint(X, dy, thre=enforce, verbose=verbose)
#     else:
#         Xnew = IC.newCartesian(X, dy, verbose=verbose)
#     # Shift to the origin
#     Xold = X.copy().reshape(-1, 3)
#     Xold_com = np.mean(Xold, axis=0)
#     Xold -= Xold_com
#     Xnew = Xnew.copy().reshape(-1, 3)
#     Xnew_com = np.mean(Xnew, axis=0)
#     Xnew -= Xnew_com
#     # Obtain the rotation
#     if align:
#         U = get_rot(Xnew, Xold)
#         Xrot = np.dot(U, Xnew.T).T.flatten()
#         Xold = np.array(Xold).flatten()
#         displacement = np.sqrt(np.sum((((Xrot-Xold)*bohr2ang).reshape(-1,3))**2, axis=1))
#     else:
#         displacement = np.sqrt(np.sum((((Xnew-Xold)*bohr2ang).reshape(-1,3))**2, axis=1))
#     rms_displacement = np.sqrt(np.mean(displacement**2))
#     max_displacement = np.max(displacement)
#     if truncate < rms_displacement:
#         ratio = truncate / rms_displacement
#         if align:
#             truncated_dX = (Xrot-Xold)*ratio
#             Xnew_trunc = Xold + truncated_dX
#             Xnew_trunc = np.dot(U.T, Xnew_trunc.reshape(-1,3).T).T
#         else:
#             truncated_dX = (Xnew-Xold)*ratio
#             Xnew_trunc = Xold + truncated_dX
#         Xnew_trunc += Xnew_com
#         Xnew_trunc = Xnew_trunc.flatten()
#         dy_new = IC.calcDiff(Xnew_trunc, Xold)
#         new_stepsize = calc_drms_dmax(Xnew_trunc, Xold, align=align)[0]
#         print("LPW debug: Truncating step by ratio: %.5f * %.5f = %.5f\n" % (rms_displacement, ratio, new_stepsize))
#         return dy_new
#     else:
#         print("LPW debug: Truncating step not needed")
#         return dy

def between(s, a, b):
    if a < b:
        return s > a and s < b
    elif a > b:
        return s > b and s < a
    else:
        raise RuntimeError('a and b must be different')

def brent_wiki(f, a, b, rel, cvg=0.1, obj=None, verbose=0):
    """
    Brent's method for finding the root of a function.

    Parameters
    ----------
    f : function
        The function containing the root to be found
    a : float
        One side of the "bracket" to start finding the root
    b : float
        The other side of the "bracket"
    rel : float
        The denominator used to calculate the fractional error (in our case, the trust radius)
    cvg : float
        The convergence threshold for the relative error
    obj : object
        Object associated with the function that we may communicate with if desired
    verbose : int
        Print diagnostic messages

    Returns
    -------
    float
        The location of the root
    """
    fa = f(a)
    fb = f(b)
    if fa*fb > 0:
        raise RuntimeError('Not bracketed')
    if np.abs(fa) < np.abs(fb):
        # Swap if |f(a)| < |f(b)|
        a, b = b, a
        fa, fb = fb, fa
    # Set c to a
    c = a
    fc = fa
    mflag = True
    delta = 1e-6
    epsilon = min(0.01, 1e-2*np.abs(a-b))
    if obj is not None: obj.brentFailed = False
    while True:
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = a*fb*fc/((fa-fb)*(fa-fc))
            s += b*fa*fc/((fb-fa)*(fb-fc))
            s += c*fa*fb/((fc-fa)*(fc-fb))
        else:
            # Secant method
            s = b-fb*(b-a)/(fb-fa)
        # Evaluate conditions
        condition1 = not between(s, (3*a+b)/4, b)
        condition2 = mflag and (np.abs(s-b) >= np.abs(b-c)/2)
        condition3 = (not mflag) and (np.abs(s-b) >= np.abs(c-d)/2)
        condition4 = mflag and (np.abs(b-c) < delta)
        condition5 = (not mflag) and (np.abs(c-d) < delta)
        if any([condition1, condition2, condition3, condition4, condition5]):
            # Bisection method
            s = (a+b)/2
            mflag = True
        else:
            mflag = False
        # Calculate f(s)
        fs = f(s)
        # print a, s, b, fs, rel, cvg
        # Successful convergence
        if np.abs(fs/rel) <= cvg:
            return s
        # Convergence failure - interval becomes
        # smaller than threshold
        if np.abs(b-a) < epsilon:
            if verbose: logger.info("returning because interval is too small\n")
            if obj is not None: obj.brentFailed = True
            return s
        # Exit before converging when
        # the function value is positive
        if hasattr(obj, 'from_above'):
            if (obj is not None and obj.from_above) and fs > 0:
                return s
        d = c; fd = fc
        c = b; fc = fb
        if fa*fs < 0:
            b = s; fb = fs
        else:
            a = s; fa = fs
        if np.abs(fa) < np.abs(fb):
            # Swap if |f(a)| < |f(b)|
            a, b = b, a
            fa, fb = fb, fa

def OneDScan(init, final, steps):
    """
    Return a list of N equally spaced values between initial and final.
    This method works with lists of numbers

    Parameters
    ----------
    init : list
        List of numbers to be interpolated
    final : np.ndarray or list
        List of final numbers, must have same shape as "init"
    steps : int
        Number of interpolation steps

    Returns
    -------
    list
        List of lists that interpolate between init and final, including endpoints.
    """
    if len(init) != len(final):
        raise RuntimeError("init and final must have the same length")
    Answer = []
    for j in range(len(init)):
        Answer.append(np.linspace(init[j], final[j], steps))
    Answer = list([list(i) for i in np.array(Answer).T])
    return Answer

def ParseConstraints(molecule, constraints_string):
    """
    Parameters
    ----------
    molecule : Molecule
        Molecule object
    constraints_string : str
        String containing the constraint specification.

    Returns
    -------
    objs : list
        List of primitive internal coordinates corresponding to the constraints
    valgrps : list
        List of lists of constraint values. (There are multiple lists when we are scanning)
    """
    mode = None
    Freezes = []
    # The key in this dictionary is for looking up the following information:
    # 1) The classes for creating the primitive coordinates corresponding to the constraint
    # 2) The number of atomic indices that are required to specify the constraint
    ClassDict = {"distance":([Distance], 2),
                 "angle":([Angle], 3),
                 "dihedral":([Dihedral], 4),
                 "x":([CartesianX], 1),
                 "y":([CartesianY], 1),
                 "z":([CartesianZ], 1),
                 "xy":([CartesianX, CartesianY], 1),
                 "xz":([CartesianX, CartesianZ], 1),
                 "yz":([CartesianY, CartesianZ], 1),
                 "xyz":([CartesianX, CartesianY, CartesianZ], 1),
                 "trans-x":([TranslationX], 1),
                 "trans-y":([TranslationY], 1),
                 "trans-z":([TranslationZ], 1),
                 "trans-xy":([TranslationX, TranslationY], 1),
                 "trans-xz":([TranslationX, TranslationZ], 1),
                 "trans-yz":([TranslationY, TranslationZ], 1),
                 "trans-xyz":([TranslationX, TranslationY, TranslationZ], 1),
                 "rotation":([RotationA, RotationB, RotationC], 1)
                 }
    AtomKeys = ["x", "y", "z", "xy", "yz", "xz", "xyz"]
    TransKeys = ["trans-x", "trans-y", "trans-z", "trans-xy", "trans-yz", "trans-xz", "trans-xyz"]
    objs = []
    vals = []
    coords = molecule.xyzs[0].flatten() * ang2bohr
    for line in constraints_string.split('\n'):
        line = line.split("#")[0].strip().lower()
        # This is a list-of-lists. The intention is to create a multidimensional grid
        # of constraint values if necessary.
        if len(line) == 0: continue
        logger.info(line+'\n')
        if line.startswith("$"):
            mode = line.replace("$","")
        else:
            if mode is None:
                raise RuntimeError("Mode ($freeze, $set, $scan) must be set before specifying any constraints")
            s = line.split()
            key = s[0]
            if ''.join(sorted(key)) in AtomKeys:
                key = ''.join(sorted(key))
            elif ''.join(sorted(key.replace('trans-',''))) in AtomKeys:
                key = 'trans-'+''.join(sorted(key.replace('trans-','')))
            classes, n_atom = ClassDict[key]
            if mode == "freeze":
                ntok = n_atom
            elif mode == "set":
                if key == 'rotation':
                    ntok = n_atom + 4
                else:
                    ntok = n_atom + len(classes)
            elif mode == "scan":
                if key == 'rotation':
                    ntok = n_atom + 6
                else:
                    ntok = n_atom + 2*len(classes) + 1
            if len(s) != (ntok+1):
                raise RuntimeError("For this line:%s\nExpected %i tokens but got %i" % (line, ntok+1, len(s)))
            if key in AtomKeys or key in TransKeys:
                # Special code that works for atom position and translation constraints.
                if isint(s[1]):
                    atoms = [int(s[1])-1]
                elif s[1] in [k.lower() for k in Elements]:
                    atoms = [i for i in range(molecule.na) if molecule.elem[i].lower() == s[1]]
                else:
                    atoms = uncommadash(s[1])
                if any([i<0 for i in atoms]):
                    raise RuntimeError("Atom numbers must start from 1")
                if any([i>=molecule.na for i in atoms]):
                    raise RuntimeError("Constraints refer to higher atom indices than the number of atoms")
            if key in AtomKeys:
                # The x-coordinate of all the atoms in a group is a
                # list of constraints that is scanned in 1-D.
                for cls in classes:
                    objs.append([cls(a, w=1.0) for a in atoms])
                if mode == "freeze":
                    for cls in classes:
                        vals.append([[None for a in atoms]])
                elif mode == "set":
                    x1 = [float(i) * ang2bohr for i in s[2:2+len(classes)]]
                    for icls, cls in enumerate(classes):
                        vals.append([[x1[icls] for a in atoms]])
                elif mode == "scan":
                    # If we're scanning it, then we add the whole list of distances to the list-of-lists
                    x1 = [float(i) * ang2bohr for i in s[2:2+len(classes)]]
                    x2 = [float(i) * ang2bohr for i in s[2+len(classes):2+2*len(classes)]]
                    nstep = int(s[2+2*len(classes)])
                    valscan = OneDScan(x1, x2, nstep)
                    for icls, cls in enumerate(classes):
                        vals.append([[v[icls] for a in atoms] for v in valscan])
            elif key in TransKeys:
                # If there is more than one atom and the mode is "set" or "scan", then the
                # center of mass is constrained, so we pick the corresponding classes.
                if len(atoms) > 1:
                    objs.append([cls(atoms, w=np.ones(len(atoms))/len(atoms)) for cls in classes])
                else:
                    objs.append([cls(atoms[0], w=1.0) for cls in classes])
                if mode == "freeze":
                    # LPW 2016-02-10:
                    # trans-x, trans-y, trans-z is a GROUP of constraints
                    # Each group of constraints gets a [[None, None, None]] appended to vals
                    vals.append([[None for cls in classes]])
                elif mode == "set":
                    # Depending on how many coordinates are constrained, we read in the corresponding
                    # number of constraint values.
                    x1 = [float(i) * ang2bohr for i in s[2:2+len(classes)]]
                    # If there's just one constraint value then we append it to the value list-of-lists
                    vals.append([x1])
                elif mode == "scan":
                    # If we're scanning it, then we add the whole list of distances to the list-of-lists
                    x1 = [float(i) * ang2bohr for i in s[2:2+len(classes)]]
                    x2 = [float(i) * ang2bohr for i in s[2+len(classes):2+2*len(classes)]]
                    nstep = int(s[2+2*len(classes)])
                    vals.append(OneDScan(x1, x2, nstep))
            elif key in ["distance", "angle", "dihedral"]:
                if len(classes) != 1:
                    raise RuntimeError("Not OK!")
                atoms = [int(i)-1 for i in s[1:1+n_atom]]
                if key == "distance" and atoms[0] > atoms[1]:
                    atoms = atoms[::-1]
                if key == "angle" and atoms[0] > atoms[2]:
                    atoms = atoms[::-1]
                if key == "dihedral" and atoms[1] > atoms[2]:
                    atoms = atoms[::-1]
                if any([i<0 for i in atoms]):
                    raise RuntimeError("Atom numbers must start from 1")
                if any([i>=molecule.na for i in atoms]):
                    raise RuntimeError("Constraints refer to higher atom indices than the number of atoms")
                objs.append([classes[0](*atoms)])
                if mode == "freeze":
                    vals.append([[None]])
                elif mode in ["set", "scan"]:
                    if key == "distance": x1 = float(s[1+n_atom]) * ang2bohr
                    else: x1 = float(s[1+n_atom])*np.pi/180.0
                    if mode == "set":
                        vals.append([[x1]])
                    else:
                        if key == "distance": x2 = float(s[2+n_atom]) * ang2bohr
                        else: x2 = float(s[2+n_atom])*np.pi/180.0
                        nstep = int(s[3+n_atom])
                        vals.append([[i] for i in list(np.linspace(x1,x2,nstep))])
            elif key in ["rotation"]:
                # User can only specify ranges of atoms
                atoms = uncommadash(s[1])
                sel = coords.reshape(-1,3)[atoms,:]  * ang2bohr
                sel -= np.mean(sel, axis=0)
                rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
                if mode == "freeze":
                    for cls in classes:
                        objs.append([cls(atoms, coords, {}, w=rg)])
                        vals.append([[None]])
                elif mode in ["set", "scan"]:
                    objs.append([cls(atoms, coords, {}, w=rg) for cls in classes])
                    # Get the axis
                    u = np.array([float(s[i]) for i in range(2, 5)])
                    u /= np.linalg.norm(u)
                    # Get the angle
                    theta1 = float(s[5]) * np.pi / 180
                    if np.abs(theta1) > np.pi * 0.9:
                        logger.info("Large rotation: Your constraint may not work\n")
                    if mode == "set":
                        c = np.cos(theta1/2.0)
                        s = np.sin(theta1/2.0)
                        q = np.array([c, u[0]*s, u[1]*s, u[2]*s])
                        fac, _ = calc_fac_dfac(c)
                        v1 = fac*q[1]*rg
                        v2 = fac*q[2]*rg
                        v3 = fac*q[3]*rg
                        vals.append([[v1, v2, v3]])
                    elif mode == "scan":
                        theta2 = float(s[6]) * np.pi / 180
                        if np.abs(theta2) > np.pi * 0.9:
                            logger.info("Large rotation: Your constraint may not work\n")
                        steps = int(s[7])
                        # To alleviate future confusion:
                        # There is one group of three constraints that we are going to scan over in one dimension.
                        # Here we create one group of constraint values.
                        # We will add triplets of constraint values to this group
                        vs = []
                        for theta in np.linspace(theta1, theta2, steps):
                            c = np.cos(theta/2.0)
                            s = np.sin(theta/2.0)
                            q = np.array([c, u[0]*s, u[1]*s, u[2]*s])
                            fac, _ = calc_fac_dfac(c)
                            v1 = fac*q[1]*rg
                            v2 = fac*q[2]*rg
                            v3 = fac*q[3]*rg
                            vs.append([v1, v2, v3])
                        vals.append(vs)
    if len(objs) != len(vals):
        raise RuntimeError("objs and vals should be the same length")
    valgrps = [list(itertools.chain(*i)) for i in list(itertools.product(*vals))]
    objs = list(itertools.chain(*objs))
    return objs, valgrps

def image_gradient_hessian(G, H, indices):
    """
    Calculate an image quadratic function 
    """
    # Sorted eigenvalues and corresponding eigenvectors of the Hessian
    Hvals, Hvecs = sorted_eigh(H, asc=True)

    # Projection of gradient along the Hessian eigenvectors
    # Gproj = np.dot(Hvecs.T, G)

    house = np.eye(G.shape[0])
    
    for i in indices:
        Hvals[i] *= -1
        house -= 2*np.outer(Hvecs[:,i], Hvecs[:,i])

    Gs = np.dot(house, G)
        
    Hs = np.zeros_like(H)
    # Gs = np.zeros_like(G)
    for i in range(H.shape[0]):
        Hs += Hvals[i] * np.outer(Hvecs[:,i], Hvecs[:,i])
        # Gs += Gproj[i] * Hvecs[:,i]
    
    return Gs, Hs

def get_delta_prime_trm(v, X, G, H, IC, verbose=0):
    """
    Returns the Newton-Raphson step given a multiple of the diagonal
    added to the Hessian, the expected decrease in the energy, and
    the derivative of the step length w/r.t. v.

    Parameters
    ----------
    v : float
        Number that is added to the Hessian diagonal
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    # LPW 2020-01-24 Testing whether the image potential can be combined with NR for transition state optimization
    # Gs, Hs = image_gradient_hessian(G, H, [0])
    if IC is not None:
        GC, HC = IC.augmentGH(X, G, H) if IC.haveConstraints() else (G, H)
    else:
        GC, HC = (G, H)
    HT = HC + v*np.eye(len(HC))
    # The constrained degrees of freedom should not have anything added to diagonal
    for i in range(len(G), len(GC)):
        HT[i, i] = 0.0
    if verbose >= 2:
        seig = sorted(np.linalg.eig(HT)[0])
        logger.info("sorted(eig) : % .5e % .5e % .5e ... % .5e % .5e % .5e\n" % (seig[0], seig[1], seig[2], seig[-3], seig[-2], seig[-1]))
    try:
        Hi = invert_svd(HT)
    except:
        logger.info("\x1b[1;91mSVD Error - increasing v by 0.001 and trying again\x1b[0m\n")
        return get_delta_prime_trm(v+0.001, X, G, H, IC)
    dyc = flat(-1 * np.dot(Hi,col(GC)))
    dy = dyc[:len(G)]

    d_prime = flat(-1 * np.dot(Hi, col(dyc)))[:len(G)]
    dy_prime = np.dot(dy,d_prime)/np.linalg.norm(dy)
    # sol = flat(0.5*row(dy)*np.matrix(H)*col(dy))[0] + np.dot(dy,G)
    sol = flat(0.5*multi_dot([row(dy),H,col(dy)]))[0] + np.dot(dy,G)
    return dy, sol, dy_prime

def rfo_gen_evp(M, a):
    """
    Solve the generalized eigenvalue problem often encountered
    in restricted step RFO, given by:

    M * v = l * A * v

    where M is any matrix (always symmetric in our case),
    v and l are the desired eigenvectors,
    and A is a scaling matrix with the form:

    [[ 1  0  ]
     [ 0 a*I ]]
    """
    # Values on the diagonal of the matrix
    av = np.ones(M.shape[0])
    av[1:] = a
    # A to the -1/2 power
    Amh = np.diag(av**-0.5)
    # Form A^-1/2 * M * A^-1/2
    Mp = multi_dot([Amh, M, Amh])
    # Solve for eigenvalues and eigenvectors, which are actually v' = (A^1/2)*v
    eigvals, eigvecsp = sorted_eigh(Mp, asc=True)
    # Get the eigenvectors of the generalized EVP as v = (A^-1/2) v'
    eigvecs = np.dot(Amh, eigvecsp)
    return eigvals, eigvecs

def get_delta_prime_rs_p_rfo(alpha, X, G, H, IC, verbose=0):
    """
    Return the restricted-step rational functional optimization
    step, given a particular value of alpha. The step is given by:
    1) Solving the generalized eigenvalue problem
    [[0 G]  = lambda * [[1 0] * vec ,
     [G H]]             [0 S]]
       where the LHS matrix is called the augmented Hessian,
       and S is alpha times the identity (starting value 1.0).
    2) Dividing vec through by the 0th element, and keeping the rest
    This function also calculates the derivative of the norm of the step
    with respect to alpha, which allows trust_step() to rapidly find
    the RS-RFO step that satisfies a desired step length.

    Currently does not work with constraints, and gives equivalent performance
    to the trust radius method.

    Parameters
    ----------
    alpha : float
        Multiple of the identity in the S-matrix
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    if IC.haveConstraints():
        raise RuntimeError("Rational function optimization does not support constraints")
    # verbose = 2
    if verbose >= 4:
        logger.info("        === RS-P-RFO method with alpha = %.5f ===\n" % alpha)

    # Sorted eigenvalues and corresponding eigenvectors of the Hessian
    Hvals, Hvecs = sorted_eigh(H, asc=True)

    # Projection of gradient along the Hessian eigenvectors
    Gproj = np.dot(Hvecs.T, G)

    # Indices of the transition vector (which forms the small RFO partition) and the other vectors
    tv = 0
    ot = np.array([i for i in range(H.shape[0]) if i != tv])

    # The P-RFO matrices follow the convention of Bofill (1998)
    # with the "extra" row in the upper left.
    # Form the small P-RFO matrix, which is a 2x2 matrix
    prfo_tv = np.array([[0, Gproj[tv]], [Gproj[tv], Hvals[tv]]])
    # Form the large P-RFO matrix, which is the same size as H itself
    prfo_ot = np.zeros_like(H)
    # Fill matrix (except top left) with the non-'tv' eigenvalues
    prfo_ot[1:, 1:] = np.diag(Hvals[ot])
    # Fill the top row and left column with the non-'tv' gradient components
    prfo_ot[0, 1:] = Gproj[ot]
    prfo_ot[1:, 0] = Gproj[ot]

    # Solve the generalized eigenvector problems
    tv_vals, tv_vecs = rfo_gen_evp(prfo_tv, alpha)
    ot_vals, ot_vecs = rfo_gen_evp(prfo_ot, alpha)

    ## Form the P-RFO step.
    dy_tv = -Gproj[tv]*Hvecs[:,tv]/(Hvals[tv]-tv_vals[-1]*alpha)
    denoms = np.zeros_like(dy_tv)
    dy_coeffs = np.zeros_like(dy_tv)
    dy_coeffs[tv] = -Gproj[tv]/(Hvals[tv]-tv_vals[-1]*alpha)
    denoms[tv] = Hvals[tv]-tv_vals[-1]*alpha
    # Compute transition vector contribution to |dy|^2/d(alpha)
    dy2_prime_tv = 2 * (tv_vals[-1]/(1+alpha*np.dot(dy_tv,dy_tv))) * (Gproj[tv]**2/(Hvals[tv]-tv_vals[-1]*alpha)**3)
    # Now for the other vectors
    dy_ot = np.zeros_like(dy_tv)
    dy2_prime_ot = 0.0
    for i in ot:
        denom = Hvals[i]-ot_vals[0]*alpha
        denoms[i] = denom
        # Numerical instabilities can occur when the shift is very close to a pole,
        # so we exclude these components.
        if np.abs(denom) > 1e-5:
            dy_ot -= Gproj[i]*Hvecs[:,i]/denom
            dy_coeffs[i] = -Gproj[i]/denom
            dy2_prime_ot += Gproj[i]**2/denom**3
    dy2_prime_ot *= 2 * (ot_vals[0]/(1+alpha*np.dot(dy_ot,dy_ot)))
    # Add the transition vector and other vector contributions together
    dy = dy_tv + dy_ot
    dy2_prime = dy2_prime_tv + dy2_prime_ot
    # Derivative of the norm of the step w/r.t. alpha
    dy_prime = dy2_prime/(2*np.linalg.norm(dy))
    # For some reason, this formula from the paper suffers from some numerical problems
    # expect = (tv_vals[-1]/tv_vecs[0,-1]**2 + ot_vals[0]/ot_vecs[0,0]**2) / 2
    # Use the quadratic approximation to get expected change in the energy
    expect = flat(0.5*multi_dot([row(dy),H,col(dy)]))[0] + np.dot(dy,G)

    if verbose >= 5:
        logger.info("        Largest / smallest eigvals of small / large P-RFO matrix: % .5f % .5f\n" % (tv_vals[-1], ot_vals[0]))
        logger.info("        Small P-RFO matrix:\n        ")
        pmat2d(prfo_tv, precision=5, format='f')
        logger.info("        Eigenvalues of small P-RFO matrix:\n        ")
        pvec1d(tv_vals, precision=5, format='f')
        logger.info("        Large P-RFO matrix:\n        ")
        pmat2d(prfo_ot, precision=5, format='f')
        logger.info("        Eigenvalues of large P-RFO matrix:\n        ")
        pvec1d(ot_vals, precision=5, format='f')
        logger.info("        l_max*alpha, l_min*alpha = %.5f %.5f\n" % (tv_vals[-1]*alpha, ot_vals[0]*alpha))
        logger.info("        Numerator   of Gproj[i]/(h[i]-l*a) along each mode:\n        ")
        pvec1d(Gproj)
        logger.info("        Denominator of Gproj[i]/(h[i]-l*a) along each mode:\n        ")
        pvec1d(denoms)
        logger.info("        Coefficients of P-RFO step along each mode:\n        ")
        pvec1d(dy_coeffs)
        logger.info("        Step obtained from P-RFO method:\n        ")
        pvec1d(dy)
    elif verbose >= 4:
        logger.info("        Largest / smallest eigvals of small / large P-RFO matrix: % .5f % .5f\n" % (tv_vals[-1], ot_vals[0]))
        logger.info("        l_max*alpha(TS), l_min*alpha(min) = %.5f %.5f\n" % (tv_vals[-1]*alpha, ot_vals[0]*alpha))
        logger.info("        Coefficients of P-RFO step along normal modes:\n")
        printIdxs = list(np.argsort(np.abs(dy_coeffs))[-4:])
        if 0 not in printIdxs: printIdxs.append(0)
        for i in sorted(printIdxs):
            logger.info("          dy[%3i] = % .6f\n" % (i, dy_coeffs[i]))
        # pvec1d(dy_coeffs, precision=3, format='f')

    return dy, expect, dy_prime

def get_delta_prime_rfo(alpha, X, G, H, IC, verbose=0):
    """
    Return the restricted-step rational functional optimization
    step, given a particular value of alpha. The step is given by:
    1) Solving the generalized eigenvalue problem
    [[0 G]  = lambda * [[1 0] * vec ,
     [G H]]             [0 S]]
       where the LHS matrix is called the augmented Hessian,
       and S is alpha times the identity (starting value 1.0).
    2) Dividing vec through by the 0th element, and keeping the rest
    This function also calculates the derivative of the norm of the step
    with respect to alpha, which allows trust_step() to rapidly find
    the RS-RFO step that satisfies a desired step length.

    Currently does not work with constraints, and gives equivalent performance
    to the trust radius method.

    Parameters
    ----------
    alpha : float
        Multiple of the identity in the S-matrix
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    try:
        import scipy
    except ImportError:
        raise ImportError("RFO optimization requires scipy package. If this becomes important in the future, scipy will become a required dependency.")
    if IC.haveConstraints():
        raise RuntimeError("Still need to implement RFO with constraints")
    S = alpha*np.eye(len(H))
    # Augmented Hessian matrix
    AH = np.zeros((H.shape[0]+1, H.shape[1]+1), dtype=float)
    AH[1:, 1:] = H
    AH[0, 1:] = G
    AH[1:, 0] = G
    B = np.zeros_like(AH)
    B[0,0] = 1.0
    B[1:,1:] = S
    # Solve the generalized eigenvalue problem
    AHeig, AHvec = scipy.linalg.eigh(AH, b=B)
    lmin = AHeig[0]
    # logger.info("AH eigenvalues: %.5e %.5e %.5e ... %.5e %.5e %.5e" % (AHeig[0],AHeig[1],AHeig[2],AHeig[-3],AHeig[-2],AHeig[-1]))
    vmin = np.array(AHvec[:, 0]).flatten()
    dy = (vmin / vmin[0])[1:]
    nu = alpha*lmin
    # Now get eigenvectors of the Hessian
    Heig, Hvec = sorted_eigh(H, asc=True)
    Hvec = np.array(Hvec)
    dyprime2 = 0
    dy2 = 0
    for i in range(H.shape[0]):
        dyprime2 += np.dot(Hvec[:,i].T,G)**2/(Heig[i]-nu)**3
        dy2 += np.dot(Hvec[:,i].T,G)**2/(Heig[i]-nu)**2
    dyprime2 *= (2*lmin)/(1+alpha*np.dot(dy,dy))
    expect = lmin/2*(1+multi_dot([row(dy),S,col(dy)]))[0]
    dyprime1 = dyprime2 / (2*np.sqrt(dy2))
    return dy, expect, dyprime1

def get_delta_prime(v, X, G, H, IC, rfo, verbose=0):
    """
    Return the internal coordinate step given a parameter "v".
    "v" refers to the multiple of the identity added to the Hessian
    in trust-radius Newton Raphson (TRM), and the multiple of the
    identity on the RHS matrix in rational function optimization (RFO).
    Note that reasonable default values are v = 0.0 in TRM and 1.0 in RFO.

    Parameters
    ----------
    v : float
        Number that is added to the Hessian diagonal
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    rfo : bool
        If True, use rational functional optimization, otherwise use trust-radius method
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    if rfo:
        method="Restricted step P-RFO"
        dy, expect, dy_prime = get_delta_prime_rs_p_rfo(v, X, G, H, IC, verbose)
    else:
        method="Trust radius Newton-Raphson"
        dy, expect, dy_prime = get_delta_prime_trm(v, X, G, H, IC, verbose)
    if verbose >= 2: logger.info("      %s step with v = %10.5f : expect-deltaE = %.5f, Internal-step = %.5f\n" % (method, v, expect, np.linalg.norm(dy)))
    return dy, expect, dy_prime

def trust_step(target, v0, X, G, H, IC, rfo, verbose=0):
    """
    Apply an iteration formula to find the trust radius step,
    given the target value of the trust radius.

    Parameters
    ----------
    target : float
        Target size of the trust radius step
    v0 : float
        Initial guess for Number that is added to the Hessian diagonal
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    rfo : bool
        If True, use rational functional optimization, otherwise use trust-radius method
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step with the desired size
    sol : float
        Expected change of the objective function
    """
    if verbose >= 2: logger.info("    trust_step targeting internal coordinate step of length %.4f\n" % target)
    dy, sol, dy_prime = get_delta_prime(v0, X, G, H, IC, rfo, verbose)
    # dy_plus, _, __ = get_delta_prime(v0+0.001, X, G, H, IC, rfo, verbose)
    # print("LPW debug:", (np.linalg.norm(dy_plus)-np.linalg.norm(dy))/0.001, dy_prime)

    ndy = np.linalg.norm(dy)
    if ndy < target:
        return dy, sol
    v = v0
    niter = 0
    ndy_last = 0
    # Store the minimum norm in case we give up
    m_ndy = ndy
    m_dy = dy.copy()
    m_sol = sol
    while True:
        # LPW 2020-01-24 : I think this was a bug, but since it was in the code for 3 years,
        # we need to keep our eyes open
        if rfo:
            # Nonlinear Newton's method solution, from Bofill (1998)
            v += (target/ndy-1)*(ndy/dy_prime) 
        else:
            # Iterative formula from Hebden (1973), equation 5.2.10 in "Practical methods of optimization" by Fletcher
            v += (1-ndy/target)*(ndy/dy_prime)
        dy, sol, dy_prime, = get_delta_prime(v, X, G, H, IC, rfo, verbose)
        ndy = np.linalg.norm(dy)
        if np.abs((ndy-target)/target) < 0.001:
            if verbose >= 3: get_delta_prime(v, X, G, H, IC, rfo, verbose+1)
            if verbose: logger.info("    trust_step Iter:  %4i, v = %.5f, dy on target:   %.5f ---> %.5f\n" % (niter, v, ndy, target))
            return dy, sol
        # With Lagrange multipliers it may be impossible to go under a target step size
        elif niter > 10 and np.abs(ndy_last-ndy)/ndy < 0.001:
            if verbose >= 3: get_delta_prime(v, X, G, H, IC, rfo, verbose+1)
            if verbose: logger.info("    trust_step Iter:  %4i, v = %.5f, dy over target: %.5f -x-> %.5f\n" % (niter, v, ndy, target))
            return dy, sol
        elif verbose >= 2:
            logger.info("    trust_step Iter:  %4i, v = %.5f, dy -> target:   %.5f ---> %.5f\n" % (niter, v, ndy, target))
        niter += 1
        ndy_last = ndy
        if ndy < m_ndy:
            m_ndy = ndy
            m_dy = dy.copy()
            m_sol = sol
        # Break out of infinite oscillation loops
        if niter%100 == 99:
            logger.info("    trust_step Iter:  %4i, randomizing\n" % niter)
            v += np.random.random() * niter / 100
        if niter%1000 == 999:
            if verbose >= 3: get_delta_prime(v, X, G, H, IC, rfo, verbose+1)
            if verbose: logger.info("    trust_step Iter:  %4i, v = %.5f, dy at max-iter: %.5f -x-> %.5f\n" % (niter, v, ndy, target))
            return m_dy, m_sol
        
class Froot(object):
    """
    Object describing a function of the internal coordinate step
    length, which returns the Cartesian coordinate step length minus
    the trust radius.

    This is an object instead of a function mainly because we want the
    Brent root-finding method to read and write extra attributes of
    this function and not just its value, for example: Did we converge
    to a root? Under what conditions are we allowed to exit the algorithm?
    """
    def __init__(self, trust, v0, X, G, H, IC, params):
        self.counter = 0
        self.stores = {}
        self.trust = trust
        self.target = trust
        self.above_flag = False
        self.stored_arg = None
        self.stored_val = None
        self.brentFailed = False
        self.params = params
        self.v0 = v0
        self.X = X
        self.G = G
        self.H = H
        self.IC = IC

    def evaluate(self, trial):
        """
        This is a one-argument "function" that is called by brent_wiki which takes
        an internal coordinate step length as input, and returns the Cartesian coordinate
        step length (minus the target) as output.
        """
        v0 = self.v0
        X = self.X
        G = self.G
        H = self.H
        IC = self.IC
        trust = self.trust
        if trial == 0.0:
            self.from_above = False
            return -trust
        else:
            if trial in self.stores:
                cnorm = self.stores[trial]
                self.from_above = False
            else:
                dy, expect = trust_step(trial, v0, X, G, H, IC, self.params.transition, self.params.verbose)
                cnorm = getCartesianNorm(X, dy, IC, self.params.enforce, self.params.verbose)
                # Early "convergence"; this signals whether we have found a valid step that is
                # above the current target, but below the original trust radius. This happens
                # when the original trust radius fails, and we reduce the target step-length
                # as a contingency
                self.from_above = (self.above_flag and not IC.bork and cnorm < trust)
                self.stores[trial] = cnorm
                self.counter += 1
            # Store the largest trial value with cnorm below the target
            if cnorm-self.target < 0:
                if self.stored_val is None or cnorm > self.stored_val:
                    self.stored_arg = trial
                    self.stored_val = cnorm
            if self.params.verbose: logger.info("  Brent Iter: %i Internal-step: %.4f Cartesian-step: %.4f ---> Trust-Radius: %.4f%s\n" % (self.counter, trial, cnorm, self.target, " (done)" if self.from_above else ""))
            return cnorm-self.target

class OptParams(object):
    """
    Container for optimization parameters.
    The parameters used to be contained in the command-line "args",
    but this was dropped in order to call Optimize() from another script.
    """
    def __init__(self, **kwargs):
        # Whether we are optimizing for a transition state. This changes a number of default parameters.
        self.transition = kwargs.get('transition', False)
        # Handle convergence criteria; this edits the kwargs
        self.convergence_criteria(**kwargs)
        # Threshold (in a.u. / rad) for activating alternative algorithm that enforces precise constraint satisfaction
        self.enforce = kwargs.get('enforce', 0.0)
        # Small eigenvalue threshold
        self.epsilon = kwargs.get('epsilon', 1e-5)
        # Interval for checking the coordinate system for changes
        self.check = kwargs.get('check', 0)
        # More verbose printout
        self.verbose = kwargs.get('verbose', False)
        # Rational function optimization (experimental)
        self.rfo = kwargs.get('rfo', False)
        # Starting value of the trust radius
        self.trust = kwargs.get('trust', 0.1)
        # Maximum value of trust radius
        self.tmax = kwargs.get('tmax', 0.3)
        self.trust = min(self.tmax, self.trust)
        # Maximum number of optimization cycles
        self.maxiter = kwargs.get('maxiter', 300)
        # Use updated constraint algorithm implemented 2019-03-20
        self.conmethod = kwargs.get('conmethod', 0)
        # Write Hessian matrix at optimized structure to text file
        self.write_cart_hess = kwargs.get('write_cart_hess', None)
        # CI optimizations sometimes require tiny steps
        self.meci = kwargs.get('meci', False)
        # Output .xyz file name may be set separately in
        # run_optimizer() prior to calling Optimize().
        self.xyzout = kwargs.get('xyzout', None)
        # Name of the qdata.txt file to be written.
        # The CLI is designed so the user passes true/false instead of the file name.
        self.qdata = 'qdata.txt' if kwargs.get('qdata', False) else None
        # Whether to calculate or read a Hessian matrix.
        self.hessian = kwargs.get('hessian', None)
        if self.hessian is None:
            # Default is to calculate Hessian in the first step if searching for a transition state.
            # Otherwise the default is to never calculate the Hessian.
            if self.transition: self.hessian = 'first'
            else: self.hessian = 'never'
        if self.hessian.startswith('file:'):
            if os.path.exists(self.hessian[5:]):
                # If a path is provided for reading a Hessian file, read it now.
                self.hess_data = np.loadtxt(self.hessian[5:])
            else:
                raise IOError("No Hessian data file found at %s" % self.hessian)
        elif self.hessian.lower() in ['never', 'first', 'each', 'exit']:
            self.hessian = self.hessian.lower()
        else:
            raise RuntimeError("Hessian command line argument can only be never, first, each, exit, or file:<path>")
        # Reset Hessian to guess whenever eigenvalues drop below epsilon
        self.reset = kwargs.get('reset', None)
        if self.reset is None: self.reset = not (self.transition or self.hessian == 'each')

    def convergence_criteria(self, **kwargs):
        criteria = kwargs.get('converge', [])
        if len(criteria)%2 != 0:
            raise RuntimeError('Please pass an even number of options to --converge')
        for i in range(int(len(criteria)/2)):
            key = 'convergence_' + criteria[2*i].lower()
            try:
                val = float(criteria[2*i+1])
                logger.info('Using convergence criteria: %s %.2e\n' % (key, val))
            except ValueError:
                # This must be a set
                val = str(criteria[2*i+1])
                logger.info('Using convergence criteria set: %s %s\n' % (key, val))
            kwargs[key] = val
        # convergence dictionary to store criteria stored in order of energy, grms, gmax, drms, dmax
        # 'GAU' contains the default convergence criteria that are used when nothing is passed.
        convergence_sets = {'GAU': [1e-6, 3e-4, 4.5e-4, 1.2e-3, 1.8e-3],
                            'NWCHEM_LOOSE': [1e-6, 3e-3, 4.5e-3, 3.6e-3, 5.4e-3],
                            'GAU_LOOSE': [1e-6, 1.7e-3, 2.5e-3, 6.7e-3, 1e-2],
                            'TURBOMOLE': [1e-6, 5e-4, 1e-3, 5.0e-4, 1e-3],
                            'INTERFRAG_TIGHT': [1e-6, 1e-5, 1.5e-5, 4.0e-4, 6.0e-4],
                            'GAU_TIGHT': [1e-6, 1e-5, 1.5e-5, 4e-5, 6e-5],
                            'GAU_VERYTIGHT': [1e-6, 1e-6, 2e-6, 4e-6, 6e-6]}
        # Q-Chem style convergence criteria (i.e. gradient and either energy or displacement)
        self.qccnv = kwargs.get('qccnv', False)
        # Molpro style convergence criteria (i.e. gradient and either energy or displacement, with different defaults)
        self.molcnv = kwargs.get('molcnv', False)
        # Check if there is a convergence set passed else use the default
        set_name = kwargs.get('convergence_set', 'GAU').upper()
        # If we have extra keywords apply them here else use the set
        # Convergence criteria in a.u. and Angstrom
        self.Convergence_energy = kwargs.get('convergence_energy', convergence_sets[set_name][0])
        self.Convergence_grms = kwargs.get('convergence_grms', convergence_sets[set_name][1])
        self.Convergence_gmax = kwargs.get('convergence_gmax', convergence_sets[set_name][2])
        self.Convergence_drms = kwargs.get('convergence_drms', convergence_sets[set_name][3])
        self.Convergence_dmax = kwargs.get('convergence_dmax', convergence_sets[set_name][4])
        # Convergence criteria that are only used if molconv is set to True
        self.Convergence_molpro_gmax = kwargs.get('convergence_molpro_gmax', 3e-4)
        self.Convergence_molpro_dmax = kwargs.get('convergence_molpro_dmax', 1.2e-3)

    def printInfo(self):
        if self.transition:
            logger.info(' Transition state optimization requested.\n')
        if self.hessian == 'first':
            logger.info(' Hessian will be computed on the first step.\n')
        elif self.hessian == 'each':
            logger.info(' Hessian will be computed for each step.\n')
        elif self.hessian == 'exit':
            logger.info(' Hessian will be computed for first step, then program will exit.\n')
        elif self.hessian.startswith('file:'):
            logger.info(' Hessian data will be read from file: %s\n' % self.hessian[5:])

class OPT_STATE(object):
    """ This describes the state of an OptObject during the optimization process
    """
    NEEDS_EVALUATION = 0  # convergence has not been evaluated -> calcualte Energy, Forces
    SKIP_EVALUATION  = 1  # We know this is not yet converged -> skip Energy
    CONVERGED        = 2
    FAILED           = 3  # optimization failed with no recovery option

class Optimizer(object):
    def __init__(self, coords, molecule, IC, engine, dirname, params):
        """
        Object representing the geometry optimization of a molecular system.

        Parameters
        ----------
        coords : np.ndarray
            Nx3 array of Cartesian coordinates in atomic units
        molecule : Molecule
            Molecule object (Units Angstrom)
        IC : InternalCoordinates
            Object describing the internal coordinate system
        engine : Engine
            Object containing methods for calculating energy and gradient
        dirname : str
            Directory name for files to be written
        params : OptParams object
            Contains optimization parameters (really just a struct)
            Includes xyzout and qdata output file names (written if not None)
            
        """
        # Copies of data passed into constructor
        self.coords = coords
        self.molecule = deepcopy(molecule)
        self.IC = IC
        self.engine = engine
        self.dirname = dirname
        self.params = params
        # Threshold for "low quality step" which decreases trust radius.
        self.ThreLQ = 0.25
        # Threshold for "high quality step" which increases trust radius.
        self.ThreHQ = 0.75
        # If the trust radius is lower than this number, do not reject steps.
        if self.params.meci:
            self.thre_rj = 1e-4
        else:
            self.thre_rj = 1e-2
        # Set initial value of the trust radius.
        self.trust = self.params.trust
        # Copies of molecule object for preserving the optimization trajectory and the last frame
        self.progress = deepcopy(self.molecule)
        self.progress.xyzs = []
        self.progress.qm_energies = []
        self.progress.qm_grads = []
        self.progress.comms = []
        # Cartesian coordinates
        self.X = self.coords.copy()
        # Loop of optimization
        self.Iteration = 0
        # Counts how many steps it has been since checking the coordinate system
        self.CoordCounter = 0
        # Current state, used to control logic of optimization loop.
        self.state = OPT_STATE.NEEDS_EVALUATION
        # Some more variables to be updated throughout the course of the optimization
        self.trustprint = "="
        self.ForceRebuild = False

    def getCartesianNorm(self, dy, verbose=None):
        if not verbose: verbose = self.params.verbose
        return getCartesianNorm(self.X, dy, self.IC, self.params.enforce, self.params.verbose)

    def get_delta_prime(self, v0, verbose=None):
        # This method can be called at a different verbose level than the master
        # because it can occur inside a nested loop
        if not verbose: verbose = self.params.verbose
        return get_delta_prime(v0, self.X, self.G, self.H, self.IC, self.params.transition, verbose)

    def createFroot(self, v0):
        return Froot(self.trust, v0, self.X, self.G, self.H, self.IC, self.params)

    def refreshCoordinates(self):
        """
        Refresh the Cartesian coordinates used to define parts of the internal coordinate system.
        These include definitions of delocalized internal coordinates and reference coordinates for rotators.
        """
        self.IC.resetRotations(self.X)
        if isinstance(self.IC, DelocalizedInternalCoordinates):
            self.IC.build_dlc(self.X)
        # With redefined internal coordinates, the Hessian needs to be rebuilt
        self.H0 = self.IC.guess_hessian(self.coords)
        self.RebuildHessian()
        # Current values of internal coordinates and IC gradient are recalculated
        self.Y = self.IC.calculate(self.X)
        self.G = self.IC.calcGrad(self.X, self.gradx)

    def checkCoordinateSystem(self, recover=False, cartesian=False):
        """
        Build a new internal coordinate system from current Cartesians and replace the current one if different.
        """
        # Reset the check counter
        self.CoordCounter = 0
        # Build a new molecule object and connectivity graph
        newmol = deepcopy(self.molecule)
        newmol.xyzs[0] = self.X.reshape(-1,3) * bohr2ang
        newmol.build_topology()
        # Build the new internal coordinate system
        if cartesian:
            if self.IC.haveConstraints():
                raise ValueError("Cannot continue a constrained optimization; please implement constrained optimization in Cartesian coordinates")
            IC1 = CartesianCoordinates(newmol)
        else:
            IC1 = self.IC.__class__(newmol, connect=self.IC.connect, addcart=self.IC.addcart, build=False, conmethod=self.IC.conmethod)
            if self.IC.haveConstraints(): IC1.getConstraints_from(self.IC)
        # Check for differences
        changed = (IC1 != self.IC)
        if changed:
            logger.info("\x1b[1;94mInternal coordinate system may have changed\x1b[0m\n")
            if self.IC.repr_diff(IC1) != "":
                logger.info(self.IC.repr_diff(IC1)+'\n')
        # Set current ICs to the new one
        if changed or recover or cartesian:
            self.IC = IC1
            self.refreshCoordinates()
            return True
        else: return False

    def trust_step(self, iopt, v0, verbose=None):
        # This method can be called at a different verbose level than the master
        # because it can occur inside a nested loop
        if not verbose: verbose = self.params.verbose
        return trust_step(iopt, v0, self.X, self.G, self.H, self.IC, self.params.transition, verbose)

    def newCartesian(self, dy):
        if self.IC.haveConstraints() and self.params.enforce:
            self.X = self.IC.newCartesian_withConstraint(self.X, dy, thre=self.params.enforce, verbose=self.params.verbose)
        else:
            self.X = self.IC.newCartesian(self.X, dy, self.params.verbose)

    def calcGradNorm(self):
        gradxc = self.IC.calcGradProj(self.X, self.gradx) if self.IC.haveConstraints() else self.gradx.copy()
        atomgrad = np.sqrt(np.sum((gradxc.reshape(-1,3))**2, axis=1))
        rms_gradient = np.sqrt(np.mean(atomgrad**2))
        max_gradient = np.max(atomgrad)
        return rms_gradient, max_gradient

    def RebuildHessian(self):
        self.H = RebuildHessian(self.IC, self.H0, self.X_hist, self.Gx_hist, self.params)

    def calcEnergyForce(self):
        """
        Calculate the energy and Cartesian gradients of the current structure.
        """
        ### Calculate Energy and Gradient ###
        # Dictionary containing single point properties (energy, gradient)
        spcalc = self.engine.calc(self.X, self.dirname)
        self.E = spcalc['energy']
        self.gradx = spcalc['gradient']
        # Calculate Hessian at the first step, or at each step if desired
        if self.params.hessian == 'each':
            # Hx is assumed to be the Cartesian Hessian at the current step.
            # Otherwise we use the variable name Hx0 to avoid almost certain confusion.
            self.Hx = CartesianHessian(self.X, self.molecule, self.engine, self.dirname, readfiles=True, verbose=self.params.verbose)
        elif self.Iteration == 0:
            if self.params.hessian in ['first', 'exit']:
                self.Hx0 = CartesianHessian(self.X, self.molecule, self.engine, self.dirname, readfiles=True, verbose=self.params.verbose)
                if self.params.hessian == 'exit':
                    logger.info("Exiting as requested after Hessian calculation.\n")
                    sys.exit(0)
            elif hasattr(self.params, 'hess_data') and self.Iteration == 0:
                self.Hx0 = self.params.hess_data.copy()
                if self.Hx0.shape != (self.X.shape[0], self.X.shape[0]):
                    raise IOError('hess_data passed in via OptParams does not have the right shape')
            # self.Hx = self.Hx0.copy()
        # Add new Cartesian coordinates, energies, and gradients to history
        self.progress.xyzs.append(self.X.reshape(-1,3) * bohr2ang)
        self.progress.qm_energies.append(self.E)
        self.progress.qm_grads.append(self.gradx.copy())
        self.progress.comms.append('Iteration %i Energy % .8f' % (self.Iteration, self.E))

    def prepareFirstStep(self):
        """
        After computing the initial set of energies and forces, carry out some preparatory tasks
        prior to entering the optimization loop.
        """
        # Initial internal coordinates (optimization variables) and internal gradient
        self.Y = self.IC.calculate(self.coords)
        self.G = self.IC.calcGrad(self.X, self.gradx).flatten()
        # Print initial iteration
        rms_gradient, max_gradient = self.calcGradNorm()
        msg = "Step %4i :" % self.Iteration
        logger.info(msg + " Gradient = %.3e/%.3e (rms/max) Energy = % .10f\n" % (rms_gradient, max_gradient, self.E))
        # Initial history
        self.X_hist = [self.X]
        self.Gx_hist = [self.gradx]
        # Initial Hessian
        if hasattr(self, 'Hx'):
            # Compute IC Hessian from Cartesian Hessian at the current step
            self.H0 = self.IC.calcHess(self.X, self.gradx, self.Hx)
        elif hasattr(self, 'Hx0'):
            # Compute IC Hessian from input Cartesian Hessian
            self.H0 = self.IC.calcHess(self.X, self.gradx, self.Hx0)
        else:
            # Form guess Hessian if initial Hessian is not provided
            self.H0 = self.IC.guess_hessian(self.coords)
        self.H = self.H0.copy()

    def SortedEigenvalues(self):
        Eig = sorted(np.linalg.eigh(self.H)[0])
        if self.params.transition and len(Eig) >= 12:
            # logger.info("Hessian Eigenvalues:  %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e ... %.3e %.3e %.3e\n" %
            #             (Eig[0],Eig[1],Eig[2],Eig[3],Eig[4],Eig[5],Eig[6],Eig[7],Eig[8],Eig[-3],Eig[-2],Eig[-1]))
            logger.info("Hessian Eigenvalues:  % .3e % .3e % .3e % .3e % .3e % .3e % .3e\n" % (Eig[0],Eig[1],Eig[2],Eig[3],Eig[4],Eig[5],Eig[6])),
            logger.info("% .3e % .3e % .3e % .3e % .3e    .....   % .3e % .3e % .3e\n" % (Eig[7],Eig[8],Eig[9],Eig[10],Eig[11],Eig[-3],Eig[-2],Eig[-1])),
        elif len(Eig) >= 6:
            logger.info("Hessian Eigenvalues: %.5e %.5e %.5e ... %.5e %.5e %.5e\n" % (Eig[0],Eig[1],Eig[2],Eig[-3],Eig[-2],Eig[-1]))
        else:
            logger.info("Hessian Eigenvalues: " + ' '.join("%.5e" % i for i in Eig) + '\n')
        return Eig
        
    def step(self):
        """
        Perform one step of the optimization.
        """
        params = self.params
        if np.isnan(self.G).any():
            raise RuntimeError("Gradient contains nan - check output and temp-files for possible errors")
        if np.isnan(self.H).any():
            raise RuntimeError("Hessian contains nan - check output and temp-files for possible errors")
        self.Iteration += 1
        if (self.Iteration%5) == 0:
            self.engine.clearCalcs()
            self.IC.clearCache()

        # At the start of the loop, the optimization variables, function value, gradient and Hessian are known.
        # (i.e. self.Y, self.E, self.G, self.H)
        if params.verbose: self.IC.Prims.printRotations(self.X)
        Eig = self.SortedEigenvalues()
        Emin = Eig[1].real
        if params.transition:
            v0 = 1.0
        elif Emin < params.epsilon:
            v0 = params.epsilon-Emin
        else:
            v0 = 0.0
        # Are we far from constraint satisfaction?
        self.farConstraints = self.IC.haveConstraints() and self.IC.getConstraintViolation(self.X) > 1e-1
        ### OBTAIN AN OPTIMIZATION STEP ###
        # The trust radius is to be computed in Cartesian coordinates.
        # First take a full-size optimization step
        if params.verbose: logger.info("  Optimizer.step : Attempting full-size optimization step\n")
        dy, _, __ = self.get_delta_prime(v0, verbose=self.params.verbose)
        # Internal coordinate step size
        inorm = np.linalg.norm(dy)
        # Cartesian coordinate step size
        self.cnorm = self.getCartesianNorm(dy)
        # If the full-size step is within the trust radius, then call get_delta_prime again with diagnostic messages if needed
        if (self.params.verbose >= 2 and self.params.verbose < 4 and self.cnorm <= 1.1*self.trust):
            self.get_delta_prime(v0, verbose=self.params.verbose+2)
        if params.verbose: logger.info("  Optimizer.step : Internal-step: %.4f Cartesian-step: %.4f Trust-radius: %.4f\n" % (inorm, self.cnorm, self.trust))
        # If the step is above the trust radius in Cartesian coordinates, then
        # do the following to reduce the step length:
        if self.cnorm > 1.1 * self.trust:
            # This is the function f(inorm) = cnorm-target that we find a root
            # for obtaining a step with the desired Cartesian step size.
            froot = self.createFroot(v0)
            froot.stores[inorm] = self.cnorm
            ### Find the internal coordinate norm that matches the desired Cartesian coordinate norm
            if params.verbose: logger.info("  Optimizer.step : Using Brent algorithm to target Cartesian trust radius\n")
            iopt = brent_wiki(froot.evaluate, 0.0, inorm, self.trust, cvg=0.1, obj=froot, verbose=params.verbose)
            if froot.brentFailed and froot.stored_arg is not None:
                # If Brent fails but we obtained an IC step that is smaller than the Cartesian trust radius, use it
                if params.verbose: logger.info("  Optimizer.step : \x1b[93mUsing stored solution at %.3e\x1b[0m\n" % froot.stored_val)
                iopt = froot.stored_arg
            elif self.IC.bork:
                # Decrease the target Cartesian step size and try again
                for i in range(3):
                    froot.target /= 2
                    if params.verbose: logger.info("  Optimizer.step : \x1b[93mReducing target to %.3e\x1b[0m\n" % froot.target)
                    froot.above_flag = True # Stop at any valid step between current target step size and trust radius
                    iopt = brent_wiki(froot.evaluate, 0.0, iopt, froot.target, cvg=0.1, verbose=params.verbose)
                    if not self.IC.bork: break
            LastForce = self.ForceRebuild
            self.ForceRebuild = False
            if self.IC.bork:
                logger.info("\x1b[91mInverse iteration for Cartesians failed\x1b[0m\n")
                # This variable is added because IC.bork is unset later.
                self.ForceRebuild = True
            else:
                if params.verbose: logger.info("  Optimizer.step : \x1b[93mBrent algorithm requires %i evaluations\x1b[0m\n" % froot.counter)
            ##### If IC failed to produce valid Cartesian step, it is "borked" and we need to rebuild it.
            if self.ForceRebuild:
                # Force a rebuild of the coordinate system and skip the energy / gradient and evaluation steps.
                if LastForce:
                    logger.warning("\x1b[1;91mFailed twice in a row to rebuild the coordinate system; continuing in Cartesian coordinates\x1b[0m\n")
                self.checkCoordinateSystem(recover=True, cartesian=LastForce)
                logger.info("\x1b[1;93mSkipping optimization step\x1b[0m\n")
                self.Iteration -= 1
                self.state = OPT_STATE.SKIP_EVALUATION
                return
            ##### End Rebuild
            # Finally, take an internal coordinate step of the desired length.
            dy, _ = self.trust_step(iopt, v0, verbose=(self.params.verbose+1 if self.params.verbose >= 2 else 0))
            self.cnorm = self.getCartesianNorm(dy)
        ### DONE OBTAINING THE STEP ###
        if isinstance(self.IC, PrimitiveInternalCoordinates):
            idx = np.argmax(np.abs(dy))
            iunit = np.zeros_like(dy)
            iunit[idx] = 1.0
            self.prim_msg = "Along %s %.3f" % (self.IC.Internals[idx], np.dot(dy/np.linalg.norm(dy), iunit))
        ### These quantities, computed previously, are no longer used.
        # Dot product of the gradient with the step direction
        # Dot = -np.dot(dy/np.linalg.norm(dy), self.G/np.linalg.norm(self.G))
        # Whether the Cartesian norm comes close to the trust radius
        # bump = cnorm > 0.8 * self.trust
        ### Before updating any of our variables, copy current variables to "previous"
        self.Yprev = self.Y.copy()
        self.Xprev = self.X.copy()
        self.Gxprev = self.gradx.copy()
        self.Gprev = self.G.copy()
        self.Eprev = self.E
        ### Update the Internal Coordinates ###
        X0 = self.X.copy()
        self.newCartesian(dy)
        ## The "actual" dy may be different from the one passed to newCartesian(),
        ## for example if we enforce constraints or don't get the step we expect.
        dy = self.IC.calcDiff(self.X, X0)
        self.Y += dy
        self.expect = flat(0.5*multi_dot([row(dy),self.H,col(dy)]))[0] + np.dot(dy,self.G)
        self.state = OPT_STATE.NEEDS_EVALUATION

    def evaluateStep(self):
        ### At this point, the state should be NEEDS_EVALUATION
        assert self.state == OPT_STATE.NEEDS_EVALUATION
        # Shorthand for self.params
        params = self.params
        # Write current optimization trajectory to file
        if self.params.xyzout is not None: self.progress.write(self.params.xyzout)
        if self.params.qdata is not None: self.progress.write(self.params.qdata, ftype='qdata')
        # Project out the degrees of freedom that are constrained
        rms_gradient, max_gradient = self.calcGradNorm()
        rms_displacement, max_displacement = calc_drms_dmax(self.X, self.Xprev)
        # The ratio of the actual energy change to the expected change
        Quality = (self.E-self.Eprev)/self.expect
        Converged_energy = np.abs(self.E-self.Eprev) < params.Convergence_energy
        Converged_grms = rms_gradient < params.Convergence_grms
        Converged_gmax = max_gradient < params.Convergence_gmax
        Converged_drms = rms_displacement < params.Convergence_drms
        Converged_dmax = max_displacement < params.Convergence_dmax
        BadStep = Quality < 0
        # Molpro defaults for convergence
        Converged_molpro_gmax = max_gradient < params.Convergence_molpro_gmax
        Converged_molpro_dmax = max_displacement < params.Convergence_molpro_dmax
        self.conSatisfied = not self.IC.haveConstraints() or self.IC.getConstraintViolation(self.X) < 1e-2
        # Print status
        msg = "Step %4i :" % self.Iteration
        msg += " Displace = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("\x1b[92m" if Converged_drms else "\x1b[0m", rms_displacement, "\x1b[92m" if Converged_dmax else "\x1b[0m", max_displacement)
        msg += " Trust = %.3e (%s)" % (self.trust, self.trustprint)
        msg += " Grad%s = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("_T" if self.IC.haveConstraints() else "", "\x1b[92m" if Converged_grms else "\x1b[0m", rms_gradient, "\x1b[92m" if Converged_gmax else "\x1b[0m", max_gradient)
        # print "Dy.G = %.3f" % Dot,
        logger.info(msg + " E (change) = % .10f (%s%+.3e\x1b[0m) Quality = %s%.3f\x1b[0m" % (self.E, "\x1b[91m" if BadStep else ("\x1b[92m" if Converged_energy else "\x1b[0m"), self.E-self.Eprev, "\x1b[91m" if BadStep else "\x1b[0m", Quality) + "\n")

        if self.IC is not None and self.IC.haveConstraints():
            self.IC.printConstraints(self.X, thre=1e-3)
        if isinstance(self.IC, PrimitiveInternalCoordinates):
            logger.info(self.prim_msg + '\n')

        ### Check convergence criteria ###
        if Converged_energy and Converged_grms and Converged_drms and Converged_gmax and Converged_dmax and self.conSatisfied:
            self.SortedEigenvalues()
            logger.info("Converged! =D\n")
            self.state = OPT_STATE.CONVERGED
            return

        if self.Iteration > params.maxiter:
            self.SortedEigenvalues()
            logger.info("Maximum iterations reached (%i); increase --maxiter for more\n" % params.maxiter)
            self.state = OPT_STATE.FAILED
            return

        if params.qccnv and Converged_grms and (Converged_drms or Converged_energy) and self.conSatisfied:
            self.SortedEigenvalues()
            logger.info("Converged! (Q-Chem style criteria requires grms and either drms or energy)\n")
            self.state = OPT_STATE.CONVERGED
            return

        if params.molcnv and Converged_molpro_gmax and (Converged_molpro_dmax or Converged_energy) and self.conSatisfied:
            self.SortedEigenvalues()
            logger.info("Converged! (Molpro style criteria requires gmax and either dmax or energy)\nThis is approximate since convergence checks are done in cartesian coordinates.\n")
            self.state = OPT_STATE.CONVERGED
            return

        assert self.state == OPT_STATE.NEEDS_EVALUATION
        ### Adjust Trust Radius and/or Reject Step ###
        if self.params.transition:
            down_trust = (Quality <= self.ThreLQ or Quality >= 2.0-self.ThreLQ)
            up_trust = (not down_trust and self.cnorm > 0.8*self.trust)
            reject_step = (Quality <= 0.0 or Quality >= 2.0) and (self.trust > self.thre_rj)
        else:
            # Define some conditions under which a step should not be rejected
            # If the trust radius is under thre_rj then do not reject.
            # This code rejects steps / reduces trust radius only if we're close to satisfying constraints;
            # it improved performance in some cases but worsened for others.
            rejectOk = (self.trust > self.thre_rj and (self.E > self.Eprev) and (Quality < -10 or not self.farConstraints))
            if self.farConstraints: rejectOk = False
            down_trust = (Quality <= self.ThreLQ)
            up_trust = (Quality >= self.ThreHQ)
            reject_step = (Quality <= -1.0) and rejectOk
        if Quality <= self.ThreLQ:
            # For bad steps, the trust radius is reduced
            if down_trust:
                self.trust = max(0.0 if params.meci else min(1.2e-3, params.Convergence_drms), self.trust/2)
                self.trustprint = "\x1b[91m-\x1b[0m"
            else:
                self.trustprint = "="
        elif up_trust:
            if self.trust < params.tmax:
                # For good steps, the trust radius is increased
                self.trust = min(np.sqrt(2)*self.trust, params.tmax)
                self.trustprint = "\x1b[92m+\x1b[0m"
            else:
                self.trustprint = "="
        else:
            self.trustprint = "="
        if reject_step:
            # Reject the step and take a smaller one from the previous iteration
            self.trust = max(0.0 if params.meci else min(1.2e-3, params.Convergence_drms), min(self.trust, self.cnorm/2))
            self.trustprint = "\x1b[1;91mx\x1b[0m"
            self.Y = self.Yprev.copy()
            self.X = self.Xprev.copy()
            self.gradx = self.Gxprev.copy()
            self.G = self.Gprev.copy()
            self.E = self.Eprev
            return

        # Steps that are bad, but are very small (under thre_rj) are not rejected.
        # This is because some systems (e.g. formate) have discontinuities on the
        # potential surface that can cause an infinite loop
        if Quality < -1:
            if self.trust < self.thre_rj: logger.info("\x1b[93mNot rejecting step - trust below %.3e\x1b[0m\n" % self.thre_rj)
            elif self.E < self.Eprev: logger.info("\x1b[93mNot rejecting step - energy decreases\x1b[0m\n")
            elif self.farConstraints: logger.info("\x1b[93mNot rejecting step - far from constraint satisfaction\x1b[0m\n")

        # Append steps to history (for rebuilding Hessian)
        self.X_hist.append(self.X)
        self.Gx_hist.append(self.gradx)

        ### Rebuild Coordinate System if Necessary ###
        UpdateHessian = (not self.params.hessian == 'each')
        if self.IC.bork:
            logger.info("Failed inverse iteration - checking coordinate system\n")
            self.checkCoordinateSystem(recover=True)
            UpdateHessian = False
        elif self.CoordCounter == (params.check - 1):
            logger.info("Checking coordinate system as requested every %i cycles\n" % params.check)
            if self.checkCoordinateSystem(): UpdateHessian = False
        else:
            self.CoordCounter += 1
        if self.IC.largeRots():
            logger.info("Large rotations - refreshing Rotator reference points and DLC vectors\n")
            self.refreshCoordinates()
            UpdateHessian = False
        self.G = self.IC.calcGrad(self.X, self.gradx).flatten()

        ### Update the Hessian ###
        if UpdateHessian:
            self.UpdateHessian()
        if hasattr(self, 'Hx'):
            self.H = self.IC.calcHess(self.X, self.gradx, self.Hx)
        # Then it's on to the next loop iteration!
        return

    def UpdateHessian(self):
        params = self.params

        if params.transition:
            cart_up = False
            if cart_up:
                Dx = col(self.X - self.Xprev)
                Dg = col(self.gradx - self.Gxprev)
                # Murtagh-Sargent-Powell update
                Xi = Dg - np.dot(self.Hx,Dx)
                dH_MS = np.dot(Xi, Xi.T)/np.dot(Dx.T, Xi)
                dH_P = np.dot(Xi, Dx.T) + np.dot(Dx, Xi.T) - np.dot(Dx, Dx.T)*np.dot(Xi.T, Dx)/np.dot(Dx.T, Dx)
                dH_P /= np.dot(Dx.T, Dx)
                phi = 1.0 - np.dot(Dx.T,Xi)**2/(np.dot(Dx.T,Dx)*np.dot(Xi.T,Xi))
                self.Hx += (1.0-phi)*dH_MS + phi*dH_P
                if params.verbose:
                    logger.info("Cartesian Hessian update: %.5f Powell + %.5f Murtagh-Sargent\n" % (phi, 1.0-phi))
            else:
                Dy   = col(self.Y - self.Yprev)
                Dg   = col(self.G - self.Gprev)
                # Murtagh-Sargent-Powell update
                Xi = Dg - np.dot(self.H,Dy)
                # ndy2 = np.dot(Dy.T,Dy)
                dH_MS = np.dot(Xi, Xi.T)/np.dot(Dy.T, Xi)
                dH_P = np.dot(Xi, Dy.T) + np.dot(Dy, Xi.T) - np.dot(Dy, Dy.T)*np.dot(Xi.T, Dy)/np.dot(Dy.T, Dy)
                dH_P /= np.dot(Dy.T, Dy)
                phi = 1.0 - np.dot(Dy.T,Xi)**2/(np.dot(Dy.T,Dy)*np.dot(Xi.T,Xi))
                phi = 1.0
                self.H += (1.0-phi)*dH_MS + phi*dH_P
                if params.verbose:
                    logger.info("Hessian update: %.5f Powell + %.5f Murtagh-Sargent\n" % (phi, 1.0-phi))
        else:
            Dy   = col(self.Y - self.Yprev)
            Dg   = col(self.G - self.Gprev)
            # Catch some abnormal cases of extremely small changes.
            if np.linalg.norm(Dg) < 1e-6: return
            if np.linalg.norm(Dy) < 1e-6: return
            # BFGS Hessian update
            Mat1 = np.dot(Dg,Dg.T)/np.dot(Dg.T,Dy)[0,0]
            Mat2 = np.dot(np.dot(self.H,Dy), np.dot(self.H,Dy).T)/multi_dot([Dy.T,self.H,Dy])[0,0]
            Eig = np.linalg.eigh(self.H)[0]
            Eig.sort()
            ndy = np.array(Dy).flatten()/np.linalg.norm(np.array(Dy))
            ndg = np.array(Dg).flatten()/np.linalg.norm(np.array(Dg))
            nhdy = np.dot(self.H,Dy).flatten()/np.linalg.norm(np.dot(self.H,Dy))
            if params.verbose:
                msg = "Denoms: %.3e %.3e" % (np.dot(Dg.T,Dy)[0,0], multi_dot((Dy.T,self.H,Dy))[0,0])
                msg +=" Dots: %.3e %.3e" % (np.dot(ndg, ndy), np.dot(ndy, nhdy))
            #H1 = H.copy()
            self.H += Mat1-Mat2
            Eig1 = np.linalg.eigh(self.H)[0]
            Eig1.sort()
            if params.verbose:
                msg += " Eig-ratios: %.5e ... %.5e" % (np.min(Eig1)/np.min(Eig), np.max(Eig1)/np.max(Eig))
                logger.info(msg+'\n')
            if np.min(Eig1) <= params.epsilon and params.reset:
                logger.info("Eigenvalues below %.4e (%.4e) - returning guess\n" % (params.epsilon, np.min(Eig1)))
                self.H = self.IC.guess_hessian(self.coords)

    def optimizeGeometry(self):
        """
        High-level optimization loop.
        This allows calcEnergyForce() to be separated from the rest of the codes
        """
        self.calcEnergyForce()
        self.prepareFirstStep()
        while self.state not in [OPT_STATE.CONVERGED, OPT_STATE.FAILED]:
            self.step()
            if self.state == OPT_STATE.NEEDS_EVALUATION:
                self.calcEnergyForce()
                self.evaluateStep()
        if self.state == OPT_STATE.FAILED:
            raise GeomOptNotConvergedError("Optimizer.optimizeGeometry() failed to converge.")
        # If we want to save the Hessian used by the optimizer (in Cartesian coordinates)
        if self.params.write_cart_hess:
            # One last Hessian update before writing it out
            self.UpdateHessian()
            logger.info("Saving current approximate Hessian (Cartesian coordinates) to %s" % self.params.write_cart_hess)
            Hx = self.IC.calcHessCart(self.X, self.G, self.H)
            np.savetxt(self.params.write_cart_hess, Hx, fmt='% 14.10f')
        return self.progress

def Optimize(coords, molecule, IC, engine, dirname, params):
    """
    Optimize the geometry of a molecule. This function used contain the whole
    optimization loop, which has since been moved to the Optimizer() class;
    now a wrapper and kept for compatibility.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of Cartesian coordinates in atomic units
    molecule : Molecule
        Molecule object
    IC : InternalCoordinates
        Object describing the internal coordinate system
    engine : Engine
        Object containing methods for calculating energy and gradient
    dirname : str
        Directory name for files to be written
    params : OptParams object
        Contains optimization parameters (really just a struct)
    hessian : np.ndarray, optional
        3Nx3N array of Cartesian Hessian of initial structure

    Returns
    -------
    progress: Molecule
        A molecule object for opt trajectory and energies
    """
    optimizer = Optimizer(coords, molecule, IC, engine, dirname, params)
    return optimizer.optimizeGeometry()

def CheckInternalGrad(coords, molecule, IC, engine, dirname, verbose=0):
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

def CheckInternalHess(coords, molecule, IC, engine, dirname, verbose=0):
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

def CartesianHessian(coords, molecule, engine, dirname, readfiles=True, verbose=0):
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

def print_msg():
    print("""
    #==========================================================================#
    #| If this code has benefited your research, please support us by citing: |#
    #|                                                                        |#
    #| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
    #| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
    #| http://dx.doi.org/10.1063/1.4952956                                    |#
    #==========================================================================#
    """, file=sys.stderr)

def WriteDisplacements(coords, M, IC, dirname, verbose):
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

def get_molecule_engine(**kwargs):
    """
    Parameters
    ----------
    args : namespace
        Command line arguments from argparse

    Returns
    -------
    Molecule
        Molecule object containing necessary optimization info
    Engine
        Engine object containing methods for calculating energy and gradient
    """
    ### Set up based on which quantum chemistry code we're using.
    engine_str = kwargs.get('engine', None)
    customengine = kwargs.get('customengine', None)
    # Path to Molpro executable (used if molpro=True)
    molproexe = kwargs.get('molproexe', None)
    # PDB file will be read for residue IDs to make TRICs for fragments
    # and provide starting coordinates in the case of OpenMM
    pdb = kwargs.get('pdb', None)
    # if frag=True, do not add a bond between residues.
    frag = kwargs.get('frag', False)
    # Number of threads to use (engine-dependent)
    nt = kwargs.get('nt', None)
    # Name of the input file.
    inputf = kwargs.get('input')

    ## MECI calculations create a custom engine that contains two other engines.
    if kwargs.get('meci', None):
        if engine_str.lower() in ['psi4', 'gmx', 'molpro', 'qcengine', 'openmm'] or customengine:
            logger.warning("MECI optimizations are not tested with engines: psi4, gmx, molpro, qcegine, openmm, customengine. Be Careful!")
        meci_sigma = kwargs.get('meci_sigma')
        meci_alpha = kwargs.get('meci_alpha')
        sub_engines = {}
        for state in [1, 2]:
            sub_kwargs = kwargs.copy()
            if state == 2:
                sub_kwargs['input'] = kwargs['meci']
            sub_kwargs['meci'] = None
            M, sub_engine = get_molecule_engine(**sub_kwargs)
            sub_engines[state] = sub_engine
        engine = ConicalIntersection(M, sub_engines[1], sub_engines[2], meci_sigma, meci_alpha)
        return M, engine

    ## Read radii from the command line.
    # Ions should have radii of zero.
    arg_radii = kwargs.get('radii', ["Na","0.0","Cl","0.0","K","0.0"])
    # logger.info(arg_radii)
    if (len(arg_radii) % 2) != 0:
        raise RuntimeError("Must have an even number of arguments for radii")
    nrad = int(len(arg_radii) / 2)
    radii = {}
    for i in range(nrad):
        radii[arg_radii[2*i].capitalize()] = float(arg_radii[2*i+1])

    if engine_str:
        engine_str = engine_str.lower()
        if engine_str not in ['tera', 'qchem', 'psi4', 'gmx', 'molpro', 'openmm', 'qcengine']:
            raise RuntimeError("Valid values of engine are: tera, qchem, psi4, gmx, molpro, openmm, qcengine")
        if customengine:
            raise RuntimeError("engine and customengine cannot simultaneously be set")
        if engine_str == 'tera':
            logger.info("TeraChem engine selected. Expecting TeraChem input for gradient calculation.\n")
            set_tcenv()
            tcin = load_tcin(inputf)
            if pdb is not None:
                M = Molecule(pdb, radii=radii, fragment=frag)
            else:
                if not os.path.exists(tcin['coordinates']):
                    raise RuntimeError("TeraChem coordinate file does not exist")
                M = Molecule(tcin['coordinates'], radii=radii, fragment=frag)
            M.charge = tcin['charge']
            M.mult = tcin.get('spinmult',1)
            engine = TeraChem(M, tcin)
            if nt is not None:
                raise RuntimeError("--nt not configured to work with terachem yet")
        elif engine_str == 'qchem':
            logger.info("Q-Chem engine selected. Expecting Q-Chem input for gradient calculation.\n")
            # The file from which we make the Molecule object
            if pdb is not None:
                # If we pass the PDB, then read both the PDB and the Q-Chem input file,
                # then copy the Q-Chem rem variables over to the PDB
                M = Molecule(pdb, radii=radii, fragment=frag)
                M1 = Molecule(inputf, radii=radii)
                for i in ['qctemplate', 'qcrems', 'elem', 'qm_ghost', 'charge', 'mult']:
                    if i in M1: M[i] = M1[i]
            else:
                M = Molecule(inputf, radii=radii)
            engine = QChem(M)
            if nt is not None:
                engine.set_nt(nt)
        elif engine_str == 'gmx':
            logger.info("Gromacs engine selected. Expecting conf.gro, topol.top and shot.mdp (exact names).\n")
            M = Molecule(inputf, radii=radii, fragment=frag)
            if pdb is not None:
                M = Molecule(pdb, radii=radii, fragment=frag)
            if 'boxes' in M.Data:
                del M.Data['boxes']
            engine = Gromacs(M)
            if nt is not None:
                raise RuntimeError("--nt not configured to work with Gromacs yet")
        elif engine_str == 'openmm':
            logger.info("OpenMM engine selected. Expecting forcefield.xml or system.xml file, and PDB passed in via --pdb.\n")
            if pdb is None:
                raise RuntimeError("Must pass a PDB with option --pdb to use OpenMM.")
            M = Molecule(pdb, radii=radii, fragment=frag)
            if 'boxes' in M.Data:
                del M.Data['boxes']
            engine = OpenMM(M, pdb, inputf)
            if nt is not None:
                raise RuntimeError("--nt not configured to work with OpenMM yet")
        elif engine_str == 'psi4':
            logger.info("Psi4 engine selected. Expecting Psi4 input for gradient calculation.\n")
            engine = Psi4()
            engine.load_psi4_input(inputf)
            if pdb is not None:
                M = Molecule(pdb, radii=radii, fragment=frag)
                M1 = engine.M
                for i in ['elem']:
                    if i in M1: M[i] = M1[i]
            else:
                M = engine.M
                M.top_settings['radii'] = radii
            if nt is not None:
                engine.set_nt(nt)
        elif engine_str == 'molpro':
            logger.info("Molpro engine selected. Expecting Molpro input for gradient calculation.\n")
            engine = Molpro()
            engine.load_molpro_input(inputf)
            M = engine.M
            if nt is not None:
                engine.set_nt(nt)
            if molproexe is not None:
                engine.set_molproexe(molproexe)
        elif engine_str == 'qcengine':
            logger.info("QCEngine selected.\n")
            schema = kwargs.get('qcschema', False)
            if schema is False:
                raise RuntimeError("QCEngineAPI option requires a QCSchema")
    
            program = kwargs.get('qce_program', False)
            if program is False:
                raise RuntimeError("QCEngineAPI option requires a qce_program option")
            engine = QCEngineAPI(schema, program)
            M = engine.M
        else:
            raise RuntimeError("Failed to create an engine object, this might be a bug in get_molecule_engine")
    elif customengine:
        logger.info("Custom engine selected.\n")
        engine = customengine
        M = engine.M

    # If --coords is provided via command line, use initial coordinates in the provided file
    # to override all previously provided coordinates.
    arg_coords = kwargs.get('coords', None)
    if arg_coords is not None:
        M1 = Molecule(arg_coords)
        M1 = M1[-1]
        M.xyzs = M1.xyzs

    return M, engine


def run_optimizer(**kwargs):
    """
    Run geometry optimization, constrained optimization, or
    constrained scan job given arguments from command line.
    """
    #==============================#
    #|   Log file configuration   |#
    #==============================#
    # By default, output should be written to <args.prefix>.log and also printed to the terminal.
    # This behavior may be changed by editing the log.ini file.
    # Output will only be written to log files after the 'logConfig' line is called!
    logIni = 'log.ini'
    if kwargs.get('logIni') is None:
        import geometric.optimize
        logIni = pkg_resources.resource_filename(geometric.optimize.__name__, logIni)
    else:
        logIni = kwargs.get('logIni')
    logfilename = kwargs.get('prefix')
    # Input file for optimization; QC input file or OpenMM .xml file
    inputf = kwargs.get('input')
    # Get calculation prefix and temporary directory name
    arg_prefix = kwargs.get('prefix', None) #prefix for output file and temporary directory
    prefix = arg_prefix if arg_prefix is not None else os.path.splitext(inputf)[0]
    logfilename = prefix + ".log"
    # Create a backup if the log file already exists
    backed_up = bak(logfilename)
    import logging.config
    logging.config.fileConfig(logIni,defaults={'logfilename': logfilename},disable_existing_loggers=False)
    #==============================#
    #| End log file configuration |#
    #==============================#

    import geometric
    logger.info('geometric-optimize called with the following command line:\n')
    logger.info(' '.join(sys.argv)+'\n')
    printLogo(logger)
    logger.info('-=# \x1b[1;94m geomeTRIC started. Version: %s \x1b[0m #=-\n' % geometric.__version__)
    if backed_up:
        logger.info('Backed up existing log file: %s -> %s\n' % (logfilename, os.path.basename(backed_up)))

    t0 = time.time()

    # Create the params object, containing data to be passed into the optimizer
    params = OptParams(**kwargs)
    params.printInfo()

    # Get the Molecule and engine objects needed for optimization
    M, engine = get_molecule_engine(**kwargs)

    #============================================#
    #|   Temporary file and folder management   |#
    #============================================#
    # LPW: Should this be refactored and moved into get_molecule_engine?
    # Probably not .. get_molecule_engine is independent of dirname.
    # We may implement Engine.initialize(dirname) in the future.
    dirname = prefix+".tmp"
    if not os.path.exists(dirname):
        # LPW: Some engines do not need the tmp-folder, but the
        # auxiliary functions might (such as writing out displacements)
        os.makedirs(dirname)
    else:
        logger.info("%s exists ; make sure nothing else is writing to the folder\n" % dirname)
        # TC-specific: Remove existing scratch files in ./run.tmp/scr to avoid confusion
        for f in ['c0', 'ca0', 'cb0']:
            if os.path.exists(os.path.join(dirname, 'scr', f)):
                os.remove(os.path.join(dirname, 'scr', f))

    # QC-specific scratch folder
    qcdir = kwargs.get('qcdir', None) # Provide an initial qchem scratch folder (e.g. supplied initial guess
    if qcdir is not None:
        if 'engine' in kwargs and kwargs['engine'].lower() != 'qchem':
            raise RuntimeError("--qcdir only valid if --qchem is specified")
        if not os.path.exists(qcdir):
            raise RuntimeError("--qcdir points to a folder that doesn't exist")
        shutil.copytree(qcdir, os.path.join(dirname, "run.d"))
        engine.M.edit_qcrems({'scf_guess':'read'})
        engine.qcdir = True
    #============================================#
    #| End temporary file and folder management |#
    #============================================#

    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() * ang2bohr

    # Read in the constraints
    constraints = kwargs.get('constraints', None) #Constraint input file (optional)

    if constraints is not None:
        Cons, CVals = ParseConstraints(M, open(constraints).read())
    else:
        Cons = None
        CVals = None

    #=========================================#
    #| Set up the internal coordinate system |#
    #=========================================#
    # First item in tuple: The class to be initialized
    # Second item in tuple: Whether to connect nonbonded fragments
    # Third item in tuple: Whether to throw in all Cartesians (no effect if second item is True)
    CoordSysDict = {'cart':(CartesianCoordinates, False, False),
                    'prim':(PrimitiveInternalCoordinates, True, False),
                    'dlc':(DelocalizedInternalCoordinates, True, False),
                    'hdlc':(DelocalizedInternalCoordinates, False, True),
                    'tric':(DelocalizedInternalCoordinates, False, False)}
    coordsys = kwargs.get('coordsys', 'tric')
    CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]

    IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVals[0] if CVals is not None else None,
                    conmethod=params.conmethod)
    #========================================#
    #| End internal coordinate system setup |#
    #========================================#

    # Auxiliary functions (will not do optimization):
    verbose = kwargs.get('verbose', False)
    displace = kwargs.get('displace', False) # Write out the displacements of the coordinates.
    if displace:
        WriteDisplacements(coords, M, IC, dirname, verbose)
        return

    fdcheck = kwargs.get('fdcheck', False) # Check internal coordinate gradients using finite difference..
    if fdcheck:
        IC.Prims.checkFiniteDifferenceGrad(coords)
        IC.Prims.checkFiniteDifferenceHess(coords)
        CheckInternalGrad(coords, M, IC.Prims, engine, dirname, verbose)
        CheckInternalHess(coords, M, IC.Prims, engine, dirname, verbose)
        return

    # Print out information about the coordinate system
    if isinstance(IC, CartesianCoordinates):
        logger.info("%i Cartesian coordinates being used\n" % (3*M.na))
    else:
        logger.info("%i internal coordinates being used (instead of %i Cartesians)\n" % (len(IC.Internals), 3*M.na))
    logger.info(IC)
    logger.info("\n")

    if Cons is None:
        # Run a standard geometry optimization
        params.xyzout = prefix+"_optim.xyz"
        progress = Optimize(coords, M, IC, engine, dirname, params)
    else:
        # Run a single constrained geometry optimization or scan over a grid of values
        if isinstance(IC, (CartesianCoordinates, PrimitiveInternalCoordinates)):
            raise RuntimeError("Constraints only work with delocalized internal coordinates")
        Mfinal = None
        for ic, CVal in enumerate(CVals):
            if len(CVals) > 1:
                logger.info("---=== Scan %i/%i : Constrained Optimization ===---\n" % (ic+1, len(CVals)))
            IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVal, conmethod=params.conmethod)
            IC.printConstraints(coords, thre=-1)
            if len(CVals) > 1:
                params.xyzout = prefix+"_scan-%03i.xyz" % ic
                # In the special case of a constraint scan, we write out multiple qdata.txt files
                if params.qdata is not None: params.qdata = 'qdata_scan-%03i.txt' % ic
            else:
                params.xyzout = prefix+"_optim.xyz"
            if ic == 0:
                progress = Optimize(coords, M, IC, engine, dirname, params)
            else:
                progress += Optimize(coords, M, IC, engine, dirname, params)
            # update the structure for next optimization in SCAN (by CNH)
            M.xyzs[0] = progress.xyzs[-1]
            coords = progress.xyzs[-1].flatten() * ang2bohr
            if Mfinal:
                Mfinal += progress[-1]
            else:
                Mfinal = progress[-1]
            cNames, cVals = IC.getConstraintTargetVals()
            comment = ', '.join(["%s = %.2f" % (cName, cVal) for cName, cVal in zip(cNames, cVals)])
            Mfinal.comms[-1] = "Scan Cycle %i/%i ; %s ; %s" % (ic+1, len(CVals), comment, progress.comms[-1])
            #print
        if len(CVals) > 1:
            Mfinal.write('scan-final.xyz')
            if params.qdata is not None: Mfinal.write('qdata-final.txt')
    print_msg()
    logger.info("Time elapsed since start of run_optimizer: %.3f seconds\n" % (time.time()-t0))
    return progress

def str2bool(v):
    """ Allows command line options such as "yes" and "True" to be converted into Booleans. """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """Read user's input"""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--coordsys', type=str, default='tric', help='Coordinate system: "cart" for Cartesian, "prim" for Primitive (a.k.a redundant), '
                        '"dlc" for Delocalized Internal Coordinates, "hdlc" for Hybrid Delocalized Internal Coordinates, "tric" for Translation-Rotation'
                        'Internal Coordinates (default).')
    parser.add_argument('--engine', type=str, default='tera', help='Specify engine for computing energies and gradients, '
                        '"tera" for TeraChem (default), "qchem" for Q-Chem, "psi4" for Psi4, "openmm" for OpenMM, "gmx" for Gromacs, "molpro" for Molpro.')
    parser.add_argument('--meci', type=str, default=None, help='Provide second input file and search for minimum-energy conical '
                        'intersection or crossing point between two SCF solutions (TeraChem and Q-Chem supported).')
    parser.add_argument('--meci_sigma', type=float, default=3.5, help='Sigma parameter for MECI optimization.')
    parser.add_argument('--meci_alpha', type=float, default=0.025, help='Alpha parameter for MECI optimization.')
    parser.add_argument('--molproexe', type=str, default=None, help='Specify absolute path of Molpro executable.')
    parser.add_argument('--molcnv', action='store_true', help='Use Molpro style convergence criteria instead of the default.')
    parser.add_argument('--prefix', type=str, default=None, help='Specify a prefix for log file and temporary directory.')
    parser.add_argument('--displace', action='store_true', help='Write out the displacements of the coordinates.')
    parser.add_argument('--fdcheck', action='store_true', help='Check internal coordinate gradients using finite difference.')
    parser.add_argument('--enforce', type=float, default=0.0, help='Enforce exact constraints when within provided tolerance (in a.u. and radian)')
    parser.add_argument('--conmethod', type=int, default=0, help='Set to 1 to enable updated constraint algorithm.')
    parser.add_argument('--write_cart_hess', type=str, default=None, help='Convert current Hessian matrix at optimized geometry to Cartesian coords and write to specified file.')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Small eigenvalue threshold.')
    parser.add_argument('--check', type=int, default=0, help='Check coordinates every N steps to see whether it has changed.')
    parser.add_argument('--verbose', type=int, default=0, help='Set to positive for more verbose printout. 1 = Basic info about optimization step. 2 = Include microiterations.'
                        '3 = Lots of printout from low-level functions.')
    parser.add_argument('--logINI',  type=str, dest='logIni', help='ini file for logging')
    parser.add_argument('--reset', type=str2bool, help='Reset Hessian when eigenvalues are under epsilon. Defaults to True for minimization and False for transition states.')
    parser.add_argument('--transition', action='store_true', help='Search for a first order saddle point / transition state.')
    parser.add_argument('--hessian', type=str, help='Specify when to calculate Cartesian Hessian from finite difference of gradient. '
                        '"never" : Do not calculate or read Hessian data. file:<path> : Read Hessian data in NumPy format from path, e.g. file:run.tmp/hessian/hessian.txt .'
                        '"first" : Calculate or read for the initial structure. "each" : Calculate for each step in the optimization (costly).'
                        '"exit" : Calculate Hessian and then exit without optimizing. Default is "never" for minimization and "initial" for transition state.')
    parser.add_argument('--rfo', action='store_true', help='Use rational function optimization (default is trust-radius Newton Raphson).')
    parser.add_argument('--trust', type=float, default=0.1, help='Starting trust radius.')
    parser.add_argument('--tmax', type=float, default=0.3, help='Maximum trust radius.')
    parser.add_argument('--maxiter', type=int, default=300, help='Maximum number of optimization steps.')
    parser.add_argument('--radii', type=str, nargs="+", default=["Na","0.0"], help='List of atomic radii for coordinate system.')
    parser.add_argument('--pdb', type=str, help='Provide a PDB file name with coordinates and resids to split the molecule.')
    parser.add_argument('--coords', type=str, help='Provide coordinates to override the TeraChem input file / PDB file. The LAST frame will be used.')
    parser.add_argument('--frag', action='store_true', help='Fragment the internal coordinate system by deleting bonds between residues.')
    parser.add_argument('--qcdir', type=str, help='Provide an initial qchem scratch folder (e.g. supplied initial guess).')
    parser.add_argument('--qccnv', action='store_true', help='Use Q-Chem style convergence criteria instead of the default.')
    parser.add_argument('--qdata', action='store_true', help='Write qdata.txt containing coordinates, energies, gradients for each structure in optimization.')
    parser.add_argument('--converge', type=str, nargs="+", default=[], help='Custom convergence criteria as key/value pairs.'
                        'Provide the name of a criteria set as "set GAU_LOOSE" or "set TURBOMOLE", and/or set specific criteria using "energy 1e-5" or "grms 1e-3')
    parser.add_argument('--nt', type=int, help='Specify number of threads for running in parallel (for TeraChem this should be number of GPUs)')
    parser.add_argument('input', type=str, help='TeraChem or Q-Chem input file')
    parser.add_argument('constraints', type=str, nargs='?', help='Constraint input file (optional)')
    args = parser.parse_args(sys.argv[1:])

    # Run the optimizer.
    try:
        run_optimizer(**vars(args))
    except EngineError:
        logger.info("EngineError:\n" + traceback.format_exc())
        sys.exit(51)
    except GeomOptNotConvergedError:
        logger.info("Geometry Converge Failed Error:\n" + traceback.format_exc())
        sys.exit(50)
    except:
        logger.info("Unknown Error:\n" + traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
