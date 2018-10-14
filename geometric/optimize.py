from __future__ import print_function, division

import argparse
import itertools
import os
import shutil
import sys

import numpy as np
from numpy.linalg import multi_dot

from .engine import set_tcenv, load_tcin, TeraChem, TeraChem_CI, Psi4, QChem, Gromacs, Molpro, QCEngineAPI
from .internal import *
from .molecule import Molecule, Elements
from .nifty import row, col, flat, invert_svd, uncommadash, isint, bohr2ang, ang2bohr
from .rotate import get_rot, sorted_eigh, calc_fac_dfac


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
    print("Rebuilding Hessian using %i gradients" % history)
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
        print("Eigenvalues below %.4e (%.4e) - returning guess" % (params.epsilon, np.min(np.linalg.eigh(H)[0])))
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

def getCartesianNorm(X, dy, IC, enforce=False, verbose=False):
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
    enforce : bool
        Enforce constraints in the internal coordinate system
    verbose : bool
        Print diagnostic messages

    Returns
    -------
    float
        The RMSD between the updated and original Cartesian coordinates
    """
    # Displacement of each atom in Angstrom
    if IC.haveConstraints() and enforce:
        Xnew = IC.newCartesian_withConstraint(X, dy, verbose=verbose)
    else:
        Xnew = IC.newCartesian(X, dy, verbose=verbose)
    rmsd, maxd = calc_drms_dmax(Xnew, X)
    return rmsd

def between(s, a, b):
    if a < b:
        return s > a and s < b
    elif a > b:
        return s > b and s < a
    else:
        raise RuntimeError('a and b must be different')

def brent_wiki(f, a, b, rel, cvg=0.1, obj=None, verbose=False):
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
    verbose : bool
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
            if verbose: print("returning because interval is too small")
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

def ftest(x):
    answer = (x+3)*(x-1)**2
    print("(x, y) = ", x, answer)
    return answer

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
        print(line)
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
                if key == "bond" and atoms[0] > atoms[1]:
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
                        print("Large rotation: Your constraint may not work")
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
                            print("Large rotation: Your constraint may not work")
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

def get_delta_prime_trm(v, X, G, H, IC, verbose=False):
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
    verbose : bool
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
    if IC is not None:
        GC, HC = IC.augmentGH(X, G, H) if IC.haveConstraints() else (G, H)
    else:
        GC, HC = (G, H)
    HT = HC + v*np.eye(len(HC))
    # The constrained degrees of freedom should not have anything added to diagonal
    for i in range(len(G), len(GC)):
        HT[i, i] = 0.0
    if verbose:
        seig = sorted(np.linalg.eig(HT)[0])
        print("sorted(eig) : % .5e % .5e % .5e ... % .5e % .5e % .5e" % (seig[0], seig[1], seig[2], seig[-3], seig[-2], seig[-1]))
    try:
        Hi = invert_svd(HT)
    except:
        print ("\x1b[1;91mSVD Error - increasing v by 0.001 and trying again\x1b[0m")
        return get_delta_prime_trm(v+0.001, X, G, H, IC)
    dyc = flat(-1 * np.dot(Hi,col(GC)))
    dy = dyc[:len(G)]
    d_prime = flat(-1 * np.dot(Hi, col(dyc)))[:len(G)]
    dy_prime = np.dot(dy,d_prime)/np.linalg.norm(dy)
    # sol = flat(0.5*row(dy)*np.matrix(H)*col(dy))[0] + np.dot(dy,G)
    sol = flat(0.5*multi_dot([row(dy),H,col(dy)]))[0] + np.dot(dy,G)
    return dy, sol, dy_prime

def get_delta_prime_rfo(alpha, X, G, H, IC, verbose=False):
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
    verbose : bool
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
    # print "AH eigenvalues: %.5e %.5e %.5e ... %.5e %.5e %.5e" % (AHeig[0],AHeig[1],AHeig[2],AHeig[-3],AHeig[-2],AHeig[-1])
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

def get_delta_prime(v, X, G, H, IC, rfo, verbose=False):
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
    verbose : bool
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
        return get_delta_prime_rfo(v, X, G, H, IC, verbose)
    else:
        return get_delta_prime_trm(v, X, G, H, IC, verbose)

def trust_step(target, v0, X, G, H, IC, rfo, verbose=False):
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
    verbose : bool
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step with the desired size
    sol : float
        Expected change of the objective function
    """
    dy, sol, dy_prime = get_delta_prime(v0, X, G, H, IC, rfo, verbose)
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
        v += (1-ndy/target)*(ndy/dy_prime)
        dy, sol, dy_prime, = get_delta_prime(v, X, G, H, IC, rfo, verbose)
        ndy = np.linalg.norm(dy)
        if verbose: print("v = %.5f dy -> target = %.5f -> %.5f" % (v, ndy, target))
        if np.abs((ndy-target)/target) < 0.001:
            return dy, sol
        # With Lagrange multipliers it may be impossible to go under a target step size
        elif niter > 10 and np.abs(ndy_last-ndy)/ndy < 0.001:
            return dy, sol
        niter += 1
        ndy_last = ndy
        if ndy < m_ndy:
            m_ndy = ndy
            m_dy = dy.copy()
            m_sol = sol
        # Break out of infinite oscillation loops
        if niter%100 == 99:
            print("trust_step hit niter = 100, randomizing")
            v += np.random.random() * niter / 100
        if niter%1000 == 999:
            print ("trust_step hit niter = 1000, giving up")
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
                dy, expect = trust_step(trial, v0, X, G, H, IC, self.params.rfo, self.params.verbose)
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
            if self.params.verbose: print("dy(i): %.4f dy(c) -> target: %.4f -> %.4f%s" % (trial, cnorm, self.target, " (done)" if self.from_above else ""))
            return cnorm-self.target

def recover(molecule, IC, X, gradx, X_hist, Gx_hist, params):
    """
    Recover from a failed optimization.

    Parameters
    ----------
    molecule : Molecule
        Molecule object for rebuilding internal coordinates
    IC : InternalCoordinates
        Object describing the current internal coordinate system
    X : np.ndarray
        Nx3 array of Cartesian coordinates in atomic units
    gradx : np.ndarray
        Nx3 array of Cartesian gradients in atomic units
    X_hist : list
        List of previous Cartesian coordinates
    Gx_hist : list
        List of previous Cartesian gradients
    params : OptParams
        Pass optimization parameters to Hessian rebuild

    Returns
    -------
    Y : np.ndarray
        New internal coordinates
    G : np.ndarray
        New internal gradients
    H : np.ndarray
        New internal Hessian
    """
    newmol = deepcopy(molecule)
    newmol.xyzs[0] = X.reshape(-1,3) * bohr2ang
    newmol.build_topology()
    IC1 = IC.__class__(newmol, connect=IC.connect, addcart=IC.addcart, build=False)
    if IC.haveConstraints(): IC1.getConstraints_from(IC)
    if IC1 != IC:
        print("\x1b[1;94mInternal coordinate system may have changed\x1b[0m")
        if IC.repr_diff(IC1) != "":
            print(IC.repr_diff(IC1))
    IC = IC1
    IC.resetRotations(X)
    if isinstance(IC, DelocalizedInternalCoordinates):
        IC.build_dlc(X)
    H0 = IC.guess_hessian(X)
    if params.reset:
        H = H0.copy()
    else:
        H = RebuildHessian(IC, H0, X_hist, Gx_hist, params)
    Y = IC.calculate(X)
    G = IC.calcGrad(X, gradx)
    return Y, G, H, IC

class OptParams(object):
    """
    Container for optimization parameters.
    The parameters used to be contained in the command-line "args",
    but this was dropped in order to call Optimize() from another script.
    """
    def __init__(self, **kwargs):
        self.enforce = kwargs.get('enforce', False)
        self.epsilon = kwargs.get('epsilon', 1e-5)
        self.check = kwargs.get('check', 0)
        self.verbose = kwargs.get('verbose', False)
        self.reset = kwargs.get('reset', False)
        self.rfo = kwargs.get('rfo', False)
        self.trust = kwargs.get('trust', 0.1)
        self.tmax = kwargs.get('tmax', 0.3)
        self.maxiter = kwargs.get('maxiter', 300)
        self.qccnv = kwargs.get('qccnv', False)
        self.molcnv = kwargs.get('molcnv', False)
        self.Convergence_energy = kwargs.get('convergence_energy', 1e-6)
        self.Convergence_grms = kwargs.get('convergence_grms', 3e-4)
        self.Convergence_gmax = kwargs.get('convergence_gmax', 4.5e-4)
        self.Convergence_drms = kwargs.get('convergence_drms', 1.2e-3)
        self.Convergence_dmax = kwargs.get('convergence_dmax', 1.8e-3)
        self.molpro_convergence_gmax = kwargs.get('molpro_convergence_gmax', 3e-4)
        self.molpro_convergence_dmax = kwargs.get('molpro_convergence_dmax', 1.2e-3)
        # CI optimizations sometimes require tiny steps
        self.meci = kwargs.get('meci', False)

def Optimize(coords, molecule, IC, engine, dirname, params, xyzout=None, xyzout2=None):
    """
    Optimize the geometry of a molecule.

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
    params : OptParams object
        Contains optimization parameters (really just a struct)
    xyzout : str, optional
        Output file name for writing the progress of the optimization.

    Returns
    -------
    progress: Molecule
        A molecule object for opt trajectory and energies
    """
    progress = deepcopy(molecule)
    progress2 = deepcopy(molecule)
    # Initial Hessian
    H0 = IC.guess_hessian(coords)
    H = H0.copy()
    # Cartesian coordinates
    X = coords.copy()
    # Initial energy and gradient
    E, gradx = engine.calc(coords, dirname)
    progress.qm_energies = [E]
    # Initial internal coordinates
    q0 = IC.calculate(coords)
    Gq = IC.calcGrad(X, gradx)
    # The optimization variables are the internal coordinates.
    Y = q0.copy()
    G = np.array(Gq).flatten()
    # Loop of optimization
    Iteration = 0
    CoordCounter = 0
    trust = params.trust
    if params.meci:
        thre_rj = 1e-4
    else:
        thre_rj = 1e-2
    # Print initial iteration
    gradxc = IC.calcGradProj(X, gradx) if IC.haveConstraints() else gradx.copy()
    atomgrad = np.sqrt(np.sum((gradxc.reshape(-1,3))**2, axis=1))
    rms_gradient = np.sqrt(np.mean(atomgrad**2))
    max_gradient = np.max(atomgrad)
    print("Step %4i :" % Iteration, end=' '),
    print("Gradient = %.3e/%.3e (rms/max) Energy = % .10f" % (rms_gradient, max_gradient, E))
    progress.xyzs = [coords.copy().reshape(-1, 3) * bohr2ang]
    progress.comms = ['Iteration %i Energy % .8f' % (Iteration, E)]
    # if IC.haveConstraints(): IC.printConstraints(X)
    # Threshold for "low quality step" which decreases trust radius.
    ThreLQ = 0.25
    # Threshold for "high quality step" which increases trust radius.
    ThreHQ = 0.75
    # Convergence criteria
    Convergence_energy = params.Convergence_energy
    Convergence_grms = params.Convergence_grms
    Convergence_gmax = params.Convergence_gmax
    Convergence_drms = params.Convergence_drms
    Convergence_dmax = params.Convergence_dmax
    # Approximate Molpro convergence criteria
    # Approximate b/c Molpro appears to evaluate criteria in normal coordinates instead of cartesian coordinates.
    molpro_convergence_gmax = params.molpro_convergence_gmax
    molpro_convergence_dmax = params.molpro_convergence_dmax
    X_hist = [X]
    Gx_hist = [gradx]
    trustprint = "="
    ForceRebuild = False
    while True:
        if np.isnan(G).any():
            raise RuntimeError("Gradient contains nan - check output and temp-files for possible errors")
        if np.isnan(H).any():
            raise RuntimeError("Hessian contains nan - check output and temp-files for possible errors")
        Iteration += 1
        if (Iteration%5) == 0:
            engine.clearCalcs()
            IC.clearCache()
        # At the start of the loop, the function value, gradient and Hessian are known.
        Eig = sorted(np.linalg.eigh(H)[0])
        Emin = min(Eig).real
        if params.rfo:
            v0 = 1.0
        elif Emin < params.epsilon:
            v0 = params.epsilon-Emin
        else:
            v0 = 0.0
        if params.verbose: IC.Prims.printRotations()
        if len(Eig) >= 6:
            print("Hessian Eigenvalues: %.5e %.5e %.5e ... %.5e %.5e %.5e" % (Eig[0],Eig[1],Eig[2],Eig[-3],Eig[-2],Eig[-1]))
        else:
            print("Hessian Eigenvalues:", ' '.join("%.5e" % i for i in Eig))
        # Are we far from constraint satisfaction?
        farConstraints = IC.haveConstraints() and IC.getConstraintViolation(X) > 1e-1
        conSatisfied = not IC.haveConstraints() or IC.getConstraintViolation(X) < 1e-2
        ### OBTAIN AN OPTIMIZATION STEP ###
        # The trust radius is to be computed in Cartesian coordinates.
        # First take a full-size Newton Raphson step
        dy, expect, _ = get_delta_prime(v0, X, G, H, IC, params.rfo)
        # Internal coordinate step size
        inorm = np.linalg.norm(dy)
        # Cartesian coordinate step size
        cnorm = getCartesianNorm(X, dy, IC, params.enforce, params.verbose)
        if params.verbose: print("dy(i): %.4f dy(c) -> target: %.4f -> %.4f" % (inorm, cnorm, trust))
        # If the step is above the trust radius in Cartesian coordinates, then
        # do the following to reduce the step length:
        if cnorm > 1.1 * trust:
            # This is the function f(inorm) = cnorm-target that we find a root
            # for obtaining a step with the desired Cartesian step size.
            froot = Froot(trust, v0, X, G, H, IC, params)
            froot.stores[inorm] = cnorm
            # Find the internal coordinate norm that matches the desired
            # Cartesian coordinate norm
            iopt = brent_wiki(froot.evaluate, 0.0, inorm, trust, cvg=0.1, obj=froot, verbose=params.verbose)
            if froot.brentFailed and froot.stored_arg is not None:
                if params.verbose: print ("\x1b[93mUsing stored solution at %.3e\x1b[0m" % froot.stored_val)
                iopt = froot.stored_arg
            elif IC.bork:
                for i in range(3):
                    froot.target /= 2
                    if params.verbose: print ("\x1b[93mReducing target to %.3e\x1b[0m" % froot.target)
                    froot.above_flag = True
                    iopt = brent_wiki(froot.evaluate, 0.0, iopt, froot.target, cvg=0.1, verbose=params.verbose)
                    if not IC.bork: break
            LastForce = ForceRebuild
            ForceRebuild = False
            if IC.bork:
                print("\x1b[91mInverse iteration for Cartesians failed\x1b[0m")
                # This variable is added because IC.bork is unset later.
                ForceRebuild = True
            else:
                if params.verbose: print("\x1b[93mBrent algorithm requires %i evaluations\x1b[0m" % froot.counter)
            ##### Force a rebuild of the coordinate system
            if ForceRebuild:
                if LastForce:
                    print("\x1b[1;91mFailed twice in a row to rebuild the coordinate system\x1b[0m")
                    if IC.haveConstraints():
                        print("Cannot continue a constrained optimization; please implement constrained optimization in Cartesian coordinates")
                        sys.exit()
                    else:
                        print("\x1b[93mContinuing in Cartesian coordinates\x1b[0m")
                        IC = CartesianCoordinates(newmol)
                CoordCounter = 0
                Y, G, H, IC = recover(molecule, IC, X, gradx, X_hist, Gx_hist, params)
                print("\x1b[1;93mSkipping optimization step\x1b[0m")
                Iteration -= 1
                continue
            ##### End Rebuild
            # Finally, take an internal coordinate step of the desired length.
            dy, expect = trust_step(iopt, v0, X, G, H, IC, params.rfo, params.verbose)
            cnorm = getCartesianNorm(X, dy, IC, params.enforce, params.verbose)
        ### DONE OBTAINING THE STEP ###
        # Dot product of the gradient with the step direction
        Dot = -np.dot(dy/np.linalg.norm(dy), G/np.linalg.norm(G))
        # Whether the Cartesian norm comes close to the trust radius
        bump = cnorm > 0.8 * trust
        # Before updating any of our variables, copy current variables to "previous"
        Yprev = Y.copy()
        Xprev = X.copy()
        Gprev = G.copy()
        Eprev = E
        ### Update the Internal Coordinates ###
        Y += dy
        if IC.haveConstraints() and params.enforce:
            X = IC.newCartesian_withConstraint(X, dy, verbose=params.verbose)
        else:
            X = IC.newCartesian(X, dy, verbose=params.verbose)
        ### Calculate Energy and Gradient ###
        E, gradx = engine.calc(X, dirname    )
        ### Check Convergence ###
        # Add new Cartesian coordinates and gradients to history
        progress.xyzs.append(X.reshape(-1,3) * bohr2ang)
        progress.qm_energies.append(E)
        progress.comms.append('Iteration %i Energy % .8f' % (Iteration, E))
        if xyzout is not None:
            progress.write(xyzout)
        # Project out the degrees of freedom that are constrained
        gradxc = IC.calcGradProj(X, gradx) if IC.haveConstraints() else gradx.copy()
        atomgrad = np.sqrt(np.sum((gradxc.reshape(-1,3))**2, axis=1))
        rms_gradient = np.sqrt(np.mean(atomgrad**2))
        rms_displacement, max_displacement = calc_drms_dmax(X, Xprev)
        max_gradient = np.max(atomgrad)
        # The ratio of the actual energy change to the expected change
        Quality = (E-Eprev)/expect
        Converged_energy = np.abs(E-Eprev) < Convergence_energy
        Converged_grms = rms_gradient < Convergence_grms
        Converged_gmax = max_gradient < Convergence_gmax
        Converged_drms = rms_displacement < Convergence_drms
        Converged_dmax = max_displacement < Convergence_dmax
        BadStep = Quality < 0
        # Molpro defaults for convergence
        molpro_converged_gmax = max_gradient < molpro_convergence_gmax
        molpro_converged_dmax = max_displacement < molpro_convergence_dmax
        # Print status
        print("Step %4i :" % Iteration, end=' '),
        print("Displace = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("\x1b[92m" if Converged_drms else "\x1b[0m", rms_displacement, "\x1b[92m" if Converged_dmax else "\x1b[0m", max_displacement), end=' '),
        print("Trust = %.3e (%s)" % (trust, trustprint), end=' '),
        print("Grad%s = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("_T" if IC.haveConstraints() else "", "\x1b[92m" if Converged_grms else "\x1b[0m", rms_gradient, "\x1b[92m" if Converged_gmax else "\x1b[0m", max_gradient), end=' '),
        # print "Dy.G = %.3f" % Dot,
        print("E (change) = % .10f (%s%+.3e\x1b[0m) Quality = %s%.3f\x1b[0m" % (E, "\x1b[91m" if BadStep else ("\x1b[92m" if Converged_energy else "\x1b[0m"), E-Eprev, "\x1b[91m" if BadStep else "\x1b[0m", Quality))
        if IC is not None and IC.haveConstraints():
            IC.printConstraints(X, thre=1e-3)
        if isinstance(IC, PrimitiveInternalCoordinates):
            idx = np.argmax(np.abs(dy))
            iunit = np.zeros_like(dy)
            iunit[idx] = 1.0
            print("Along %s %.3f" % (IC.Internals[idx], np.dot(dy/np.linalg.norm(dy), iunit)))
        if Converged_energy and Converged_grms and Converged_drms and Converged_gmax and Converged_dmax and conSatisfied:
            print("Converged! =D")
            # _exec("touch energy.txt") #JS these two lines used to make a energy.txt file using the final energy
            if dirname is not None:
                with open("energy.txt","w") as f:
                    print("% .10f" % E, file=f)
            progress2.xyzs = [X.reshape(-1,3) * bohr2ang] #JS these two lines used to make a opt.xyz file along with the if statement below.
            progress2.comms = ['Iteration %i Energy % .8f' % (Iteration, E)]
            if xyzout2 is not None:
                progress2.write(xyzout2) #This contains the last frame of the trajectory.
            break
        if Iteration > params.maxiter:
            print("Maximum iterations reached (%i); increase --maxiter for more" % params.maxiter)
            break
        if params.qccnv and Converged_grms and (Converged_drms or Converged_energy) and conSatisfied:
            print("Converged! (Q-Chem style criteria requires grms and either drms or energy)")
            # _exec("touch energy.txt") #JS these two lines used to make a energy.txt file using the final energy
            with open("energy.txt","w") as f:
                print("% .10f" % E, file=f)
            progress2.xyzs = [X.reshape(-1,3) * bohr2ang] #JS these two lines used to make a opt.xyz file along with the if statement below.
            progress2.comms = ['Iteration %i Energy % .8f' % (Iteration, E)]
            if xyzout2 is not None:
                progress2.write(xyzout2) #This contains the last frame of the trajectory.
            break
        if params.molcnv and molpro_converged_gmax and (molpro_converged_dmax or Converged_energy) and conSatisfied:
            print("Converged! (Molpro style criteria requires gmax and either dmax or energy) This is approximate since convergence checks are done in cartesian coordinates.")
            # _exec("touch energy.txt") #JS these two lines used to make a energy.txt file using the final energy
            with open("energy.txt","w") as f:
                print("% .10f" % E, file=f)
            progress2.xyzs = [X.reshape(-1,3) * 0.529177] #JS these two lines used to make a opt.xyz file along with the if statement below.
            progress2.comms = ['Iteration %i Energy % .8f' % (Iteration, E)]
            if xyzout2 is not None:
                progress2.write(xyzout2) #This contains the last frame of the trajectory.
            break

        ### Adjust Trust Radius and/or Reject Step ###
        # If the trust radius is under thre_rj then do not reject.
        # This code rejects steps / reduces trust radius only if we're close to satisfying constraints;
        # it improved performance in some cases but worsened for others.
        rejectOk = (trust > thre_rj and E > Eprev and (Quality < -10 or not farConstraints))
        # This statement was added to prevent
        # some occasionally observed infinite loops
        if farConstraints: rejectOk = False
        # rejectOk = (trust > thre_rj and E > Eprev)
        if Quality <= ThreLQ:
            # For bad steps, the trust radius is reduced
            if not farConstraints:
                trust = max(0.0 if params.meci else Convergence_drms, trust/2)
                trustprint = "\x1b[91m-\x1b[0m"
            else:
                trustprint = "="
        elif Quality >= ThreHQ: # and bump:
            if trust < params.tmax:
                # For good steps, the trust radius is increased
                trust = min(np.sqrt(2)*trust, params.tmax)
                trustprint = "\x1b[92m+\x1b[0m"
            else:
                trustprint = "="
        else:
            trustprint = "="
        if Quality < -1 and rejectOk:
            # Reject the step and take a smaller one from the previous iteration
            trust = max(0.0 if params.meci else Convergence_drms, min(trust, cnorm/2))
            trustprint = "\x1b[1;91mx\x1b[0m"
            Y = Yprev.copy()
            X = Xprev.copy()
            G = Gprev.copy()
            E = Eprev
            continue

        # Steps that are bad, but are very small (under thre_rj) are not rejected.
        # This is because some systems (e.g. formate) have discontinuities on the
        # potential surface that can cause an infinite loop
        if Quality < -1:
            if trust < thre_rj: print("\x1b[93mNot rejecting step - trust below %.3e\x1b[0m" % thre_rj)
            elif E < Eprev: print("\x1b[93mNot rejecting step - energy decreases\x1b[0m")
            elif farConstraints: print("\x1b[93mNot rejecting step - far from constraint satisfaction\x1b[0m")
        # Append steps to history (for rebuilding Hessian)
        X_hist.append(X)
        Gx_hist.append(gradx)
        ### Rebuild Coordinate System if Necessary ###
        # Check to see whether the coordinate system has changed
        check = False
        # Reinitialize certain variables (i.e. DLC and rotations)
        reinit = False
        if IC.largeRots():
            print("Large rotations - reinitializing coordinates")
            reinit = True
        if IC.bork:
            print("Failed inverse iteration - reinitializing coordinates")
            check = True
            reinit = True
        # Check the coordinate system every (N) steps
        if (CoordCounter == (params.check - 1)) or check:
            newmol = deepcopy(molecule)
            newmol.xyzs[0] = X.reshape(-1,3) * bohr2ang

            newmol.build_topology()
            IC1 = IC.__class__(newmol, build=False, connect=IC.connect, addcart=IC.addcart)
            if IC.haveConstraints(): IC1.getConstraints_from(IC)
            if IC1 != IC:
                print("\x1b[1;94mInternal coordinate system may have changed\x1b[0m")
                if IC.repr_diff(IC1) != "":
                    print(IC.repr_diff(IC1))
                reinit = True
                IC = IC1
            CoordCounter = 0
        else:
            CoordCounter += 1
        # Reinitialize the coordinates (may happen even if coordinate system does not change)
        UpdateHessian = True
        if reinit:
            IC.resetRotations(X)
            if isinstance(IC, DelocalizedInternalCoordinates):
                IC.build_dlc(X)
            H0 = IC.guess_hessian(coords)
            H = RebuildHessian(IC, H0, X_hist, Gx_hist, params)
            UpdateHessian = False
            Y = IC.calculate(X)
        Gq = IC.calcGrad(X, gradx)
        G = np.array(Gq).flatten()

        ### Update the Hessian ###
        if UpdateHessian:
            # BFGS Hessian update
            Dy   = col(Y - Yprev)
            Dg   = col(G - Gprev)
            # Catch some abnormal cases of extremely small changes.
            if np.linalg.norm(Dg) < 1e-6: continue
            if np.linalg.norm(Dy) < 1e-6: continue
            # Mat1 = (Dg*Dg.T)/(Dg.T*Dy)[0,0]
            # Mat2 = ((H*Dy)*(H*Dy).T)/(Dy.T*H*Dy)[0,0]
            Mat1 = np.dot(Dg,Dg.T)/np.dot(Dg.T,Dy)[0,0]
            Mat2 = np.dot(np.dot(H,Dy),np.dot(H,Dy).T)/multi_dot([Dy.T,H,Dy])[0,0]
            Eig = np.linalg.eigh(H)[0]
            Eig.sort()
            ndy = np.array(Dy).flatten()/np.linalg.norm(np.array(Dy))
            ndg = np.array(Dg).flatten()/np.linalg.norm(np.array(Dg))
            nhdy = np.dot(H,Dy).flatten()/np.linalg.norm(np.dot(H,Dy))
            if params.verbose:
                print("Denoms: %.3e %.3e" % (np.dot(Dg.T,Dy)[0,0], multi_dot(Dy.T,H,Dy)[0,0]), end=''),
                print("Dots: %.3e %.3e" % (np.dot(ndg, ndy), np.dot(ndy, nhdy)), end=''),
            H1 = H.copy()
            H += Mat1-Mat2
            Eig1 = np.linalg.eigh(H)[0]
            Eig1.sort()
            if params.verbose:
                print("Eig-ratios: %.5e ... %.5e" % (np.min(Eig1)/np.min(Eig), np.max(Eig1)/np.max(Eig)))
            if np.min(Eig1) <= params.epsilon and params.reset:
                print("Eigenvalues below %.4e (%.4e) - returning guess" % (params.epsilon, np.min(Eig1)))
                H = IC.guess_hessian(coords)
            # Then it's on to the next loop iteration!
    return progress

def CheckInternalGrad(coords, molecule, IC, engine, dirname, verbose=False):
    """ Check the internal coordinate gradient using finite difference. """
    # Initial energy and gradient
    E, gradx = engine.calc(coords, dirname)
    # Initial internal coordinates
    q0 = IC.calculate(coords)
    Gq = IC.calcGrad(coords, gradx)
    for i in range(len(q0)):
        dq = np.zeros_like(q0)
        dq[i] += 1e-4
        x1 = IC.newCartesian(coords, dq, verbose)
        EPlus, _ = engine.calc(x1, dirname)
        dq[i] -= 2e-4
        x1 = IC.newCartesian(coords, dq, verbose)
        EMinus, _ = engine.calc(x1, dirname)
        fdiff = (EPlus-EMinus)/2e-4
        print("%s : % .6e % .6e % .6e" % (IC.Internals[i], Gq[i], fdiff, Gq[i]-fdiff))

def CalcInternalHess(coords, molecule, IC, engine, dirname, verbose=False):
    """
    Calculate the internal coordinate Hessian using finite difference.
    Don't remember when was the last time I used it.
    """
    # Initial energy and gradient
    E, gradx = engine.calc(coords, dirname)
    # Initial internal coordinates
    q0 = IC.calculate(coords)
    for i in range(len(q0)):
        dq = np.zeros_like(q0)
        dq[i] += 1e-4
        x1 = IC.newCartesian(coords, dq, verbose)
        EPlus, _ = engine.calc(x1, dirname)
        dq[i] -= 2e-4
        x1 = IC.newCartesian(coords, dq, verbose)
        EMinus, _ = engine.calc(x1, dirname)
        fdiff = (EPlus+EMinus-2*E)/1e-6
        print("%s : % .6e" % (IC.Internals[i], fdiff))

def print_msg():
    print("""
    #==========================================================================#
    #| If this code has benefited your research, please support us by citing: |#
    #|                                                                        |#
    #| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
    #| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
    #| http://dx.doi.org/10.1063/1.4952956                                    |#
    #==========================================================================#
    """)

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
    verbose : bool
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
            displacement = np.sqrt(np.sum((((x1-coords) * bohr2ang).reshape(-1,3))**2, axis=1))
            rms_displacement = np.sqrt(np.mean(displacement**2))
            max_displacement = np.max(displacement)
            if j != 0:
                dx = (x1-coords)*np.abs(j)*2/max_displacement
            else:
                dx = 0.0
            x.append((coords+dx).reshape(-1,3) * bohr2ang)
            print(i, j, "Displacement (rms/max) = %.5f / %.5f" % (rms_displacement, max_displacement), "(Bork)" if IC.bork else "(Good)")
        M.xyzs = x
        M.write("%s/ic_%03i.xyz" % (dirname, i))

def get_molecule_engine(**kwargs):
    """
    Parameters
    ----------
    args : namespace
        Command line arguments from argparse
    Changed to

    Returns
    -------
    Molecule
        Molecule object containing necessary optimization info
    Engine
        Engine object containing methods for calculating energy and gradient
    """
    ## Read radii from the command line.
    # Ions should have radii of zero.
    arg_radii = kwargs.get('radii', ["Na","0.0","Cl","0.0","K","0.0"])
    # print(arg_radii)
    if (len(arg_radii) % 2) != 0:
        raise RuntimeError("Must have an even number of arguments for radii")
    nrad = int(len(arg_radii) / 2)
    radii = {}
    for i in range(nrad):
        radii[arg_radii[2*i].capitalize()] = float(arg_radii[2*i+1])

    ### Set up based on which quantum chemistry code we're using.
    qchem = kwargs.get('qchem', False)
    psi4 = kwargs.get('psi4', False)
    gmx = kwargs.get('gmx', False)
    molpro = kwargs.get('molpro', False)
    qcengine = kwargs.get('qcengine', False)
    molproexe = kwargs.get('molproexe', None)
    pdb = kwargs.get('pdb', None)
    frag = kwargs.get('frag', False)
    inputf = kwargs.get('input')
    meci = kwargs.get('meci', False)
    meci_sigma = kwargs.get('meci_sigma')
    meci_alpha = kwargs.get('meci_alpha')
    nt = kwargs.get('nt', None)

    if sum([qchem, psi4, gmx, molpro, qcengine]) > 1:
        raise RuntimeError("Do not specify more than one of --qchem, --psi4, --gmx, --molpro, --qcengine")
    if sum([qchem, psi4, gmx, molpro, qcengine, meci]) > 1:
        raise RuntimeError("Do not specify --qchem, --psi4, --gmx, --molpro, --qcengine with --meci")
    if qchem:
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
    elif gmx:
        M = Molecule(inputf, radii=radii, fragment=frag)
        if pdb is not None:
            M = Molecule(pdb, radii=radii, fragment=frag)
        if 'boxes' in M.Data:
            del M.Data['boxes']
        engine = Gromacs(M)
        if nt is not None:
            raise RuntimeError("--nt not configured to work with --gmx yet")
    elif psi4:
        engine = Psi4()
        engine.load_psi4_input(inputf)
        M = engine.M
        if nt is not None:
            engine.set_nt(nt)
    elif molpro:
        engine = Molpro()
        engine.load_molpro_input(inputf)
        M = engine.M
        if nt is not None:
            engine.set_nt(nt)
        if molproexe is not None:
            engine.set_molproexe(molproexe)
    elif qcengine:
        schema = kwargs.get('qcschema', False)
        if schema is False:
            raise RuntimeError("QCEngineAPI option requires a QCSchema")

        program = kwargs.get('qce_program', False)
        if program is False:
            raise RuntimeError("QCEngineAPI option requires a qce_program option")

        engine = QCEngineAPI(schema, program)
        M = engine.M
    else:
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
        if meci:
            engine = TeraChem_CI(M, tcin, meci_sigma, meci_alpha)
        else:
            engine = TeraChem(M, tcin)
            if 'guess' in tcin:
                for f in tcin['guess'].split():
                    if not os.path.exists(f):
                        raise RuntimeError("TeraChem input file specifies guess %s but it does not exist\nPlease include this file in the same folder as your input" % f)
        if nt is not None:
            raise RuntimeError("--nt not configured to work with terachem yet")

    arg_coords = kwargs.get('coords', None)
    if arg_coords is not None:
        M1 = Molecule(arg_coords)
        M1 = M1[-1]
        M.xyzs = M1.xyzs

    return M, engine


def run_optimizer(**kwargs):

    params = OptParams(**kwargs)

    # Get the Molecule and engine objects needed for optimization
    M, engine = get_molecule_engine(**kwargs)

    # Get calculation prefix and temporary directory name
    arg_prefix = kwargs.get('prefix', None)
    inputf = kwargs.get('input')
    prefix = arg_prefix if arg_prefix is not None else os.path.splitext(inputf)[0]
    dirname = prefix+".tmp"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        print("%s exists ; make sure nothing else is writing to the folder" % dirname)
        # Remove existing scratch files in ./run.tmp/scr to avoid confusion
        for f in ['c0', 'ca0', 'cb0']:
            if os.path.exists(os.path.join(dirname, 'scr', f)):
                os.remove(os.path.join(dirname, 'scr', f))

    # QC-specific scratch folder
    qcdir = kwargs.get('qdir', None)
    qchem = kwargs.get('qchem', False)
    if qcdir is not None:
        if not qchem:
            raise RuntimeError("--qcdir only valid if --qchem is specified")
        if not os.path.exists(qcdir):
            raise RuntimeError("--qcdir points to a folder that doesn't exist")
        shutil.copytree(qcdir, os.path.join(dirname, "run.d"))
        engine.M.edit_qcrems({'scf_guess':'read'})
        engine.qcdir = True

    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() * ang2bohr

    # Read in the constraints
    constraints = kwargs.get('constraints', None)
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

    IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVals[0] if CVals is not None else None)

    # Auxiliary functions (will not do optimization)
    displace = kwargs.get('discplace', False)
    verbose = kwargs.get('verbose', False)
    if displace:
        WriteDisplacements(coords, M, IC, dirname, verbose)
        sys.exit()

    fdcheck = kwargs.get('fdcheck', False)
    if fdcheck:
        IC.Prims.checkFiniteDifference(coords)
        CheckInternalGrad(coords, M, IC.Prims, engine, dirname, verbose)
        sys.exit()

    # Print out information about the coordinate system
    if isinstance(IC, CartesianCoordinates):
        print("%i Cartesian coordinates being used" % (3*M.na))
    else:
        print("%i internal coordinates being used (instead of %i Cartesians)" % (len(IC.Internals), 3*M.na))
    print(IC)

    if Cons is None:
        # Run a standard geometry optimization
        if prefix == os.path.splitext(inputf)[0]:
            xyzout = prefix+"_optim.xyz"
            xyzout2="opt.xyz"
        else:
            xyzout = prefix+".xyz"
            xyzout2="opt.xyz"
        progress = Optimize(coords, M, IC, engine, dirname, params, xyzout,xyzout2)
    else:
        # Run a constrained geometry optimization
        if isinstance(IC, (CartesianCoordinates, PrimitiveInternalCoordinates)):
            raise RuntimeError("Constraints only work with delocalized internal coordinates")
        for ic, CVal in enumerate(CVals):
            if len(CVals) > 1:
                print("---=== Scan %i/%i : Constrained Optimization ===---" % (ic+1, len(CVals)))
            IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVal)
            IC.printConstraints(coords, thre=-1)

            if len(CVals) > 1:
                xyzout = prefix+"_scan-%03i.xyz" % ic
                xyzout2="opt.xyz"
            elif prefix == os.path.splitext(kwargs['input'])[0]:
                xyzout = prefix+"_optim.xyz"
                xyzout2="opt.xyz"
            else:
                xyzout = prefix+".xyz"
                xyzout2="opt.xyz"
            progress = Optimize(coords, M, IC, engine, dirname, params, xyzout,xyzout2)
            # update the structure for next optimization in SCAN (by CNH)
            M.xyzs[0] = progress.xyzs[-1]
            coords = progress.xyzs[-1].flatten() * ang2bohr
            print
    print_msg()
    return progress

def main():
    """Read user's input"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--coordsys', type=str, default='tric', help='Coordinate system: "cart" for Cartesian, "prim" for Primitive (a.k.a redundant), '
                        '"dlc" for Delocalized Internal Coordinates, "hdlc" for Hybrid Delocalized Internal Coordinates, "tric" for Translation-Rotation'
                        'Internal Coordinates (default).')
    parser.add_argument('--qchem', action='store_true', help='Run optimization in Q-Chem (pass Q-Chem input).')
    parser.add_argument('--psi4', action='store_true', help='Compute gradients in Psi4.')
    parser.add_argument('--gmx', action='store_true', help='Compute gradients in Gromacs (requires conf.gro, topol.top, shot.mdp).')
    parser.add_argument('--meci', action='store_true', help='Compute minimum-energy conical intersection or crossing point between two SCF solutions (TeraChem only).')
    parser.add_argument('--meci_sigma', type=float, default=3.5, help='Sigma parameter for MECI optimization.')
    parser.add_argument('--meci_alpha', type=float, default=0.025, help='Alpha parameter for MECI optimization.')
    parser.add_argument('--molpro', action='store_true', help='Compute gradients in Molpro.')
    parser.add_argument('--molproexe', type=str, default=None, help='Specify absolute path of Molpro executable.')
    parser.add_argument('--molcnv', action='store_true', help='Use Molpro style convergence criteria instead of the default.')
    parser.add_argument('--prefix', type=str, default=None, help='Specify a prefix for output file and temporary directory.')
    parser.add_argument('--displace', action='store_true', help='Write out the displacements of the coordinates.')
    parser.add_argument('--fdcheck', action='store_true', help='Check internal coordinate gradients using finite difference..')
    parser.add_argument('--enforce', action='store_true', help='Enforce exact constraints (activated when constraints are almost satisfied)')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Small eigenvalue threshold.')
    parser.add_argument('--check', type=int, default=0, help='Check coordinates every N steps to see whether it has changed.')
    parser.add_argument('--verbose', action='store_true', help='Write out the displacements.')
    parser.add_argument('--reset', action='store_true', help='Reset Hessian when eigenvalues are under epsilon.')
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
    parser.add_argument('--nt', type=int, help='Specify number of threads for running in parallel (for TeraChem this should be number of GPUs)')
    parser.add_argument('input', type=str, help='TeraChem or Q-Chem input file')
    parser.add_argument('constraints', type=str, nargs='?', help='Constraint input file (optional)')
    print('geometric-optimize called with the following command line:')
    print(' '.join(sys.argv))
    args = parser.parse_args(sys.argv[1:])
    # Run the optimizer.
    run_optimizer(**vars(args))

if __name__ == "__main__":
    main()
