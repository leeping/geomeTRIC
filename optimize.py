#!/usr/bin/env python

from __future__ import division
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from internal import PrimitiveInternalCoordinates, DelocalizedInternalCoordinates, RotationA, RotationB, RotationC, Distance, Angle, Dihedral, OutOfPlane, CartesianX, CartesianY, CartesianZ, TranslationX, TranslationY, TranslationZ
from rotate import get_rot, sorted_eigh
from forcebalance.molecule import Molecule, Elements
from forcebalance.nifty import row, col, flat, invert_svd, uncommadash, isint
from scipy import optimize
import scipy
import traceback
import argparse
import subprocess
import itertools
import os, sys, shutil

parser = argparse.ArgumentParser()
parser.add_argument('--cart', action='store_true', help='Use Cartesian coordinate system.')
parser.add_argument('--connect', action='store_true', help='Connect noncovalent molecules into a network.')
parser.add_argument('--constraints', type=str, help='Provide a text file specifying geometry constraints.')
parser.add_argument('--redund', action='store_true', help='Use redundant coordinate system.')
parser.add_argument('--terachem', action='store_true', help='Run optimization in TeraChem (pass xyz as first argument).')
parser.add_argument('--dftd', action='store_true', help='Turn on dispersion correction in TeraChem.')
parser.add_argument('--double', action='store_true', help='Run TeraChem in double precision mode.')
parser.add_argument('--prefix', type=str, default=None, help='Specify a prefix for output file and temporary directory.')
parser.add_argument('--displace', action='store_true', help='Write out the displacements.')
parser.add_argument('--epsilon', type=float, default=1e-5, help='Small eigenvalue threshold.')
parser.add_argument('-c', '--check', type=int, default=10, help='Check coordinates every N steps (-1 for no check).')
parser.add_argument('-v', '--verbose', action='store_true', help='Write out the displacements.')
parser.add_argument('--reseth', action='store_true', help='Reset Hessian when eigenvalues are under epsilon.')
parser.add_argument('--rfo', action='store_true', help='Use rational function optimization (leave off = trust radius Newton Raphson).')
parser.add_argument('--trust', type=float, default=0.1, help='Starting trust radius.')
parser.add_argument('--tmax', type=float, default=0.3, help='Maximum trust radius.')

print ' '.join(sys.argv)

args, sys.argv = parser.parse_known_args(sys.argv)

TeraChem = args.terachem

TCHome = "/home/leeping/src/terachem/production/build"

if args.prefix is None:
    prefix = os.path.splitext(sys.argv[1])[0]+"_optim"
else:
    prefix = args.prefix

dirname = prefix+".tmp"
if os.path.exists(dirname):
    raise RuntimeError("Please delete temporary folder %s before proceeding" % dirname)
os.makedirs(dirname)

os.environ['TeraChem'] = TCHome
os.environ['PATH'] = os.path.join(TCHome,'bin')+":"+os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = os.path.join(TCHome,'lib')+":"+os.environ['LD_LIBRARY_PATH']

# For compactness, this script always reads in a Q-Chem input file and translates it
# to the corresponding TeraChem input file.
TeraTemp = """
coordinates start.xyz
run gradient
basis {basis}
method {ur}{method}
charge {charge}
spinmult {mult}
precision {precision}
convthre 1.0e-6
threall 1.0e-16
mixguess 0.0
scf diis+a
{guess}
{dftd}
"""

eps = args.epsilon

# Read in the molecule
M = Molecule(sys.argv[1], radii={'Na':0.0})
if 'method' in M.qcrems[0]:
    method = method
else:
    method = M.qcrems[0]['exchange']

def calc_terachem(coords):
    coord_hash = tuple(list(coords))
    if coord_hash in calc_terachem.stored_calcs:
        # print "Reading stored values"
        energy = calc_terachem.stored_calcs[coord_hash]['energy']
        gradient = calc_terachem.stored_calcs[coord_hash]['gradient']
        return energy, gradient
    # print "Calling TeraChem for energy and gradient"
    guesses = []
    for f in ['c0', 'ca0', 'cb0']:
        if os.path.exists(os.path.join(dirname, 'scr', f)):
            shutil.copy2(os.path.join(dirname, 'scr', f), os.path.join(dirname, f))
            guesses.append(f)
            have_guess = True
    else: have_guess = False
    with open("%s/run.in" % dirname, "w") as f:
        print >> f, TeraTemp.format(basis = M.qcrems[0]['basis'],
                                    ur = "u" if M.mult != 1 else "r",
                                    method = method,
                                    charge = str(M.charge), mult=str(M.mult),
                                    precision = "double" if args.double else "dynamic",
                                    guess = ("guess " + ' '.join(guesses) if have_guess else ""),
                                    dftd = ("dispersion yes" if args.dftd else ""))
    # Convert coordinates back to the xyz file
    M.xyzs[0] = coords.reshape(-1, 3) * 0.529
    M[0].write(os.path.join(dirname, 'start.xyz'))
    # Run TeraChem
    subprocess.call('terachem run.in > run.out', cwd=dirname, shell=True)
    # Extract energy and gradient
    subprocess.call("awk '/FINAL ENERGY/ {print $3}' run.out > energy.txt", cwd=dirname, shell=True)
    subprocess.call("awk '/Gradient units are Hartree/,/Net gradient/ {if ($1 ~ /^-?[0-9]/) {print}}' run.out > grad.txt", cwd=dirname, shell=True)
    energy = float(open(os.path.join(dirname,'energy.txt')).readlines()[0].strip())
    gradient = np.loadtxt(os.path.join(dirname,'grad.txt')).flatten()
    if len(calc_terachem.stored_calcs.keys()) > 0:
        PrevHash = calc_terachem.stored_calcs.keys()[-1]
        displacement = coords - calc_terachem.stored_calcs[PrevHash]['coords']
    calc_terachem.stored_calcs[coord_hash] = {'coords':coords,'energy':energy,'gradient':gradient}
    return energy, gradient
calc_terachem.calcnum = 0
calc_terachem.stored_calcs = OrderedDict()

def calc_qchem(coords):
    coord_hash = tuple(list(coords))
    if coord_hash in calc_qchem.stored_calcs:
        energy = calc_qchem.stored_calcs[coord_hash]['energy']
        gradient = calc_qchem.stored_calcs[coord_hash]['gradient']
        return energy, gradient
    # print "Calling Qchem for energy and gradient"
    if not os.path.exists(dirname): os.makedirs(dirname)
    # Convert coordinates back to the xyz file
    M.xyzs[0] = coords.reshape(-1, 3) * 0.529
    M.edit_qcrems({'jobtype':'force'})
    M[0].write(os.path.join(dirname, 'run.in'))
    # Run Qchem
    subprocess.call('%s/runqc run.in run.out &> run.log' % os.path.dirname(os.path.abspath(__file__)), cwd=dirname, shell=True)
    M1 = Molecule('%s/run.out' % dirname)
    energy = M1.qm_energies[0]
    gradient = M1.qm_grads[0].flatten()
    if len(calc_qchem.stored_calcs.keys()) > 0:
        PrevHash = calc_qchem.stored_calcs.keys()[-1]
        displacement = coords - calc_qchem.stored_calcs[PrevHash]['coords']
    calc_qchem.stored_calcs[coord_hash] = {'coords':coords,'energy':energy,'gradient':gradient}
    return energy, gradient
calc_qchem.calcnum = 0
calc_qchem.stored_calcs = OrderedDict()

def calc(coords):
    if args.terachem:
        e, g = calc_terachem(coords)
    else:
        e, g = calc_qchem(coords)
    return e, g

def calc_energy(coords):
    print "Getting energy:",
    return calc(coords)[0]

def calc_grad(coords):
    print "Getting gradient:",
    return calc(coords)[1]

def energy_internal(q):
    IC = energy_internal.IC
    x0 = energy_internal.x0
    q0 = IC.calculate(x0)
    dQ = IC.subtractInternal(q,q0)
    newxyz = IC.newCartesian(x0, dQ, verbose=args.verbose)
    energy_internal.x0 = newxyz
    return calc(newxyz)[0]
energy_internal.IC = None
energy_internal.x0 = None

def gradient_internal(q):
    IC = gradient_internal.IC
    x0 = gradient_internal.x0
    q0 = IC.calculate(x0)
    dQ = IC.subtractInternal(q,q0)
    newxyz = IC.newCartesian(x0, dQ, verbose=args.verbose)
    gradient_internal.x0 = newxyz
    Gx = np.matrix(calc(newxyz)[1]).T
    Ginv = IC.GInverse(newxyz)
    Bmat = IC.wilsonB(newxyz)
    Gq = np.matrix(Ginv)*np.matrix(Bmat)*Gx
    return np.array(Gq).flatten()
gradient_internal.IC = None
gradient_internal.x0 = None

def RebuildHessian(IC, H0, coord_seq, grad_seq, trust=0.3):
    Na = len(coord_seq[0])/3
    history = 0
    for i in range(2, len(coord_seq)+1):
        disp = 0.529*(coord_seq[-i]-coord_seq[-1])
        rmsd = np.sqrt(np.sum(disp**2)/Na)
        # print -i, rmsd
        if rmsd > trust: break
        history += 1
    if history < 1:
        return H0.copy()
    print "Rebuilding Hessian using %i gradients" % history
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
        Mat1 = (Dg*Dg.T)/(Dg.T*Dy)[0,0]
        Mat2 = ((H*Dy)*(H*Dy).T)/(Dy.T*H*Dy)[0,0]
        Hstor = H.copy()
        H += Mat1-Mat2
    if np.min(np.linalg.eigh(H)[0]) < eps and args.reseth:
        print "Eigenvalues below %.4e (%.4e) - returning guess" % (eps,np.min(np.linalg.eigh(H)[0]))
        return H0.copy()
    return H

def getCartesianNorm(X, dy, IC):
    # Displacement of each atom in Angstrom
    Xnew = IC.newCartesian(X, dy, verbose=args.verbose)
    Xmat = np.matrix(Xnew.reshape(-1,3))
    U = get_rot(Xmat, X.reshape(-1,3))
    Xrot = np.array((U*Xmat.T).T).flatten()
    displacement = np.sqrt(np.sum((((Xrot-X)*0.529).reshape(-1,3))**2, axis=1))
    rms_displacement = np.sqrt(np.mean(displacement**2))
    return rms_displacement

def between(s, a, b):
    if a < b:
        return s > a and s < b
    elif a > b:
        return s > b and s < a
    else:
        raise RuntimeError('a and b must be different')

def brent_wiki(f, a, b, rel, cvg=0.1):
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
        # Check convergence
        if np.abs(fs/rel) <= cvg:
            return s
        if np.abs(b-a) < epsilon:
            if args.verbose: print "returning because interval is too small"
            f.small_interval = True
            return s
        if hasattr(f, 'from_above'):
            if f.from_above and fs > 0:
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
    print "(x, y) = ", x, answer
    return answer

#brent_wiki(ftest, -4, 4/3, 1, 1e-8)
#sys.exit()

# def getConstraint(X, IC, cPrim):
#     if type(IC) is DelocalizedInternalCoordinates:
#         Prims = IC.Prims.Internals
#         # Get the "primitive number" in the delocalized internal coordinates.
#         iPrim = Prims.index(cPrim)
#         return cPrim.value(X), np.array(IC.Vecs[iPrim, :]).flatten()
#     else:
#         raise RuntimeError('Spoo!')

def OneDScan(init, final, steps):
    # Return a list of values 
    if len(init) != len(final):
        raise RuntimeError("init and final must have the same length")
    Answer = []
    for j in range(len(init)):
        Answer.append(np.linspace(init[j], final[j], steps))
    Answer = list([list(i) for i in np.array(Answer).T])
    print Answer
    return Answer

def ParseConstraints(molecule, cFile):
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
                 "xyz":([CartesianX, CartesianY, CartesianZ], 1)
                 }
    CDict_Trans = {"x":([TranslationX], 1), 
                   "y":([TranslationY], 1), 
                   "z":([TranslationZ], 1),
                   "xy":([TranslationX, TranslationY], 1), 
                   "xz":([TranslationX, TranslationZ], 1), 
                   "yz":([TranslationY, TranslationZ], 1), 
                   "xyz":([TranslationX, TranslationY, TranslationZ], 1)
                   }
    AtomKeys = ["x", "y", "z", "xy", "yz", "xz", "xyz"]
    objs = []
    vals = []
    for line in open(cFile).readlines():
        line = line.split("#")[0].strip().lower()
        # This is a list-of-lists. The intention is to create a multidimensional grid
        # of constraint values if necessary.
        if len(line) == 0: continue
        print line
        if line.startswith("$"):
            mode = line.replace("$","")
        else:
            if mode is None:
                raise RuntimeError("Mode ($freeze, $set, $scan) must be set before specifying any constraints")
            s = line.split()
            key = s[0]
            if ''.join(sorted(key)) in AtomKeys:
                key = ''.join(sorted(key))
            classes, n_atom = ClassDict[key]
            if mode == "freeze":
                ntok = n_atom
            elif mode == "set":
                ntok = n_atom + len(classes)
            elif mode == "scan":
                ntok = n_atom + 2*len(classes) + 1
            if len(s) != (ntok+1):
                raise RuntimeError("For this line:%s\nExpected %i tokens but got %i" % (line, ntok+1, len(s)))
            if key in AtomKeys:
                # Special code that works for atom position constraints.
                # First figure out the range of atoms.
                if isint(s[1]):
                    atoms = [int(s[1])-1]
                elif s[1] in [k.lower() for k in Elements]:
                    atoms = [i for i in range(molecule.na) if molecule.elem[i].lower() == key]
                else:
                    atoms = uncommadash(s[1])
                if any([i<0 for i in atoms]):
                    raise RuntimeError("Atom numbers must start from 1")
                if any([i>=molecule.na for i in atoms]):
                    raise RuntimeError("Constraints refer to higher atom indices than the number of atoms")
                if mode in ["set", "scan"]:
                    # If there is more than one atom and the mode is "set" or "scan", then the
                    # center of mass is constrained, so we pick the corresponding classes.
                    if len(atoms) > 1:
                        objs.append([cls(atoms, w=np.ones(len(atoms))/len(atoms)) for cls in CDict_Trans[key][0]])
                        # for cls in CDict_Trans[key][0]:
                        #     objs.append(cls(atoms, w=np.ones(len(atoms))/len(atoms)))
                    else:
                        objs.append([cls(atoms[0], w=1.0) for cls in classes])
                        # for cls in classes:
                        #     objs.append(cls(atoms[0], w=1.0))
                    # Depending on how many coordinates are constrained, we read in the corresponding
                    # number of constraint values.
                    x1 = [float(i)/0.529 for i in s[2:2+len(classes)]]
                    # If there's just one constraint value then we append it to the value list-of-lists
                    if mode == "set":
                        vals.append([x1])
                    elif mode == "scan":
                        # If we're scanning it, then we add the whole list of distances to the list-of-lists
                        x2 = [float(i)/0.529 for i in s[2+len(classes):2+2*len(classes)]]
                        nstep = int(s[2+2*len(classes)])
                        vals.append(OneDScan(x1, x2, nstep))
                elif mode == "freeze":
                    # Freezing atoms works a bit differently, we add one constraint for each
                    # atom in the range and append None to the values.
                    for a in atoms:
                        for cls in classes:
                            objs.append([cls(a, w=1.0)])
                            vals.append([[None]])
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
                    if key == "distance": x1 = float(s[1+n_atom])/0.529
                    else: x1 = float(s[1+n_atom])*np.pi/180.0
                    if mode == "set":
                        vals.append([[x1]])
                    else:
                        if key == "distance": x2 = float(s[2+n_atom])/0.529
                        else: x2 = float(s[2+n_atom])*np.pi/180.0
                        nstep = int(s[3+n_atom])
                        vals.append([[i] for i in list(np.linspace(x1,x2,nstep))])
            else:
                raise RuntimeError("Line not supported: %s" % line)
    if len(objs) != len(vals):
        raise RuntimeError("objs and vals should be the same length")
    valgrps = [list(itertools.chain(*i)) for i in list(itertools.product(*vals))]
    # print valgrps
    objs = list(itertools.chain(*objs))
    return objs, valgrps

# Read in the constraints
if args.constraints is not None:
    Cons, CVals = ParseConstraints(M, args.constraints)
else:
    Cons = None
    CVals = None
            
def Optimize(coords, molecule, IC=None, xyzout=None, printIC=True):
    progress = deepcopy(molecule)
    # Initial Hessian
    if IC is not None:
        if printIC:
            print "%i internal coordinates being used (rather than %i Cartesians)" % (len(IC.Internals), 3*molecule.na)
            print IC
        internal = True
        H0 = IC.guess_hessian(coords)
    else:
        internal = False
        H0 = np.eye(len(coords))
    H = H0.copy()
    # Cartesian coordinates
    X = coords.copy()
    # Initial energy and gradient
    E, gradx = calc(coords)
    if internal:
        # Initial internal coordinates
        q0 = IC.calculate(coords)
        Gq = IC.calcGrad(X, gradx)
        # The optimization variables are the internal coordinates.
        Y = q0.copy()
        G = np.array(Gq).flatten()
    else:
        # The optimization variables are the Cartesian coordinates.
        Y = coords.copy()
        G = gradx.copy()

    # Loop of optimization
    Iteration = 0
    CoordCounter = 0
    trust = args.trust
    thre_rj = 0.01
    # Print initial iteration
    gradxc = IC.calcGradProj(X, gradx)
    atomgrad = np.sqrt(np.sum((gradxc.reshape(-1,3))**2, axis=1))
    rms_gradient = np.sqrt(np.mean(atomgrad**2))
    max_gradient = np.max(atomgrad)
    print "Step %4i :" % Iteration,
    print "Gradient = %.3e/%.3e (rms/max) Energy = % .10f" % (rms_gradient, max_gradient, E)
    if IC is not None and IC.haveConstraints():
        IC.printConstraints(X)
    # Threshold for "low quality step" which decreases trust radius.
    ThreLQ = 0.25
    # Threshold for "high quality step" which increases trust radius.
    ThreHQ = 0.75
    # print np.diag(H)
    Convergence_energy = 1e-6
    Convergence_grms = 3e-4
    Convergence_gmax = 4.5e-4
    Convergence_drms = 1.2e-3
    Convergence_dmax = 1.8e-3
    X_hist = [X]
    Gx_hist = [gradx]
    CartesianTrust = True
    trustprint = "="
    while 1:
        Iteration += 1
        # Force Hessian to have positive eigenvalues.
        Eig = np.linalg.eigh(H)[0]
        Emin = min(Eig).real
        if args.rfo:
            v0 = 1.0
        elif Emin < eps:
            v0 = eps-Emin
        else:
            v0 = 0.0
        Eig = np.linalg.eigh(H)[0]
        Eig = sorted(Eig)
        print "Hessian Eigenvalues: %.5e %.5e %.5e ... %.5e %.5e %.5e" % (Eig[0],Eig[1],Eig[2],Eig[-3],Eig[-2],Eig[-1])
        # Functions that help us to find the trust radius step.
        # This returns the Newton-Raphson step given a multiple of the diagonal
        # added to the Hessian, the expected decrease in the energy, and
        # the derivative of the step length w/r.t. v.
        def get_delta_prime_trm(v):
            HC, GC = IC.augmentGH(X, G, H)
            # print HC, GC
            # raw_input()
            HT = HC + v*np.eye(len(HC))
            try:
                Hi = invert_svd(np.matrix(HT))
            except:
                print "SVD Error - increasing v by 0.001 and trying again"
                return get_delta_prime_trm(v+0.001)
            dyc = flat(-1 * Hi * col(GC))
            dy = dyc[:len(G)]
            d_prime = flat(-1 * Hi * col(dyc))[:len(G)]
            dy_prime = np.dot(dy,d_prime)/np.linalg.norm(dy)
            sol = flat(0.5*row(dy)*np.matrix(H)*col(dy))[0] + np.dot(dy,G)
            return dy, sol, dy_prime

        def get_delta_prime_rfo(alpha):
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
            """
            S = alpha*np.matrix(np.eye(len(H)))
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
            expect = lmin/2*(1+row(dy)*S*col(dy))[0]
            dyprime1 = dyprime2 / (2*np.sqrt(dy2))
            return dy, expect, dyprime1

        def get_delta_prime(v):
            """
            Return the internal coordinate step given a parameter "v". 
            "v" refers to the multiple of the identity added to the Hessian
            in trust-radius Newton Raphson (TRM), and the multiple of the
            identity on the RHS matrix in rational function optimization (RFO).
            Note that reasonable default values are v = 0.0 in TRM and 1.0 in RFO.
            """
            if args.rfo:
                return get_delta_prime_rfo(v)
            else:
                return get_delta_prime_trm(v)

        # this applies an iteration formula to find the trust radius step,
        # given the target value of the trust radius.
        def trust_step(target, v0):
            dy, sol, dy_prime = get_delta_prime(v0)
            ndy = np.linalg.norm(dy)
            if ndy < target:
                return dy, sol
            v = v0
            # print "v: %.4e ndy -> target: %.4e -> %.4e" % (v, ndy, target)
            while True:
                v += (1-ndy/target)*(ndy/dy_prime)
                dy, sol, dy_prime, = get_delta_prime(v)
                ndy = np.linalg.norm(dy)
                # print "v: %.4e ndy -> target: %.4e -> %.4e" % (v, ndy, target)
                if np.abs((ndy-target)/target) < 0.001:
                    return dy, sol

        # If our trust radius is to be computed in Cartesian coordinates,
        # then we use an outer loop to find the appropriate step
        if CartesianTrust:
            dy, expect, _ = get_delta_prime(v0)
            inorm = np.linalg.norm(dy)
            dynorm = getCartesianNorm(X, dy, IC)
            if args.verbose: print "dy(i): %.4f dy(c) -> target: %.4f -> %.4f" % (inorm, dynorm, trust)
            if dynorm > 1.1 * trust:
                def froot(trial):
                    if trial == 0.0: 
                        froot.from_above = False
                        return -trust
                    else:
                        if trial in froot.stores:
                            dynorm = froot.stores[trial]
                            froot.from_above = False
                        else:
                            dy, expect = trust_step(trial, v0)
                            dynorm = getCartesianNorm(X, dy, IC)
                            froot.from_above = (froot.above_flag and not IC.bork)
                            froot.stores[trial] = dynorm
                            froot.counter += 1
                        # Store the largest trial value with dynorm below the target
                        if froot.current_val is None:
                            if dynorm-froot.target < 0:
                                froot.current_arg = trial
                                froot.current_val = dynorm
                        elif dynorm-froot.target < 0:
                            if dynorm > froot.current_val:
                                froot.current_arg = trial
                                froot.current_val = dynorm
                        if args.verbose: print "dy(i) %.4f dy(c) -> target: %.4f -> %.4f%s" % (trial, dynorm, froot.target, " (done)" if froot.from_above else "")
                        return dynorm-froot.target
                froot.counter = 0
                froot.stores = {inorm : dynorm}
                froot.target = trust
                froot.above_flag = False
                froot.current_arg = None
                froot.current_val = None
                froot.small_interval = False
                iopt = brent_wiki(froot, 0, inorm, trust, cvg=0.1)
                if froot.small_interval:
                    iopt = froot.current_arg
                for i in range(3):
                    if (not froot.small_interval) and IC.bork:
                        froot.target /= 2
                        if args.verbose: print "\x1b[93mReducing target to %.3e\x1b[0m" % froot.target
                        froot.above_flag = True
                        iopt = brent_wiki(froot, 0, iopt, froot.target, cvg=0.1)
                    else: break
                if IC.bork:
                    print "\x1b[91mInverse iteration for Cartesians failed\x1b[0m"
                    iopt = 1e-3
                else:
                    if args.verbose: print "\x1b[93mBrent algorithm requires %i evaluations\x1b[0m" % froot.counter
                if iopt is None:
                    iopt = 1e-3
                dy, expect = trust_step(iopt, v0)
        else:
            # This code is rarely used; trust radius in internal coordinates
            dy, expect = trust_step(trust, v0)
            dynorm = np.linalg.norm(dy)

        Dot = -np.dot(dy/np.linalg.norm(dy), G/np.linalg.norm(G))
        bump = dynorm > 0.8 * trust
        # Get the previous iteration stuff
        Yprev = Y.copy()
        Xprev = X.copy()
        Gprev = G.copy()
        Eprev = E
        Y += dy
        if internal:
            X = IC.newCartesian(X, dy, verbose=args.verbose)
        else:
            X = Y.copy()
        E, gradx = calc(X)
        # Calculate quantities for convergence
        Xmat = X.reshape(-1,3)
        U = get_rot(Xmat, Xprev.reshape(-1,3))
        Xrot = np.array((U*Xmat.T).T).flatten()
        displacement = np.sqrt(np.sum((((Xrot-Xprev)*0.529).reshape(-1,3))**2, axis=1))
        # Add new Cartesian coordinates and gradients to history
        progress.xyzs.append(X.reshape(-1,3) * 0.529)
        progress.comms.append('Iteration %i Energy % .8f' % (Iteration, E))
        progress.write(xyzout)
        # Project out the degrees of freedom that are constrained
        gradxc = IC.calcGradProj(X, gradx)
        atomgrad = np.sqrt(np.sum((gradxc.reshape(-1,3))**2, axis=1))
        rms_displacement = np.sqrt(np.mean(displacement**2))
        rms_gradient = np.sqrt(np.mean(atomgrad**2))
        max_displacement = np.max(displacement)
        max_gradient = np.max(atomgrad)
        Quality = (E-Eprev)/expect
        Converged_energy = np.abs(E-Eprev) < Convergence_energy
        Converged_grms = rms_gradient < Convergence_grms
        Converged_gmax = max_gradient < Convergence_gmax
        Converged_drms = rms_displacement < Convergence_drms
        Converged_dmax = max_displacement < Convergence_dmax
        #BadStep = (E-Eprev) > 0
        BadStep = Quality < 0
            
        print "Step %4i :" % Iteration,
        print "Displace = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("\x1b[92m" if Converged_drms else "\x1b[0m", rms_displacement, "\x1b[92m" if Converged_dmax else "\x1b[0m", max_displacement),
        print "Trust = %.3e (%s)" % (trust, trustprint), 
        print "Grad%s = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("_T" if IC.haveConstraints() else "", "\x1b[92m" if Converged_grms else "\x1b[0m", rms_gradient, "\x1b[92m" if Converged_gmax else "\x1b[0m", max_gradient),
        print "Dy.G = %.3f" % Dot,
        print "E (change) = % .10f (%s%+.3e\x1b[0m) Quality = %s%.3f\x1b[0m" % (E, "\x1b[91m" if BadStep else ("\x1b[92m" if Converged_energy else "\x1b[0m"), E-Eprev, "\x1b[91m" if BadStep else "\x1b[0m", Quality)
        if IC is not None and IC.haveConstraints():
            IC.printConstraints(X)
        if IC is not None:
            idx = np.argmax(np.abs(dy))
            iunit = np.zeros_like(dy)
            iunit[idx] = 1.0
            if type(IC) is PrimitiveInternalCoordinates:
                print "Along %s %.3f" % (IC.Internals[idx], np.dot(dy/np.linalg.norm(dy), iunit))
        if Converged_energy and Converged_grms and Converged_drms and Converged_gmax and Converged_dmax:
            print "Converged! =D"
            break
            # if (trust > Convergence_dmax) or (not bump):
            #     print "Converged! =D"
            #     break
            # else:
            #     print "Trust radius is too small to converge"
        rejectOk = trust > thre_rj
        if Quality <= ThreLQ:
            trust = max(Convergence_drms, min(trust, dynorm)/2)
            trustprint = "\x1b[91m-\x1b[0m"
        elif Quality >= ThreHQ and bump:
            if trust < args.tmax:
                trust = min(np.sqrt(2)*trust, args.tmax)
                trustprint = "\x1b[92m+\x1b[0m"
            else:
                trustprint = "="
        else:
            trustprint = "="
        if Quality < -1 and rejectOk:
            trustprint = "\x1b[1;91mx\x1b[0m"
            Y = Yprev.copy()
            X = Xprev.copy()
            G = Gprev.copy()
            E = Eprev
        else:
            if Quality < -1:
                print "\x1b[93mNot rejecting step - trust below %.3e\x1b[0m" % thre_rj
            X_hist.append(X)
            Gx_hist.append(gradx)
            skipBFGS=False
            if internal:
                check = False
                reinit = False
                if IC.largeRots():
                    print "Large rotations - reinitializing coordinates"
                    reinit = True
                if IC.bork:
                    print "Failed inverse iteration - reinitializing coordinates"
                    check = True
                    reinit = True
                if CoordCounter == (args.check - 1):
                    check = True
                if check:
                    newmol = deepcopy(molecule)
                    newmol.xyzs[0] = X.reshape(-1,3)*0.529
                    newmol.build_topology()
                    if type(IC) is PrimitiveInternalCoordinates:
                        IC1 = PrimitiveInternalCoordinates(newmol, connect=args.connect)
                    elif type(IC) is DelocalizedInternalCoordinates:
                        IC1 = DelocalizedInternalCoordinates(newmol, build=False, connect=args.connect)
                        IC1.getConstraints_from(IC)
                    else:
                        raise RuntimeError('Spoo!')
                    if IC1 != IC:
                        print "\x1b[1;94mInternal coordinate system may have changed\x1b[0m"
                        if IC.repr_diff(IC1) != "":
                            print IC.repr_diff(IC1)
                        reinit = True
                        IC = IC1
                    CoordCounter = 0
                else:
                    CoordCounter += 1
                if reinit:
                    IC.resetRotations(X)
                    if type(IC) is DelocalizedInternalCoordinates:
                        IC.build_dlc(X)
                    H0 = IC.guess_hessian(coords)
                    # H0 = np.eye(len(IC.Internals))
                    H = RebuildHessian(IC, H0, X_hist, Gx_hist, 0.3)
                    # H = H0.copy()
                    Y = IC.calculate(X)
                    skipBFGS = True
                Gq = IC.calcGrad(X, gradx)
                G = np.array(Gq).flatten()
                rms_gq = np.sqrt(np.mean(G**2))
                max_gq = np.max(np.abs(G))
                if max_gq > 1:
                    print "Gq = %.3e/%.3e (rms/max)" % (rms_gq, max_gq),
            else:
                G = gradx.copy()

            if not skipBFGS:
                # BFGS Hessian update
                Dy   = col(Y - Yprev)
                Dg   = col(G - Gprev)
                Mat1 = (Dg*Dg.T)/(Dg.T*Dy)[0,0]
                Mat2 = ((H*Dy)*(H*Dy).T)/(Dy.T*H*Dy)[0,0]
                Eig = np.linalg.eigh(H)[0]
                Eig.sort()

                ndy = np.array(Dy).flatten()/np.linalg.norm(np.array(Dy))
                ndg = np.array(Dg).flatten()/np.linalg.norm(np.array(Dg))
                nhdy = np.array(H*Dy).flatten()/np.linalg.norm(np.array(H*Dy))
                if args.verbose: 
                    print "Denoms: %.3e %.3e" % ((Dg.T*Dy)[0,0], (Dy.T*H*Dy)[0,0]),
                    print "Dots: %.3e %.3e" % (np.dot(ndg, ndy), np.dot(ndy, nhdy)),
                H1 = H.copy()
                H += Mat1-Mat2
                Eig1 = np.linalg.eigh(H)[0]
                Eig1.sort()
                if args.verbose:
                    print "Eig-ratios: %.5e ... %.5e" % (np.min(Eig1)/np.min(Eig), np.max(Eig1)/np.max(Eig))
                if np.min(Eig1) <= eps and args.reseth:
                    print "Eigenvalues below %.4e (%.4e) - returning guess" % (eps, np.min(Eig1))
                    H = IC.guess_hessian(coords)
        # Then it's on to the next loop iteration!
    return X

def CheckInternalGrad(coords, molecule, IC):
    # Initial energy and gradient
    E, gradx = calc(coords)
    # Initial internal coordinates
    q0 = IC.calculate(coords)
    Gq = IC.calcGrad(coords, gradx)
    for i in range(len(q0)):
        dq = np.zeros_like(q0)
        dq[i] += 1e-4
        x1 = IC.newCartesian(coords, dq, verbose=args.verbose)
        EPlus, _ = calc(x1)
        dq[i] -= 2e-4
        x1 = IC.newCartesian(coords, dq, verbose=args.verbose)
        EMinus, _ = calc(x1)
        fdiff = (EPlus-EMinus)/2e-4
        print "%s : % .6e % .6e % .6e" % (IC.Internals[i], Gq[i], fdiff, Gq[i]-fdiff)

def CalcInternalHess(coords, molecule, IC):
    # Initial energy and gradient
    E, gradx = calc(coords)
    # Initial internal coordinates
    q0 = IC.calculate(coords)
    for i in range(len(q0)):
        dq = np.zeros_like(q0)
        dq[i] += 1e-4
        x1 = IC.newCartesian(coords, dq, verbose=args.verbose)
        EPlus, _ = calc(x1)
        dq[i] -= 2e-4
        x1 = IC.newCartesian(coords, dq, verbose=args.verbose)
        EMinus, _ = calc(x1)
        fdiff = (EPlus+EMinus-2*E)/1e-6
        print "%s : % .6e" % (IC.Internals[i], fdiff)

def main():
    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() / 0.529
    if args.cart:
        IC = None
    elif args.redund:
        IC = PrimitiveInternalCoordinates(M, connect=args.connect)
    else:
        IC = DelocalizedInternalCoordinates(M, connect=args.connect)
        IC.build_dlc(coords)
    if args.displace:
        # if not args.redund:
        #     IC.weight_vectors(coords)
        for i in range(len(IC.Internals)):
            x = []
            for j in np.linspace(-1.0, 1.0, 11):
                if j != 0:
                    dq = np.zeros(len(IC.Internals))
                    dq[i] = j
                    x1 = IC.newCartesian(coords, dq, verbose=args.verbose)
                else:
                    x1 = coords.copy()
                displacement = np.sqrt(np.sum((((x1-coords)*0.529).reshape(-1,3))**2, axis=1))
                rms_displacement = np.sqrt(np.mean(displacement**2))
                max_displacement = np.max(displacement)
                x.append(x1.reshape(-1,3) * 0.529)
                print i, j, "Displacement (rms/max) = %.5f / %.5f" % (rms_displacement, max_displacement), "(Bork)" if IC.bork else "(Good)"
            M.xyzs = x
            M.write("%s/ic_%03i.xyz" % (dirname, i))
        sys.exit()
                
    FDCheck = False
    if FDCheck:
        IC.checkFiniteDifference(coords)
        CheckInternalGrad(coords, M, IC)
        sys.exit()

    if Cons is None:
        xyzout = prefix+".xyz"
        opt_coords = Optimize(coords, M, IC, xyzout=xyzout)
    else:
        # ccoord = []
        for ic, CVal in enumerate(CVals):
            IC = DelocalizedInternalCoordinates(M, connect=args.connect, constraints=Cons, cvals=CVal)
            IC.build_dlc(coords)
            if len(CVals) > 1:
                xyzout = prefix+"scan_%03i.xyz" % ic
            else:
                xyzout = prefix+".xyz"
            coords = IC.applyConstraints(coords)
            coords = Optimize(coords, M, IC, xyzout=xyzout, printIC=(ic==0))
    
if __name__ == "__main__":
    main()

