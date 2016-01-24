#!/usr/bin/env python

from __future__ import division
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from internal import *
from rotate import get_rot, sorted_eigh, calc_fac_dfac
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
parser.add_argument('--coordsys', type=str, default='xdlc', help='Coordinate system: "cart" for Cartesian, "prim" for Primitive (a.k.a redundant), '
                    '"dlc" for Delocalized Internal Coordinates, "hdlc" for Hybrid Delocalized Internal Coordinates, "xdlc" for explicit translations'
                    'and rotations added to DLC (default).')
parser.add_argument('--qchem', action='store_true', help='Run optimization in Q-Chem (pass Q-Chem input).')
parser.add_argument('--psi4', action='store_true', help='Compute gradients in Psi4.')
parser.add_argument('--prefix', type=str, default=None, help='Specify a prefix for output file and temporary directory.')
parser.add_argument('--displace', action='store_true', help='Write out the displacements of the coordinates.')
parser.add_argument('--enforce', action='store_true', help='Enforce exact constraints (activated when constraints are almost satisfied)')
parser.add_argument('--epsilon', type=float, default=1e-5, help='Small eigenvalue threshold.')
parser.add_argument('--check', type=int, default=0, help='Check coordinates every N steps to see whether it has changed.')
parser.add_argument('--verbose', action='store_true', help='Write out the displacements.')
parser.add_argument('--reset', action='store_true', help='Reset Hessian when eigenvalues are under epsilon.')
parser.add_argument('--rfo', action='store_true', help='Use rational function optimization (default is trust-radius Newton Raphson).')
parser.add_argument('--trust', type=float, default=0.1, help='Starting trust radius.')
parser.add_argument('--tmax', type=float, default=0.3, help='Maximum trust radius.')
parser.add_argument('input', type=str, help='TeraChem or Q-Chem input file')
parser.add_argument('constraints', type=str, nargs='?', help='Constraint input file (optional)')

print ' '.join(sys.argv)

args = parser.parse_args(sys.argv[1:])

def edit_tcin(fin=None, fout=None, options={}, defaults={}):
    """
    Parse a TeraChem input file.

    Parameters
    ----------
    fin : str, optional
        Name of the TeraChem input file to be read
    fout : str, optional
        Name of the TeraChem output file to be written, if desired
    options : dict, optional
        Dictionary of options to overrule TeraChem input file
    defaults : dict, optional
        Dictionary of options to add to the end
    
    Returns
    -------
    dictionary
        Keys mapped to values as strings.  Certain keys will be changed to integers (e.g. charge, spinmult).
        Keys are standardized to lowercase.
    """
    intkeys = ['charge', 'spinmult']
    Answer = OrderedDict()
    # Read from the input if provided
    if fin is not None:
        for line in open(args.input).readlines():
            line = line.split("#")[0].strip()
            if len(line) == 0: continue
            s = line.split(' ', 1)
            k = s[0].lower()
            v = s[1].strip()
            if k == 'coordinates':
                if not os.path.exists(v.strip()):
                    raise RuntimeError("TeraChem coordinate file does not exist")
            if k in intkeys:
                v = int(v)
            if k in Answer:
                raise RuntimeError("Found duplicate key in TeraChem input file: %s" % k)
            Answer[k] = v
    # Replace existing keys with ones from options
    for k, v in options.items():
        Answer[k] = v
    # Append defaults to the end
    for k, v in defaults.items():
        if k not in Answer.keys():
            Answer[k] = v
    # Print to the output if provided
    if fout is not None:
        with open(fout, 'w') as f:
            for k, v in Answer.items():
                print >> f, "%s %s" % (k, str(v))
    return Answer

### Set up based on which quantum chemistry code we're using.

if args.qchem:
    if args.psi4: raise RuntimeError("Do not specify both --qchem and --psi4")
    # The file from which we make the Molecule object
    M = Molecule(args.input, radii={'Na':0.0})
else:
    if args.psi4:
        Psi4exe = which('psi4')
        if len(Psi4exe) == 0: raise RuntimeError("Please make sure psi4 executable is in your PATH")
    else:
        if 'TeraChem' not in os.environ:
            raise RuntimeError('Please set TeraChem environment variable')
        TCHome = os.environ['TeraChem']
        os.environ['PATH'] = os.path.join(TCHome,'bin')+":"+os.environ['PATH']
        os.environ['LD_LIBRARY_PATH'] = os.path.join(TCHome,'lib')+":"+os.environ['LD_LIBRARY_PATH']
    tcdef = OrderedDict()
    tcdef['convthre'] = "1.0e-6"
    tcdef['threall'] = "1.0e-16"
    tcdef['mixguess'] = "0.0"
    tcdef['scf'] = "diis+a"
    tcdef['maxit'] = "50"
    tcdef['dftgrid'] = "1"
    tcdef['precision'] = "mixed"
    tcdef['threspdp'] = "1.0e-8"
    tcin = edit_tcin(fin=args.input, options={'run':'gradient'}, defaults=tcdef)
    M = Molecule(tcin['coordinates'], radii={'Na':0.0})
    M.charge = tcin['charge']
    M.mult = tcin.get('spinmult',1)

prefix = args.prefix if args.prefix is not None else os.path.splitext(args.input)[0]

dirname = prefix+".tmp"
if not os.path.exists(dirname):
    os.makedirs(dirname)
else:
    print "%s exists ; make sure nothing else is writing to the folder" % dirname
    
# First item in tuple: The class to be initialized
# Second item in tuple: Whether to connect nonbonded fragments
# Third item in tuple: Whether to throw in all Cartesians (no effect if second item is True)
CoordSysDict = {'cart':(CartesianCoordinates, False, False),
                'prim':(PrimitiveInternalCoordinates, True, False),
                'dlc':(DelocalizedInternalCoordinates, True, False),
                'hdlc':(DelocalizedInternalCoordinates, False, True),
                'xdlc':(DelocalizedInternalCoordinates, False, False)}
CoordClass, connect, addcart = CoordSysDict[args.coordsys.lower()]

### Above this line: Global variables that should go into main()
### Below this line: function definitions

def calc_terachem(coords):
    """ 
    Run a TeraChem energy and gradient calculation.
    
    Parameters
    ----------
    coords : np.ndarray
        Cartesian coordinates in atomic units

    Returns
    -------
    energy : float
        Total energy in atomic units
    gradient : np.ndarray
        Gradient of energy in atomic units
    """
    coord_hash = tuple(list(coords))
    if coord_hash in calc_terachem.stored_calcs:
        # print "Reading stored values"
        energy = calc_terachem.stored_calcs[coord_hash]['energy']
        gradient = calc_terachem.stored_calcs[coord_hash]['gradient']
        return energy, gradient
    # print "Calling TeraChem for energy and gradient"
    guesses = []
    have_guess = False
    for f in ['c0', 'ca0', 'cb0']:
        if os.path.exists(os.path.join(dirname, 'scr', f)):
            shutil.copy2(os.path.join(dirname, 'scr', f), os.path.join(dirname, f))
            guesses.append(f)
            have_guess = True
    tcin['coordinates'] = 'start.xyz'
    tcin['run'] = 'gradient'
    if have_guess:
        tcin['guess'] = ' '.join(guesses)
        tcin['purify'] = 'no'
    edit_tcin(fout="%s/run.in" % dirname, options=tcin)
    # Convert coordinates back to the xyz file
    M.xyzs[0] = coords.reshape(-1, 3) * 0.529177
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
    """ 
    Run a Q-Chem energy and gradient calculation.
    
    Parameters
    ----------
    coords : np.ndarray
        Cartesian coordinates in atomic units

    Returns
    -------
    energy : float
        Total energy in atomic units
    gradient : np.ndarray
        Gradient of energy in atomic units
    """
    coord_hash = tuple(list(coords))
    if coord_hash in calc_qchem.stored_calcs:
        energy = calc_qchem.stored_calcs[coord_hash]['energy']
        gradient = calc_qchem.stored_calcs[coord_hash]['gradient']
        return energy, gradient
    # print "Calling Qchem for energy and gradient"
    if not os.path.exists(dirname): os.makedirs(dirname)
    # Convert coordinates back to the xyz file
    M.xyzs[0] = coords.reshape(-1, 3) * 0.529177
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

Psi4Temp = """
molecule {{
{charge} {mult}
{geometry}
}}

set basis {basis}

gradient('{method}{dftd}')

"""

def calc_psi4(coords):
    coord_hash = tuple(list(coords))
    if coord_hash in calc_psi4.stored_calcs:
        energy = calc_psi4.stored_calcs[coord_hash]['energy']
        gradient = calc_psi4.stored_calcs[coord_hash]['gradient']
        return energy, gradient
    # print "Calling Psi4 for energy and gradient"
    if not os.path.exists(dirname): os.mkdir(dirname)
    # Convert coordinates back to the xyz file
    M.xyzs[0] = coords.reshape(-1, 3) * 0.529177
    M[0].write(os.path.join(dirname, 'start.xyz'))
    # Write Psi4 input.dat
    with open(os.path.join(dirname,'input.dat'),'w') as inputfile:
        geometrylines = []
        with open(os.path.join(dirname, 'start.xyz'),'r') as xyzinput:
            xyzinput.readline()
            xyzinput.readline()
            for line in xyzinput:
                geometrylines.append(line)
        print >>inputfile, Psi4Temp.format(geometry = ''.join(geometrylines),
                                           charge = str(M.charge), mult=str(M.mult),
                                           basis = tcin['basis'],
                                           method = tcin['method'][1:],
                                           dftd = ('-d' if tcin.get('dispersion', 'no').lower() != 'no' else ""))
    # Run Psi4
    subprocess.call(Psi4exe, cwd = dirname, shell=True)
    # Read energy and gradients from Psi4 output
    energy = 0
    gradient = []
    with open(os.path.join(dirname,'output.dat'), 'r') as psi4output:
        grad_block = False
        with open(os.path.join(dirname,'grad.txt'), 'w') as gradoutput:
            for line in psi4output:
                if '-Total Gradient:' in line:
                    grad_block = True
                if grad_block:
                    if len(line.split()) == 4 and line.split()[0].isdigit():
                        gradoutput.write('   '.join(line.split()[1:]) + '\n')
                    elif len(line.split()) == 0:
                        grad_block = False
                if 'CURRENT ENERGY:' in line and len(line.split()) == 3:
                    energy = float(line.split()[-1])
        gradient = np.loadtxt(os.path.join(dirname,'grad.txt')).flatten()
    if len(calc_psi4.stored_calcs.keys()) > 0:
        PrevHash = calc_psi4.stored_calcs.keys()[-1]
        displacement = coords - calc_psi4.stored_calcs[PrevHash]['coords']
    calc_psi4.stored_calcs[coord_hash] = {'coords':coords,'energy':energy,'gradient':gradient}
    return energy, gradient
calc_psi4.calcnum = 0
calc_psi4.stored_calcs = OrderedDict()

def calc(coords):
    """ Run the selected quantum chemistry code. """
    if args.qchem:
        e, g = calc_qchem(coords)
    elif args.psi4:
        e, g = calc_psi4(coords)
    else:
        e, g = calc_terachem(coords)
    return e, g

def RebuildHessian(IC, H0, coord_seq, grad_seq, trust=0.3):
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
    trust : float
        Include only the geometries with RMSD < trust of the last point
    
    Returns
    -------
    np.ndarray
        Internal coordinate Hessian updated with series of internal coordinate gradients
    """
    Na = len(coord_seq[0])/3
    history = 0
    for i in range(2, len(coord_seq)+1):
        disp = 0.529177*(coord_seq[-i]-coord_seq[-1])
        rmsd = np.sqrt(np.sum(disp**2)/Na)
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
    if np.min(np.linalg.eigh(H)[0]) < args.epsilon and args.reset:
        print "Eigenvalues below %.4e (%.4e) - returning guess" % (args.epsilon, np.min(np.linalg.eigh(H)[0]))
        return H0.copy()
    return H

def calc_drms_dmax(Xnew, Xold):
    # Shift to the origin
    Xold = Xold.copy().reshape(-1, 3)
    Xold -= np.mean(Xold, axis=0)
    Xnew = Xnew.copy().reshape(-1, 3)
    Xnew -= np.mean(Xnew, axis=0)
    # Obtain the rotation
    U = get_rot(Xnew, Xold)
    Xrot = np.array((U*np.matrix(Xnew).T).T).flatten()
    Xold = np.array(Xold).flatten()
    displacement = np.sqrt(np.sum((((Xrot-Xold)*0.529177).reshape(-1,3))**2, axis=1))
    rms_displacement = np.sqrt(np.mean(displacement**2))
    max_displacement = np.max(displacement)
    return rms_displacement, max_displacement

def getCartesianNorm(X, dy, IC):
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

    Returns
    -------
    float
        The RMSD between the updated and original Cartesian coordinates
    """
    # Displacement of each atom in Angstrom
    if IC.haveConstraints() and args.enforce:
        Xnew = IC.newCartesian_withConstraint(X, dy, verbose=args.verbose)
    else:
        Xnew = IC.newCartesian(X, dy, verbose=args.verbose)
    rmsd, maxd = calc_drms_dmax(Xnew, X)
    return rmsd

def between(s, a, b):
    if a < b:
        return s > a and s < b
    elif a > b:
        return s > b and s < a
    else:
        raise RuntimeError('a and b must be different')

def brent_wiki(f, a, b, rel, cvg=0.1, obj=None):
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
            if args.verbose: print "returning because interval is too small"
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
    print "(x, y) = ", x, answer
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

def ParseConstraints(molecule, cFile):
    """
    Parameters
    ----------
    molecule : Molecule
        Molecule object
    cFile : str
        File containing the constraint specification.
    
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
                 "rotation":([RotationA, RotationB, RotationC], 1)
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
    coords = molecule.xyzs[0].flatten() / 0.529177
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
            if key in AtomKeys:
                # Special code that works for atom position constraints.
                # First figure out the range of atoms.
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
                    x1 = [float(i)/0.529177 for i in s[2:2+len(classes)]]
                    # If there's just one constraint value then we append it to the value list-of-lists
                    if mode == "set":
                        vals.append([x1])
                    elif mode == "scan":
                        # If we're scanning it, then we add the whole list of distances to the list-of-lists
                        x2 = [float(i)/0.529177 for i in s[2+len(classes):2+2*len(classes)]]
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
                    if key == "distance": x1 = float(s[1+n_atom])/0.529177
                    else: x1 = float(s[1+n_atom])*np.pi/180.0
                    if mode == "set":
                        vals.append([[x1]])
                    else:
                        if key == "distance": x2 = float(s[2+n_atom])/0.529177
                        else: x2 = float(s[2+n_atom])*np.pi/180.0
                        nstep = int(s[3+n_atom])
                        vals.append([[i] for i in list(np.linspace(x1,x2,nstep))])
            elif key in ["rotation"]:
                # User can only specify ranges of atoms
                atoms = uncommadash(s[1])
                sel = coords.reshape(-1,3)[atoms,:] / 0.529177
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
                        print "Large rotation: Your constraint may not work"
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
                            print "Large rotation: Your constraint may not work"
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

def get_delta_prime_trm(v, X, G, H, IC):
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

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    GC, HC = IC.augmentGH(X, G, H) if IC.haveConstraints() else (G, H)
    HT = HC + v*np.eye(len(HC))
    # The constrained degrees of freedom should not have anything added to diagonal
    for i in range(len(G), len(GC)):
        HT[i, i] = 0.0
    if args.verbose:
        seig = sorted(np.linalg.eig(HT)[0])
        print "sorted(eig) : % .5e % .5e % .5e ... % .5e % .5e % .5e" % (seig[0], seig[1], seig[2], seig[-3], seig[-2], seig[-1])
    try:
        Hi = invert_svd(np.matrix(HT))
    except:
        print "\x1b[1;91mSVD Error - increasing v by 0.001 and trying again\x1b[0m"
        return get_delta_prime_trm(v+0.001, X, G, H, IC)
    dyc = flat(-1 * Hi * col(GC))
    dy = dyc[:len(G)]
    d_prime = flat(-1 * Hi * col(dyc))[:len(G)]
    dy_prime = np.dot(dy,d_prime)/np.linalg.norm(dy)
    sol = flat(0.5*row(dy)*np.matrix(H)*col(dy))[0] + np.dot(dy,G)
    return dy, sol, dy_prime

def get_delta_prime_rfo(alpha, X, G, H, IC):
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
        raise RuntimeError("Still need to implement RFO with constraints")
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

def get_delta_prime(v, X, G, H, IC):
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

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    if args.rfo:
        return get_delta_prime_rfo(v, X, G, H, IC)
    else:
        return get_delta_prime_trm(v, X, G, H, IC)

# this applies an iteration formula to find the trust radius step,
# given the target value of the trust radius.
def trust_step(target, v0, X, G, H, IC):
    dy, sol, dy_prime = get_delta_prime(v0, X, G, H, IC)
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
        dy, sol, dy_prime, = get_delta_prime(v, X, G, H, IC)
        ndy = np.linalg.norm(dy)
        if args.verbose: print "v = %.5f dy -> target = %.5f -> %.5f" % (v, ndy, target)
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
            print "trust_step hit niter = 100, randomizing"
            v += np.random.random() * niter / 100
        if niter%1000 == 999:
            print "trust_step hit niter = 1000, giving up"
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
    def __init__(self, trust, v0, X, G, H, IC):
        self.counter = 0
        self.stores = {}
        self.trust = trust
        self.target = trust
        self.above_flag = False
        self.stored_arg = None
        self.stored_val = None
        self.brentFailed = False
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
                dy, expect = trust_step(trial, v0, X, G, H, IC)
                cnorm = getCartesianNorm(X, dy, IC)
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
            if args.verbose: print "dy(i): %.4f dy(c) -> target: %.4f -> %.4f%s" % (trial, cnorm, self.target, " (done)" if self.from_above else "")
            return cnorm-self.target    

def recover(molecule, IC, X, gradx, X_hist, Gx_hist, connect, addcart):
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
    connect : bool
        Connect molecules when building internal coordinates ("prim" and "dlc" options)
    addcart : bool
        Add all Cartesians when building internal coordinates ("hdlc" option)

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
    newmol.xyzs[0] = X.reshape(-1,3)*0.529177
    newmol.build_topology()
    IC1 = IC.__class__(newmol, connect=connect, addcart=addcart, build=False)
    if IC.haveConstraints(): IC1.getConstraints_from(IC)
    if IC1 != IC:
        print "\x1b[1;94mInternal coordinate system may have changed\x1b[0m"
        if IC.repr_diff(IC1) != "":
            print IC.repr_diff(IC1)
    IC = IC1
    IC.resetRotations(X)
    if type(IC) is DelocalizedInternalCoordinates:
        IC.build_dlc(X)
    H0 = IC.guess_hessian(X)
    if args.reset:
        H = H0.copy()
    else:
        H = RebuildHessian(IC, H0, X_hist, Gx_hist, 0.3)
    Y = IC.calculate(X)
    G = IC.calcGrad(X, gradx)
    return Y, G, H, IC
        
def Optimize(coords, molecule, IC, xyzout):
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
    xyzout : str
        Output file name for writing the progress of the optimization.

    Returns
    -------
    np.ndarray
        Nx3 array of optimized Cartesian coordinates in atomic units
    """
    progress = deepcopy(molecule)
    # Initial Hessian
    H0 = IC.guess_hessian(coords)
    H = H0.copy()
    # Cartesian coordinates
    X = coords.copy()
    # Initial energy and gradient
    E, gradx = calc(coords)
    # Initial internal coordinates
    q0 = IC.calculate(coords)
    Gq = IC.calcGrad(X, gradx)
    # The optimization variables are the internal coordinates.
    Y = q0.copy()
    G = np.array(Gq).flatten()
    # Loop of optimization
    Iteration = 0
    CoordCounter = 0
    trust = args.trust
    thre_rj = 0.01
    # Print initial iteration
    gradxc = IC.calcGradProj(X, gradx) if IC.haveConstraints() else gradx.copy()
    atomgrad = np.sqrt(np.sum((gradxc.reshape(-1,3))**2, axis=1))
    rms_gradient = np.sqrt(np.mean(atomgrad**2))
    max_gradient = np.max(atomgrad)
    print "Step %4i :" % Iteration,
    print "Gradient = %.3e/%.3e (rms/max) Energy = % .10f" % (rms_gradient, max_gradient, E)
    # if IC.haveConstraints(): IC.printConstraints(X)
    # Threshold for "low quality step" which decreases trust radius.
    ThreLQ = 0.25
    # Threshold for "high quality step" which increases trust radius.
    ThreHQ = 0.75
    # Convergence criteria
    Convergence_energy = 1e-6
    Convergence_grms = 3e-4
    Convergence_gmax = 4.5e-4
    Convergence_drms = 1.2e-3
    Convergence_dmax = 1.8e-3
    X_hist = [X]
    Gx_hist = [gradx]
    trustprint = "="
    ForceRebuild = False
    while 1:
        if np.isnan(G).any():
            raise RuntimeError("Gradient contains nan - check output and temp-files for possible errors")
        if np.isnan(H).any():
            raise RuntimeError("Hessian contains nan - check output and temp-files for possible errors")
        Iteration += 1
        # At the start of the loop, the function value, gradient and Hessian are known.
        Eig = sorted(np.linalg.eigh(H)[0])
        Emin = min(Eig).real
        if args.rfo:
            v0 = 1.0
        elif Emin < args.epsilon:
            v0 = args.epsilon-Emin
        else:
            v0 = 0.0
        print "Hessian Eigenvalues: %.5e %.5e %.5e ... %.5e %.5e %.5e" % (Eig[0],Eig[1],Eig[2],Eig[-3],Eig[-2],Eig[-1])
        ### OBTAIN AN OPTIMIZATION STEP ###
        # The trust radius is to be computed in Cartesian coordinates.
        # First take a full-size Newton Raphson step
        dy, expect, _ = get_delta_prime(v0, X, G, H, IC)
        # Internal coordinate step size
        inorm = np.linalg.norm(dy)
        # Cartesian coordinate step size
        cnorm = getCartesianNorm(X, dy, IC)
        if args.verbose: print "dy(i): %.4f dy(c) -> target: %.4f -> %.4f" % (inorm, cnorm, trust)
        # If the step is above the trust radius in Cartesian coordinates, then
        # do the following to reduce the step length:
        if cnorm > 1.1 * trust:
            # This is the function f(inorm) = cnorm-target that we find a root
            # for obtaining a step with the desired Cartesian step size.
            froot = Froot(trust, v0, X, G, H, IC)
            froot.stores[inorm] = cnorm
            # Find the internal coordinate norm that matches the desired
            # Cartesian coordinate norm
            iopt = brent_wiki(froot.evaluate, 0.0, inorm, trust, cvg=0.1, obj=froot)
            if froot.brentFailed and froot.stored_arg is not None:
                if args.verbose: print "\x1b[93mUsing stored solution at %.3e\x1b[0m" % froot.stored_val
                iopt = froot.stored_arg
            elif IC.bork:
                for i in range(3):
                    froot.target /= 2
                    if args.verbose: print "\x1b[93mReducing target to %.3e\x1b[0m" % froot.target
                    froot.above_flag = True
                    iopt = brent_wiki(froot.evaluate, 0.0, iopt, froot.target, cvg=0.1)
                    if not IC.bork: break
            LastForce = ForceRebuild
            ForceRebuild = False
            if IC.bork:
                print "\x1b[91mInverse iteration for Cartesians failed\x1b[0m"
                # This variable is added because IC.bork is unset later.
                ForceRebuild = True
            else:
                if args.verbose: print "\x1b[93mBrent algorithm requires %i evaluations\x1b[0m" % froot.counter
            ##### Force a rebuild of the coordinate system
            if ForceRebuild:
                if LastForce:
                    print "\x1b[1;91mFailed twice in a row to rebuild the coordinate system\x1b[0m"
                    if IC.haveConstraints():
                        print "Cannot continue a constrained optimization; please implement constrained optimization in Cartesian coordinates"
                        sys.exit()
                    else:
                        print "\x1b[93mContinuing in Cartesian coordinates\x1b[0m"
                        IC = CartesianCoordinates(newmol)
                CoordCounter = 0
                Y, G, H, IC = recover(molecule, IC, X, gradx, X_hist, Gx_hist, connect, addcart)
                print "\x1b[1;93mSkipping optimization step\x1b[0m"
                Iteration -= 1
                continue
            ##### End Rebuild
            # Finally, take an internal coordinate step of the desired length.
            dy, expect = trust_step(iopt, v0, X, G, H, IC)
            cnorm = getCartesianNorm(X, dy, IC)
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
        if IC.haveConstraints() and args.enforce:
            X = IC.newCartesian_withConstraint(X, dy, verbose=args.verbose)
        else:
            X = IC.newCartesian(X, dy, verbose=args.verbose)
        ### Calculate Energy and Gradient ###
        E, gradx = calc(X)
        ### Check Convergence ###
        # Add new Cartesian coordinates and gradients to history
        progress.xyzs.append(X.reshape(-1,3) * 0.529177)
        progress.comms.append('Iteration %i Energy % .8f' % (Iteration, E))
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
        # Print status
        print "Step %4i :" % Iteration,
        print "Displace = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("\x1b[92m" if Converged_drms else "\x1b[0m", rms_displacement, "\x1b[92m" if Converged_dmax else "\x1b[0m", max_displacement),
        print "Trust = %.3e (%s)" % (trust, trustprint), 
        print "Grad%s = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("_T" if IC.haveConstraints() else "", "\x1b[92m" if Converged_grms else "\x1b[0m", rms_gradient, "\x1b[92m" if Converged_gmax else "\x1b[0m", max_gradient),
        # print "Dy.G = %.3f" % Dot,
        print "E (change) = % .10f (%s%+.3e\x1b[0m) Quality = %s%.3f\x1b[0m" % (E, "\x1b[91m" if BadStep else ("\x1b[92m" if Converged_energy else "\x1b[0m"), E-Eprev, "\x1b[91m" if BadStep else "\x1b[0m", Quality)
        if IC is not None and IC.haveConstraints():
            IC.printConstraints(X, thre=1e-3)
        if type(IC) is PrimitiveInternalCoordinates:
            idx = np.argmax(np.abs(dy))
            iunit = np.zeros_like(dy)
            iunit[idx] = 1.0
            print "Along %s %.3f" % (IC.Internals[idx], np.dot(dy/np.linalg.norm(dy), iunit))
        if Converged_energy and Converged_grms and Converged_drms and Converged_gmax and Converged_dmax:
            print "Converged! =D"
            break

        ### Adjust Trust Radius and/or Reject Step ###
        # If the trust radius is under thre_rj then do not reject.
        rejectOk = (trust > thre_rj and E > Eprev)
        if Quality <= ThreLQ:
            # For bad steps, the trust radius is reduced
            trust = max(Convergence_drms, trust/2)
            trustprint = "\x1b[91m-\x1b[0m"
        elif Quality >= ThreHQ and bump:
            if trust < args.tmax:
                # For good steps, the trust radius is increased
                trust = min(np.sqrt(2)*trust, args.tmax)
                trustprint = "\x1b[92m+\x1b[0m"
            else:
                trustprint = "="
        else:
            trustprint = "="
        if Quality < -1 and rejectOk:
            # Reject the step and take a smaller one from the previous iteration
            trust = max(Convergence_drms, min(trust, cnorm/2))
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
            if trust < thre_rj: print "\x1b[93mNot rejecting step - trust below %.3e\x1b[0m" % thre_rj
            elif E < Eprev: print "\x1b[93mNot rejecting step - energy decreases\x1b[0m"
        # Append steps to history (for rebuilding Hessian)
        X_hist.append(X)
        Gx_hist.append(gradx)
        ### Rebuild Coordinate System if Necessary ###
        # Check to see whether the coordinate system has changed
        check = False
        # Reinitialize certain variables (i.e. DLC and rotations)
        reinit = False
        if IC.largeRots():
            print "Large rotations - reinitializing coordinates"
            reinit = True
        if IC.bork:
            print "Failed inverse iteration - reinitializing coordinates"
            check = True
            reinit = True
        # Check the coordinate system every (N) steps
        if (CoordCounter == (args.check - 1)) or check:
            newmol = deepcopy(molecule)
            newmol.xyzs[0] = X.reshape(-1,3)*0.529177
            newmol.build_topology()
            IC1 = IC.__class__(newmol, build=False, connect=connect, addcart=addcart)
            if IC.haveConstraints(): IC1.getConstraints_from(IC)
            if IC1 != IC:
                print "\x1b[1;94mInternal coordinate system may have changed\x1b[0m"
                if IC.repr_diff(IC1) != "":
                    print IC.repr_diff(IC1)
                reinit = True
                IC = IC1
            CoordCounter = 0
        else:
            CoordCounter += 1
        # Reinitialize the coordinates (may happen even if coordinate system does not change)
        UpdateHessian = True
        if reinit:
            IC.resetRotations(X)
            if type(IC) is DelocalizedInternalCoordinates:
                IC.build_dlc(X)
            H0 = IC.guess_hessian(coords)
            H = RebuildHessian(IC, H0, X_hist, Gx_hist, 0.3)
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
            if np.min(Eig1) <= args.epsilon and args.reset:
                print "Eigenvalues below %.4e (%.4e) - returning guess" % (args.epsilon, np.min(Eig1))
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
    coords = M.xyzs[0].flatten() / 0.529177
    
    # Read in the constraints
    if args.constraints is not None:
        Cons, CVals = ParseConstraints(M, args.constraints)
    else:
        Cons = None
        CVals = None
                
    IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVals[0] if CVals is not None else None)
    if args.displace:
        for i in range(len(IC.Internals)):
            x = []
            for j in np.linspace(-0.3, 0.3, 7):
                if j != 0:
                    dq = np.zeros(len(IC.Internals))
                    dq[i] = j
                    x1 = IC.newCartesian(coords, dq, verbose=args.verbose)
                else:
                    x1 = coords.copy()
                displacement = np.sqrt(np.sum((((x1-coords)*0.529177).reshape(-1,3))**2, axis=1))
                rms_displacement = np.sqrt(np.mean(displacement**2))
                max_displacement = np.max(displacement)
                if j != 0:
                    dx = (x1-coords)*np.abs(j)*2/max_displacement
                else:
                    dx = 0.0
                x.append((coords+dx).reshape(-1,3) * 0.529177)
                print i, j, "Displacement (rms/max) = %.5f / %.5f" % (rms_displacement, max_displacement), "(Bork)" if IC.bork else "(Good)"
            M.xyzs = x
            M.write("%s/ic_%03i.xyz" % (dirname, i))
        sys.exit()
                
    FDCheck = False
    if FDCheck:
        IC.checkFiniteDifference(coords)
        CheckInternalGrad(coords, M, IC)
        sys.exit()

    if type(IC) is CartesianCoordinates:
        print "%i Cartesian coordinates being used" % (3*M.na)
    else:
        print "%i internal coordinates being used (instead of %i Cartesians)" % (len(IC.Internals), 3*M.na)
    print IC

    if Cons is None:
        if prefix == os.path.splitext(args.input)[0]:
            xyzout = prefix+"_optim.xyz"
        else:
            xyzout = prefix+".xyz"
        opt_coords = Optimize(coords, M, IC, xyzout)
    else:
        if type(IC) in [CartesianCoordinates, PrimitiveInternalCoordinates]:
            raise RuntimeError("Constraints only work with delocalized internal coordinates")
        for ic, CVal in enumerate(CVals):
            if len(CVals) > 1:
                print "---=== Scan %i/%i : Constrained Optimization ===---" % (ic+1, len(CVals))
            IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVal)
            # IC = DelocalizedInternalCoordinates(M, connect=args.connect, constraints=Cons, cvals=CVal)
            # IC.build_dlc(coords)
            IC.printConstraints(coords, thre=-1)

            if len(CVals) > 1:
                xyzout = prefix+"_scan-%03i.xyz" % ic
            elif prefix == os.path.splitext(args.input)[0]:
                xyzout = prefix+"_optim.xyz"
            else:
                xyzout = prefix+".xyz"

                # We may explicitly enforce the constraints here if we want to.
            # coords = IC.applyConstraints(coords)
            coords = Optimize(coords, M, IC, xyzout)
            print
    
if __name__ == "__main__":
    main()

