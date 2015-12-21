#!/usr/bin/env python

from __future__ import division
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from internal import Distance, Angle, Dihedral, OutOfPlane, DelocalizedInternalCoordinates, RedundantInternalCoordinates, CartesianX, CartesianY, CartesianZ
from forcebalance.molecule import Molecule
from forcebalance.nifty import row, col, flat, invert_svd
from scipy import optimize
import argparse
import subprocess
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--cart', action='store_true', help='Use Cartesian coordinate system.')
parser.add_argument('--redund', action='store_true', help='Use redundant coordinate system.')
parser.add_argument('--terachem', action='store_true', help='Run optimization in TeraChem.')

print ' '.join(sys.argv)

args, sys.argv = parser.parse_known_args(sys.argv)

TeraChem = args.terachem

# TeraChem options
TeraTemp = """
# General options
coordinates start.xyz
charge      0
run         gradient
method      rb3lyp
basis       6-31g*
gpus        8
convthre    1e-8
threall     1e-16
"""

TCHome = "/home/leeping/src/terachem/production/build"

xyzout = os.path.splitext(sys.argv[1])[0]+"_optim.xyz"

os.environ['TeraChem'] = TCHome
os.environ['PATH'] = os.path.join(TCHome,'bin')+":"+os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = os.path.join(TCHome,'lib')+":"+os.environ['LD_LIBRARY_PATH']

# Read in the molecule
M = Molecule(sys.argv[1])

def calc_terachem(coords):
    coord_hash = tuple(list(coords))
    if coord_hash in calc_terachem.stored_calcs:
        # print "Reading stored values"
        energy = calc_terachem.stored_calcs[coord_hash]['energy']
        gradient = calc_terachem.stored_calcs[coord_hash]['gradient']
        return energy, gradient
    # print "Calling TeraChem for energy and gradient"
    # dirname = "calc_%04i" % calc_terachem.calcnum
    dirname = os.path.splitext(sys.argv[1])[0]+".tmp"
    if not os.path.exists(dirname): os.makedirs(dirname)
    # Write input files to directory
    with open(os.path.join(dirname, 'run.in'), 'w') as f: print >> f, TeraTemp
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
    # print "RMS gradient     : % .4e" % (np.sqrt(np.mean(gradient**2)))
    # print "Max gradient     : % .4e" % (np.max(np.abs(gradient)))
    if len(calc_terachem.stored_calcs.keys()) > 0:
        PrevHash = calc_terachem.stored_calcs.keys()[-1]
        displacement = coords - calc_terachem.stored_calcs[PrevHash]['coords']
        # print "Energy change    : % .4e" % (energy - calc_terachem.stored_calcs[PrevHash]['energy'])
        # print "RMS displacement : % .4e" % (np.sqrt(np.mean(displacement**2)))
        # print "Max displacement : % .4e" % (np.max(np.abs(displacement)))
    calc_terachem.stored_calcs[coord_hash] = {'coords':coords,'energy':energy,'gradient':gradient}
    return energy, gradient
calc_terachem.calcnum = 0
calc_terachem.stored_calcs = OrderedDict()

def calc_qchem(coords):
    coord_hash = tuple(list(coords))
    if coord_hash in calc_qchem.stored_calcs:
        # print "Reading stored values"
        energy = calc_qchem.stored_calcs[coord_hash]['energy']
        gradient = calc_qchem.stored_calcs[coord_hash]['gradient']
        return energy, gradient
    # print "Calling Qchem for energy and gradient"
    dirname = os.path.splitext(sys.argv[1])[0]+".tmp"
    # dirname = "calc_%04i" % calc_qchem.calcnum
    if not os.path.exists(dirname): os.makedirs(dirname)
    # Convert coordinates back to the xyz file
    M.xyzs[0] = coords.reshape(-1, 3) * 0.529
    M.edit_qcrems({'jobtype':'force'})
    M[0].write(os.path.join(dirname, 'run.in'))
    # Run Qchem
    subprocess.call('%s/runqc run.in run.out &> run.log' % os.getcwd(), cwd=dirname, shell=True)
    M1 = Molecule('%s/run.out' % dirname)
    energy = M1.qm_energies[0]
    gradient = M1.qm_grads[0].flatten()
    # print "RMS gradient     : % .4e" % (np.sqrt(np.mean(gradient**2)))
    # print "Max gradient     : % .4e" % (np.max(np.abs(gradient)))
    if len(calc_qchem.stored_calcs.keys()) > 0:
        PrevHash = calc_qchem.stored_calcs.keys()[-1]
        displacement = coords - calc_qchem.stored_calcs[PrevHash]['coords']
        # print "Energy change    : % .4e" % (energy - calc_qchem.stored_calcs[PrevHash]['energy'])
        # print "RMS displacement : % .4e" % (np.sqrt(np.mean(displacement**2)))
        # print "Max displacement : % .4e" % (np.max(np.abs(displacement)))
    calc_qchem.stored_calcs[coord_hash] = {'coords':coords,'energy':energy,'gradient':gradient}
    return energy, gradient
calc_qchem.calcnum = 0
calc_qchem.stored_calcs = OrderedDict()

def calc(coords):
    if TeraChem:
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
    newxyz = IC.newCartesian(x0, dQ)
    energy_internal.x0 = newxyz
    return calc(newxyz)[0]
energy_internal.IC = None
energy_internal.x0 = None

def gradient_internal(q):
    IC = gradient_internal.IC
    x0 = gradient_internal.x0
    q0 = IC.calculate(x0)
    dQ = IC.subtractInternal(q,q0)
    newxyz = IC.newCartesian(x0, dQ)
    gradient_internal.x0 = newxyz
    Gx = np.matrix(calc(newxyz)[1]).T
    Ginv = IC.GInverse(newxyz)
    Bmat = IC.wilsonB(newxyz)
    Gq = np.matrix(Ginv)*np.matrix(Bmat)*Gx
    return np.array(Gq).flatten()
gradient_internal.IC = None
gradient_internal.x0 = None

def Rebuild(IC, H0, coord_seq, grad_seq, trust=0.3):
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
        if np.min(np.linalg.eig(H)[0]) < 0:
            print "Negative eigenvalues at iteration %i - returning previous" % i
            return Hstor
    return H

def getNorm(X, dy, IC=None, CartesianTrust=False):
    if IC is not None and CartesianTrust:
        # Displacement of each atom in Angstrom
        try:
            Xnew = IC.newCartesian(X, dy)
            disp = 0.529*(Xnew-X)
        except np.linalg.LinAlgError:
            return np.sum(dy**2)*1e10
    else:
        disp = 0.529*dy
    # Number of atoms
    Na = len(X)/3
    return np.sqrt(np.sum(disp**2)/Na)
        
def Optimize(coords, molecule, IC=None):
    progress = deepcopy(molecule)
    # Initial Hessian
    if IC is not None:
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
    trust = 0.1
    # Adaptive trust radius
    trust0 = 1.0
    adapt_fac = 1.0
    adapt_damp = 0.5
    # Threshold for "low quality step" which decreases trust radius.
    ThreLQ = 0.25
    # Threshold for "high quality step" which increases trust radius.
    ThreHQ = 0.75
    print np.diag(H)
    Convergence_energy = 1e-6
    Convergence_grms = 3e-4
    Convergence_gmax = 4.5e-4
    Convergence_drms = 1.2e-3
    Convergence_dmax = 1.8e-3
    X_hist = [X]
    Gx_hist = [gradx]
    CartesianTrust = True
    while 1:
        # Force Hessian to have positive eigenvalues.
        Eig = np.linalg.eigh(H)[0]
        Emin = min(Eig).real
        eps = 1e-5
        if Emin < eps:
            Adj = eps-Emin
        else:
            Adj = 0
        H += Adj * np.eye(H.shape[0])
        Eig = np.linalg.eigh(H)[0]
        Eig = sorted(Eig)
        print "Hessian Eigenvalues: %.5f %.5f %.5f ... %.5f %.5f %.5f" % (Eig[0],Eig[1],Eig[2],Eig[-3],Eig[-2],Eig[-1])
        # Define two functions that help us to find the trust radius step.
        def solver(L):
            HT = H + (L-1)**2*np.eye(len(H))
            Hi = invert_svd(np.matrix(HT))
            dy = flat(-1 * Hi * col(G))
            sol = flat(0.5*row(dy)*np.matrix(H)*col(dy))[0] + np.dot(dy,G)
            return dy, sol
        def trust_fun(L):
            try:
                dy = solver(L)[0]
            except:
                print "\x1b[1;91mError inverting Hessian - L = %.3f\x1b[0m" % L
                return 1e10*(L-1)**2
            N = getNorm(X, dy, IC, CartesianTrust)
            # print "Finding trust radius: H%+.4f*I, length %.4e (target %.4e)" % ((L-1)**2,N,trust)
            return (N - trust)**2
        # This is the normal step from inverting the Hessian
        dy, expect = solver(1)
        dynorm = getNorm(X, dy, IC, CartesianTrust)
        # If the step is larger than the trust radius, then restrict it to the trust radius
        if dynorm > trust:
            LOpt = optimize.brent(trust_fun,brack=(1.0, 4.0),tol=1e-4)
            dy, expect = solver(LOpt)
            dynorm = getNorm(X, dy, IC, CartesianTrust)
            # print "Trust-radius step found (length %.4e), % .4e added to Hessian diagonal" % (dynorm, (LOpt-1)**2)
        bump = dynorm > 0.8 * trust
        # Get the previous iteration stuff
        Yprev = Y.copy()
        Xprev = X.copy()
        Gprev = G.copy()
        Eprev = E
        Y += dy
        if internal:
            X = IC.newCartesian(X, dy)
        else:
            X = Y.copy()
        E, gradx = calc(X)
        # Add new Cartesian coordinates and gradients to history
        progress.xyzs.append(X.reshape(-1,3) * 0.529)
        progress.comms.append('Iteration %i Energy % .8f' % (Iteration, E))
        progress.write(xyzout)
        # Calculate quantities for convergence
        displacement = np.sqrt(np.sum((((X-Xprev)*0.529).reshape(-1,3))**2, axis=1))
        atomgrad = np.sqrt(np.sum((gradx.reshape(-1,3))**2, axis=1))
        rms_displacement = np.sqrt(np.mean(displacement**2))
        rms_gradient = np.sqrt(np.mean(atomgrad**2))
        max_displacement = np.max(displacement)
        max_gradient = np.max(atomgrad)
        Quality = (E-Eprev)/expect
        Converged_energy = ((E-Eprev) < 0 and np.abs(E-Eprev) < Convergence_energy)
        Converged_grms = rms_gradient < Convergence_grms
        Converged_gmax = max_gradient < Convergence_gmax
        Converged_drms = rms_displacement < Convergence_drms
        Converged_dmax = max_displacement < Convergence_dmax
        BadStep = (E-Eprev) > 0
            
        print "Iteration %4i :" % Iteration,
        print "Displacement = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("\x1b[92m" if Converged_drms else "\x1b[0m", rms_displacement, "\x1b[92m" if Converged_dmax else "\x1b[0m", max_displacement),
        print "Gradient = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("\x1b[92m" if Converged_grms else "\x1b[0m", rms_gradient, "\x1b[92m" if Converged_gmax else "\x1b[0m", max_gradient),
        print "Energy (change) = % .10f (%s%+.3e\x1b[0m) Quality = %s%.3f\x1b[0m" % (E, "\x1b[91m" if BadStep else ("\x1b[92m" if Converged_energy else "\x1b[0m"), E-Eprev, "\x1b[91m" if BadStep else "\x1b[0m", Quality),
        if IC is not None:
            print "(Bork)" if IC.bork else "(Good)"
        else:
            print
        if IC is not None:
            idx = np.argmax(np.abs(dy))
            iunit = np.zeros_like(dy)
            iunit[idx] = 1.0
            if type(IC) is RedundantInternalCoordinates:
                print "Along %s %.3f" % (IC.Internals[idx], np.dot(dy/np.linalg.norm(dy), iunit))
        if Converged_energy and Converged_grms and Converged_drms and Converged_gmax and Converged_dmax:
            print "Converged! =D"
            break
        # Update the trust radius
        if Quality <= ThreLQ:
            trust = dynorm/4
            # trust = dynorm/(1+adapt_fac)
            trustprint = "Decreasing trust radius to % .4e" % trust
            print_trust = True
        elif Quality >= ThreHQ and bump:
            if trust < 0.3:
                trust = min(2*trust, 0.3)
                trustprint = "Increasing trust radius to % .4e" % trust
                print_trust = True
        else:
            print_trust = False
        if Quality < 0:
            print "%s and rejecting step" % trustprint
            Y = Yprev.copy()
            X = Xprev.copy()
            G = Gprev.copy()
            E = Eprev
        else:
            X_hist.append(X)
            Gx_hist.append(gradx)
            skipBFGS=False
            if print_trust:
                print trustprint
            if internal:
                if (IC.bork or CoordCounter == 1):
                    newmol = deepcopy(molecule)
                    newmol.xyzs[0] = X.reshape(-1,3)*0.529
                    newmol.build_topology()
                    if type(IC) is RedundantInternalCoordinates:
                        IC1 = RedundantInternalCoordinates(newmol)
                    elif type(IC) is DelocalizedInternalCoordinates:
                        IC1 = DelocalizedInternalCoordinates(newmol, build=False)
                    else:
                        raise RuntimeError('Spoo!')
                    if IC.update(IC1):
                        # if IC1 != IC:
                        print "\x1b[1;94mInternal coordinate system may have changed\x1b[0m"
                        # print IC.repr_diff(IC1)
                        # IC = IC1
                        if type(IC) is DelocalizedInternalCoordinates:
                            IC.build_dlc(X)
                        H0 = IC.guess_hessian(coords)
                        # H0 = np.eye(len(IC.Internals))
                        H = Rebuild(IC, H0, X_hist, Gx_hist, 0.3)
                        # H = H0.copy()
                        Y = IC.calculate(X)
                        skipBFGS = True
                        CoordCounter = 0 
                else:
                    CoordCounter += 1
                Gq = IC.calcGrad(X, gradx)
                G = np.array(Gq).flatten()
            else:
                G = gradx.copy()
            if not skipBFGS:
                # BFGS Hessian update
                Dy   = col(Y - Yprev)
                Dg   = col(G - Gprev)
                Mat1 = (Dg*Dg.T)/(Dg.T*Dy)[0,0]
                Mat2 = ((H*Dy)*(H*Dy).T)/(Dy.T*H*Dy)[0,0]
                H += Mat1-Mat2
        # Then it's on to the next loop iteration!
        Iteration += 1
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
        x1 = IC.newCartesian(coords, dq)
        EPlus, _ = calc(x1)
        dq[i] -= 2e-4
        x1 = IC.newCartesian(coords, dq)
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
        x1 = IC.newCartesian(coords, dq)
        EPlus, _ = calc(x1)
        dq[i] -= 2e-4
        x1 = IC.newCartesian(coords, dq)
        EMinus, _ = calc(x1)
        fdiff = (EPlus+EMinus-2*E)/1e-6
        print "%s : % .6e" % (IC.Internals[i], fdiff)
            
def main():
    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() / 0.529
    if args.cart:
        IC = None
    elif args.redund:
        IC = RedundantInternalCoordinates(M)
    else:
        IC = DelocalizedInternalCoordinates(M)
    FDCheck = False
    if FDCheck:
        IC.checkFiniteDifference(coords)
        CheckInternalGrad(coords, M, IC)
    opt_coords = Optimize(coords, M, IC)
    M.xyzs[0] = opt_coords.reshape(-1,3) * 0.529
    # IC = RedundantInternalCoordinates(M)
    # CalcInternalHess(opt_coords, M, IC)
    # if FDCheck:
    #     IC.checkFiniteDifference(opt_coords)
    #     CheckInternalGrad(opt_coords, M, IC)
    
if __name__ == "__main__":
    main()

