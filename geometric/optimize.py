"""
optimize.py: Driver and core functions for geometry optimization

This code is part of geomeTRIC.

Copyright 2016-2024 Regents of the University of California and the Authors

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

from __future__ import print_function, division

import os
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime

import numpy as np
import scipy
import warnings
from scipy import optimize
from numpy.linalg import multi_dot

from .info import print_logo, print_citation
from .internal import CartesianCoordinates, PrimitiveInternalCoordinates, DelocalizedInternalCoordinates
from .ic_tools import check_internal_grad, check_internal_hess, write_displacements
from .normal_modes import calc_cartesian_hessian, frequency_analysis
from .step import brent_wiki, Froot, calc_drms_dmax, get_cartesian_norm, get_delta_prime, trust_step, force_positive_definite, update_hessian
from .prepare import get_molecule_engine, parse_constraints
from .params import OptParams, parse_optimizer_args
from .nifty import row, col, flat, bohr2ang, ang2bohr, logger, bak, createWorkQueue, destroyWorkQueue, printcool_dictionary
from .errors import InputError, HessianExit, EngineError, IRCError, GeomOptNotConvergedError, GeomOptStructureError, LinearTorsionError
from .config import config_dir

class Optimizer(object):
    def __init__(self, coords, molecule, IC, engine, dirname, params, print_info=True):
        """
        Object representing the geometry optimization of a molecular system.

        Parameters
        ----------
        coords : np.ndarray
            1-dimensional array of shape (3*N_atoms) containing atomic coordinates in Bohr
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
        print_info : bool
            Print information about the optimization at end of constructor call
        """
        # Copies of data passed into constructor
        self.coords = coords
        self.molecule = deepcopy(molecule)
        self.IC0 = IC
        self.IC = IC
        self.engine = engine
        self.dirname = dirname
        self.params = params
        # Set initial value of the trust radius.
        self.trust = self.params.trust
        # Copies of molecule object for preserving the optimization trajectory and the last frame
        self.progress = deepcopy(self.molecule)
        self.progress.xyzs = []
        self.progress.qm_energies = []
        self.progress.qm_grads = []
        self.progress.comms = []
        self.viz_rotations = False
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
        # Sanity check - if there's only one atom, it will probably crash
        if self.molecule.na < 2:
            raise InputError("Geometry optimizer assumes there are at least two atoms in the system")
        # Detect poor quality steps dominated by net translation and rotation.
        # Minimum ratio of (total displacement / aligned displacement), hard-coded parameter.
        self.lowq_tr_thre  = 5.0
        # Counter of poor quality steps dominated by net translation/rotation.
        self.lowq_tr_count = 0
        # Number of poor quality steps before starting to project out net force/torque, hard-coded parameter.
        self.lowq_tr_limit = 1
        # Recalculate the Hessian after a trigger - for example if the energy changes by a lot during TS optimization.
        self.recalcHess = False
        # IRC related information
        self.IRC_direction = self.params.irc_direction
        self.IRC_info = {"direction" : 1, "opt" : False, "total_disp" : 0.0,
                        "substep_success" : True, "IC_changed" : False}
        if print_info:
            self.print_info()
        
    def print_info(self):
        params = self.params
        logger.info("> ===== Optimization Info: ====\n")
        if params.transition:
            logger.info("> Job type: Transition state optimization\n")
        elif params.irc:
            logger.info("> Job type: Intrinsic Reaction Coordinate method\n")
        else:
            logger.info("> Job type: Energy minimization\n")
            
        logger.info("> Maximum number of optimization cycles: %i\n" % params.maxiter)
        logger.info("> Initial / maximum trust radius (Angstrom): %.3f / %.3f\n" % (params.trust, params.tmax))
        logger.info("> Convergence Criteria:\n")
        if params.qccnv:
            logger.info("> Q-Chem style convergence criteria requested.\n")
            logger.info("> Will converge when 2 out of 3 criteria are reached:\n")
            logger.info(">  |Delta-E| < %.2e\n" % params.Convergence_energy)
            if self.IC.haveConstraints():
                logger.info(">  RMS-Ortho-Grad < %.2e\n" % params.Convergence_grms)
            else:
                logger.info(">  RMS-Grad  < %.2e\n" % params.Convergence_grms)
            logger.info(">  RMS-Disp  < %.2e\n" % params.Convergence_drms)
        else:
            logger.info("> Will converge when all 5 criteria are reached:\n")
            logger.info(">  |Delta-E| < %.2e\n" % params.Convergence_energy)
            if self.IC.haveConstraints():
                logger.info(">  RMS-Ortho-Grad < %.2e\n" % params.Convergence_grms)
                logger.info(">  Max-Ortho-Grad < %.2e\n" % params.Convergence_gmax)
            else:
                logger.info(">  RMS-Grad  < %.2e\n" % params.Convergence_grms)
                logger.info(">  Max-Grad  < %.2e\n" % params.Convergence_gmax)
            logger.info(">  RMS-Disp  < %.2e\n" % params.Convergence_drms)
            logger.info(">  Max-Disp  < %.2e\n" % params.Convergence_dmax)

        if params.molcnv:
            logger.info("> \n")
            logger.info("> Molpro style convergence criteria requested.\n")
            logger.info("> Will also converge when both of the following are satisfied:\n")
            if self.IC.haveConstraints():
                logger.info(">  Max-Ortho-Grad < %.2e\n" % params.Convergence_molpro_gmax)
            else:
                logger.info(">  Max-Grad < %.2e\n" % params.Convergence_molpro_gmax)
            logger.info(">  Max-Disp < %.2e -OR- |Delta-E| < %.2e\n" % (params.Convergence_molpro_dmax, params.Convergence_energy))
            
        if self.IC.haveConstraints():
            logger.info("> \n")
            logger.info("> Constraints are requested. The following criterion is added:\n")
            logger.info(">  Max Constraint Violation (in Angstroms/degrees) < %.2e \n" % self.params.Convergence_cmax)

        if params.Converge_maxiter:
            logger.info(">  Converge-on-maxiter set: Will exit with success if maximum number of iterations (%i) is reached.\n" % params.maxiter)

        logger.info("> === End Optimization Info ===\n")
        
    def get_cartesian_norm(self, dy, verbose=None):
        if not verbose: verbose = self.params.verbose
        return get_cartesian_norm(self.X, dy, self.IC, self.params.enforce, self.params.verbose, self.params.usedmax)

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
        logger.info("Refreshing coordinate system and resetting rotations\n")
        # Resetting of rotations
        self.IC.resetRotations(self.X)
        if isinstance(self.IC, DelocalizedInternalCoordinates):
            self.IC.build_dlc(self.X)
        # With redefined internal coordinates, the Hessian needs to be rebuilt
        self.rebuild_hessian()
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
            IC1 = self.IC.__class__(newmol, connect=self.IC.connect, addcart=self.IC.addcart, build=False, conmethod=self.IC.conmethod, rigid=self.IC.rigid)
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
        if self.IC.rigid:
            mol = deepcopy(self.molecule)
            mol.xyzs = [self.X.reshape(-1, 3)*bohr2ang]
            mol.qm_grads = [gradxc.reshape(-1,3)]
            atomgrad = []
            # print("Net forces / torques:")
            netfrcs = []
            torques = []
            for i, frag in enumerate(self.IC.frags):
                frag_mol = mol.atom_select(frag)
                netfrc, torque = frag_mol.calc_netforce_torque(mass=True)
                netfrcs.append(netfrc[0])
                torques.append(torque[0])
                # print("Frag %i: % 9.3e % 9.3e % 9.3e ; % 9.3e % 9.3e % 9.3e" % (i, *netfrc[0], *torque[0]))
            atomgrad = np.sqrt(np.sum((np.array(netfrcs + torques).reshape(-1,3))**2, axis=1))
        else:
            atomgrad = np.sqrt(np.sum((gradxc.reshape(-1,3))**2, axis=1))
        rms_gradient = np.sqrt(np.mean(atomgrad**2))
        max_gradient = np.max(atomgrad)
        return rms_gradient, max_gradient

    def rebuild_hessian(self):
        if self.params.irc:
            self.H0 = self.IC.calcHess(self.coords.copy(), self.Gx_init, self.Hx0)
        else:
            self.H0 = self.IC.guess_hessian(self.coords)
        self.H = update_hessian(self.IC, self.H0, self.X_hist, self.Gx_hist, self.params, trust_limit=True, max_updates=100)

    def frequency_analysis(self, hessian, suffix, afterOpt):
        do_wigner = False
        if self.params.wigner:
            # Wigner sampling should only be performed on the final Hessian calculation of a run
            if self.params.hessian in ['last', 'first+last', 'each'] and afterOpt:
                do_wigner = True
            elif self.params.hessian in ['first', 'stop']:
                do_wigner = True
        if do_wigner:
            logger.info("Requesting %i samples from Wigner distribution.\n" % self.params.wigner)
        prefix = self.params.xyzout.replace("_optim.xyz", "").replace(".xyz", "")
        # Call the frequency analysis function with an input Hessian, with most arguments populated from self.params
        self.TSWavenum, self.TSNormal_modes_x, _ = frequency_analysis(self.X, hessian, self.molecule.elem, energy=self.E, temperature=self.params.temperature, pressure=self.params.pressure, verbose=self.params.verbose,
                           outfnm='%s.vdata_%s' % (prefix, suffix), note='Iteration %i Energy % .8f%s' % (self.Iteration, self.E, ' (Optimized Structure)' if afterOpt else ''),
                           wigner=((self.params.wigner, os.path.join(self.dirname, 'wigner')) if do_wigner else None), ignore=self.params.ignore_modes)

    def calcEnergyForce(self):
        """
        Calculate the energy and Cartesian gradients of the current structure.
        """
        # Check to confirm that the structure has nothing that would cause a cryptic error
        self.checkStructure()
        ### Calculate Energy and Gradient ###
        # Dictionary containing single point properties (energy, gradient)
        # For frequency calculations and multi-step jobs, the gradient from an existing
        # output file may be read in.
        spcalc = self.engine.calc(self.X, self.dirname, read_data=(self.Iteration==0))
        if self.params.subfrctor == 2 or (self.params.subfrctor == 1 and (self.lowq_tr_count >= self.lowq_tr_limit)):
            # Subtract out the overall translational and rotational components of the force
            netfrc_torque_mol = deepcopy(self.molecule)
            netfrc_torque_mol.xyzs = [self.X.reshape(-1, 3)*bohr2ang]
            netfrc_torque_mol.qm_grads = [spcalc['gradient'].reshape(-1,3)]
            qm_grads_proj = netfrc_torque_mol.remove_netforce_torque(mass=True)[0].flatten()
            ## Currently unclear whether mass weighting makes a different when projecting out the torques.
            # print("Net force/torque before projection:", netfrc_torque_mol.calc_netforce_torque())
            # netfrc_torque_mol.qm_grads = [qm_grads_proj.reshape(-1,3)]
            # print("Net force/torque after  projection:", netfrc_torque_mol.calc_netforce_torque(mass=False))
            # print("Force difference: %.3e" % np.linalg.norm(spcalc['gradient'] - qm_grads_proj))
            spcalc['gradient'] = qm_grads_proj
        self.E = spcalc['energy']
        self.gradx = spcalc['gradient']
        # gx = self.gradx.reshape(-1, 3)
        # for i in range(gx.shape[0]):
        #     print("% 10.6f % 10.6f % 10.6f" % (gx[i, 0], gx[i, 1], gx[i, 2]))
        # sys.exit()
        # Calculate Hessian at the first step, or at each step if desired
        if self.params.hessian == 'each' or self.recalcHess:
            # Hx is assumed to be the Cartesian Hessian at the current step.
            # Otherwise we use the variable name Hx0 to avoid almost certain confusion.
            self.Hx = calc_cartesian_hessian(self.X, self.molecule, self.engine, self.dirname, read_data=True, bigchem=self.params.bigchem, verbose=self.params.verbose)
            if self.params.frequency:
                self.frequency_analysis(self.Hx, 'iter%03i' % self.Iteration, False)
            if self.recalcHess:
                logger.info(">>> Recomputed the Hessian from scratch <<<\n")
                self.H0 = self.IC.calcHess(self.X, self.gradx, self.Hx)
                self.H = self.H0.copy()
                if self.params.hessian != 'each':
                    delattr(self, 'Hx')
                self.recalcHess = False
        elif self.Iteration == 0:
            if self.params.hessian in ['first', 'stop', 'first+last'] and not hasattr(self.params, 'hess_data'):
                self.Hx0 = calc_cartesian_hessian(self.X, self.molecule, self.engine, self.dirname, read_data=True, bigchem=self.params.bigchem, verbose=self.params.verbose)
                logger.info(">> Initial Cartesian Hessian Eigenvalues\n")
                self.SortedEigenvalues(self.Hx0)
                if self.params.frequency:
                    self.frequency_analysis(self.Hx0, 'first', False)
                if self.params.hessian == 'stop':
                    logger.info("Exiting as requested after Hessian calculation.\n")
                    logger.info("Cartesian Hessian is stored in %s/hessian/hessian.txt.\n" % self.dirname)
                    raise HessianExit
                    # sys.exit(0)
            elif hasattr(self.params, 'hess_data'):
                logger.info(">> Using Hessian data provided via params...\n")
                self.Hx0 = self.params.hess_data.copy()
                logger.info(">> Initial Cartesian Hessian Eigenvalues\n")
                self.SortedEigenvalues(self.Hx0)
                if self.params.frequency:
                    self.frequency_analysis(self.Hx0, 'first', False)
                if self.Hx0.shape != (self.X.shape[0], self.X.shape[0]):
                    raise IOError('hess_data passed in via OptParams does not have the right shape')
            if self.params.maxiter == 0:
                logger.info("Maximum iterations reached (%i); increase --maxiter for more\n" % self.params.maxiter)
                if self.params.Converge_maxiter:
                    logger.info("Exiting normally because --converge maxiter was set.\n")
                    self.state = OPT_STATE.CONVERGED
                else:
                    self.state = OPT_STATE.FAILED

            # self.Hx = self.Hx0.copy()
        # Add new Cartesian coordinates, energies, and gradients to history
        if self.viz_rotations:
            if hasattr(self, 'progress_with_r'):
                tmpMol = self.IC.visualizeRotations(self.X)
                tmpMol.comms = ['Iteration %i Energy % .8f' % (self.Iteration, self.E)]
                try:
                    self.progress_with_r += tmpMol
                except RuntimeError:
                    bak(os.path.splitext(self.params.xyzout)[0]+"_with_r.xyz")
                    self.progress_with_r = tmpMol
            else:
                self.progress_with_r = self.IC.visualizeRotations(self.X)
                self.progress_with_r.comms = ['Iteration %i Energy % .8f' % (self.Iteration, self.E)]
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
        self.Gx_init = self.gradx.copy()
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

    def SortedEigenvalues(self, H):
        Eig = sorted(np.linalg.eigh(H)[0])
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

    def find_lambda(self, Lambda, Heig, Hvec, g_M, p_M):
        """
        Equation 26 from Gonzalez & Schlegel (1990)
        """
        a = Heig * np.dot(p_M, Hvec) - np.dot(g_M, Hvec)
        b = Heig - Lambda
        c = (a / b) ** 2
        return np.sum(c) - (0.5 * self.IRC_stepsize) ** 2

    def guess_g(self, g, H, disp):
        """
        Guess the gradient using the quadratic expansion
        """
        g_new = g + np.dot(H, disp)
        return g_new

    def IRC_step(self):
        self.farConstraints = self.IC.haveConstraints() and self.IC.maxConstraintViolation(self.X) > 1e-1

        if self.params.verbose:
            logger.info("IRC sub-step 1: Finding the pivot point (q*_{k+1})\n")

        # Need to take a step towards the pivot point
        self.IC.clearCache()
        MWGMat = self.IC.GMatrix(self.X, invMW=True)

        # Save the initial Cartesian coordinate
        X0 = self.X.copy()

        # Vector to the pivot point
        if self.Iteration == 0:
            # If it's the very first step, pick the eigenvector of the imaginary frequency and pick the direction
            logger.info('First, following the imaginary mode vector\n')
            if self.TSWavenum[1] < 0:
                if np.isclose(self.TSWavenum[0], self.TSWavenum[1], atol = 1):
                    logger.warning("Warning: More than one imaginary mode is detected. The two lowest wavenumbers are degenerate; following the lowest one.\n")
                else:
                    raise IRCError("There are more than one imaginary vibrational mode. Please optimize the structure and try again.\n")
            elif self.TSWavenum[0] >= 0:
                raise IRCError("No imaginary mode detected. Please optimize the structure and try again.\n")

            self.IRC_adjfactor = np.linalg.norm(self.TSNormal_modes_x[0] * np.sqrt(self.IC.mass))
            IRC_init_step = self.trust * ang2bohr * self.IRC_adjfactor
            self.IRC_std_step = 0.1 * ang2bohr * self.IRC_adjfactor
            logger.info("Initial step-size: %.5f \n" %IRC_init_step)

            # Following the imaginary mode vector
            Im_mode = self.TSNormal_modes_x[0]
            if self.IRC_direction == 'both' or self.IRC_direction == 'forward':
                Im_mode *= self.IRC_info.get("direction")
            elif self.IRC_direction == 'backward':
                Im_mode *= -self.IRC_info.get("direction")
            else:
                raise IRCError('IRC direction should either be forward or backward.\n')
            v = self.IC.calcDiff(X0 + Im_mode, X0)
            X = self.IC.newCartesian(X0, v)
            v = self.IC.calcDiff(X, X0)
  
            #if self.IRC_init_v is None:
            #    #MWGMat_sqrt_inv, MWGMat_sqrt = self.IC.GInverse_SVD(self.X, sqrt=True, invMW=True)
            #    #H_M = np.dot(np.dot(MWGMat_sqrt, self.H), MWGMat_sqrt.T)
            #    #_, MW_IC_vecs = np.linalg.eigh(H_M)
            #    #v = MW_IC_vecs[0]
            #    self.IRC_init_v = self.IC.calcDiff(X0 - Im_mode, X0)
            #else:
            #    v = self.IRC_init_v

        else:
            v = self.G.copy() # Internal coordinate gradients

        invMW_v = np.dot(MWGMat, v) # Inverse mass-weighted vector
        # Normalization factor
        N = 1 / np.sqrt(np.dot(v.T, np.dot(MWGMat, v)))
        self.IRC_stepsize = self.trust * ang2bohr * self.IRC_adjfactor
        # Step towards the pivot point
        dy_to_pivot = -0.5*self.IRC_stepsize*N*invMW_v

        # Move to the pivot point
        X_pivot = self.IC.newCartesian(X0, dy_to_pivot, self.params.verbose)
        dy_to_pivot = self.IC.calcDiff(X_pivot, X0)
        # Calculating sqrt(mass) weighted Cartesian coordinate
        MWGMat_sqrt_inv, MWGMat_sqrt = self.IC.GInverse_SVD(X_pivot, sqrt=True, invMW=True)
        mwdx_1 = np.dot(MWGMat_sqrt_inv, dy_to_pivot)

        if self.params.verbose:
            logger.info("Half step dy     = %.5f\n" %np.linalg.norm(dy_to_pivot))
            logger.info("Half step mw-dx  = %.5f Bohr*sqrt(amu)\n" %np.linalg.norm(mwdx_1))

        # We are at the pivot point
            logger.info('\nIRC sub-step 2: Finding the next point (q_{k+1})\n')
        v1 = v.copy()
        irc_sub_iteration = 0
        irc_reset_iteration = 0
        p_prime = dy_to_pivot.copy()
        # Finding the next point
        while True:
            X = self.IC.newCartesian(X_pivot, p_prime, self.params.verbose)
            # Now we are at the guessed point, define mass-weighted G matrix at the guessed point
            self.IC.clearCache()
            MWGMat_sqrt_inv, MWGMat_sqrt = self.IC.GInverse_SVD(X, sqrt=True, invMW=True)
            # Mass weighted displacement, gradients, and Hessian
            g_M = np.dot(MWGMat_sqrt, self.guess_g(self.G, self.H, dy_to_pivot + p_prime))
            H_M = np.dot(np.dot(MWGMat_sqrt, self.H), MWGMat_sqrt.T)
            p_M = np.dot(MWGMat_sqrt_inv, p_prime)

            Heig, Hvecs = np.linalg.eigh(H_M)
            init_guess = 1.01*Heig[0] if Heig[0] < 0 else 0.99*Heig[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                min_lambda = scipy.optimize.fsolve(self.find_lambda, init_guess, (Heig, Hvecs, g_M, p_M), xtol=1e-14, maxfev=500)
            LambdaI = min_lambda[0]*np.eye(len(self.IC.Internals))
            del_q_M1 = np.linalg.pinv(H_M - LambdaI)
            del_q_M2 = g_M - min_lambda[0]*p_M
            del_q_M = -np.dot(del_q_M1, del_q_M2)
            dq_new = np.dot(MWGMat_sqrt, del_q_M)

            if np.linalg.norm(dq_new) < 1e-6 or irc_sub_iteration > 100:
                dy = dy_to_pivot + p_prime
                dy_norm = np.linalg.norm(dy)
                v2 = p_prime/np.linalg.norm(p_prime)
                deg = np.degrees(np.arccos(np.dot(v1/np.linalg.norm(v1),v2)))
                mwdx = np.linalg.norm(np.dot(MWGMat_sqrt_inv, dy))
                half_mwdx = np.linalg.norm(np.dot(MWGMat_sqrt_inv, p_prime))
                self.IRC_dystep = dy_norm
                self.IRC_mwdxstep = mwdx
                self.IRC_info["total_disp"] += mwdx
                const = self.find_lambda(min_lambda, Heig, Hvecs, g_M, p_M)
                cnorm = self.get_cartesian_norm(dy) # Angstrom
                if ((const > 1 or min_lambda > Heig[0]) and irc_sub_iteration > 100) or mwdx > self.IRC_stepsize*1.5:
                    #logger.info("mwdx_1: %f \n" %np.linalg.norm(mwdx_1))
                    #logger.info("mwdx: %f \n" %mwdx)
                    #logger.info("cnorm: %f \n" %cnorm)
                    if self.IRC_info.get("substep_success"):
                        logger.info("IRC second sub-step failed. Rejecting the step.\n")
                        self.IRC_info["substep_success"] = False
                    else:
                        logger.info("IRC second sub-step failed. Taking a half-step.\n")
                        dy = dy_to_pivot
                        self.IRC_info["substep_success"] = True
                    logger.info('X Failed half step dy     = %.5f \n' % np.linalg.norm(p_prime))
                    logger.info('X Failed half step mw-dx  = %.5f Bohr*sqrt(amu)\n\n' % half_mwdx)
                    logger.info('X Failed total step dy: %.5f \n' %dy_norm)
                    logger.info('X Failed total step mw-dx = %.5f Bohr*sqrt(amu)\n' % mwdx)
                    logger.info('X Failed total cnorm: %.5f \n' %cnorm)
                    logger.info('Constraint: %.5f \n' %const)
                    logger.info('dq_new: %.5f \n' %np.linalg.norm(dq_new))
                else:
                    self.IRC_info["substep_success"] = True
                break
            irc_sub_iteration += 1
            p_prime += dq_new

        if self.params.verbose:
            logger.info('Angle between v1 and v2: %2.f \n' % deg)
            logger.info('Half step dy     = %.5f \n' % np.linalg.norm(p_prime))
            logger.info('Half step mw-dx  = %.5f Bohr*sqrt(amu)\n\n' % half_mwdx)
        logger.info('=> Total step dy    = %.5f \n' % dy_norm)
        logger.info('=> Total step mw-dx = %.5f Bohr*sqrt(amu)\n\n' % mwdx)
        self.Iteration += 1
        return dy

    def optimize_step(self):
        # This function is identical to the previous version without IRC (a copy of lines 426-504)
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
        if params.verbose: self.IC.printRotations(self.X)
        Eig = self.SortedEigenvalues(self.H)
        Emin = Eig[0].real
        if params.transition:
            v0 = 1.0
        elif Emin < params.epsilon:
            v0 = params.epsilon-Emin
        else:
            v0 = 0.0
        # Are we far from constraint satisfaction?
        self.farConstraints = self.IC.haveConstraints() and self.IC.maxConstraintViolation(self.X) > 1e-1
        ### OBTAIN AN OPTIMIZATION STEP ###
        # The trust radius is to be computed in Cartesian coordinates.
        # First take a full-size optimization step
        if params.verbose: logger.info("  Optimizer.step : Attempting full-size optimization step\n")
        dy, _, __ = self.get_delta_prime(v0, verbose=self.params.verbose)
        # Internal coordinate step size
        inorm = np.linalg.norm(dy)
        # Cartesian coordinate step size
        self.cnorm = self.get_cartesian_norm(dy)
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

        return dy


    def step(self):
        if self.params.irc:
            """
            Take an IRC step.
            """
            if self.IRC_info.get("opt"):
                # If the IRC switches to optimization, take an optimization step. () 
                dy = self.optimize_step()
            else:
                dy = self.IRC_step()
        else:
            """
            Take an optimization step.
            """
            dy = self.optimize_step()

        if dy is None: return

        ### Before updating any of our variables, copy current variables to "previous"
        self.cnorm = self.get_cartesian_norm(dy)
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
        # dyp = self.IC.Prims.calcDiff(self.X, X0)
        # print("Actual dy:", dy)
        self.Y += dy
        self.expect = flat(0.5*multi_dot([row(dy),self.H,col(dy)]))[0] + np.dot(dy,self.G)
        # self.expectdG = np.dot(self.H, col(dy).flatten())
        self.state = OPT_STATE.NEEDS_EVALUATION

    def reset_irc(self):
        self.dirname = self.dirname.replace('forward','backward')
        self.IRC_info["direction"] = -1
        self.Iteration = 0
        self.X = self.X_hist[0].copy()
        self.IC = self.IC0
        self.E = self.progress.qm_energies[0]
        self.gradx = self.progress.qm_grads[0]
        self.progress = self.progress[::-1]
        self.trust = self.params.trust
        self.IRC_info["opt"] = False
        self.trustprint = "="
        self.IRC_info["substep_success"] = True 
        self.IRC_info["IC_changed"] = False
        self.IRC_info["total_disp"] = 0.0
        self.prepareFirstStep()


    def evaluate_IRC_step(self, params, step_state, criteria_met, IRC_converged):

        if self.Iteration > params.maxiter:
            if self.IRC_direction == 'both':
                if self.IRC_info.get("direction") == -1:
                    logger.info("\nIRC backward direction reached maximum iteration number\n")
                    logger.info("Terminating IRC\n")
                    self.state = OPT_STATE.FAILED
                else:
                    logger.info("\nIRC forward direction reached maximum iteration number\n")
                    logger.info("IRC backward direction starts here\n\n")
                    self.reset_irc()
            else:
                logger.info("\nReached the maximum iteration number\n")
                logger.info("Terminating IRC\n")
                self.state = OPT_STATE.FAILED
            return True, step_state
        
        self.IC_check = False
        if step_state in (StepState.Reject, StepState.Poor):
            step_state = StepState.Reject
            if np.isclose(self.trust, params.tmin):
                logger.info("IRC stuck with the minimum step-size and bad quality step. Forcing it to take a step.\n")
                step_state = StepState.Okay
            if self.Iteration < 3:
                logger.info("Bad quality IRC step detected near the starting point. Decreasing the step-size to the minimum.\n")
                self.trust = params.tmin
            self.IC_check = True
            
        if not self.IRC_info.get("substep_success"):
            step_state = StepState.Reject
            self.IC_check = True

        if self.IRC_info["total_disp"] > 5*self.IRC_std_step:
            if criteria_met:
                if self.IRC_direction == 'both':
                    if self.IRC_info.get("direction") == 1:
                        logger.info("\nIRC forward direction converged\n")
                        logger.info("IRC backward direction starts here\n\n")
                        self.reset_irc()
                    elif self.IRC_info.get("direction") == -1:
                        self.SortedEigenvalues(self.H)
                        logger.info("Converged! =D\n")
                        self.state = OPT_STATE.CONVERGED
                else:
                    self.SortedEigenvalues(self.H)
                    logger.info("Converged! =D\n")
                    self.state = OPT_STATE.CONVERGED
                return True, step_state
            elif IRC_converged and not self.IRC_info.get("opt"):
                self.IRC_info["opt"] = True
                #if self.cnorm > 1e-4:
                #    self.trust = self.cnorm
                #self.params.tmax = self.params.trust*3
                logger.info("Switching to optimization\n")

        return False, step_state

    def evaluate_OPT_step(self, params, step_state, criteria_met, Converged_grms, Converged_drms, Converged_energy,
                          Converged_molpro_gmax, Converged_molpro_dmax):
        if criteria_met and self.conSatisfied:
            if params.irc:
                if self.IRC_info.get("direction") == 1 and self.IRC_direction == 'both':
                    logger.info("\nIRC forward direction converged\n")
                    logger.info("IRC backward direction starts here\n\n")
                    self.reset_irc()
                else:
                    self.SortedEigenvalues(self.H)
                    logger.info("Converged! =D\n")
                    self.state = OPT_STATE.CONVERGED
            else:
                self.SortedEigenvalues(self.H)
                logger.info("Converged! =D\n")
                self.state = OPT_STATE.CONVERGED
            return True, step_state

        if self.Iteration >= params.maxiter:
            self.SortedEigenvalues(self.H)
            logger.info("Maximum iterations reached (%i); increase --maxiter for more\n" % params.maxiter)
            if params.Converge_maxiter:
                logger.info("Exiting normally because --converge maxiter was set.\n")
                self.state = OPT_STATE.CONVERGED
            else:
                self.state = OPT_STATE.FAILED
            return True, step_state

        if params.qccnv and Converged_grms and (Converged_drms or Converged_energy) and self.conSatisfied:
            self.SortedEigenvalues(self.H)
            logger.info("Converged! (Q-Chem style criteria requires grms and either drms or energy)\n")
            self.state = OPT_STATE.CONVERGED
            return True, step_state

        if params.molcnv and Converged_molpro_gmax and (Converged_molpro_dmax or Converged_energy) and self.conSatisfied:
            self.SortedEigenvalues(self.H)
            logger.info("Converged! (Molpro style criteria requires gmax and either dmax or energy)\nThis is approximate since convergence checks are done in cartesian coordinates.\n")
            self.state = OPT_STATE.CONVERGED
            return True, step_state

        return False, step_state


    def evaluateStep(self):
        ### At this point, the state should be NEEDS_EVALUATION
        assert self.state == OPT_STATE.NEEDS_EVALUATION
        # Shorthand for self.params
        params = self.params
        # Write current optimization trajectory to file
        if self.params.xyzout is not None: 
            self.progress.write(self.params.xyzout)
            if self.viz_rotations:
                self.progress_with_r.write(os.path.splitext(self.params.xyzout)[0]+"_with_r.xyz")
        if self.params.qdata is not None: self.progress.write(self.params.qdata, ftype='qdata')
        # Project out the degrees of freedom that are constrained
        rms_gradient, max_gradient = self.calcGradNorm()
        rms_displacement, max_displacement = calc_drms_dmax(self.X, self.Xprev)
        rms_displacement_noalign, max_displacement_noalign = calc_drms_dmax(self.X, self.Xprev, align=False)
        del_E = self.E-self.Eprev
        # The ratio of the actual energy change to the expected change
        Quality = del_E/self.expect
        # The internal coordinate gradient (actually not really used in this function)
        self.G = self.IC.calcGrad(self.X, self.gradx).flatten()
        # For transition states, the quality factor decreases in both directions
        if params.transition and Quality > 1.0:
            Quality = 2.0 - Quality
        # Check convergence criteria
        Converged_energy = np.abs(del_E) < params.Convergence_energy
        Converged_grms = rms_gradient < params.Convergence_grms
        Converged_gmax = max_gradient < params.Convergence_gmax
        Converged_drms = rms_displacement < params.Convergence_drms
        Converged_dmax = max_displacement < params.Convergence_dmax
        # Set step state and log colors
        # 2020-03-10: Step quality thresholds are hard-coded here.
        colors = {}
        if Quality > 0.75: step_state = StepState.Good
        elif Quality > (0.5 if params.transition or params.irc else 0.25): step_state = StepState.Okay
        elif Quality > 0.0: step_state = StepState.Poor
        else:
            colors['energy'] = "\x1b[92m" if Converged_energy else "\x1b[91m"
            colors['quality'] = "\x1b[91m"
            step_state = StepState.Reject if (Quality < -1.0 or params.transition or params.irc) else StepState.Poor
        if 'energy' not in colors: colors['energy'] = "\x1b[92m" if Converged_energy else "\x1b[0m"
        if 'quality' not in colors: colors['quality'] = "\x1b[0m"
        colors['grms'] = "\x1b[92m" if Converged_grms else "\x1b[0m"
        colors['gmax'] = "\x1b[92m" if Converged_gmax else "\x1b[0m"
        colors['drms'] = "\x1b[92m" if Converged_drms else "\x1b[0m"
        colors['dmax'] = "\x1b[92m" if Converged_dmax else "\x1b[0m"
        # Molpro defaults for convergence
        Converged_molpro_gmax = max_gradient < params.Convergence_molpro_gmax
        Converged_molpro_dmax = max_displacement < params.Convergence_molpro_dmax
        self.conSatisfied = not self.IC.haveConstraints() or self.IC.maxConstraintViolation(self.X) < params.Convergence_cmax
        # Print status
        msg = "Step %4i :" % self.Iteration
        msg += " Displace = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % (colors['drms'], rms_displacement, colors['dmax'], max_displacement)
        msg += " Trust = %.3e (%s)" % (self.trust, self.trustprint)
        msg += " Grad%s = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("_T" if self.IC.haveConstraints() else "", colors['grms'], rms_gradient, colors['gmax'], max_gradient)
        logger.info(msg + " E (change) = % .10f (%s%+.3e\x1b[0m) Quality = %s%.3f\x1b[0m" % (self.E, colors['energy'], self.E-self.Eprev, colors['quality'], Quality) + "\n")

        if self.IC is not None and self.IC.haveConstraints():
            self.IC.printConstraints(self.X, thre=1e-3)
        if isinstance(self.IC, PrimitiveInternalCoordinates):
            logger.info(self.prim_msg + '\n')

        ### Check convergence criteria ###
        criteria_met = Converged_energy and Converged_grms and Converged_drms and Converged_gmax and Converged_dmax

        # The IRC method often oscillates near the end point without converging. 
        # If the energy, grms, and gmax converge, simple optimization procedure will be initiated. 
        IRC_converged = Converged_energy and Converged_grms and Converged_gmax

        if params.irc and not self.IRC_info.get("opt"):
            terminate, step_state = self.evaluate_IRC_step(params, step_state, criteria_met, IRC_converged)

            # When IRC deals with a small linear moleule, the 2nd part of the substep cancels the 1st part near convergence.
            # To help it with convergence, trust radius is decreased to the minimum. 
            if rms_displacement < 1e-7 and max_displacement < 1e-7:
                step_state = StepState.Okay
                self.trust = params.tmin
        else:
            terminate, step_state = self.evaluate_OPT_step(params, step_state, criteria_met, Converged_grms,
                                        Converged_drms, Converged_energy, Converged_molpro_gmax, Converged_molpro_dmax)

        if terminate: return

        assert self.state == OPT_STATE.NEEDS_EVALUATION
        
        ### Adjust Trust Radius and/or Reject Step ###
        prev_trust = self.trust
        # logger.info(" Check force/torque: rmsd = %.5f rmsd_noalign = %.5f ratio = %.5f\n" %
        #             (rms_displacement, rms_displacement_noalign, rms_displacement_noalign / rms_displacement))
        # LPW 2023-05-24: Hack for caterpillar. Enable via CLI later.
        # if step_state in (StepState.Okay, StepState.Poor, StepState.Reject) and params.transition:
        #     logger.info("LPW: Recalculating Hessian\n")
        #     self.recalcHess = True
        if step_state in (StepState.Poor, StepState.Reject):
            new_trust = max(params.tmin, min(self.trust, self.cnorm)/2)
            # if (Converged_grms or Converged_gmax) or (params.molcnv and Converged_molpro_gmax):
            #     new_trust = max(new_trust, self.params.Convergence_dmax if self.params.usedmax else self.params.Convergence_drms)
            self.trustprint = "\x1b[91m-\x1b[0m" if new_trust < self.trust else "="
            self.trust = new_trust
            # A poor quality step that is dominated by overall translation/rotation
            # is a sign that projecting out the net force and torque may be needed
            if self.params.subfrctor == 1 and ((rms_displacement_noalign / rms_displacement) > self.lowq_tr_thre):
                self.lowq_tr_count += 1
                if self.lowq_tr_count == self.lowq_tr_limit :
                    logger.info("Poor-quality step dominated by net translation/rotation detected; ")
                    logger.info("will project out net forces and torques past this point.\n")
        elif step_state == StepState.Good:
            new_trust = min(params.tmax, np.sqrt(2)*self.trust)
            self.trustprint = "\x1b[92m+\x1b[0m" if new_trust > self.trust else "="
            self.trust = new_trust
        elif step_state == StepState.Okay:
            self.trustprint = "="

        if step_state == StepState.Reject:
            if params.irc and not self.IRC_info.get("opt") and self.Iteration > 1:
                self.IRC_info["total_disp"] -= self.IRC_mwdxstep
                self.Iteration -= 1
                logger.info("\x1b[93mRejecting step - low quality IRC step\x1b[0m\n")
                self.trustprint = "\x1b[1;91mx\x1b[0m"
                self.X_rj = self.X.copy()
                self.Y = self.Yprev.copy()
                self.X = self.Xprev.copy()
                self.gradx = self.Gxprev.copy()
                self.G = self.Gprev.copy()
                self.E = self.Eprev
                self.engine.load_guess_files(self.dirname)
                self.recalcHess = False
                self.progress = self.progress[:-1]
                #if self.IC.bork:
                #    logger.info("Failed inverse iteration - checking coordinate system\n")
                #    self.checkCoordinateSystem(recover=True, cartesian=isinstance(self.IC, CartesianCoordinates))
                #else:
                #    self.X_hist = self.X_hist[:-1]
                #    self.Gx_hist = self.Gx_hist[:-1]
                if self.IC_check:
                    logger.info("Checking coordinate system\n")
                    IC_changed = self.checkCoordinateSystem(cartesian=isinstance(self.IC, CartesianCoordinates))
                    if IC_changed and not self.IRC_info.get("IC_changed"):
                        self.IRC_info["IC_changed"] = IC_changed
                return
            if hasattr(self, 'X_rj') and np.allclose(self.X_rj, self.X, atol=1e-6):
                logger.info("\x1b[93mA previously rejected step was repeated; accepting to avoid infinite loop\x1b[0m\n")
                delattr(self, 'X_rj')
            elif prev_trust <= params.tmin:
                logger.info("\x1b[93mNot rejecting step - trust below tmin = %.3e\x1b[0m\n" % params.tmin)
            elif rms_displacement <= 1.2*params.tmin:
                # Prevents rejecting / repeating the step and then accepting the step based on trust <= params.tmin
                # Suppose a step is taken whose length is close to tmin but trust is actually above tmin. Without this rule, the step would be rejected.
                # Then trust would be decreased, eventually to tmin, then the exact same step would be accepted.
                # The "1.2" is because the actual step can be larger than the trust radius by a small amount.
                logger.info("\x1b[93mNot rejecting step - RMS displacement close to tmin = %.3e\x1b[0m\n" % (params.tmin))
            # elif (not params.transition) and self.E < self.Eprev:
            elif self.E < self.Eprev and not params.transition:
                logger.info("\x1b[93mNot rejecting step - energy decreases during minimization\x1b[0m\n")
            elif Converged_energy:
                logger.info("\x1b[93mNot rejecting step - energy change meets convergence criteria\x1b[0m\n")
            elif self.farConstraints:
                logger.info("\x1b[93mNot rejecting step - far from constraint satisfaction\x1b[0m\n")
            else:
                logger.info("\x1b[93mRejecting step - quality is lower than %.1f\x1b[0m\n" % (0.0 if params.transition or params.irc else -1.0))
                self.trustprint = "\x1b[1;91mx\x1b[0m"
                # Store the rejected step.  In case the next step is identical to the rejected one, the next step should be accepted to avoid infinite loops.
                self.X_rj = self.X.copy()
                self.Y = self.Yprev.copy()
                self.X = self.Xprev.copy()
                self.gradx = self.Gxprev.copy()
                self.G = self.Gprev.copy()
                self.E = self.Eprev
                self.engine.load_guess_files(self.dirname)
                self.recalcHess = False
                return

        # Append steps to history (for rebuilding Hessian)
        self.X_hist.append(self.X)
        self.Gx_hist.append(self.gradx)
        self.engine.save_guess_files(self.dirname)

        ### Rebuild Coordinate System if Necessary ###
        UpdateHessian = (not self.params.hessian == 'each')
        if self.IC.bork and not params.irc:
            logger.info("Failed inverse iteration - checking coordinate system\n")
            self.checkCoordinateSystem(recover=True)
            UpdateHessian = False
        elif self.CoordCounter == (params.check - 1):
            logger.info("Checking coordinate system as requested every %i cycles\n" % params.check)
            if self.checkCoordinateSystem(recover=False): UpdateHessian = False
        else:
            self.CoordCounter += 1

        ## Save the regularization quaternions, used to lift rotation degeneracies for linear molecules.
        ## This function also repositions "e0" for linear angles.
        if self.IC.setRegularization(self.X) and UpdateHessian:
            self.rebuild_hessian()
            UpdateHessian = False

        # Check for large rotations (debugging purposes)
        if self.params.verbose >= 1: self.IC.largeRots()

        ### Update the Hessian ###
        if UpdateHessian:
            self.UpdateHessian()
        if hasattr(self, 'Hx'):
            self.H = self.IC.calcHess(self.X, self.gradx, self.Hx)
        # Then it's on to the next loop iteration!
        return

    def UpdateHessian(self):
        self.H = update_hessian(self.IC, self.H, [self.X, self.Xprev], [self.gradx, self.Gxprev], self.params, trust_limit=False, max_updates=1)
        
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
            if self.recalcHess:
                # If a Hessian recalculation is needed at this point, we need to
                # call calcEnergyForce() which will compute the cartesian Hessian,
                # convert it to IC, and then store it.
                self.calcEnergyForce()
        if self.state == OPT_STATE.FAILED:
            raise GeomOptNotConvergedError("Optimizer.optimizeGeometry() failed to converge.")
        # If we want to save the Hessian used by the optimizer (in Cartesian coordinates)
        if self.params.write_cart_hess:
            # One last Hessian update before writing it out
            self.UpdateHessian()
            logger.info("Saving current approximate Hessian (Cartesian coordinates) to %s" % self.params.write_cart_hess)
            Hx = self.IC.calcHessCart(self.X, self.G, self.H)
            np.savetxt(self.params.write_cart_hess, Hx, fmt='% 14.10f')
        if self.params.hessian in ['last', 'first+last', 'each']:
            Hx = calc_cartesian_hessian(self.X, self.molecule, self.engine, self.dirname, read_data=False, bigchem=self.params.bigchem, verbose=self.params.verbose)
            if self.params.frequency:
                self.frequency_analysis(Hx, 'last', True)
        return self.progress

    def checkStructure(self):
        """
        A function that checks for problematic structures and throws an error before
        calling any QC method.
        """
        # Check for three consecutive atoms in torsion angle becoming linear
        torsion_constraint_linear_angles = self.IC.torsionConstraintLinearAngles(self.X)
        if torsion_constraint_linear_angles:
            errorStr = "> Atoms Angle\n"
            for key, val in torsion_constraint_linear_angles.items():
                errorStr += "> %i-%i-%i %6.2f\n" % (key[0]+1, key[1]+1, key[2]+1, val)
            raise LinearTorsionError("A constrained torsion has three consecutive atoms\n"
                                     "forming a nearly linear angle, making the torsion angle poorly defined.\n"+errorStr)
        

class OPT_STATE(object):
    """ This describes the state of an OptObject during the optimization process
    """
    NEEDS_EVALUATION = 0  # convergence has not been evaluated -> calcualte Energy, Forces
    SKIP_EVALUATION  = 1  # We know this is not yet converged -> skip Energy
    CONVERGED        = 2
    FAILED           = 3  # optimization failed with no recovery option

class StepState(object):
    """ This describes the state of an OptObject during the optimization process
    """
    Reject  = 0 # Reject the step
    Poor    = 1 # Poor step; decrease the trust radius down to the lower limit.
    Okay    = 2 # Okay step; do not change the trust radius.
    Good    = 3 # Good step; increase the trust radius up to the limit.
    
def Optimize(coords, molecule, IC, engine, dirname, params, print_info=True):
    """
    Optimize the geometry of a molecule. This function used to contain the whole
    optimization loop, which has since been moved to the Optimizer() class;
    now a wrapper and kept for compatibility.

    Parameters
    ----------
    coords : np.ndarray
        1-dimensional array of shape (3*N_atoms) containing atomic coordinates in Bohr
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
    optimizer = Optimizer(coords, molecule, IC, engine, dirname, params, print_info)
    return optimizer.optimizeGeometry()

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
    if kwargs.get('logIni') is None:
        logIni = os.path.join(config_dir, 'log.ini')
    else:
        logIni = kwargs.get('logIni')
    logfilename = kwargs.get('prefix')
    # Input file for optimization; QC input file or OpenMM .xml file
    inputf = kwargs.get('input')
    verbose = kwargs.get('verbose', False)
    # Get calculation prefix and temporary directory name
    arg_prefix = kwargs.get('prefix', None) #prefix for output file and temporary directory
    prefix = arg_prefix if arg_prefix is not None else os.path.splitext(inputf)[0]
    logfilename = rf"{prefix}.log"
    # Create a backup if the log file already exists
    backed_up = bak(logfilename)
    import logging.config
    logging.config.fileConfig(logIni,defaults={'logfilename': logfilename},disable_existing_loggers=False)
    #==============================#
    #| End log file configuration |#
    #==============================#

    import geometric
    logger.info('geometric-optimize called with the following command line:\n')
    argv_print = []
    # When geometric-optimize is called on the command line with an argument such as 
    # --ase-kwargs='{"method":"GFN2-xTB"}'
    # the shell strips away the single quotes resulting in printing out
    # --ase-kwargs={"method":"GFN2-xTB"}
    # making the copy-pasted command invalid.
    # This is a dirty hack to put the single quotes back.
    for arg in sys.argv:
        arg = arg.replace('\'{','{').replace('{','\'{')
        arg = arg.replace('}\'','}').replace('}','}\'')
        argv_print.append(arg)
    logger.info(' '.join(argv_print)+'\n')
    print_logo(logger)
    now = datetime.now()
    logger.info('-=# \x1b[1;94m geomeTRIC started. Version: %s \x1b[0m #=-\n' % (geometric.__version__))
    logger.info('Current date and time: %s\n' % now.strftime("%Y-%m-%d %H:%M:%S"))
    printcool_dictionary(kwargs, 'Arguments passed to driver run_optimizer():')
    
    if backed_up:
        logger.info('Backed up existing log file: %s -> %s\n' % (logfilename, os.path.basename(backed_up)))

    t0 = time.time()

    # Create the params object, containing data to be passed into the optimizer
    params = OptParams(**kwargs)
    params.printInfo()

    # Create "dirname" folder for writing
    dirname = prefix+".tmp"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if params.irc:
        if params.irc_direction == 'both':
            prefix += '_forward'
            dirname = os.path.join(dirname, "irc_forward.tmp")
        else:
            prefix += '_%s' % params.irc_direction

    kwargs['dirname'] = dirname
    
    # Get the Molecule and engine objects needed for optimization
    M, engine = get_molecule_engine(**kwargs)
    
    # Create Work Queue object
    if kwargs.get('wqport', 0):
        logger.info("Creating Work Queue object for distributed Hessian calculation\n")
        createWorkQueue(kwargs['wqport'], debug=verbose>1)

    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() * ang2bohr

    # Read in the constraints
    constraints = kwargs.get('constraints', None) # Constraint input file (optional)
    conmethod = kwargs.get('conmethod', 0) # Constraint algorithm - 0, original; 1, alternative
    rigid = kwargs.get('rigid', False) # Whether to keep molecules rigid during optimization (TRIC only)

    if constraints is not None:
        Cons, CVals = parse_constraints(M, open(constraints).read())
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
                    'tric-p':(PrimitiveInternalCoordinates, False, False),
                    'tric':(DelocalizedInternalCoordinates, False, False)}
    coordsys = kwargs.get('coordsys', 'tric')
    CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]

    # Perform an initial single-point QM calculation to determine bonding & fragments, if using TRIC
    if hasattr(engine, 'calc_bondorder') and coordsys.lower() in ['hdlc', 'tric'] and params.bothre > 1e-3:
        bothre = params.bothre
        logger.info("Calculating QM bond order and forming bonds using criterion of %.2f\n" % bothre)
        qm_bo = engine.calc_bondorder(coords, dirname)
        M.qm_bondorder = [qm_bo]
        M.build_topology(bond_order=bothre)
        if len(M.molecules) == 1 and bothre < 0.75:
            logger.info("Only one fragment found; increasing threshold\n")
            # Increase the bond order threshold until there are at least 2 fragments of size >1
            while bothre < 0.75 and len([m for m in M.molecules if len(m.e()) > 1]) == 1:
                bothre += 0.01
                M.build_topology(bond_order=bothre)
            logger.info("Using threshold of %.2f, there are now %i molecules\n" % (bothre, len(M.molecules)))
        M.top_settings['read_bonds'] = True
        # Delete the QM bond order to avoid problems when more structures are added
        del M.Data['qm_bondorder']
    else:
        if coordsys.lower() in ['hdlc', 'tric'] and params.bothre > 1e-3:
            logger.info("Requested bond order-based connectivity but it is not available in the current engine\n")
        logger.info("Bonds will be generated from interatomic distances less than %.2f times sum of covalent radii\n" % M.top_settings['Fac'])

    # TRIC IRC will fail when there is only one molecule due to the small Hessian eigenvalues.
    if params.irc and len(M.molecules) == 1 and coordsys == 'tric':
        coordsys = 'dlc'

    CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]

    IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVals[0] if CVals is not None else None,
                    conmethod=conmethod, rigid=rigid)
    
    #========================================#
    #| End internal coordinate system setup |#
    #========================================#

    # Auxiliary functions (will not do optimization):
    displace = kwargs.get('displace', False) # Write out the displacements of the coordinates.
    if displace:
        write_displacements(coords, M, IC, dirname, verbose)
        return

    fdcheck = kwargs.get('fdcheck', False) # Check internal coordinate gradients using finite difference..
    if fdcheck:
        IC.Prims.checkFiniteDifferenceGrad(coords)
        IC.Prims.checkFiniteDifferenceHess(coords)
        check_internal_grad(coords, M, IC.Prims, engine, dirname, verbose)
        check_internal_hess(coords, M, IC.Prims, engine, dirname, verbose)
        return

    # Print out information about the coordinate system
    if isinstance(IC, CartesianCoordinates):
        logger.info("%i Cartesian coordinates being used\n" % (3*M.na))
    else:
        logger.info("%i internal coordinates being used (instead of %i Cartesians)\n" % (len(IC.Internals), 3*M.na))
    logger.info(IC)
    logger.info("\n")

    # Print out a note if DFT is used for non-fragmented systems; recommend --dlc and --subfrctor 2.
    if engine.detect_dft() and coordsys != "dlc" and len(IC.frags) == 1:
        logger.info("#===================================================================================#\n")
        logger.info("#| \x1b[91mNote: Detected the use of DFT for a system containing only one fragment.\x1b[0m        |#\n")
        logger.info("#|                                                                                 |#\n")
        logger.info("#| DFT calculations that use small to medium-sized grids can sometimes result in   |#\n")
        logger.info("#| energies and/or gradients that are not translationally/rotationally invariant,  |#\n")
        logger.info("#| which varies depending on the QC program used.                                  |#\n")
        logger.info("#|                                                                                 |#\n")
        logger.info("#| Spurious translation/rotation contributions to the energy and/or gradient       |#\n")
        logger.info("#| may cause convergence failure that is observable as 'movement' or 'tumbling'    |#\n")
        logger.info("#| of the molecule in the optimization output. If observed, rerun the calculation  |#\n")
        logger.info("#| using --coordsys dlc and --subfrctor 2. It will project out overall translation |#\n")
        logger.info("#| and rotation from the optimization space.                                       |#\n")
        logger.info("#===================================================================================#\n")

    if Cons is None:
        # Run a standard geometry optimization
        add = "_optim.xyz"
        if params.irc:
            if params.irc_direction == 'both':
                prefix = prefix.replace('_forward', '')
            add = "_irc.xyz"
        params.xyzout = prefix+add
        progress = Optimize(coords, M, IC, engine, dirname, params)
    else:
        # Run a single constrained geometry optimization or scan over a grid of values
        if isinstance(IC, (CartesianCoordinates, PrimitiveInternalCoordinates)):
            raise RuntimeError("Constraints only work with delocalized internal coordinates")
        Mfinal = None
        for ic, CVal in enumerate(CVals):
            if len(CVals) > 1:
                logger.info("---=== Scan %i/%i : Constrained Optimization ===---\n" % (ic+1, len(CVals)))
            IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVal, conmethod=conmethod, rigid=rigid)
            IC.printConstraints(coords, thre=-1)
            if len(CVals) > 1:
                params.xyzout = prefix+"_scan-%03i.xyz" % (ic+1)
                # In the special case of a constraint scan, we write out multiple qdata.txt files
                if params.qdata is not None: params.qdata = 'qdata_scan-%03i.txt' % (ic+1)
            else:
                params.xyzout = prefix+"_optim.xyz"
            if ic == 0:
                progress = Optimize(coords, M, IC, engine, dirname, params)
            else:
                progress += Optimize(coords, M, IC, engine, dirname, params, print_info=False)
            # update the structure for next optimization in SCAN (by CNH)
            M.xyzs[0] = progress.xyzs[-1]
            coords = progress.xyzs[-1].flatten() * ang2bohr
            if Mfinal:
                Mfinal += progress[-1]
            else:
                Mfinal = progress[-1]
            cNames = IC.getConstraintNames()
            cVals = IC.getConstraintTargetVals()
            comment = ', '.join(["%s = %.2f" % (cName, cVal) for cName, cVal in zip(cNames, cVals)])
            Mfinal.comms[-1] = "Scan Cycle %i/%i ; %s ; %s" % (ic+1, len(CVals), comment, progress.comms[-1])
            #print
        if len(CVals) > 1:
            Mfinal.write('scan-final.xyz')
            if params.qdata is not None: Mfinal.write('qdata-final.txt')
    print_citation(logger)
    logger.info("Time elapsed since start of run_optimizer: %.3f seconds\n" % (time.time()-t0))
    if kwargs.get('wqport', 0):
        destroyWorkQueue()
    return progress

def main(): # pragma: no cover
    # Read user input (look in params.py for full list of options).
    # args is a dictionary containing only user-specified arguments
    # (i.e. keys without provided values are removed.)
    args = parse_optimizer_args(sys.argv[1:])

    # Run the optimizer.
    try:
        run_optimizer(**args)
    except EngineError:
        logger.info("EngineError:\n" + traceback.format_exc())
        sys.exit(51)
    except GeomOptNotConvergedError:
        logger.info("Geometry Converge Failed Error:\n" + traceback.format_exc())
        sys.exit(50)
    except GeomOptStructureError:
        logger.info("Structure Error:\n" + traceback.format_exc())
        sys.exit(50)
    except HessianExit:
        logger.info("Exiting normally.\n")
        sys.exit(0)
    except:
        logger.info("Unknown Error:\n" + traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
