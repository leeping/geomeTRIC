"""
optimize.py: Driver and core functions for geometry optimization

Copyright 2016-2020 Regents of the University of California and the Authors

Authors: Lee-Ping Wang, Chenchen Song

Contributors: Yudong Qiu, Daniel G. A. Smith, Alberto Gobbi, Josh Horton

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
import pkg_resources
from copy import deepcopy

import numpy as np
from numpy.linalg import multi_dot

import geometric
from .info import print_logo, print_citation
from .internal import CartesianCoordinates, PrimitiveInternalCoordinates, DelocalizedInternalCoordinates
from .ic_tools import check_internal_grad, check_internal_hess, write_displacements
from .normal_modes import calc_cartesian_hessian, frequency_analysis
from .step import brent_wiki, Froot, calc_drms_dmax, get_cartesian_norm, rebuild_hessian, get_delta_prime, trust_step, force_positive_definite
from .prepare import get_molecule_engine, parse_constraints
from .params import OptParams, parse_optimizer_args
from .nifty import row, col, flat, bohr2ang, ang2bohr, logger, bak, createWorkQueue
from .errors import InputError, HessianExit, EngineError, GeomOptNotConvergedError, GeomOptStructureError, LinearTorsionError

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
        # Sanity check - if there's only one atom, it will probably crash
        if self.molecule.na < 2:
            raise InputError("Geometry optimizer assumes there are at least two atoms in the system")

    def get_cartesian_norm(self, dy, verbose=None):
        if not verbose: verbose = self.params.verbose
        return get_cartesian_norm(self.X, dy, self.IC, self.params.enforce, self.params.verbose)

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
        self.H0 = self.IC.guess_hessian(self.coords)
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

    def rebuild_hessian(self):
        self.H = rebuild_hessian(self.IC, self.H0, self.X_hist, self.Gx_hist, self.params)

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
        frequency_analysis(self.X, hessian, self.molecule.elem, energy=self.E, temperature=self.params.temperature, pressure=self.params.pressure, verbose=self.params.verbose, 
                           outfnm='%s.vdata_%s' % (prefix, suffix), note='Iteration %i Energy % .8f%s' % (self.Iteration, self.E, ' (Optimized Structure)' if afterOpt else ''),
                           wigner=((self.params.wigner, os.path.join(self.dirname, 'wigner')) if do_wigner else None))


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
        self.E = spcalc['energy']
        self.gradx = spcalc['gradient']
        # Calculate Hessian at the first step, or at each step if desired
        if self.params.hessian == 'each':
            # Hx is assumed to be the Cartesian Hessian at the current step.
            # Otherwise we use the variable name Hx0 to avoid almost certain confusion.
            self.Hx = calc_cartesian_hessian(self.X, self.molecule, self.engine, self.dirname, read_data=True, verbose=self.params.verbose)
            if self.params.frequency:
                self.frequency_analysis(self.Hx, 'iter%03i' % self.Iteration, False)
        elif self.Iteration == 0:
            if self.params.hessian in ['first', 'stop', 'first+last']:
                self.Hx0 = calc_cartesian_hessian(self.X, self.molecule, self.engine, self.dirname, read_data=True, verbose=self.params.verbose)
                if self.params.frequency:
                    self.frequency_analysis(self.Hx0, 'first', False)
                if self.params.hessian == 'stop':
                    logger.info("Exiting as requested after Hessian calculation.\n")
                    logger.info("Cartesian Hessian is stored in %s/hessian/hessian.txt.\n" % self.dirname)
                    raise HessianExit
                    # sys.exit(0)
            elif hasattr(self.params, 'hess_data') and self.Iteration == 0:
                self.Hx0 = self.params.hess_data.copy()
                if self.params.frequency:
                    self.frequency_analysis(self.Hx0, 'first', False)
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
        if params.verbose: self.IC.printRotations(self.X)
        Eig = self.SortedEigenvalues()
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
        # dyp = self.IC.Prims.calcDiff(self.X, X0)
        # print("Actual dy:", dy)
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
        colors = {}
        colors['quality'] = "\x1b[0m"
        # 2020-03-10: Step quality thresholds are hard-coded here.
        # At the moment, no need to set them as variables.
        if params.transition:
            if Quality > 0.8 and Quality < 1.2: step_state = StepState.Good
            elif Quality > 0.5 and Quality < 1.5: step_state = StepState.Okay
            elif Quality > 0.0 and Quality < 2.0: step_state = StepState.Poor
            else:
                colors['energy'] = "\x1b[91m"
                colors['quality'] = "\x1b[91m"
                step_state = StepState.Reject
        else:
            if Quality > 0.75: step_state = StepState.Good
            elif Quality > 0.25: step_state = StepState.Okay
            elif Quality > 0.0: step_state = StepState.Poor
            else:
                colors['energy'] = "\x1b[91m"
                colors['quality'] = "\x1b[91m"
                step_state = StepState.Poor if Quality > -1.0 else StepState.Reject
        # Check convergence criteria
        Converged_energy = np.abs(self.E-self.Eprev) < params.Convergence_energy
        Converged_grms = rms_gradient < params.Convergence_grms
        Converged_gmax = max_gradient < params.Convergence_gmax
        Converged_drms = rms_displacement < params.Convergence_drms
        Converged_dmax = max_displacement < params.Convergence_dmax
        if 'energy' not in colors: colors['energy'] = "\x1b[92m" if Converged_energy else "\x1b[0m"
        colors['grms'] = "\x1b[92m" if Converged_grms else "\x1b[0m"
        colors['gmax'] = "\x1b[92m" if Converged_gmax else "\x1b[0m"
        colors['drms'] = "\x1b[92m" if Converged_drms else "\x1b[0m"
        colors['dmax'] = "\x1b[92m" if Converged_dmax else "\x1b[0m"
        # Molpro defaults for convergence
        Converged_molpro_gmax = max_gradient < params.Convergence_molpro_gmax
        Converged_molpro_dmax = max_displacement < params.Convergence_molpro_dmax
        self.conSatisfied = not self.IC.haveConstraints() or self.IC.maxConstraintViolation(self.X) < 1e-2
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
        prev_trust = self.trust
        if step_state in (StepState.Poor, StepState.Reject):
            new_trust = max(params.tmin, min(self.trust, self.cnorm)/2)
            self.trustprint = "\x1b[91m-\x1b[0m" if new_trust < self.trust else "="
            self.trust = new_trust
        elif step_state == StepState.Good:
            new_trust = min(params.tmax, np.sqrt(2)*self.trust)
            self.trustprint = "\x1b[92m+\x1b[0m" if new_trust > self.trust else "="
            self.trust = new_trust
        elif step_state == StepState.Okay:
            self.trustprint = "="

        if step_state == StepState.Reject:
            if prev_trust <= params.thre_rj:
                logger.info("\x1b[93mNot rejecting step - trust below %.3e\x1b[0m\n" % params.thre_rj)
            elif (not params.transition) and self.E < self.Eprev:
                logger.info("\x1b[93mNot rejecting step - energy decreases during minimization\x1b[0m\n")
            elif self.farConstraints:
                logger.info("\x1b[93mNot rejecting step - far from constraint satisfaction\x1b[0m\n")
            else:
                logger.info("\x1b[93mStep Is Rejected\x1b[0m\n")
                self.trustprint = "\x1b[1;91mx\x1b[0m"
                self.Y = self.Yprev.copy()
                self.X = self.Xprev.copy()
                self.gradx = self.Gxprev.copy()
                self.G = self.Gprev.copy()
                self.E = self.Eprev
                return

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
        # Check for large rotations (debugging purposes)
        if self.params.verbose >= 1: self.IC.largeRots()
        # Check for large rotations in linear molecules
        if self.IC.linearRotCheck():
            logger.info("Large rotations in linear molecules - refreshing Rotator reference points and DLC vectors\n")
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
            ts_bfgs = False
            if ts_bfgs: # pragma: no cover
                logger.info("TS-BFGS Hessian update\n")
                # yk = Dg; dk = Dy
                dk = col(self.Y - self.Yprev)
                yk = col(self.G - self.Gprev)
                jk = yk - np.dot(self.H, dk)
                B = force_positive_definite(self.H)
                # Scalar 1: dk^T |Bk| dk
                s1 = multi_dot([dk.T, B, dk])
                # Scalar 2: (yk^T dk)^2 + (dk^T |Bk| dk)^2
                s2 = np.dot(yk.T, dk)**2 + s1**2
                # Vector quantities
                v2 = np.dot(yk.T, dk)*yk + s1*np.dot(B, dk)
                uk = v2/s2
                Ek = np.dot(jk, uk.T) + np.dot(uk, jk.T) + np.dot(jk.T, dk) * np.dot(uk, uk.T)
                self.H += Ek
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
                # phi = 1.0
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
        if self.params.hessian in ['last', 'first+last', 'each']:
            Hx = calc_cartesian_hessian(self.X, self.molecule, self.engine, self.dirname, read_data=False, verbose=self.params.verbose)
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
    
def Optimize(coords, molecule, IC, engine, dirname, params):
    """
    Optimize the geometry of a molecule. This function used to contain the whole
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


def set_up_coordinate_system(molecule, coordsys, conmethod, CVals, Cons=None):
    # First item in tuple: The class to be initialized
    # Second item in tuple: Whether to connect non-bonded fragments
    # Third item in tuple: Whether to throw in all Cartesians (no effect if second item is True)
    coord_sys_dict = {'cart': (CartesianCoordinates, False, False),
                      'prim': (PrimitiveInternalCoordinates, True, False),
                      'dlc': (DelocalizedInternalCoordinates, True, False),
                      'hdlc': (DelocalizedInternalCoordinates, False, True),
                      'tric-p': (PrimitiveInternalCoordinates, False, False),
                      'tric': (DelocalizedInternalCoordinates, False, False)}
    coord_class, connect, addcart = coord_sys_dict[coordsys.lower()]

    return coord_class(molecule, build=True, connect=connect, addcart=addcart, constraints=Cons,
                       cvals=CVals,
                       conmethod=conmethod)


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
        import geometric.optimize
        logIni = pkg_resources.resource_filename(geometric.optimize.__name__, 'config/log.ini')
    else:
        logIni = kwargs.get('logIni')
    logfilename = kwargs.get('prefix')
    # Input file for optimization; QC input file or OpenMM .xml file
    inputf = kwargs.get('input')
    verbose = kwargs.get('verbose', False)
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
    print_logo(logger)
    logger.info('-=# \x1b[1;94m geomeTRIC started. Version: %s \x1b[0m #=-\n' % geometric.__version__)
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
    kwargs['dirname'] = dirname
    
    # Get the Molecule and engine objects needed for optimization
    M, engine = get_molecule_engine(**kwargs)

    # Create Work Queue object
    if kwargs.get('port', 0):
        logger.info("Creating Work Queue object for distributed Hessian calculation\n")
        createWorkQueue(kwargs['port'], debug=verbose>1)

    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() * ang2bohr

    # Read in the constraints
    constraints = kwargs.get('constraints', None) #Constraint input file (optional)

    if constraints is not None:
        Cons, CVals = parse_constraints(M, open(constraints).read())
    else:
        Cons = None
        CVals = None

    #=========================================#
    #| Set up the internal coordinate system |#
    #=========================================#
    coordsys = kwargs.get('coordsys', 'tric')
    IC = set_up_coordinate_system(M, coordsys, params.conmethod,
                                  CVals[0] if CVals is not None else None, Cons)

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
            IC = set_up_coordinate_system(M, coordsys=coordsys, conmethod=params.conmethod, CVals=CVals, Cons=Cons)
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
                progress += Optimize(coords, M, IC, engine, dirname, params)
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
