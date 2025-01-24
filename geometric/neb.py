"""
neb.py: NEB method

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


from __future__ import print_function
from __future__ import division

import os, sys, time
from collections import OrderedDict
import numpy as np
from copy import deepcopy
from datetime import datetime
from .info import print_logo, print_citation
from .prepare import get_molecule_engine
from .optimize import Optimize
from .params import OptParams, NEBParams, parse_neb_args
from .step import get_delta_prime_trm, brent_wiki, trust_step, calc_drms_dmax
from .engine import Blank
from .internal import CartesianCoordinates, PrimitiveInternalCoordinates, DelocalizedInternalCoordinates, ChainCoordinates
from .nifty import flat, row, col, createWorkQueue, getWorkQueue, wq_wait, ang2bohr, bohr2ang, kcal2au, au2kcal, au2evang, logger
from .molecule import EqualSpacing
from .errors import NEBStructureError, NEBChainShapeError, NEBBandTangentError, NEBBandGradientError
from .config import config_dir

def print_forces(chain, avgg, maxg):
    """Check average and maximum chain forces and return color coded string"""
    avgg_print = "%.4f" %chain.avgg
    maxg_print = "%.4f" %chain.maxg

    # Else statements were added to ensure the same string lengths
    if chain.avgg < avgg:
        avgg_print = "\x1b[92m%s\x1b[0m" % avgg_print
    elif chain.avgg > 100*avgg:
        avgg_print = "\x1b[91m%s\x1b[0m" % avgg_print
    else:
        avgg_print = "\x1b[0m%s" % avgg_print

    if chain.maxg < maxg:
        maxg_print = "\x1b[92m%s\x1b[0m" % maxg_print
    elif chain.maxg > 100*maxg:
        maxg_print = "\x1b[91m%s\x1b[0m" % maxg_print
    else:
        maxg_print = "\x1b[0m%s" % maxg_print

    return avgg_print, maxg_print

def rms_gradient(gradx):
    """Return the RMS of a Cartesian gradient."""
    atomgrad = np.sqrt(np.sum((gradx.reshape(-1, 3)) ** 2, axis=1))
    return np.sqrt(np.mean(atomgrad**2))

def CoordinateSystem(M, coordtype, chain=False, guessw=0.1):
    """
    Parameters
    ----------
    M : Molecule object
        Contains all structures of the input chain
    coordtype : string
        Pass in 'cart', 'prim', 'dlc', 'hdlc', or 'tric'
    chain : bool
        True will return a chain object
    guessw : float
        Guessed weight value for the chain coordinate

    Returns
    -------
    InternalCoordinates
        The corresponding internal coordinate system
    """
    # First item in tuple: The class to be initialized
    # Second item in tuple: Whether to connect nonbonded fragments
    # Third item in tuple: Whether to throw in all Cartesians (no effect if second item is True)
    # Fourth item in tuple: Build a chain coordinate system
    CoordSysDict = {
        "cart": (CartesianCoordinates, False, False),
        "prim": (PrimitiveInternalCoordinates, True, False),
        "dlc": (DelocalizedInternalCoordinates, True, False),
        "hdlc": (DelocalizedInternalCoordinates, False, True),
        "tric": (DelocalizedInternalCoordinates, False, False),
        "tric-p": (PrimitiveInternalCoordinates, False, False),
    }  # Primitive TRIC, i.e. not delocalized
    CoordClass, connect, addcart = CoordSysDict[coordtype]
    if CoordClass is DelocalizedInternalCoordinates:
        IC = CoordClass(M, build=True, connect=connect, addcart=addcart)
    elif chain:
        IC = ChainCoordinates(M, connect=connect, addcart=addcart, cartesian=(coordtype == "cart"),
                               guessw=guessw)
    else:
        IC = CoordClass(M, build=True, connect=connect, addcart=addcart, chain=False)
    IC.coordtype = coordtype
    return IC


class Structure(object):
    """Class representing a single structure in a chain."""

    def __init__(self, molecule, engine, tmpdir, coordtype, coords=None):
        """
        Create a Structure object.

        Parameters
        ----------
        molecule : Molecule object, 1 frame
            Contains the properties of the molecule (elements, connectivity etc).
        engine : Engine object
            Wrapper around quantum chemistry code (currently not a ForceBalance engine)
        tmpdir : str
            The folder in which to run calculations
        coordtype : string, optional
            Choice of coordinate system (Either cart, prim, dlc, hdlc, or tric) ; defaults to Cartesian
        coords : np.ndarray, optional
            Flat array in a.u. containing coordinates (will overwrite what we have in molecule)
            This is convenient if we are keeping a reference Molecule object but updating coordinates
        """
        # Keep a copy of the Molecule object
        # LPW: This copy operation is checked and deemed necessary
        self.M = deepcopy(molecule)
        if len(self.M) != 1:
            raise NEBStructureError("Please pass in a Molecule object with just one frame")
        # Overwrite coordinates if we are passing in new ones
        if coords is not None:
            self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
            self.M.build_topology()
        # Set initial Cartesian coordinates
        self.cartesians = self.M.xyzs[0].flatten() * ang2bohr
        # The default engine used to calculate energies
        self.engine = engine
        # Temporary folder for running calculations
        self.tmpdir = tmpdir
        # The type of internal coordinate system.
        self.coordtype = coordtype
        # The internal coordinate system.
        IC = CoordinateSystem(self.M, self.coordtype)
        self.set_IC(IC)
        # The values of internal coordinates corresponding to the Cartesians
        self.CalcInternals()
        # The number of atoms
        self.na = self.M.na
        # The potential energy calculated using an electronic structure method
        self.energy = None
        # The gradient in Cartesian coordinates
        self.grad_cartesian = None
        # The gradient in internal coordinates
        self.grad_internal = None

    def clearCalcs(self, clearEngine=True):
        self.energy = None
        self.grad_cartesian = None
        self.grad_internal = None
        self.IC.clearCache()
        if clearEngine:
            self.engine.clearCalcs()
        self.CalcInternals()

    def set_cartesians(self, value):
        if not isinstance(value, np.ndarray):
            raise NEBStructureError("Please pass in a flat NumPy array in a.u.")
        if len(value.shape) != 1:
            raise NEBStructureError("Please pass in a flat NumPy array in a.u.")
        if value.shape[0] != (3 * self.M.na):
            raise NEBStructureError("Input array dimensions should be 3x number of atoms")
        # Multiplying by bohr2ang copies it
        self.M.xyzs[0] = value.reshape(-1, 3) * bohr2ang
        # LPW: This copy operation is checked and deemed necessary
        self._cartesians = value.copy()

    def get_cartesians(self):
        return self._cartesians

    cartesians = property(get_cartesians, set_cartesians)

    def set_IC(self, IC):
        self._IC = IC
        self.coordtype = IC.coordtype
        self.nICs = len(IC.Internals)
        self.CalcInternals()

    def get_IC(self):
        return self._IC

    IC = property(get_IC, set_IC)

    def CalcInternals(self, coords=None):
        """
        Calculate the internal coordinates using the Structure's internal coordinate system
        for some input Cartesian coordinates.

        Parameters
        ----------
        coords : optional, np.ndarray
            Flat array containing input cartesian coordinates in a.u.
            If not provided, will use stored cartesian coordinates and set self.internals

        Returns
        -------
        np.ndarray
            Flat array containing internal coordinates
        """
        if coords is None:
            internals = self.IC.calculate(self.cartesians)
            self.internals = internals
        else:
            internals = self.IC.calculate(coords)
        return internals

    def ComputeEnergyGradient(self, result=None):
        """Compute energies and Cartesian gradients for the current structure."""
        # If the result (energy and gradient in a dictionary) is provided, skip the calculations.
        if result is None:
            result = self.engine.calc(self.cartesians, self.tmpdir)
        self.energy = result["energy"]
        self.grad_cartesian = np.array(result["gradient"])
        self.grad_internal = self.IC.calcGrad(self.cartesians, self.grad_cartesian)

    def QueueEnergyGradient(self):
        self.engine.calc_wq(self.cartesians, self.tmpdir)

    def GetEnergyGradient(self):
        """Add energy and gradient attributes"""
        result = self.engine.read_wq(self.cartesians, self.tmpdir)
        self.energy = result["energy"]
        self.grad_cartesian = result["gradient"]
        self.grad_internal = self.IC.calcGrad(self.cartesians, self.grad_cartesian)

    def OptimizeGeometry(self, gtol=None):
        """Optimize the geometry of this Structure."""
        opt_params = OptParams()
        if gtol != None and opt_params.Convergence_grms * au2evang > gtol:
            opt_params.Convergence_grms = gtol / au2evang
            opt_params.Convergence_gmax = 1.5 * gtol / au2evang

        self.IC = CoordinateSystem(self.M, "tric")

        optProg = Optimize(self.cartesians, self.M, self.IC, self.engine, self.tmpdir, opt_params)

        self.cartesians = np.array(optProg[-1].xyzs).flatten()
        self.M = optProg[-1]
        self.IC = CoordinateSystem(self.M, self.coordtype)
        self.CalcInternals()


class Chain(object):
    """Class representing a chain of states."""
    def __init__(self, molecule, engine, tmpdir, params, coords=None):
        """
        Create a Chain object.

        Parameters
        ----------
        molecule : Molecule object, N frames
            Contains the properties of the molecule (elements, connectivity etc).
        engine : Engine object
            Wrapper around quantum chemistry code (currently not a ForceBalance engine)
        tmpdir : string
            Temporary directory (relative to root) that the calculation temporary files will be written to.
        parmas : NEB parameter object
            params.NEBparams()
        coords : np.ndarray, optional
            Array in a.u. containing coordinates (will overwrite what we have in molecule)
            Pass either an array with shape (len(M), 3*M.na), or a flat array with the same number of elements.
        """
        self.params = params
        # LPW: This copy operation is checked and deemed necessary
        self.M = deepcopy(molecule)
        self.engine = engine
        self.tmpdir = tmpdir
        # HP 5/2/2023: coordtype is set to 'cart' here.
        self.coordtype='cart'

        # coords is a 2D array with dimensions (N_image, N_atomx3) in atomic units
        if coords is not None:
            # Reshape the array if we passed in a flat array
            if len(coords.shape) == 1:
                coords = coords.reshape(len(self), 3 * self.M.na)
            if coords.shape != (len(self), 3 * self.M.na):
                raise NEBChainShapeError("Coordinates do not have the right shape")
            for i in range(len(self)):
                self.M.xyzs[i] = coords[i, :].reshape(-1, 3) * bohr2ang
        self.na = self.M.na
        # The Structures are what store the individual Cartesian coordinates for each frame.
        self.Structures = [Structure(self.M[i], engine, os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i),
                self.coordtype) for i in range(len(self))]
        # The total number of variables being optimized
        # self.nvars = sum([self.Structures[n].nICs for n in range(1, len(self)-1)])
        # Locked images (those having met the convergence criteria) are not updated
        self.locks = [True] + [False for n in range(1, len(self) - 1)] + [True]
        self.haveCalcs = False
        self.GlobalIC = CoordinateSystem(self.M, self.coordtype, chain=True, guessw=self.params.guessw)
        self.nvars = len(self.GlobalIC.Internals)
        # raw_input()

    def __len__(self):
        """Return the length of the chain."""
        return len(self.M)

    def ComputeEnergyGradient(self, cyc=None, result=None):
        """Compute energies and gradients for each structure."""
        # This is the parallel point.
        wq = getWorkQueue()
        if wq:  # If work queue is available, handle jobs with the work queue.
            for i in range(len(self)):
                self.Structures[i].QueueEnergyGradient()
            wq_wait(wq, print_time=600)
            for i in range(len(self)):
                self.Structures[i].GetEnergyGradient()
        elif self.params.bigchem:
            # If BigChem is available, it will be used to carry the single-point calculations.
            from qcio import Molecule as qcio_Molecule, ProgramInput
            from bigchem import compute, group
            elems = self.Structures[0].M.elem
            molecules = []
            
            # Creating a list with qcio Molecule objects and submitting calculations.
            for Structure in self.Structures:
                molecules.append(qcio_Molecule(symbols=elems, geometry=Structure.cartesians.reshape(-1,3)))

            outputs = group(compute.s(self.engine.__class__.__name__.lower(),
                            ProgramInput(molecule=qcio_M, calctype="gradient",
                                         model={"method":self.engine.method, "basis": self.engine.basis},
                                         extras={"order":i}),
                            ) for i, qcio_M in enumerate(molecules)).apply_async()

            # Getting the records
            records = outputs.get()
            
            # Passing the results to chain.ComputeEnergyGradient()
            for i in range(len(self)):
                assert records[i].input_data.extras["order"] == i
                result = {"energy": records[i].results.energy, "gradient": np.array(records[i].results.gradient).ravel()}
                self.Structures[i].ComputeEnergyGradient(result=result)

            # Deleting the records
            outputs.forget()

        else:
            for i in range(len(self)):
                if result:
                    self.Structures[i].ComputeEnergyGradient(result=result[i])
                else:
                    self.Structures[i].ComputeEnergyGradient()
        # When BigChem is used, it does not write output files to disk.
        if cyc is not None and not self.params.bigchem:
            for i in range(len(self)):
                self.Structures[i].engine.number_output(self.Structures[i].tmpdir, cyc)
        self.haveCalcs = True

    def get_cartesian_all(self, endpts=False):
        """Return the internal coordinates of images 1 .. N-2."""
        if endpts:
            return np.hstack(tuple([self.Structures[i].cartesians for i in range(len(self))]))
        else:
            return np.hstack(tuple([self.Structures[i].cartesians for i in range(1, len(self) - 1)]))

    def get_internal_all(self):
        """Return the internal coordinates of images 1 .. N-2."""
        return self.GlobalIC.calculate(np.hstack([self.Structures[i].cartesians for i in range(len(self))]).flatten())

    def getCartesianNorm(self, dy, verbose=False):
        """
        Get the norm of the optimization step in Cartesian coordinates.

        Parameters
        ----------
        dy : np.ndarray
            Array of internal coordinate displacements for each image in the chain
        verbose : bool
            Print diagnostic messages

        Returns
        -------
        float
            The RMSD between the updated and original Cartesian coordinates
        """
        cplus = self.TakeStep(dy, verbose=False)
        return ChainRMSD(self, cplus)

    def clearCalcs(self, clearEngine=True):
        for s in self.Structures:
            s.clearCalcs(clearEngine=clearEngine)
        self.haveCalcs = False
        self.GlobalIC.clearCache()

    def anybork(self):
        return self.GlobalIC.bork

    def CalcInternalStep(self, trust, HW, HP, finish=False):
        """
        Parameters
        ----------
        trust : float
            Trust radius for the internal coordinate step
        HW : np.ndarray
            "Working Hessian" for the optimization.
            This depends on what kind of optimization is being done
            (plain elastic band, nudged elastic band or hybrid).
        HP: np.ndarray
            "Plain Hessian"; second derivatives of the
            plain elastic band energy, used to calculate
            expected increase / decrease of the energy.
        """
        X = self.get_cartesian_all(endpts=True)
        Y = self.get_internal_all()
        G = self.get_global_grad("total", "working")
        H = HW.copy()
        Eig = sorted(np.linalg.eigh(H)[0])
        EigP = sorted(np.linalg.eigh(HP)[0])
        Emin = min(Eig).real
        if Emin < self.params.epsilon:
            v0 = self.params.epsilon - Emin
        else:
            v0 = 0.0

        logger.info("Hessian Eigenvalues (Working) :"+" ".join(["% .4e" % i for i in Eig[:5]]) + \
            "..." + \
            " ".join(["% .4e" % i for i in Eig[-5:]])+ '\n')

        if np.sum(np.array(Eig) < 0.0) > 5:
            logger.info("%i Negative Eigenvalues \n" % (np.sum(np.array(Eig) < 0.0)))

        if Eig[0] != EigP[0]:
            logger.info("Hessian Eigenvalues (Plain)   :"+" ".join(["% .4e" % i for i in EigP[:5]]) + \
                "..." +\
                " ".join(["% .4e" % i for i in EigP[-5:]])+'\n')
            if np.sum(np.array(EigP) < 0.0) > 5:
                logger.info("%i Negative Eigenvalues \n" % (np.sum(np.array(EigP) < 0.0)))
        if finish:
            return

        if Eig[0] < 0.0:
            dy, expect, _ = get_delta_prime_trm(0.0, X, G, np.eye(len(G)), None, False)
            logger.info(
                "\x1b[95mTaking steepest descent rather than Newton-Raphson step\x1b[0m \n"
            )
            ForceRebuild = True
        else:
            dy, expect, _ = get_delta_prime_trm(v0, X, G, H, None, False)
            ForceRebuild = False
        # Internal coordinate step size
        inorm = np.linalg.norm(dy)
        # Cartesian coordinate step size
        # params.verbose = True
        cnorm = self.getCartesianNorm(dy, self.params.verbose)
        # Flag that determines whether internal coordinates need to be rebuilt
        if self.params.verbose:
            logger.info("dy(i): %.4f dy(c) -> target: %.4f -> %.4f \n" % (inorm, cnorm, trust))
        if cnorm > 1.1 * trust:
            # This is the function f(inorm) = cnorm-target that we find a root
            # for obtaining a step with the desired Cartesian step size.
            # We had to copy the original Froot class in optimize.py and make some modifications
            froot = Froot(self, trust, v0, H, self.params)
            froot.stores[inorm] = cnorm
            # Find the internal coordinate norm that matches the desired
            # Cartesian coordinate norm
            iopt = brent_wiki(froot.evaluate, 0.0, inorm, trust, cvg=0.1, obj=froot, verbose=self.params.verbose)
            # Check failure modes
            if froot.brentFailed and froot.stored_arg is not None:
                # 1) Brent optimization failed to converge,
                # but we stored a solution below the trust radius
                if self.params.verbose:
                    logger.info("\x1b[93mUsing stored solution at %.3e\x1b[0m \n" % froot.stored_val)
                iopt = froot.stored_arg
            elif self.anybork():
                # 2) If there is no stored solution,
                # then reduce trust radius by 50% and try again
                # (up to three times)
                for i in range(3):
                    froot.target /= 2
                    if self.params.verbose:
                        logger.info("\x1b[93mReducing target to %.3e\x1b[0m \n" % froot.target)
                    froot.above_flag = True
                    iopt = brent_wiki(froot.evaluate, 0.0, iopt, froot.target, cvg=0.1, verbose=self.params.verbose)
                    if not self.anybork():
                        break
            if self.anybork():
                logger.info("\x1b[91mInverse iteration for Cartesians failed\x1b[0m \n")
                # This variable is added because IC.bork is unset later.
                ForceRebuild = True
            else:
                if self.params.verbose:
                    logger.info(
                        "\x1b[93mBrent algorithm requires %i evaluations\x1b[0m \n"
                        % froot.counter
                    )
            dy, expect = trust_step(iopt, v0, X, G, H, None, False, self.params.verbose)
        # Expected energy change should be calculated from PlainGrad
        GP = self.get_global_grad("total", "plain")
        expect = flat(0.5 * np.linalg.multi_dot([row(dy), HP, col(dy)]))[0] + np.dot(
            dy, GP
        )
        expectG = flat(np.dot(np.array(H), col(dy))) + G
        return dy, expect, expectG, ForceRebuild

    def TakeStep(self, dy, verbose=False):
        """
        Return a new Chain object that contains the internal coordinate step.

        Parameters
        ----------
        dy : np.ndarray
            Array of internal coordinate displacements for each image in the chain
        verbose : bool
            Print diagnostic messages

        Returns
        -------
        NEB
            A new Chain object containing the updated coordinates
        """
        currvar = 0
        # LPW 2017-04-08: Because we're creating a new Chain object that represents the step taken, this copy operation is deemed necessary
        Cnew = deepcopy(self)
        Xnew = self.GlobalIC.newCartesian(
            self.get_cartesian_all(endpts=True), dy, verbose=verbose
        )
        Xnew = Xnew.reshape(-1, 3 * self.na)
        for n in range(1, len(self) - 1):
            if not self.locks[n]:
                Cnew.M.xyzs[n] = Xnew[n].reshape(-1, 3) * bohr2ang
                Cnew.Structures[n].cartesians = Xnew[n]
        Cnew.clearCalcs(clearEngine=False)
        Cnew.haveMetric = True
        return Cnew

    def align(self):
        self.M.align()
        self.Structures = [Structure(self.M[i], self.engine, os.path.join(self.tmpdir, "struct_%%0%ii" %  len(str(len(self))) % i),
                          self.coordtype) for i in range(len(self))]
        self.clearCalcs()

    def respace(self, thresh):
        """
        Space-out NEB images that are closer than the threshold.
        """
        respaced = False
        OldSpac = " ".join(["%6.3f " % i for i in self.calc_spacings()])
        merge_images = []
        for i, spac in enumerate(self.calc_spacings()):
            if spac < thresh:
                merge_images.append(i)
                merge_images.append(i + 1)
                if i > 0:
                    merge_images.append(i - 1)
                if i < len(self) - 2:
                    merge_images.append(i + 2)
        merge_images = sorted(list(set(merge_images)))
        in_segment = False
        merge_segments = []
        for i, im in enumerate(merge_images):
            if not in_segment:
                merge_left = im
                in_segment = True
            if in_segment:
                if im + 1 in merge_images:
                    continue
                merge_right = im
                in_segment = False
                merge_segments.append((merge_left, merge_right))
        for s in merge_segments:
            Mspac = deepcopy(self.Structures[s[0]].M)
            for i in range(s[0] + 1, s[1] + 1):
                Mspac.xyzs += self.Structures[i].M.xyzs
            Mspac_eq = EqualSpacing(Mspac, frames=len(Mspac), RMSD=True, align=False)
            for i in range(len(Mspac_eq)):
                self.Structures[s[0] + i] = Structure(
                    Mspac_eq[i],
                    self.engine,
                    os.path.join(
                        self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % (s[0] + i)
                    ),
                    self.coordtype,
                )
            logger.info("Respaced images %s \n" % (list(range(s[0], s[1] + 1))))
        if len(merge_segments) > 0:
            respaced = True
            self.clearCalcs(clearEngine=False)
            logger.info("Image Number          :"+" ".join(["  %3i  " % i for i in range(len(self))])+'\n')
            logger.info("Spacing (Ang)     Old :"+" " * 5+OldSpac+'\n')
            logger.info("                  New :"+" " * 5+" ".join(["%6.3f " % i for i in self.calc_spacings()])+'\n')
        return respaced

    def delete_insert(self, thresh):
        """
        Second algorithm for deleting images and inserting new ones.
        """
        respaced = False
        OldSpac = " ".join(["%6.3f " % i for i in self.calc_spacings()])

        nloop = 0
        while True:
            spac_nn = np.array(self.calc_spacings())
            spac_nnn = spac_nn[1:] + spac_nn[:-1]
            # This is the frame "i" with the minimum arc-length to frame "i+2"
            left_del = np.argmin(spac_nnn)
            spac_del = spac_nnn[left_del]
            # This is the frame "j" with the maximum arc-length to frame "j+1"
            left_ins = np.argmax(spac_nn)
            spac_ins = spac_nn[left_ins]
            if thresh * spac_del < spac_ins:
                # The space (j) -- (j+1) is greater than (i) -- (i+2) times a threshold
                xyzs = [self.Structures[i].M.xyzs[0].copy() for i in range(len(self))]
                deli = left_del + 1
                insi = left_ins
                insj = left_ins + 1
                xavg = 0.5 * (xyzs[insi] + xyzs[insj])
                xyzs.insert(insj, xavg)
                if insj > deli:
                    del xyzs[deli]
                else:
                    del xyzs[deli + 1]
                for i in range(len(self)):
                    Mtmp = deepcopy(self.Structures[0].M)
                    Mtmp.xyzs = [xyzs[i]]
                    self.Structures[i] = Structure(Mtmp, self.engine, os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i
                        ), self.coordtype)
                logger.info(
                    "Evening out spacings: Deleted image %2i and added a new image between %2i and %2i \n"
                    % (deli, insi, insj)
                )
                respaced = True
            else:
                break
            nloop += 1
            if nloop > len(self):
                logger.info("Spacing out images could not be completed within %i iterations. NEB will be performed with the last iterated chain. \n" %nloop)
                break
        if respaced:
            self.clearCalcs(clearEngine=False)
            logger.info("Image Number          :"+" ".join(["  %3i  " % i for i in range(len(self))])+'\n')
            logger.info("Spacing (Ang)     Old :"+" " * 5+OldSpac+'\n')
            logger.info("                  New :"+" " * 5+" ".join(["%6.3f " % i for i in self.calc_spacings()])+'\n')
        return respaced

    def SaveToDisk(self, fout="chain.xyz"):
        # Mout should be garbage-collected, right?
        Mout = deepcopy(self.Structures[0].M)
        for i in range(1, len(self)):
            Mout.xyzs += self.Structures[i].M.xyzs
        enes = np.array([s.energy for s in self.Structures])
        eneKcal = au2kcal * (enes - np.min(enes))
        # enes -= np.min(enes)
        # enes *= au2kcal
        Mout.comms = [
            "Image %i/%i, Energy = % 16.10f (%+.3f kcal/mol)"
            % (i+1, len(enes), enes[i], eneKcal[i])
            for i in range(len(enes))
        ]
        Mout.write(os.path.join(self.tmpdir, fout))


class ElasticBand(Chain):
    """Using the Chain class to define a band object. Total force varies based on the band types."""
    def __init__(self, molecule, engine, tmpdir, params, coords=None, plain=0):
        super(ElasticBand, self).__init__(molecule, engine, tmpdir, params, coords=coords)
        # convert kcal/mol/Ang^2 to Hartree/Bohr^2
        self.k = self.params.nebk * kcal2au * (bohr2ang**2)
        # Number of atoms
        self.na = molecule.na
        # Use plain elastic band?
        self.plain = plain
        # The tangent vectors are only defined for images 2 .. N-1 (not the endpoints)
        # The weight of each individual displacement in the tangent.
        # Each image has 2 numbers for the weights of drprev and drnext
        self._tangents = [None for i in range(len(self))]
        # Internal storage of gradients along the path
        # Gradients have two energy components (spring, potential)
        # and can either be "plain" (the total gradient) or "projected"
        # (perpendicular component if potential, parallel component if spring)
        # The "working" component depends on which calculation we're running.
        # In plain elastic band (option plain=2), both spring and potential are "plain".
        # In the hybrid method (option plain=1), spring is "plain" and potential is "projected".
        # In nudged elastic band (option plain=0), both spring and potential are "projected".
        self._grads = OrderedDict()
        self._global_grads = OrderedDict()
        self.haveMetric = False
        self.climbSet = False

    def clearCalcs(self, clearEngine=True):
        super(ElasticBand, self).clearCalcs(clearEngine=clearEngine)
        self._grads = OrderedDict()
        self._global_grads = OrderedDict()
        self.haveMetric = False

    def RebuildIC(self, result=None):
        Cnew = ElasticBand(self.M, self.engine, self.tmpdir, self.params, None, plain=self.plain)
        Cnew.ComputeChain(result=result)
        return Cnew

    def set_tangent(self, i, value):
        if i < 1 or i > (len(self) - 2):
            raise NEBBandTangentError(
                "Tangents are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)"
            )
        self._tangents[i] = value

    def get_tangent(self, i):
        if i < 1 or i > (len(self) - 2):
            raise NEBBandTangentError(
                "Tangents are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)"
            )
        return self._tangents[i]

    def get_tangent_all(self):
        return np.hstack(tuple(self._tangents[1 : len(self) - 1]))

    def set_global_grad(self, value, component, projection):
        if value.ndim != 1:
            raise NEBBandGradientError("Please pass a 1D array")
        if value.shape[0] != len(self.GlobalIC.Internals):
            raise NEBBandGradientError(
                "Dimensions of array being passed are wrong (%i ICs expected)"
                % (len(self.GlobalIC.Internals))
            )
        if component not in ["potential", "spring", "total"]:
            raise NEBBandGradientError(
                "Please set the component argument to potential, spring, or total"
            )
        if projection not in ["plain", "projected", "working"]:
            raise NEBBandGradientError(
                "Please set the projection argument to plain, projected, or working"
            )
        self._global_grads[(component, projection)] = value.copy()
        self._global_grads[(component, projection)].flags.writeable = False

    def get_global_grad(self, component, projection):
        if component not in ["potential", "spring", "total"]:
            raise NEBBandGradientError(
                "Please set the component argument to potential, spring, or total"
            )
        if projection not in ["plain", "projected", "working"]:
            raise NEBBandGradientError(
                "Please set the projection argument to plain, projected, or working"
            )
        if projection == "working":
            if component == "potential":
                # Plain = 0: Projected potential force, projected spring force
                # Plain = 1: Projected potential force, full spring force
                # Plain = 2: Full potential force, full spring force
                if self.plain < 2:
                    return self.get_global_grad(component, "projected")
                else:
                    return self.get_global_grad(component, "plain")
            elif component == "spring":
                if self.plain < 1:
                    return self.get_global_grad(component, "projected")
                else:
                    return self.get_global_grad(component, "plain")
            elif component == "total":
                if self.plain == 2:
                    return self.get_global_grad(
                        "potential", "plain"
                    ) + self.get_global_grad("spring", "plain")
                elif self.plain == 1:
                    return self.get_global_grad(
                        "potential", "projected"
                    ) + self.get_global_grad("spring", "plain")
                elif self.plain == 0:
                    return self.get_global_grad(
                        "potential", "projected"
                    ) + self.get_global_grad("spring", "projected")
        elif component == "total":
            return self.get_global_grad("potential", projection) + self.get_global_grad(
                "spring", projection
            )
        if (component, projection) not in self._global_grads or self._global_grads[
            (component, projection)
        ] is None:
            raise NEBBandGradientError("Gradient has not been set")
        # print "Getting gradient for image", i, component, projection
        # LPW 2017-04-08: Removed copy operation, hope flags.writeable = False prevents unwanted edits
        return self._global_grads[(component, projection)]

    def calc_spacings(self):
        rmsds = []
        for i in range(1, len(self)):
            rmsd, maxd = calc_drms_dmax(self.Structures[i].cartesians, self.Structures[i - 1].cartesians, align=False)
            rmsds.append(rmsd)
        return rmsds

    def calc_straightness(self, xyz0, analyze=False):
        xyz = xyz0.reshape(len(self), -1)
        xyz.flags.writeable = False
        straight = [1.0]
        for n in range(1, len(self) - 1):
            drplus = xyz[n + 1] - xyz[n]
            drminus = xyz[n - 1] - xyz[n]
            drplus /= np.linalg.norm(drplus)
            drminus /= np.linalg.norm(drminus)
            straight.append(np.dot(drplus, -drminus))
        straight.append(1.0)
        return straight

    def SaveClimbingImages(self, cycle):
        if not self.climbSet:
            return
        enes = np.array([s.energy for s in self.Structures])
        eneKcal = au2kcal * (enes - np.min(enes))
        M = None
        for i, n in enumerate(self.climbers):
            if M is None:
                M = deepcopy(self.Structures[n].M)
            else:
                M += self.Structures[n].M
            grms = rms_gradient(self.Structures[n].grad_cartesian) * au2evang
            M.comms[i] = (
                "Climbing Image - Chain %i Image %i Energy % 16.10f (%+.3f kcal/mol) RMSGrad %.3f eV/Ang"
                % (cycle, n, enes[n], eneKcal[n], grms)
            )

        if self.params.prefix == None:
            # M.write("chains.tsClimb.xyz")
            M.write(".".join(self.tmpdir.split(".")[:-1]) + ".tsClimb.xyz")
        else:
            M.write(self.params.prefix + ".tsClimb.xyz")

    def PrintStatus(self):
        enes = np.array([s.energy for s in self.Structures])
        enes -= np.min(enes)
        enes *= au2kcal
        symbols = ["(min)"]
        for i in range(1, len(self) - 1):
            if enes[i - 1] == enes[i]:
                if enes[i] == enes[i + 1]:
                    symbols.append("  =  ")  # This may be used when all energies are zero
                elif enes[i] < enes[i + 1]:
                    symbols.append("= -->")  # This symbol should pretty much never be used
                else:
                    symbols.append("= <--")  # This symbol should pretty much never be used
            elif enes[i - 1] < enes[i]:
                if enes[i] == enes[i + 1]:
                    symbols.append("--> =")  # This symbol should pretty much never be used
                elif enes[i] < enes[i + 1]:
                    symbols.append("---->")
                else:
                    if self.climbSet and i in self.climbers:
                        symbols.append("(^_^)")
                    else:
                        symbols.append("(max)")
            else:
                if enes[i] == enes[i + 1]:
                    symbols.append("<-- =")  # This symbol should pretty much never be used
                elif enes[i] > enes[i + 1]:
                    symbols.append("<----")
                else:
                    symbols.append("(min)")
        symbols.append("(min)")
        symcolors = []
        for i in range(len(symbols)):
            if self.locks[i]:
                symcolors.append(("\x1b[94m", "\x1b[0m"))
            else:
                symcolors.append(("", ""))
        logger.info("Image Number          :"+" ".join(["  %3i  " % i for i in range(len(self))])+'\n')
        logger.info("                       "+" ".join(
                [
                    "%s%7s%s" % (symcolors[i][0], s, symcolors[i][1])
                    for i, s in enumerate(symbols)
                ]
            )+'\n')
        logger.info("Energies  (kcal/mol)  :")
        logger.info(" ".join(["%7.3f" % n for n in enes]) + '\n')
        logger.info("Spacing   (Ang)       :")
        logger.info(" " * 4+" ".join(["%6.3f " % i for i in self.calc_spacings()])+'\n')

        xyz = self.get_cartesian_all(endpts=True)
        Bmat = self.GlobalIC.wilsonB(xyz)

        def GetCartesianGradient(component, projection):
            answer = np.dot(
                np.array(Bmat.T),
                np.array(self.get_global_grad(component, projection)).T,
            ).flatten()
            answer = answer.reshape(len(self), -1)
            if component in ["total", "potential"]:
                answer[0] = self.Structures[0].grad_cartesian
                answer[-1] = self.Structures[-1].grad_cartesian
            return answer

        totGrad = GetCartesianGradient("total", "working")
        vGrad = GetCartesianGradient("potential", "working")
        spGrad = GetCartesianGradient("spring", "working")
        straight = self.calc_straightness(xyz)  # , analyze=True)

        # The average gradient in eV/Angstrom
        avgg = (
            np.mean([rms_gradient(totGrad[n]) for n in range(1, len(totGrad) - 1)])
            * au2evang
        )
        maxg = (
            np.max([rms_gradient(totGrad[n]) for n in range(1, len(totGrad) - 1)])
            * au2evang
        )
        logger.info("Gradients (eV/Ang)    :")  # % avgg
        logger.info(
            " ".join(
                [
                    "%7.3f" % (rms_gradient(totGrad[n]) * au2evang)
                    for n in range(len(totGrad))
                ]
            ) + '\n'
        )
        logger.info("Straightness          :")  # % avgg
        logger.info(" ".join(["%7.3f" % (straight[n]) for n in range(len(totGrad))]) + '\n')
        self.avgg = avgg
        self.maxg = maxg
        return
        #printDiffs = False
        #if not printDiffs:
        #    return
        #for n in range(1, len(self) - 1):
        #    ICP = self.GlobalIC.ImageICs[n].Internals
        #    drplus = self.GlobalIC.calcDisplacement(xyz, n + 1, n)
        #    drminus = self.GlobalIC.calcDisplacement(xyz, n - 1, n)
        #    logger.info(
        #        "Largest IC devs (%i - %i); norm % 8.4f \n"
        #        % (n + 1, n, np.linalg.norm(drplus))
        #    )
        #    for i in np.argsort(np.abs(drplus))[::-1][:5]:
        #        logger.info("%30s % 8.4f \n" % (repr(ICP[i]), (drplus)[i]))
        #    logger.info(
        #        "Largest IC devs (%i - %i); norm % 8.4f \n"
        #        % (n - 1, n, np.linalg.norm(drminus))
        #    )
        #    for i in np.argsort(np.abs(drminus))[::-1][:5]:
        #        logger.info("%30s % 8.4f \n" % (repr(ICP[i]), (drminus)[i]))

    def CalcRMSCartGrad(self, igrad):
        xyz = self.get_cartesian_all(endpts=True)
        Bmat = self.GlobalIC.wilsonB(xyz)
        cGrad = np.dot(np.array(Bmat.T), np.array(igrad).T).reshape(len(self), -1)
        # The average gradient in eV/Angstrom
        avgg = (
            np.mean([rms_gradient(cGrad[n]) for n in range(1, len(cGrad) - 1)])
            * au2evang
        )
        return avgg

    def SetFlags(self):
        """Set locks on images that are below the convergence tolerance 'gtol' and are connected to the ends through locked images."""
        xyz = self.get_cartesian_all(endpts=True)
        Bmat = self.GlobalIC.wilsonB(xyz)

        def GetCartesianGradient(component, projection):
            answer = np.dot(
                np.array(Bmat.T),
                np.array(self.get_global_grad(component, projection)).T,
            ).flatten()
            return answer.reshape(len(self), -1)

        totGrad = GetCartesianGradient("total", "working")

        rmsGrad = [rms_gradient(totGrad[n]) * au2evang for n in range(len(totGrad))]
        maxGrad = [max(totGrad[n]) * au2evang for n in range(len(totGrad))]

        avgg = np.mean(rmsGrad[1:-1])
        maxg = np.max(rmsGrad[1:-1])

        recompute = False
        if maxg < self.params.climb or self.climbSet:
            enes = np.array([s.energy for s in self.Structures])
            enes -= np.min(enes)
            enes *= au2kcal
            climbers = []
            cenes = []
            for i in range(1, len(self) - 1):
                if enes[i - 1] < enes[i] and enes[i] > enes[i + 1]:
                    # Image "i" is a local maximum / climbing image
                    climbers.append(i)
                    cenes.append(enes[i])
            # Keep up to "ncimg" climbing images
            climbers = sorted(
                list(
                    np.array(climbers)[np.argsort(np.array(cenes))[::-1]][
                        : self.params.ncimg
                    ]
                )
            )
            if self.climbSet and climbers != self.climbers:
                recompute = True
            if len(climbers) > 0:
                # Note: Climbers may turn on or off over the course of the optimization - look out
                if not self.climbSet or climbers != self.climbers:
                    recompute = True
                    logger.info(
                        "--== Images set to Climbing Mode: %s ==-- \n"
                        % (",".join([str(i) for i in climbers]))
                    )
                self.climbSet = True
                self.climbers = climbers
            # Not sure if the Hessian also needs to be affected for the climbing image

        # Because the gradient depends on the neighboring images, a locked image may get unlocked if a neighboring image moves.
        # This is a bit problematic and we should revisit it in the future.
        # new_scheme = True
        # if new_scheme:
        newLocks = (
            [True] + [False for n in range(1, len(self) - 1)] + [True]
        )  # self.locks[:]
        for n in range(1, len(self)):
            # HP: Locking images considers maxg as well.
            if rmsGrad[n] < self.params.avgg and maxGrad[n] < self.params.maxg:
                newLocks[n] = True
        if False not in newLocks:
            # HP: In case all the images are locked before NEB converges, unlock a few.
            logger.info(
                "All the images got locked, unlocking some images with tighter average gradient value. \n"
            )
            factor = 1.0
            while False not in newLocks:
                factor *= 0.9
                for n in range(1, len(self)):
                    if rmsGrad[n] > self.params.avgg * factor:
                        newLocks[n] = False
        self.locks = newLocks[:]
        return recompute

    def ComputeTangent(self):
        if not self.haveCalcs:
            raise NEBBandTangentError("Calculate energies before tangents")
        enes = [s.energy for s in self.Structures]
        grads = [s.grad_cartesian for s in self.Structures]
        xyz = self.get_cartesian_all(endpts=True)
        for n in range(1, len(self) - 1):
            cc_next = self.Structures[n + 1].cartesians
            cc_curr = self.Structures[n].cartesians
            cc_prev = self.Structures[n - 1].cartesians
            drplus = cc_next - cc_curr
            drminus = cc_curr - cc_prev
            # Energy differences along the band
            dvplus = enes[n + 1] - enes[n]
            dvminus = enes[n] - enes[n - 1]
            if dvplus > 0 and dvminus > 0:
                # Case 1: The energy is increasing as we go along the chain
                # We use the displacement vector pointing "upward" as the
                # tangent.
                dr = drplus
            elif dvplus < 0 and dvminus < 0:
                # Case 2: The energy is decreasing as we go along the chain
                dr = drminus
            else:
                # Case 3: We're at a local maximum or a local minimum.
                # We compute an energy-weighted tangent
                absdvplus = abs(dvplus)
                absdvminus = abs(dvminus)
                absdvmax = max(absdvplus, absdvminus)
                absdvmin = min(absdvplus, absdvminus)
                if enes[n + 1] > enes[n - 1]:
                    # The energy of "next" exceeds "previous".
                    dr = drplus * absdvmax + drminus * absdvmin
                else:
                    # The energy of "previous" exceeds "next".
                    dr = drplus * absdvmin + drminus * absdvmax
            ndr = np.linalg.norm(dr)
            tau = dr / ndr
            self.set_tangent(n, tau)

    def ComputeBandEnergy(self):
        self.SprBandEnergy = 0.0
        self.PotBandEnergy = 0.0
        self.TotBandEnergy = 0.0
        xyz = self.get_cartesian_all(endpts=True)
        energies = np.array([self.Structures[n].energy for n in range(len(self))])

        for n in range(1, len(self) - 1):
            cc_next = self.Structures[n + 1].cartesians
            cc_curr = self.Structures[n].cartesians
            cc_prev = self.Structures[n - 1].cartesians
            ndrplus = np.linalg.norm(cc_next - cc_curr)
            ndrminus = np.linalg.norm(cc_curr - cc_prev)
            # The spring constant connecting each pair of images
            # should have the same strength.  This rather confusing "if"
            # statement ensures the spring constant is not incorrectly
            # doubled for non-end springs
            fplus = 0.5 if n == (len(self) - 2) else 0.25
            fminus = 0.5 if n == 1 else 0.25
            k_new = self.k
            self.SprBandEnergy += fplus * k_new * ndrplus**2
            self.SprBandEnergy += fminus * k_new * ndrminus**2
            self.PotBandEnergy += self.Structures[n].energy
        self.TotBandEnergy = self.SprBandEnergy + self.PotBandEnergy

    def ComputeProjectedGrad(self):
        # HP 5/3/2023: ComputeProjectedGrad_IC was deleted
        self.ComputeProjectedGrad_CC()

    def ComputeProjectedGrad_CC(self):
        xyz = self.get_cartesian_all(endpts=True).reshape(len(self), -1)
        grad_v_c = np.array(
            [self.Structures[n].grad_cartesian for n in range(len(self))]
        )
        grad_v_i = self.GlobalIC.calcGrad(xyz, grad_v_c.flatten())
        grad_v_p_c = np.zeros_like(grad_v_c)
        energies = np.array([self.Structures[n].energy for n in range(len(self))])
        force_s_c = np.zeros_like(grad_v_c)
        force_s_p_c = np.zeros_like(grad_v_c)
        straight = self.calc_straightness(xyz)
        factor_out = "Factors:"
        for n in range(1, len(self) - 1):
            tau = self.get_tangent(n)
            # Force from the potential
            fplus = 1.0 if n == (len(self) - 2) else 0.5
            fminus = 1.0 if n == 1 else 0.5
            force_v = -self.Structures[n].grad_cartesian
            cc_next = self.Structures[n + 1].cartesians
            cc_curr = self.Structures[n].cartesians
            cc_prev = self.Structures[n - 1].cartesians
            ndrplus = np.linalg.norm(cc_next - cc_curr)
            ndrminus = np.linalg.norm(cc_curr - cc_prev)
            # Plain elastic band force
            k_new = self.k
            force_s = k_new * (cc_prev + cc_next - 2 * cc_curr)
            force_s_para = np.dot(force_s, tau) * tau
            force_s_ortho = force_s - force_s_para
            factor = 256 * (1.0 - straight[n]) ** 4
            # Force from the spring in the tangent direction
            force_s_p = k_new * (ndrplus - ndrminus) * tau
            # Now get the perpendicular component of the force from the potential
            force_v_p = force_v - np.dot(force_v, tau) * tau
            if self.climbSet and n in self.climbers:
                # The climbing image feels no spring forces at all,
                # and the force in the direction of the tangent is reversed.
                # Note: We make the choice to change both the plain and the
                # projected spring force.
                force_v_p = force_v - 2 * np.dot(force_v, tau) * tau
                force_s_p *= 0.0
                force_s *= 0.0
            # We should be working with gradients
            grad_s = -1.0 * force_s
            grad_s_p = -1.0 * force_s_p
            grad_v = -1.0 * force_v
            grad_v_p = -1.0 * force_v_p
            grad_v_p_c[n] = grad_v_p
            force_s_p_c[n] = force_s
            force_s_c[n] = force_s

        grad_s_i = self.GlobalIC.calcGrad(xyz, -force_s_c.flatten())
        grad_v_p_i = self.GlobalIC.calcGrad(xyz, grad_v_p_c.flatten())
        grad_s_p_i = self.GlobalIC.calcGrad(xyz, -force_s_p_c.flatten())
        self.set_global_grad(grad_v_i, "potential", "plain")
        self.set_global_grad(grad_v_p_i, "potential", "projected")
        self.set_global_grad(grad_s_i, "spring", "plain")
        self.set_global_grad(grad_s_p_i, "spring", "projected")

    def ComputeGuessHessian(self, blank=False):
        # self.ComputeSpringHessian()
        self.spring_hessian_plain = np.zeros((self.nvars, self.nvars), dtype=float)
        self.spring_hessian_projected = np.zeros(
            (self.nvars, self.nvars), dtype=float
        )
        if not blank:
            guess_hessian_potential = self.GlobalIC.guess_hessian(
                self.get_cartesian_all(endpts=True), k=self.params.guessk
            )
        else:
            # guess_hessian_potential *= 0.0
            guess_hessian_potential = (
                np.eye(self.spring_hessian_plain.shape[0]) * self.params.guessk
            )
        self.guess_hessian_plain = (
            guess_hessian_potential + self.spring_hessian_plain
        )
        self.guess_hessian_projected = (
            guess_hessian_potential + self.spring_hessian_projected
        )
        # Symmetrize
        self.guess_hessian_plain = 0.5 * (
            self.guess_hessian_plain + self.guess_hessian_plain.T
        )
        self.guess_hessian_projected = 0.5 * (
            self.guess_hessian_projected + self.guess_hessian_projected.T
        )
        self.guess_hessian_plain.flags.writeable = False
        self.guess_hessian_projected.flags.writeable = False
        # When plain is set to 1 or 2, we do not project out the perpendicular component of the spring force.
        if self.plain >= 1:
            self.guess_hessian_working = self.guess_hessian_plain
        else:
            self.guess_hessian_working = self.guess_hessian_projected

    def ComputeChain(self, order=1, cyc=None, result=None):
        self.ComputeEnergyGradient(cyc=cyc, result=result)
        if order >= 1:
            self.ComputeTangent()
            self.ComputeProjectedGrad()
        if self.SetFlags():
            self.ComputeProjectedGrad()
        self.ComputeBandEnergy()

    def SaveToDisk(self, fout="chain.xyz"):
        super(ElasticBand, self).SaveToDisk(fout)

    def OptimizeEndpoints(self, gtol=None):
        self.Structures[0].OptimizeGeometry(gtol)
        self.Structures[-1].OptimizeGeometry(gtol)
        logger.info("Endpoints were optimized. \n")
        self.M.xyzs[0] = self.Structures[0].M.xyzs[0]
        self.M.xyzs[-1] = self.Structures[-1].M.xyzs[0]
        # The Structures are what store the individual Cartesian coordinates for each frame.
        self.Structures = [
            Structure(
                self.M[i],
                self.engine,
                os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i),
                self.coordtype,
            )
            for i in range(len(self))
        ]


def ChainRMSD(chain1, chain2):
    rmsds = []
    for n in range(1, len(chain1) - 1):
        rmsd, maxd = calc_drms_dmax(
            chain1.Structures[n].cartesians,
            chain2.Structures[n].cartesians,
            align=False,
        )
        rmsds.append(rmsd)
    # The RMSD of the chain is tentatively defined as the maximum RMSD over all images.
    return np.max(np.array(rmsds))


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

    def __init__(self, chain, trust, v0, H, params):
        self.chain = chain
        self.trust = trust
        self.v0 = v0
        self.H = H
        self.params = params
        # Initialize some other values
        self.counter = 0
        self.stores = {}
        self.target = trust
        self.above_flag = False
        self.stored_arg = None
        self.stored_val = None
        self.brentFailed = False

    def evaluate(self, trial):
        """
        This is a one-argument "function" that is called by brent_wiki which takes
        an internal coordinate step length as input, and returns the Cartesian coordinate
        step length (minus the target) as output.
        """
        chain = self.chain
        v0 = self.v0
        X = chain.get_cartesian_all()
        G = chain.get_global_grad("total", "working")
        H = self.H
        trust = self.trust
        if trial == 0.0:
            self.from_above = False
            return -trust
        else:
            if trial in self.stores:
                cnorm = self.stores[trial]
                self.from_above = False
            else:
                # These two lines are different from Froot in optimize.py
                dy, expect = trust_step(
                    trial, v0, X, G, H, None, False, self.params.verbose
                )
                cnorm = chain.getCartesianNorm(dy, self.params.verbose)
                # Early "convergence"; this signals whether we have found a valid step that is
                # above the current target, but below the original trust radius. This happens
                # when the original trust radius fails, and we reduce the target step-length
                # as a contingency
                bork = chain.GlobalIC.bork
                # bork = any([chain.Structures[n].IC.bork for n in range(1, len(self.chain)-1)])
                self.from_above = self.above_flag and not bork and cnorm < trust
                self.stores[trial] = cnorm
                self.counter += 1
            # Store the largest trial value with cnorm below the target
            if cnorm - self.target < 0:
                if self.stored_val is None or cnorm > self.stored_val:
                    self.stored_arg = trial
                    self.stored_val = cnorm
            if self.params.verbose:
                logger.info(
                    "dy(i): %.4f dy(c) -> target: %.4f -> %.4f%s \n"
                    % (trial, cnorm, self.target, " (done)" if self.from_above else "")
                )
            return cnorm - self.target


def recover(chain_hist, result=None):
    """
    Recover from a failed optimization.

    Parameters
    ----------
    chain_hist : list
        List of previous Chain objects;
        the last element is the current chain
    result : dict
        Dictionary with energies and gradients

    Returns
    -------
    new_chain : Chain
        New chain of states
    Y : np.ndarray
        New internal coordinates
    G : np.ndarray
        New internal gradients
    HW : np.ndarray
        New internal "working" Hessian
    HP : np.ndarray
        New internal "plain" Hessian
    """
    newchain = chain_hist[-1].RebuildIC(result=result)
    newchain.SetFlags()
    newchain.ComputeGuessHessian()
    HW = newchain.guess_hessian_working.copy()
    HP = newchain.guess_hessian_plain.copy()
    Y = newchain.get_internal_all()
    GW = newchain.get_global_grad("total", "working")
    GP = newchain.get_global_grad("total", "plain")
    return newchain, Y, GW, GP, HW, HP


def BFGSUpdate(Y, old_Y, G, old_G, H, params):
    """Update a Hessian using the BFGS method"""
    verbose = params.verbose
    # BFGS Hessian update
    Dy = col(Y - old_Y)
    Dg = col(G - old_G)
    # Catch some abnormal cases of extremely small changes.
    if np.linalg.norm(Dg) < 1e-6 or np.linalg.norm(Dy) < 1e-6:
        return False
    Mat1 = np.dot(Dg, Dg.T) / np.dot(Dg.T, Dy)[0, 0]
    Mat2 = np.dot(np.dot(H, Dy), np.dot(H, Dy).T) / np.dot(np.dot(Dy.T, H), Dy)[0, 0]
    if verbose:
        Eig = np.linalg.eigh(H)[0]
        Eig.sort()
    ndy = np.array(Dy).flatten() / np.linalg.norm(np.array(Dy))
    ndg = np.array(Dg).flatten() / np.linalg.norm(np.array(Dg))
    nhdy = np.dot(H, Dy).flatten() / np.linalg.norm(np.dot(H, Dy))
    if verbose:
        logger.info("Denoms: %.3e %.3e \n" % ((Dg.T * Dy)[0, 0], (Dy.T * H * Dy)[0, 0]))
        logger.info("Dots: %.3e %.3e \n" % (np.dot(ndg, ndy), np.dot(ndy, nhdy)))
    H += Mat1 - Mat2
    Eig1 = np.linalg.eigh(H)[0]
    Eig1.sort()
    if verbose:
        logger.info(
            "Eig-ratios: %.5e ... %.5e \n"
            % (np.min(Eig1) / np.min(Eig), np.max(Eig1) / np.max(Eig))
        )
    return Eig1
    # Then it's on to the next loop iteration!


def updatehessian(old_chain, chain, HP, HW, Y, old_Y, GW, old_GW, GP, old_GP, LastForce, params, result):
    """
    This function updates the Hessians and check their eigenvalues.
    """
    HP_bak = HP.copy()
    HW_bak = HW.copy()
    BFGSUpdate(Y, old_Y, GP, old_GP, HP, params)
    Eig1 = BFGSUpdate(Y, old_Y, GW, old_GW, HW, params)
    if np.min(Eig1) <= params.epsilon:
        if params.skip:
            logger.info(
                "Eigenvalues below %.4e (%.4e) - skipping Hessian update \n"
                % (params.epsilon, np.min(Eig1))
            )
            HP = HP_bak.copy()
            HW = HW_bak.copy()
        else:
            logger.info(
                "Eigenvalues below %.4e (%.4e) - will reset the Hessian \n"
                % (params.epsilon, np.min(Eig1))
            )
            chain, Y, GW, GP, HW, HP = recover([old_chain], LastForce, result)

    del HP_bak
    del HW_bak
    return chain, Y, GW, GP, HP, HW, old_Y, old_GP, old_GW


def qualitycheck(old_chain, new_chain, trust, Quality, ThreLQ, ThreRJ, ThreHQ, Y, GW, GP, old_Y, old_GW, old_GP, params_tmax):
    """
    This function checks quality of the step and rejects (decreases stepsize) a step with poor quality.
    """
    rejectOk = trust > ThreRJ and new_chain.TotBandEnergy - old_chain.TotBandEnergy
    if Quality <= ThreLQ:
        # For bad steps, the trust radius is reduced
        trust = max(
            ThreRJ / 10, min(ChainRMSD(new_chain, old_chain), trust) / 2
        )  # Division
        trustprint = "\x1b[91m-\x1b[0m"
    elif Quality >= ThreHQ:
        if trust < params_tmax:
            # For good steps, the trust radius is increased
            trust = min(np.sqrt(2) * trust, params_tmax)
            trustprint = "\x1b[92m+\x1b[0m"
        else:
            trustprint = "="
    else:
        trustprint = "="
    # LP-Experiment: Trust radius should be smaller than half of chain spacing
    # Otherwise kinks can (and do) appear!
    trust = min(trust, min(new_chain.calc_spacings()))

    if Quality < -1 and rejectOk:
        # Reject the step and take a smaller one from the previous iteration
        Y = old_Y.copy()
        GW = old_GW.copy()
        GP = old_GP.copy()
        # LPW 2017-04-08: Removed deepcopy to save memory.
        # If unexpected behavior appears, check here.
        chain = old_chain
        good = False
        logger.info("Reducing trust radius to %.1e and rejecting step \n" % trust)
    else:
        chain = new_chain
        good = True
    return chain, trust, trustprint, Y, GW, GP, good


def compare(old_chain, new_chain, ThreHQ, ThreLQ, old_GW, HW, HP, respaced, optCycle, expect, expectG, trust, trustprint,
    params_avgg, params_maxg, Quality):
    """
    Two chain objects are compared here to see the quality of the next step.
    """
    Y = new_chain.get_internal_all()
    GW = new_chain.get_global_grad("total", "working")
    GP = new_chain.get_global_grad("total", "plain")

    # Print Optimization Status
    new_chain.PrintStatus()
    try:
        new_chain.SaveClimbingImages(optCycle)
    except:
        pass  # When the NEB ran by QCFractal, it can't (does not need to) save the climbing images in disk.
    if respaced:
        avgg_print, maxg_print = print_forces(new_chain, params_avgg, params_maxg)
        logger.info("Respaced images - skipping trust radius update \n")
        logger.info(
            "\n@%13s %13s %13s %13s %11s %13s %13s \n"
            % ("GAvg(eV/Ang)", "GMax(eV/Ang)", "Length(Ang)", "DeltaE(kcal)", "RMSD(Ang)", "TrustRad(Ang)", "Step Quality")
        )
        logger.info(
            "@%13s %13s %13s \n"
            % (
                "   %s  " %avgg_print,
                "     %s  " %maxg_print,
                "% 8.4f  " % sum(new_chain.calc_spacings()),
            )
        )
        HW = new_chain.guess_hessian_working.copy()
        HP = new_chain.guess_hessian_plain.copy()
        c_hist = [new_chain]
        return (new_chain, Y, GW, GP, HW, HP, c_hist, Quality)

    dE = new_chain.TotBandEnergy - old_chain.TotBandEnergy
    if dE > 0.0 and expect > 0.0 and dE > expect:
        Quality = (2 * expect - dE) / expect
    else:
        Quality = dE / expect
    GC = new_chain.CalcRMSCartGrad(GW)
    eGC = new_chain.CalcRMSCartGrad(expectG)
    GPC = new_chain.CalcRMSCartGrad(old_GW)
    QualityG = 2.0 - GC / max(eGC, GPC / 2, params_avgg / 2)
    Quality = max(Quality, QualityG)
    Describe = (
        "Good"
        if Quality > ThreHQ
        else ("Okay" if Quality > ThreLQ else ("Poor" if Quality > -1.0 else "Reject"))
    )
    avgg_print, maxg_print = print_forces(new_chain, params_avgg, params_maxg)
    logger.info(
        "\n%13s %13s %13s %13s %11s %14s %13s \n"
        % ("GAvg(eV/Ang)", "GMax(eV/Ang)", "Length(Ang)", "DeltaE(kcal)", "RMSD(Ang)", "TrustRad(Ang)", "Step Quality")
    )
    logger.info(
        "@%13s %13s %13s %13s %11s  %8.4f (%s)  %13s \n"
        % (
            "   %s  " % avgg_print,
            "     %s  " % maxg_print,
            "% 8.4f  " % sum(new_chain.calc_spacings()),
            "% 8.4f  " % (au2kcal * dE / len(new_chain)),
            "% 10.4f  " % (ChainRMSD(new_chain, old_chain)),
            trust,
            trustprint,
            "% 6.2f (%s)" % (Quality, Describe),
        )
    )
    return (new_chain, Y, GW, GP, np.array(HW), np.array(HP), [old_chain], Quality)


def converged(chain_maxg, chain_avgg, params_avgg, params_maxg, optCycle, params_maxcyc):
    """
    Checking to see whether the chain is converged.
    """
    if chain_maxg < params_maxg and chain_avgg < params_avgg:
        logger.info("--== Optimization Converged. ==-- \n")
        return True
    if optCycle >= params_maxcyc:
        logger.info("--== Maximum optimization cycles reached. ==-- \n")
        return True
    return False


def takestep(c_hist, chain, optCycle, LastForce, ForceRebuild, trust, Y, GW, GP, HW, HP, result):
    """
    Take a step to move the chain based on the gradients and Hessians.
    """
    LastForce += ForceRebuild
    dy, expect, expectG, ForceRebuild = chain.CalcInternalStep(trust, HW, HP)
    # If ForceRebuild is True, the internal coordinate system
    # is rebuilt and the optimization skips a step
    if ForceRebuild:
        if LastForce == 0:
            pass
        elif LastForce == 1:
            logger.info(
                "\x1b[1;91mFailed twice in a row to rebuild the coordinate system\x1b[0m \n"
            )
            logger.info("\x1b[93mResetting Hessian\x1b[0m \n")
        elif LastForce == 2:
            logger.info(
                "\x1b[1;91mFailed three times to rebuild the coordinate system\x1b[0m \n"
            )
            logger.info("\x1b[93mContinuing in Cartesian coordinates\x1b[0m \n")
        else:
            raise NEBStructureError("Coordinate system has failed too many times")
        chain, Y, GW, GP, HW, HP = recover(c_hist, LastForce == 2, result)
        logger.info("\x1b[1;93mSkipping optimization step\x1b[0m \n")
        optCycle -= 1
    else:
        LastForce = 0
    # Before updating any of our variables, copy current variables to "previous"
    old_Y = Y.copy()
    old_GW = GW.copy()
    old_GP = GP.copy()
    # Cartesian norm of the step
    # Whether the Cartesian norm comes close to the trust radius
    # Update the internal coordinates
    # Obtain a new chain with the step applied
    new_chain = chain.TakeStep(dy)
    respaced = new_chain.delete_insert(1.5)
    return (chain, new_chain, expect, expectG, ForceRebuild, LastForce, old_Y, old_GW, old_GP, respaced, optCycle)


def OptimizeChain(chain, engine, params):
    """
    Main optimization function.
    """
    # Thresholds for low and high quality steps
    ThreLQ = 0.0
    ThreHQ = 0.5
    # Threshold below which chains should not be rejected
    ThreRJ = 0.001
    # Optimize the endpoints of the chain
    if params.optep:
        logger.info("Optimizing endpoint images \n")
        chain.OptimizeEndpoints(params.maxg)

    # Align images
    if params.align:
        logger.info("Aligning images \n")
        chain.align()

    logger.info("Optimizing the input chain \n")
    chain.respace(0.01)
    chain.delete_insert(1.0)
    chain.ComputeChain(cyc=0)
    t0 = time.time()
    # chain.SetFlags(params.gtol, params.climb)
    # Obtain the guess Hessian matrix
    chain.ComputeGuessHessian(blank=isinstance(engine, Blank))
    # Print the status of the zeroth iteration
    logger.info("\n-=# Chain optimization cycle 0 #=- \n")
    logger.info("Spring Force: %.2f kcal/mol/Ang^2 \n" % params.nebk)
    chain.PrintStatus()

    avgg_print, maxg_print = print_forces(chain, params.avgg, params.maxg)
    logger.info("\n-= Chain Properties =- \n")
    logger.info(
        "%13s %13s %13s %13s %11s %13s %13s \n"
        % ("GAvg(eV/Ang)", "GMax(eV/Ang)", "Length(Ang)", "DeltaE(kcal)", "RMSD(Ang)", "TrustRad(Ang)", "Step Quality")
    )
    logger.info(
        "@%13s %13s %13s \n"
        % (
            "   %s  " % avgg_print,
            "     %s  " % maxg_print,
            "% 8.4f  " % sum(chain.calc_spacings()),
        )
    )
    chain.SaveToDisk(fout="chain_%04i.xyz" % 0)
    Y = chain.get_internal_all()
    GW = chain.get_global_grad("total", "working")
    GP = chain.get_global_grad("total", "plain")
    HW = chain.guess_hessian_working.copy()
    HP = chain.guess_hessian_plain.copy()
    # == Initialize some variable
    # Trust radius
    trust = params.trust
    trustprint = "="
    # Whether the coordinate system was forced to rebuild
    ForceRebuild = False
    LastForce = 0
    # How many steps since we have checked the coordinate system
    CoordCounter = 0
    # History of chains, including the current one
    c_hist = [chain]
    # == Enter the main optimization loop
    optCycle = 0
    LastRebuild = 1
    respaced = False
    Quality = None
    while True:

        # ======================================================#
        # | At the top of the loop, our coordinates, energies, |#
        # | gradients, and Hessians are synchronized.          |#
        # ======================================================#

        optCycle += 1
        logger.info("\n-=# Chain optimization cycle %i #=- \n" % (optCycle))

        # =======================================#
        # |    Obtain an optimization step      |#
        # |    and update Cartesian coordinates |#
        # =======================================#

        logger.info("Time since last ComputeChain: %.3f s \n" % (time.time() - t0))

        (chain_prev, chain, expect, expectG, ForceRebuild, LastForce, Y_prev, GW_prev, GP_prev, respaced, optCycle) \
        = takestep(c_hist, chain, optCycle, LastForce, ForceRebuild, trust, Y, GW, GP, HW, HP, None)

        chain.ComputeChain(cyc=optCycle)
        t0 = time.time()
        # ----------------------------------------------------------
        chain, Y, GW, GP, HW, HP, c_hist, Quality = compare(chain_prev, chain, ThreHQ, ThreLQ, GW, HW, HP, respaced,
            optCycle, expect, expectG, trust, trustprint, params.avgg, params.maxg, Quality)

        if respaced:
            chain.SaveToDisk(fout="chain_%04i.xyz" % optCycle)
            continue
        chain.SaveToDisk(fout="chain_%04i.xyz" % optCycle)

        # =======================================#
        # |    Check convergence criteria       |#
        # =======================================#

        if converged(chain.maxg, chain.avgg, params.avgg, params.maxg, optCycle, params.maxcyc):
            break

        # =======================================#
        # |  Adjust Trust Radius / Reject Step  |#
        # =======================================#

        chain, trust, trustprint, Y, GW, GP, good = qualitycheck(chain_prev, chain, trust, Quality, ThreLQ, ThreRJ,
            ThreHQ, Y, GW, GP, Y_prev, GW_prev, GP_prev, params.tmax)
        if not good:
            continue
        c_hist.append(chain)
        c_hist = c_hist[-params.history :]

        # =======================================#
        # |      Update the Hessian Matrix      |#
        # =======================================#

        chain, Y, GW, GP, HP, HW, Y_prev, GP_prev, GW_prev = updatehessian(chain_prev, chain, HP, HW, Y, Y_prev, GW,
                                                                 GW_prev, GP, GP_prev, LastForce, params, None)
    return chain, optCycle




def main():
    args = parse_neb_args(sys.argv[1:])
    args["neb"] = True
    params = NEBParams(**args)

    if args.get('logIni') is None:
        logIni = os.path.join(config_dir, 'log.ini')
    else:
        logIni = args.get('logIni')

    inputf = args.get('input')
    verbose = args.get('verbose', False)
    # Get calculation prefix and temporary directory name
    arg_prefix = args.get('prefix', None) #prefix for output file and temporary directory
    prefix = arg_prefix if arg_prefix is not None else os.path.splitext(inputf)[0]
    logfilename = rf"{prefix}.log"
    # Create a backup if the log file already exists
    import logging.config
    import geometric
    logging.config.fileConfig(logIni,defaults={'logfilename': logfilename},disable_existing_loggers=False)
    logger.info('geometric-neb called with the following command line:\n')
    logger.info(' '.join(sys.argv) + '\n')
    print_logo(logger)
    now = datetime.now()
    logger.info('-=# \x1b[1;94m geomeTRIC started. Version: %s \x1b[0m #=-\n' % (geometric.__version__))
    logger.info('Current date and time: %s\n' % now.strftime("%Y-%m-%d %H:%M:%S"))

    M, engine = get_molecule_engine(**args)

    if params.bigchem:
        logger.info("BigChem will be used to carry singlepoint calculations. \n")

    if args.get('wqport', 0):
        createWorkQueue(args.get('wqport'), debug=params.verbose)

    if params.prefix is None:
        tmpdir = os.path.splitext(args["input"])[0] + ".tmp"
    else:
        tmpdir = params.prefix + ".tmp"

    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    # Make the initial chain
    chain = ElasticBand(M, engine=engine, tmpdir=tmpdir, params=params, plain=params.plain)
    t0 = time.time()
    OptimizeChain(chain, engine, params)
    print_citation(logger)
    logger.info("Time elapsed since start of OptimizeChain: %.3f seconds\n" % (time.time()-t0))

if __name__ == "__main__":
    main()
