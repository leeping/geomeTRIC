from __future__ import print_function
from __future__ import division

import os, sys
import numpy as np
from scipy.linalg import sqrtm
from .prepare import get_molecule_engine
from .optimize import Optimize
from .params import OptParams, parse_optimizer_args
from .step import get_delta_prime_trm, brent_wiki, trust_step, calc_drms_dmax
from .engine import (
    Blank,
)
from .internal import *
from .nifty import (
    flat,
    row,
    col,
    createWorkQueue,
    getWorkQueue,
    wq_wait,
    ang2bohr,
    bohr2ang,
    kcal2au,
    au2kcal,
    au2evang,
)
from .molecule import Molecule, EqualSpacing
from .errors import (
    EngineError,
    CheckCoordError,
    Psi4EngineError,
    QChemEngineError,
    TeraChemEngineError,
    ConicalIntersectionEngineError,
    OpenMMEngineError,
    GromacsEngineError,
    MolproEngineError,
    QCEngineAPIEngineError,
    GaussianEngineError,
    QCAIOptimizationError,
)
from copy import deepcopy


def rms_gradient(gradx):
    """Return the RMS of a Cartesian gradient."""
    atomgrad = np.sqrt(np.sum((gradx.reshape(-1, 3)) ** 2, axis=1))
    return np.sqrt(np.mean(atomgrad**2))


def CoordinateSystem(M, coordtype, chain=False, ic_displace=False, guessw=0.1):
    """
    Parameters
    ----------
    coordtype : string
    Pass in 'cart', 'prim', 'dlc', 'hdlc', or 'tric'

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
        "trim": (PrimitiveInternalCoordinates, False, False),
    }  # Primitive TRIC, i.e. not delocalized
    CoordClass, connect, addcart = CoordSysDict[coordtype]
    if CoordClass is DelocalizedInternalCoordinates:
        IC = CoordClass(
            M,
            build=True,
            connect=connect,
            addcart=addcart,
            cartesian=(coordtype == "cart"),
            chain=chain,
            ic_displace=ic_displace,
            guessw=guessw,
        )
    elif chain:
        IC = ChainCoordinates(
            M,
            connect=connect,
            addcart=addcart,
            cartesian=(coordtype == "cart"),
            ic_displace=ic_displace,
            guessw=guessw,
        )
    else:
        IC = CoordClass(
            M,
            build=True,
            connect=connect,
            addcart=addcart,
            chain=False,
            ic_displace=ic_displace,
        )
    IC.coordtype = coordtype
    return IC


class Structure(object):
    """
    Class representing a single structure.
    """

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
            raise RuntimeError("Please pass in a Molecule object with just one frame")
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
            raise RuntimeError("Please pass in a flat NumPy array in a.u.")
        if len(value.shape) != 1:
            raise RuntimeError("Please pass in a flat NumPy array in a.u.")
        if value.shape[0] != (3 * self.M.na):
            raise RuntimeError("Input array dimensions should be 3x number of atoms")
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

    def ConvertICGradToCart(self, gradq, xyz=None):
        """
        Given a gradient in internal coordinates, convert it back to a Cartesian gradient.
        Unfortunate "reversal" in the interface with respect to IC.calcGrad which takes xyz first!
        """
        if xyz is None:
            xyz = self.cartesians
        Bmat = self.IC.wilsonB(xyz)
        Gx = np.dot(Bmat.T, np.array(gradq).T).flatten()
        return Gx

    def ConvertCartGradToIC(self, gradx, xyz=None):
        """
        Given a gradient in Cartesian coordinates, convert it back to an internal gradient.
        Unfortunate "reversal" in the interface with respect to IC.calcGrad which takes xyz first!
        """
        if xyz is None:
            xyz = self.cartesians
        return self.IC.calcGrad(self.cartesians, gradx)

    def ComputeEnergyGradient(self, result=None):
        """Compute energies and Cartesian gradients for the current structure."""
        # If the result (energy and gradient in a dictionary) is provided, skip calculations
        if result is None:
            result = self.engine.calc(self.cartesians, self.tmpdir)
        self.energy = result["energy"]
        self.grad_cartesian = np.array(result["gradient"])
        self.grad_internal = self.IC.calcGrad(self.cartesians, self.grad_cartesian)

    def QueueEnergyGradient(self):
        self.engine.calc_wq(self.cartesians, self.tmpdir)

    def GetEnergyGradient(self):
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
        force_tric = True
        if force_tric:
            self.IC = CoordinateSystem(self.M, "tric")
        else:
            self.IC = CoordinateSystem(self.M, self.coordtype)

        optProg = Optimize(
            self.cartesians, self.M, self.IC, self.engine, self.tmpdir, opt_params
        )  # , xyzout=os.path.join(self.tmpdir,'optimize.xyz'))

        self.cartesians = np.array(optProg[-1].xyzs).flatten()
        self.M = optProg[-1]
        self.IC = CoordinateSystem(self.M, self.coordtype)
        self.CalcInternals()


class Chain(object):
    """Class representing a chain of states."""

    def __init__(
        self,
        molecule,
        engine,
        tmpdir,
        coordtype,
        params,
        coords=None,
        ic_displace=False,
    ):
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
        coords : np.ndarray, optional
            Array in a.u. containing coordinates (will overwrite what we have in molecule)
            Pass either an array with shape (len(M), 3*M.na), or a flat array with the same number of elements.
        coordtype : string, optional
            Choice of coordinate system (Either cart, prim, dlc, hdlc, or tric) ; defaults to Cartesian
        """
        self.params = params
        # LPW: This copy operation is checked and deemed necessary
        self.M = deepcopy(molecule)
        self.engine = engine
        self.tmpdir = tmpdir
        self.coordtype = coordtype
        self.ic_displace = ic_displace
        # coords is a 2D array with dimensions (N_image, N_atomx3) in atomic units
        if coords is not None:
            # Reshape the array if we passed in a flat array
            if len(coords.shape) == 1:
                coords = coords.reshape(len(self), 3 * self.M.na)
            if coords.shape != (len(self), 3 * self.M.na):
                raise RuntimeError("Coordinates do not have the right shape")
            for i in range(len(self)):
                self.M.xyzs[i] = coords[i, :].reshape(-1, 3) * bohr2ang
        self.na = self.M.na
        # The Structures are what store the individual Cartesian coordinates for each frame.
        self.Structures = [
            Structure(
                self.M[i],
                engine,
                os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i),
                self.coordtype,
            )
            for i in range(len(self))
        ]
        # The total number of variables being optimized
        # self.nvars = sum([self.Structures[n].nICs for n in range(1, len(self)-1)])
        # Locked images (those having met the convergence criteria) are not updated
        self.locks = [True] + [False for n in range(1, len(self) - 1)] + [True]
        self.haveCalcs = False
        ### Test ###
        # print("Starting Test")
        self.GlobalIC = CoordinateSystem(
            self.M,
            self.coordtype,
            chain=True,
            ic_displace=self.ic_displace,
            guessw=self.params.guessw,
        )
        # print xyz.shape
        # self.GlobalIC.checkFiniteDifference(self.get_cartesian_all(endpts=True))
        self.nvars = len(self.GlobalIC.Internals)
        # raw_input()

    def UpdateTempDir(self, iteration):
        self.tmpdir = os.path.join(
            os.path.split(self.tmpdir)[0], "chain_%04i" % iteration
        )
        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir)
        for i, s in enumerate(self.Structures):
            s.tmpdir = os.path.join(
                self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i
            )

    def __len__(self):
        """Return the length of the chain."""
        return len(self.M)

    def ComputeEnergyGradient(self, cyc=None, result=None):
        """Compute energies and gradients for each structure."""
        # This is the parallel point.
        wq = getWorkQueue()
        if wq is None:
            for i in range(len(self)):
                if result:
                    self.Structures[i].ComputeEnergyGradient(result=result[i])
                else:
                    self.Structures[i].ComputeEnergyGradient()
        else:  # If work queue is available, handle jobs with the work queue.
            for i in range(len(self)):
                self.Structures[i].QueueEnergyGradient()
            wq_wait(wq, print_time=600)
            for i in range(len(self)):
                self.Structures[i].GetEnergyGradient()
        if cyc is not None:
            for i in range(len(self)):
                self.Structures[i].engine.number_output(self.Structures[i].tmpdir, cyc)
        self.haveCalcs = True

    def CopyEnergyGradient(self, other):
        for i in range(len(self)):
            self.Structures[i].energy = other.Structures[i].energy
            self.Structures[i].grad_cartesian = other.Structures[
                i
            ].grad_cartesian.copy()
            self.Structures[i].grad_internal = other.Structures[i].grad_internal.copy()
        self.haveCalcs = True

    def get_cartesian_all(self, endpts=False):
        """Return the internal coordinates of images 1 .. N-2."""
        if endpts:
            return np.hstack(
                tuple([self.Structures[i].cartesians for i in range(len(self))])
            )
        else:
            return np.hstack(
                tuple([self.Structures[i].cartesians for i in range(1, len(self) - 1)])
            )

    def get_internal_all(self):
        """Return the internal coordinates of images 1 .. N-2."""
        return self.GlobalIC.calculate(
            np.hstack(
                [self.Structures[i].cartesians for i in range(len(self))]
            ).flatten()
        )

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
        print(
            "Hessian Eigenvalues (Working) :",
            " ".join(["% .4e" % i for i in Eig[:5]]),
            "...",
            " ".join(["% .4e" % i for i in Eig[-5:]]),
        )
        if np.sum(np.array(Eig) < 0.0) > 5:
            print("%i Negative Eigenvalues" % (np.sum(np.array(Eig) < 0.0)))
        if Eig[0] != EigP[0]:
            print(
                "Hessian Eigenvalues (Plain)   :",
                " ".join(["% .4e" % i for i in EigP[:5]]),
                "...",
                " ".join(["% .4e" % i for i in EigP[-5:]]),
            )
            if np.sum(np.array(EigP) < 0.0) > 5:
                print("%i Negative Eigenvalues" % (np.sum(np.array(EigP) < 0.0)))
        if finish:
            return

        if Eig[0] < 0.0:
            dy, expect, _ = get_delta_prime_trm(0.0, X, G, np.eye(len(G)), None, False)
            print(
                "\x1b[95mTaking steepest descent rather than Newton-Raphson step\x1b[0m"
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
            print("dy(i): %.4f dy(c) -> target: %.4f -> %.4f" % (inorm, cnorm, trust))
        if cnorm > 1.1 * trust:
            # This is the function f(inorm) = cnorm-target that we find a root
            # for obtaining a step with the desired Cartesian step size.
            # We had to copy the original Froot class in optimize.py and make some modifications
            froot = Froot(self, trust, v0, H, self.params)
            froot.stores[inorm] = cnorm
            # Find the internal coordinate norm that matches the desired
            # Cartesian coordinate norm
            iopt = brent_wiki(
                froot.evaluate,
                0.0,
                inorm,
                trust,
                cvg=0.1,
                obj=froot,
                verbose=self.params.verbose,
            )
            # Check failure modes
            if froot.brentFailed and froot.stored_arg is not None:
                # 1) Brent optimization failed to converge,
                # but we stored a solution below the trust radius
                if self.params.verbose:
                    print(
                        "\x1b[93mUsing stored solution at %.3e\x1b[0m"
                        % froot.stored_val
                    )
                iopt = froot.stored_arg
            elif self.anybork():
                # 2) If there is no stored solution,
                # then reduce trust radius by 50% and try again
                # (up to three times)
                for i in range(3):
                    froot.target /= 2
                    if self.params.verbose:
                        print("\x1b[93mReducing target to %.3e\x1b[0m" % froot.target)
                    froot.above_flag = True
                    iopt = brent_wiki(
                        froot.evaluate,
                        0.0,
                        iopt,
                        froot.target,
                        cvg=0.1,
                        verbose=self.params.verbose,
                    )
                    if not self.anybork():
                        break
            if self.anybork():
                print("\x1b[91mInverse iteration for Cartesians failed\x1b[0m")
                # This variable is added because IC.bork is unset later.
                ForceRebuild = True
            else:
                if self.params.verbose:
                    print(
                        "\x1b[93mBrent algorithm requires %i evaluations\x1b[0m"
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

    def align(self, qcf=False):
        self.M.align()
        self.Structures = [
            Structure(
                self.M[i],
                self.engine,
                os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i),
                self.coordtype,
            )
            for i in range(len(self))
        ]
        if not qcf:
            self.clearCalcs()

    def TakeStep(self, dy, verbose=False, printStep=False):
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
            A new NEB object containing the updated coordinates
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
        if printStep:
            Xold = self.get_cartesian_all(endpts=True)
            if hasattr(Cnew.GlobalIC, "Prims"):
                plist = Cnew.GlobalIC.Prims.Internals
                prim = Cnew.GlobalIC.Prims
            else:
                plist = Cnew.GlobalIC.Internals
                prim = Cnew.GlobalIC
            icdiff = prim.calcDiff(Xnew, Xold)
            sorter = icdiff**2
            if hasattr(Cnew.GlobalIC, "Prims"):
                dsort = np.argsort(dy**2)[::-1]
                print("Largest components of step (%i DLC total):" % len(dsort))
                print(" ".join(["%6i" % d for d in dsort[:10]]))
                print(" ".join(["%6.3f" % dy[d] for d in dsort[:10]]))
                for d in dsort[:3]:
                    print("Largest components of DLC %i (coeff % .3f):" % (d, dy[d]))
                    for i in np.argsort(
                        np.array(self.GlobalIC.Vecs[:, d]).flatten() ** 2
                    )[::-1][:5]:
                        p = plist[i]
                        print("%40s % .4f" % (p, self.GlobalIC.Vecs[i, d]))
            print("Largest Displacements:")
            for i in np.argsort(sorter)[::-1][:10]:
                p = plist[i]
                print(
                    "%40s % .3e % .3e % .3e"
                    % (p, p.value(Xnew), p.value(Xold), icdiff[i])
                )
        return Cnew

    def respace(self, thresh):
        """
        Space-out NEB images that have merged.
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
            print("Respaced images %s" % (list(range(s[0], s[1] + 1))))
        if len(merge_segments) > 0:
            respaced = True
            self.clearCalcs(clearEngine=False)
            print(
                "Image Number          :",
                " ".join(["  %3i  " % i for i in range(len(self))]),
            )
            print("Spacing (Ang)     Old :", " " * 4, OldSpac)
            print(
                "                  New :",
                " " * 4,
                " ".join(["%6.3f " % i for i in self.calc_spacings()]),
            )
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
                thresh = 1.0
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
                    self.Structures[i] = Structure(
                        Mtmp,
                        self.engine,
                        os.path.join(
                            self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i
                        ),
                        self.coordtype,
                    )
                print(
                    "Evening out spacings: Deleted image %2i and added a new image between %2i and %2i"
                    % (deli, insi, insj)
                )
                respaced = True
            else:
                break
            nloop += 1
            if nloop > len(self):
                raise RuntimeError(
                    "Stuck in a loop, bug likely! Try again with more number of images."
                )
        if respaced:
            self.clearCalcs(clearEngine=False)
            print(
                "Image Number          :",
                " ".join(["  %3i  " % i for i in range(len(self))]),
            )
            print("Spacing (Ang)     Old :", " " * 4, OldSpac)
            print(
                "                  New :",
                " " * 4,
                " ".join(["%6.3f " % i for i in self.calc_spacings()]),
            )
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
            % (i, len(enes), enes[i], eneKcal[i])
            for i in range(len(enes))
        ]
        Mout.write(os.path.join(self.tmpdir, fout))


class ElasticBand(Chain):
    def __init__(
        self,
        molecule,
        engine,
        tmpdir,
        coordtype,
        params,
        coords=None,
        plain=0,
        ic_displace=False,
    ):
        super(ElasticBand, self).__init__(
            molecule,
            engine,
            tmpdir,
            coordtype,
            params,
            coords=coords,
            ic_displace=ic_displace,
        )
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

    def RebuildIC(self, coordtype, result=None):
        Cnew = ElasticBand(
            self.M,
            self.engine,
            self.tmpdir,
            coordtype,
            self.params,
            None,
            plain=self.plain,
            ic_displace=self.ic_displace,
        )
        Cnew.ComputeChain(result=result)
        return Cnew

    def set_tangent(self, i, value):
        if i < 1 or i > (len(self) - 2):
            raise RuntimeError(
                "Tangents are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)"
            )
        self._tangents[i] = value

    def get_tangent(self, i):
        if i < 1 or i > (len(self) - 2):
            raise RuntimeError(
                "Tangents are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)"
            )
        return self._tangents[i]

    def get_tangent_all(self):
        return np.hstack(tuple(self._tangents[1 : len(self) - 1]))

    def set_grad(self, i, value, component, projection):
        """
        Set a component of the gradient for a specific image.
        The value being set is copied.

        Parameters
        ----------
        i : int
            Index of the image for which the gradient is being set
        value : np.ndarray
            1D array containing
        component : str
            Choose "potential", "spring", "total" referring to the
            corresponding energy component
        projection : str
            Choose "plain", "projected", "working" indicating whether
            this is the "plain" (i.e. total) gradient, "projected" according
            to the NEB method, or the "working" gradient being used for optimization
        """
        if value.ndim != 1:
            raise RuntimeError("Please pass a 1D array")
        if i < 1 or i > (len(self) - 2):
            raise RuntimeError(
                "Spring_Grads are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)"
            )
        if value.shape[0] != self.Structures[i].nICs:
            raise RuntimeError(
                "Dimensions of array being passed are wrong (%i ICs expected for image %i)"
                % (self.Structures[i].nICs, i)
            )
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError(
                "Please set the component argument to potential, spring, or total"
            )
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError(
                "Please set the projection argument to plain, projected, or working"
            )
        self._grads.setdefault(
            (component, projection), [None for j in range(len(self))]
        )[i] = value.copy()
        self._grads[(component, projection)][i].flags.writeable = False

    def set_global_grad(self, value, component, projection):
        if value.ndim != 1:
            raise RuntimeError("Please pass a 1D array")
        if value.shape[0] != len(self.GlobalIC.Internals):
            raise RuntimeError(
                "Dimensions of array being passed are wrong (%i ICs expected)"
                % (len(self.GlobalIC.Internals))
            )
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError(
                "Please set the component argument to potential, spring, or total"
            )
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError(
                "Please set the projection argument to plain, projected, or working"
            )
        self._global_grads[(component, projection)] = value.copy()
        self._global_grads[(component, projection)].flags.writeable = False

    def add_grad(self, i, value, component, projection):
        """
        Add to a component of the gradient for a specific image.

        Parameters
        ----------
        i : int
            Index of the image for which the gradient is being set
        value : np.ndarray
            1D array containing gradient to be added
        component : str
            Choose "potential", "spring", "total" referring to the
            corresponding energy component
        projection : str
            Choose "plain", "projected", "working" indicating whether
            this is the "plain" (i.e. total) gradient, "projected" according
            to the NEB method, or the "working" gradient being used for optimization
        """
        if value.ndim != 1:
            raise RuntimeError("Please pass a 1D array")
        if i < 1 or i > (len(self) - 2):
            raise RuntimeError(
                "Spring_Grads are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)"
            )
        if value.shape[0] != self.Structures[i].nICs:
            raise RuntimeError(
                "Dimensions of array being passed are wrong (%i ICs expected for image %i)"
                % (self.Structures[i].nICs, i)
            )
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError(
                "Please set the component argument to potential, spring, or total"
            )
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError(
                "Please set the projection argument to plain, projected, or working"
            )
        if (component, projection) not in self._grads or self._grads[
            (component, projection)
        ][i] is None:
            # print "Setting gradient for image", i, component, projection
            self.set_grad(i, value, component, projection)
        else:
            # print "Adding gradient for image", i, component, projection
            self._grads[(component, projection)][i].flags.writeable = True
            self._grads[(component, projection)][i] += value
            self._grads[(component, projection)][i].flags.writeable = False

    def get_grad(self, i, component, projection):
        """
        Get a component of the gradient.
        The returned value is copied.

        Parameters
        ----------
        i : int
            Index of the image for which the gradient is requested
        component : str
            Choose "potential", "spring", "total" referring to the
            corresponding energy component
        projection : str
            Choose "plain", "projected", "working" indicating whether
            this is the "plain" (i.e. total) gradient, "projected" according
            to the NEB method, or the "working" gradient being used for optimization

        Returns
        -------
        np.ndarray
            1D array containing the requested gradient in units of a.u.
        """
        if i < 1 or i > (len(self) - 2):
            raise RuntimeError(
                "Spring_Grads are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)"
            )
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError(
                "Please set the component argument to potential, spring, or total"
            )
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError(
                "Please set the projection argument to plain, projected, or working"
            )
        if projection == "working":
            if component == "potential":
                # Plain = 0: Projected potential force, projected spring force
                # Plain = 1: Projected potential force, full spring force
                # Plain = 2: Full potential force, full spring force
                if self.plain < 2:
                    return self.get_grad(i, component, "projected")
                else:
                    return self.get_grad(i, component, "plain")
            elif component == "spring":
                if self.plain < 1:
                    return self.get_grad(i, component, "projected")
                else:
                    return self.get_grad(i, component, "plain")
            elif component == "total":
                if self.plain == 2:
                    return self.get_grad(i, "potential", "plain") + self.get_grad(
                        i, "spring", "plain"
                    )
                elif self.plain == 1:
                    return self.get_grad(i, "potential", "projected") + self.get_grad(
                        i, "spring", "plain"
                    )
                elif self.plain == 0:
                    return self.get_grad(i, "potential", "projected") + self.get_grad(
                        i, "spring", "projected"
                    )
        elif component == "total":
            return self.get_grad(i, "potential", projection) + self.get_grad(
                i, "spring", projection
            )
        if (component, projection) not in self._grads or self._grads[
            (component, projection)
        ][i] is None:
            raise RuntimeError("Gradient has not been set")
        # print "Getting gradient for image", i, component, projection
        # LPW 2017-04-08: Removed copy operation, hope flags.writeable = False prevents unwanted edits
        return self._grads[(component, projection)][i]

    def get_global_grad(self, component, projection):
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError(
                "Please set the component argument to potential, spring, or total"
            )
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError(
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
            raise RuntimeError("Gradient has not been set")
        # print "Getting gradient for image", i, component, projection
        # LPW 2017-04-08: Removed copy operation, hope flags.writeable = False prevents unwanted edits
        return self._global_grads[(component, projection)]

    def get_grad_all(self, component, projection):
        """
        Get a component of the gradient, for all of the images.

        Parameters
        ----------
        component : str
            Choose "potential", "spring", "total" referring to the
            corresponding energy component
        projection : str
            Choose "plain", "projected", "working" indicating whether
            this is the "plain" (i.e. total) gradient, "projected" according
            to the NEB method, or the "working" gradient being used for optimization

        Returns
        -------
        np.ndarray
            2D array containing the requested gradient for each image
            (leading index is the image number)
        """
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError(
                "Please set the component argument to potential, spring, or total"
            )
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError(
                "Please set the projection argument to plain, projected, or working"
            )
        return np.hstack(
            tuple(
                [
                    self.get_grad(i, component, projection)
                    for i in range(1, len(self) - 1)
                ]
            )
        )

    def calc_spacings(self):
        rmsds = []
        for i in range(1, len(self)):
            rmsd, maxd = calc_drms_dmax(
                self.Structures[i].cartesians,
                self.Structures[i - 1].cartesians,
                align=False,
            )
            rmsds.append(rmsd)
        return rmsds

    def calc_straightness(self, xyz0, analyze=False):
        xyz = xyz0.reshape(len(self), -1)
        xyz.flags.writeable = False
        straight = [1.0]
        for n in range(1, len(self) - 1):
            if self.ic_displace:
                drplus = self.GlobalIC.calcDisplacement(xyz, n + 1, n)
                drminus = self.GlobalIC.calcDisplacement(xyz, n - 1, n)
                if analyze:
                    ndrplus = drplus / np.linalg.norm(drplus)
                    ndrminus = drminus / np.linalg.norm(drminus)
                    vsum = ndrplus + ndrminus
                    dsum = drplus + drminus
                    dsort = np.argsort(vsum**2)[::-1]
                    if hasattr(self.GlobalIC, "Prims"):
                        plist = self.GlobalIC.Prims.ImageICs[n].Internals
                    else:
                        plist = self.GlobalIC.ImageICs[n].Internals
                    print("Image %i Kink:" % n)
                    for d in dsort[:5]:
                        print("%40s % .4f" % (plist[d], dsum[d]))
            else:
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
                    symbols.append(
                        "  =  "
                    )  # This may be used when all energies are zero
                elif enes[i] < enes[i + 1]:
                    symbols.append(
                        "= -->"
                    )  # This symbol should pretty much never be used
                else:
                    symbols.append(
                        "= <--"
                    )  # This symbol should pretty much never be used
            elif enes[i - 1] < enes[i]:
                if enes[i] == enes[i + 1]:
                    symbols.append(
                        "--> ="
                    )  # This symbol should pretty much never be used
                elif enes[i] < enes[i + 1]:
                    symbols.append("---->")
                else:
                    if self.climbSet and i in self.climbers:
                        symbols.append("(^_^)")
                    else:
                        symbols.append("(max)")
            else:
                if enes[i] == enes[i + 1]:
                    symbols.append(
                        "<-- ="
                    )  # This symbol should pretty much never be used
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
        print(
            "Image Number          :",
            " ".join(["  %3i  " % i for i in range(len(self))]),
        )
        print(
            "                       ",
            " ".join(
                [
                    "%s%7s%s" % (symcolors[i][0], s, symcolors[i][1])
                    for i, s in enumerate(symbols)
                ]
            ),
        )
        print("Energies  (kcal/mol)  :", end=" ")
        print(" ".join(["%7.3f" % n for n in enes]))
        print("Spacing (Ang)         :", end=" ")
        print(" " * 4, " ".join(["%6.3f " % i for i in self.calc_spacings()]))
        # print "Angles                :",
        # print " "*13, ' '.join(["%5.2f  " % (180.0/np.pi*np.arccos(np.dot(taus[n], taus[n+1]))) for n in range(len(taus)-1)])

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
        print("Gradients (eV/Ang)    :", end=" ")  # % avgg
        print(
            " ".join(
                [
                    "%7.3f" % (rms_gradient(totGrad[n]) * au2evang)
                    for n in range(len(totGrad))
                ]
            )
        )
        # print("Potential Gradient    :", end=' ')
        # print(' '.join(["%7.3f" % (rms_gradient(vGrad[n]) * au2evang) for n in range(len(vGrad))]))
        # print("Spring Gradient       :", end=' ')
        # print(' '.join(["%7.3f" % (rms_gradient(spGrad[n]) * au2evang) for n in range(len(spGrad))]))
        print("Straightness          :", end=" ")  # % avgg
        print(" ".join(["%7.3f" % (straight[n]) for n in range(len(totGrad))]))
        self.avgg = avgg
        self.maxg = maxg
        printDiffs = False
        if not printDiffs:
            return
        for n in range(1, len(self) - 1):
            ICP = self.GlobalIC.ImageICs[n].Internals
            drplus = self.GlobalIC.calcDisplacement(xyz, n + 1, n)
            drminus = self.GlobalIC.calcDisplacement(xyz, n - 1, n)
            print(
                "Largest IC devs (%i - %i); norm % 8.4f"
                % (n + 1, n, np.linalg.norm(drplus))
            )
            for i in np.argsort(np.abs(drplus))[::-1][:5]:
                print("%30s % 8.4f" % (repr(ICP[i]), (drplus)[i]))
            print(
                "Largest IC devs (%i - %i); norm % 8.4f"
                % (n - 1, n, np.linalg.norm(drminus))
            )
            for i in np.argsort(np.abs(drminus))[::-1][:5]:
                print("%30s % 8.4f" % (repr(ICP[i]), (drminus)[i]))

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
                    print(
                        "--== Images set to Climbing Mode: %s ==--"
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
            # HP: In case all of the images are locked before NEB converges, unlock a few.
            print(
                "All the images got locked, unlocking some images with tighter average gradient value."
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
            raise RuntimeError("Calculate energies before tangents")
        enes = [s.energy for s in self.Structures]
        if self.ic_displace:
            grads = [s.grad_internal for s in self.Structures]
        else:
            grads = [s.grad_cartesian for s in self.Structures]
        xyz = self.get_cartesian_all(endpts=True)
        for n in range(1, len(self) - 1):
            if self.ic_displace:
                drplus = self.GlobalIC.calcDisplacement(xyz, n + 1, n)
                drminus = -self.GlobalIC.calcDisplacement(xyz, n - 1, n)
            else:
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
            if self.ic_displace:
                drplus = self.GlobalIC.calcDisplacement(xyz, n + 1, n)
                drminus = self.GlobalIC.calcDisplacement(xyz, n - 1, n)
            else:
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
            if self.params.nebew:
                # HP: Energy weighted NEB
                E_i = energies[n]
                E_ref = min(
                    energies[0], energies[-1]
                )  # Reference energy can be either reactant or product. Lower energy is picked here.
                E_max = max(energies)
                k_max = self.k
                k_min = self.k / 2
                a = (E_max - energies[n]) / (E_max - E_ref)
                if E_i > E_ref:
                    k_new = (1 - a) * k_max + a * k_min
                else:
                    k_new = k_min
            else:
                k_new = self.k
            if self.ic_displace:
                self.SprBandEnergy += fplus * k_new * np.dot(drplus, drplus)
                self.SprBandEnergy += fminus * k_new * np.dot(drminus, drminus)
            else:
                self.SprBandEnergy += fplus * k_new * ndrplus**2
                self.SprBandEnergy += fminus * k_new * ndrminus**2
            self.PotBandEnergy += self.Structures[n].energy
        self.TotBandEnergy = self.SprBandEnergy + self.PotBandEnergy

    def ComputeProjectedGrad(self):
        if self.ic_displace:
            self.ComputeProjectedGrad_IC()
        else:
            self.ComputeProjectedGrad_CC()

    def ComputeProjectedGrad_IC(self):
        xyz = self.get_cartesian_all(endpts=True).reshape(len(self), -1)
        grad_v_c = np.array(
            [self.Structures[n].grad_cartesian for n in range(len(self))]
        )
        energies = np.array([self.Structures[n].energy for n in range(len(self))])
        print("energies in Projected Grad", energies)
        grad_v_i = self.GlobalIC.calcGrad(xyz, grad_v_c.flatten())
        grad_v_p_c = np.zeros_like(grad_v_c)
        force_s_c = np.zeros_like(grad_v_c)
        force_s_p_c = np.zeros_like(grad_v_c)
        straight = self.calc_straightness(xyz)

        for n in range(1, len(self) - 1):
            fplus = 1.0 if n == (len(self) - 2) else 0.5
            fminus = 1.0 if n == 1 else 0.5
            drplus = self.GlobalIC.calcDisplacement(xyz, n + 1, n)
            drminus = self.GlobalIC.calcDisplacement(xyz, n - 1, n)
            if self.params.nebew:
                # HP: Energy weighted NEB
                E_i = energies[n]
                E_ref = min(
                    energies[0], energies[-1]
                )  # Reference energy can be either reactant or product. Lower energy is picked here.
                E_max = max(energies)
                k_max = self.k
                k_min = self.k / 2
                a = (E_max - energies[n]) / (E_max - E_ref)
                if E_i > E_ref:
                    k_new = (1 - a) * k_max + a * k_min
                else:
                    k_new = k_min
            else:
                k_new = self.k
            force_s_Plus = fplus * k_new * drplus
            force_s_Minus = fminus * k_new * drminus
            factor = 1.0 + 16 * (1.0 - straight[n]) ** 2
            force_s_c[n] += self.GlobalIC.applyCartesianGrad(
                xyz, factor * (force_s_Plus + force_s_Minus), n, n
            )
            tau = self.get_tangent(n)
            # Force from the spring in the tangent direction
            ndrplus = np.linalg.norm(drplus)
            ndrminus = np.linalg.norm(drminus)
            force_s_p_c[n] = self.GlobalIC.applyCartesianGrad(
                xyz, k_new * (ndrplus - ndrminus) * tau, n, n
            )
            # Now get the perpendicular component of the force from the potential
            grad_v_im = self.GlobalIC.ImageICs[n].calcGrad(xyz[n], grad_v_c[n])
            grad_v_p_c[n] = self.GlobalIC.applyCartesianGrad(
                xyz, grad_v_im - np.dot(grad_v_im, tau) * tau, n, n
            )

            if self.climbSet and n in self.climbers:
                # The force in the direction of the tangent is reversed
                grad_v_p_c[n] = self.GlobalIC.applyCartesianGrad(
                    xyz, grad_v_im - 2 * np.dot(grad_v_im, tau) * tau, n, n
                )

            if n > 1:
                force_s_c[n - 1] -= self.GlobalIC.applyCartesianGrad(
                    xyz, force_s_Minus, n - 1, n
                )
            if n < len(self) - 2:
                force_s_c[n + 1] -= self.GlobalIC.applyCartesianGrad(
                    xyz, force_s_Plus, n + 1, n
                )

        for n in range(1, len(self) - 1):
            if self.climbSet and n in self.climbers:
                # The climbing image feels no spring forces at all,
                # Note: We make the choice to change both the plain and the
                # projected spring force.
                force_s_c[n] *= 0.0
                force_s_p_c[n] *= 0.0

        xyz = self.get_cartesian_all(endpts=True).reshape(len(self), -1)
        grad_v_c = np.array(
            [self.Structures[n].grad_cartesian for n in range(len(self))]
        )
        grad_v_i = self.GlobalIC.calcGrad(xyz, grad_v_c.flatten())
        grad_s_i = self.GlobalIC.calcGrad(xyz, -force_s_c.flatten())
        grad_v_p_i = self.GlobalIC.calcGrad(xyz, grad_v_p_c.flatten())
        grad_s_p_i = self.GlobalIC.calcGrad(xyz, -force_s_p_c.flatten())

        self.set_global_grad(grad_v_i, "potential", "plain")
        self.set_global_grad(grad_v_p_i, "potential", "projected")
        self.set_global_grad(grad_s_i, "spring", "plain")
        self.set_global_grad(grad_s_p_i, "spring", "projected")

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

            if self.params.nebew is not None:
                # HP: Energy weighted NEB
                E_i = energies[n]
                E_ref = min(
                    energies[0], energies[-1]
                )  # Reference energy can be either reactant or product. Lower energy is picked here.
                E_max = max(energies)
                k_max = self.k
                k_min = self.k / self.params.nebew
                a = (E_max - energies[n]) / (E_max - E_ref)
                if E_i > E_ref:
                    k_new = (1 - a) * k_max + a * k_min
                else:
                    k_new = k_min
            else:
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

    def FiniteDifferenceTest(self):
        self.ComputeChain()
        E = self.TotBandEnergy
        G = self.get_global_grad("total", "plain")
        h = 1e-5
        for i in range(self.nvars):
            dy = np.zeros(self.nvars, dtype=float)
            dy[i] += h
            cplus = self.TakeStep(dy, verbose=False)
            cplus.ComputeChain(order=0)
            eplus = cplus.TotBandEnergy
            dy[i] -= 2 * h
            cminus = self.TakeStep(dy, verbose=False)
            cminus.ComputeChain(order=0)
            eminus = cminus.TotBandEnergy
            fdiff = (eplus - eminus) / (2 * h)
            print(
                "\r%30s%5i : % .6e % .6e % .6e                   "
                % (repr(self.GlobalIC.Internals[i]), i, G[i], fdiff, G[i] - fdiff)
            )

    def ComputeSpringHessian(self):
        # Compute both the plain and projected spring Hessian.
        self.spring_hessian_plain = np.zeros((self.nvars, self.nvars))
        self.spring_hessian_projected = np.zeros((self.nvars, self.nvars))
        h = 1e-6
        t0 = time.time()
        for i in range(self.nvars):
            dy = np.zeros(self.nvars, dtype=float)
            dy[i] += h
            cplus = self.TakeStep(dy, verbose=False)
            # An error will be thrown if grad_cartesian does not exist
            cplus.CopyEnergyGradient(self)
            cplus.ComputeTangent()
            cplus.ComputeProjectedGrad()
            gplus_plain = cplus.get_global_grad("spring", "plain")
            gplus_proj = cplus.get_global_grad("spring", "projected")
            dy[i] -= 2 * h
            cminus = self.TakeStep(dy, verbose=False)
            cminus.CopyEnergyGradient(self)
            cminus.ComputeTangent()
            cminus.ComputeProjectedGrad()
            gminus_plain = cminus.get_global_grad("spring", "plain")
            gminus_proj = cminus.get_global_grad("spring", "projected")
            self.spring_hessian_plain[i, :] = (gplus_plain - gminus_plain) / (2 * h)
            self.spring_hessian_projected[i, :] = (gplus_proj - gminus_proj) / (2 * h)
        print("Spring Hessian took %.3f seconds" % (time.time() - t0))

    def ComputeGuessHessian(self, full=False, blank=False):
        if full:
            # Compute both the plain and projected spring Hessian.
            self.guess_hessian_plain = np.zeros((self.nvars, self.nvars))
            self.guess_hessian_working = np.zeros((self.nvars, self.nvars))
            h = 1e-5
            t0 = time.time()
            for i in range(self.nvars):
                print("\rCoordinate %i/%i" % (i, self.nvars) + " " * 100)
                dy = np.zeros(self.nvars, dtype=float)
                dy[i] += h
                cplus = self.TakeStep(dy, verbose=False)
                # An error will be thrown if grad_cartesian does not exist
                cplus.ComputeChain()
                gplus_plain = cplus.get_global_grad("total", "plain")
                gplus_work = cplus.get_global_grad("total", "working")
                dy[i] -= 2 * h
                cminus = self.TakeStep(dy, verbose=False)
                cminus.ComputeChain()
                gminus_plain = cminus.get_global_grad("total", "plain")
                gminus_work = cminus.get_global_grad("total", "working")
                self.guess_hessian_plain[i, :] = (gplus_plain - gminus_plain) / (2 * h)
                self.guess_hessian_working[i, :] = (gplus_work - gminus_work) / (2 * h)
            self.guess_hessian_plain = 0.5 * (
                self.guess_hessian_plain + self.guess_hessian_plain.T
            )
            self.guess_hessian_working = 0.5 * (
                self.guess_hessian_working + self.guess_hessian_working.T
            )
            print("Full Hessian took %.3f seconds" % (time.time() - t0))
        else:
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

    def ComputeGuessHessian_full(self):
        # Compute both the plain and projected spring Hessian.
        self.guess_hessian_plain = np.zeros((self.nvars, self.nvars))
        self.guess_hessian_projected = np.zeros((self.nvars, self.nvars))
        self.guess_hessian_working = np.zeros((self.nvars, self.nvars))
        h = 1e-5
        t0 = time.time()
        for i in range(self.nvars):
            dy = np.zeros(self.nvars, dtype=float)
            dy[i] += h
            cplus = self.TakeStep(dy, verbose=False)
            # An error will be thrown if grad_cartesian does not exist
            cplus.ComputeChain()
            cplus.ComputeProjectedGrad()
            spgrad_plus_plain = cplus.get_global_grad("spring", "plain")
            spgrad_plus_proj = cplus.get_global_grad("spring", "projected")
            vgrad_plus_plain = cplus.get_global_grad("potential", "plain")
            vgrad_plus_proj = cplus.get_global_grad("potential", "projected")
            dy[i] -= 2 * h
            cminus = self.TakeStep(dy, verbose=False)
            cminus.ComputeChain()
            cminus.ComputeProjectedGrad()
            spgrad_minus_plain = cminus.get_global_grad("spring", "plain")
            spgrad_minus_proj = cminus.get_global_grad("spring", "projected")
            vgrad_minus_plain = cminus.get_global_grad("potential", "plain")
            vgrad_minus_proj = cminus.get_global_grad("potential", "projected")

            gplus_plain = spgrad_plus_plain + vgrad_plus_plain
            gplus_proj = spgrad_plus_proj + vgrad_plus_proj
            gminus_plain = spgrad_minus_plain + vgrad_minus_plain
            gminus_proj = spgrad_minus_proj + vgrad_minus_proj

            self.guess_hessian_plain[i, :] = (gplus_plain - gminus_plain) / (2 * h)
            self.guess_hessian_projected[i, :] = (gplus_proj - gminus_proj) / (2 * h)
            if self.plain == 0:
                self.guess_hessian_working[i, :] = (gplus_proj - gminus_proj) / (2 * h)
            elif self.plain == 1:
                self.guess_hessian_working[i, :] = (
                    vgrad_plus_proj
                    + spgrad_plus_plain
                    - vgrad_minus_proj
                    - spgrad_minus_plain
                ) / (2 * h)
            elif self.plain == 2:
                self.guess_hessian_working[i, :] = (gplus_plain - gminus_plain) / (
                    2 * h
                )

        # Symmetrize
        self.guess_hessian_plain = 0.5 * (
            self.guess_hessian_plain + self.guess_hessian_plain.T
        )
        self.guess_hessian_projected = 0.5 * (
            self.guess_hessian_projected + self.guess_hessian_projected.T
        )
        self.guess_hessian_working = 0.5 * (
            self.guess_hessian_working + self.guess_hessian_working.T
        )

        print("Guess Hessian took %.3f seconds" % (time.time() - t0))

    def ComputeMetric(self):
        if self.haveMetric:
            return

        t0 = time.time()
        self.kmats = [None for i in range(len(self))]
        self.metrics = [None for i in range(len(self))]
        errs = []
        for n in range(1, len(self) - 1):
            self.kmats[n] = np.eye(len(self.GlobalIC.ImageICs[n].Internals))
            self.metrics[n] = np.array(sqrtm(self.kmats[n]))
            errs.append(
                np.linalg.norm(
                    np.abs(np.dot(self.metrics[n], self.metrics[n]) - self.kmats[n])
                )
            )
        print(
            "Metric completed in %.3f seconds, maxerr = %.3e"
            % (time.time() - t0, max(errs))
        )
        self.haveMetric = True

    def ComputeChain(self, order=1, cyc=None, result=None):
        if order >= 1:
            self.ComputeMetric()
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
        print("Optimizing End Points are done.")
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
                print(
                    "dy(i): %.4f dy(c) -> target: %.4f -> %.4f%s"
                    % (trial, cnorm, self.target, " (done)" if self.from_above else "")
                )
            return cnorm - self.target


def RebuildHessian(H, chain, chain_hist, params, projection):
    """
    Rebuild the Hessian after making a change to the internal coordinate system.

    Parameters
    ----------
    H : np.ndarray
        Initial guess Hessian matrix. LPW 2017-04-08: THIS VARIABLE IS MODIFIED
    chain : Chain
        The current chain of states
    chain_hist : list
        List of previous Chain objects
    params : OptParams object
        Uses trust, epsilon, and reset
        trust : Only recover using previous geometries within the trust radius
        epsilon : Small eigenvalue threshold
        reset : Revert to the guess Hessian if eigenvalues smaller than threshold
    projection : str
        Either "plain", "projected" or "working".  Choose which gradient projection to use
        for rebuilding the Hessian.

    Returns
    -------
    np.ndarray
        Internal coordinate Hessian updated with series of internal coordinate gradients
    """
    history = 0
    for i in range(2, len(chain_hist) + 1):
        rmsd = ChainRMSD(chain_hist[-i], chain_hist[-1])
        if rmsd > params.trust:
            break
        history += 1
    if history < 1:
        return
    print("Rebuilding Hessian using %i gradients" % history)

    y_seq = [c.get_internal_all() for c in chain_hist[-history - 1 :]]
    g_seq = [c.get_global_grad("total", projection) for c in chain_hist[-history - 1 :]]

    Yprev = y_seq[0]
    Gprev = g_seq[0]
    H0 = H.copy()
    for i in range(1, len(y_seq)):
        Y = y_seq[i]
        G = g_seq[i]
        Yprev = y_seq[i - 1]
        Gprev = g_seq[i - 1]
        Dy = col(Y - Yprev)
        Dg = col(G - Gprev)
        Mat1 = np.dot(Dg, Dg.T) / np.dot(Dg.T, Dy)[0, 0]
        Mat2 = (
            np.dot(np.dot(np.array(H), Dy), np.dot(np.array(H), Dy).T)
            / np.dot((np.dot(Dy.T, np.array(H)), Dy))[0, 0]
        )
        print("Mats in RebuildHessian", Mat1, Mat2)
        # Hstor = H.copy()
        H += Mat1 - Mat2
    if np.min(np.linalg.eigh(H)[0]) < params.epsilon and params.reset:
        print(
            "Eigenvalues below %.4e (%.4e) - returning guess"
            % (params.epsilon, np.min(np.linalg.eigh(H)[0]))
        )
        H = H0.copy()


def recover(chain_hist, params, forceCart, result=None):
    """
    Recover from a failed optimization.

    Parameters
    ----------
    chain_hist : list
        List of previous Chain objects;
        the last element is the current chain
    params : ChainOptParams
        Job parameters
    forceCart : bool
        Whether to use Cartesian coordinates or
        adopt the IC system of the current chain
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
    newchain = chain_hist[-1].RebuildIC(
        "cart" if forceCart else chain_hist[-1].coordtype, result=result
    )
    for ic, c in enumerate(chain_hist):
        # Copy operations here may allow old chains to be properly erased when dereferenced
        c.kmats = deepcopy(newchain.kmats)
        c.metrics = deepcopy(newchain.metrics)
        c.GlobalIC = deepcopy(newchain.GlobalIC)
        for i in range(len(newchain)):
            c.Structures[i].IC = deepcopy(newchain.Structures[i].IC)
        c.ComputeChain(result=result)
    newchain.SetFlags()
    newchain.ComputeGuessHessian()
    HW = newchain.guess_hessian_working.copy()
    HP = newchain.guess_hessian_plain.copy()
    if not params.reset:
        RebuildHessian(HW, newchain, chain_hist, params, "working")
        RebuildHessian(HP, newchain, chain_hist, params, "plain")
    Y = newchain.get_internal_all()
    GW = newchain.get_global_grad("total", "working")
    GP = newchain.get_global_grad("total", "plain")
    return newchain, Y, GW, GP, HW, HP


def BFGSUpdate(Y, Yprev, G, Gprev, H, params):
    verbose = params.verbose
    # BFGS Hessian update
    Dy = col(Y - Yprev)
    Dg = col(G - Gprev)
    # Catch some abnormal cases of extremely small changes.
    if np.linalg.norm(Dg) < 1e-6 or np.linalg.norm(Dy) < 1e-6:
        return False
    Mat1 = np.dot(Dg, Dg.T) / np.dot(Dg.T, Dy)[0, 0]
    Mat2 = np.dot(np.dot(H, Dy), np.dot(H, Dy).T) / np.dot(np.dot(Dy.T, H), Dy)[0, 0]
    Eig = np.linalg.eigh(H)[0]
    Eig.sort()
    ndy = np.array(Dy).flatten() / np.linalg.norm(np.array(Dy))
    ndg = np.array(Dg).flatten() / np.linalg.norm(np.array(Dg))
    nhdy = np.array(H * Dy).flatten() / np.linalg.norm(np.array(H * Dy))
    if verbose:
        print("Denoms: %.3e %.3e" % ((Dg.T * Dy)[0, 0], (Dy.T * H * Dy)[0, 0]), end=" ")
        print("Dots: %.3e %.3e" % (np.dot(ndg, ndy), np.dot(ndy, nhdy)), end=" ")
    H += Mat1 - Mat2
    Eig1 = np.linalg.eigh(H)[0]
    Eig1.sort()
    if verbose:
        print(
            "Eig-ratios: %.5e ... %.5e"
            % (np.min(Eig1) / np.min(Eig), np.max(Eig1) / np.max(Eig))
        )
    return Eig1
    # Then it's on to the next loop iteration!


def updatehessian(
    chain,
    old_chain,
    HP,
    HW,
    Y,
    Y_prev,
    GW,
    GW_prev,
    GP,
    GP_prev,
    LastForce,
    params,
    result,
):
    """
    This function was part of the 'while' loop in OptimizeChain(). It updates hessian.
    """
    H_reset = False
    HP_bak = HP.copy()
    HW_bak = HW.copy()
    BFGSUpdate(Y, Y_prev, GP, GP_prev, HP, params)
    Eig1 = BFGSUpdate(Y, Y_prev, GW, GW_prev, HW, params)
    if np.min(Eig1) <= params.epsilon:
        if params.reset:
            H_reset = True
            print(
                "Eigenvalues below %.4e (%.4e) - will reset the Hessian"
                % (params.epsilon, np.min(Eig1))
            )
            chain, Y, GW, GP, HW, HP = recover([old_chain], params, LastForce, result)
            Y_prev = Y.copy()
            GP_prev = GP.copy()
            GW_prev = GW.copy()
        elif params.skip:
            print(
                "Eigenvalues below %.4e (%.4e) - skipping Hessian update"
                % (params.epsilon, np.min(Eig1))
            )
            Y_prev = Y.copy()
            GP_prev = GP.copy()
            GW_prev = GW.copy()
            HP = HP_bak.copy()
            HW = HW_bak.copy()
    del HP_bak
    del HW_bak
    return chain, Y, GW, GP, HP, HW, Y_prev, GP_prev, GW_prev, H_reset


def qualitycheck(
    trust,
    new_chain,
    old_chain,
    Quality,
    ThreLQ,
    ThreRJ,
    ThreHQ,
    Y,
    GW,
    GP,
    Y_prev,
    GW_prev,
    GP_prev,
    params_tmax,
):
    """
    This function was part of the 'while' loop in OptimizeChain(). This function checks quality of the step.
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
            trust = min(2 * trust, params_tmax)
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
        Y = Y_prev.copy()
        GW = GW_prev.copy()
        GP = GP_prev.copy()
        # LPW 2017-04-08: Removed deepcopy to save memory.
        # If unexpected behavior appears, check here.
        chain = old_chain
        good = False
        print("Reducing trust radius to %.1e and rejecting step" % trust)
    else:
        chain = new_chain
        good = True

    return chain, trust, trustprint, Y, GW, GP, good


def compare(
    old_chain,
    new_chain,
    ThreHQ,
    ThreLQ,
    GW_prev,
    HW,
    HP,
    respaced,
    optCycle,
    expect,
    expectG,
    trust,
    trustprint,
    params_avgg,
    Quality_old=None,
):
    """
    This function was part of the 'while' loop in OptimizeChain(). Two chain objects are being compared.
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
        print("Respaced images - skipping trust radius update")
        print(
            "@%13s %13s %13s %13s %11s %13s %13s"
            % (
                "GAvg(eV/Ang)",
                "GMax(eV/Ang)",
                "Length(Ang)",
                "DeltaE(kcal)",
                "RMSD(Ang)",
                "TrustRad(Ang)",
                "Step Quality",
            )
        )
        print(
            "@%13s %13s %13s"
            % (
                "% 8.4f  " % new_chain.avgg,
                "% 8.4f  " % new_chain.maxg,
                "% 8.4f  " % sum(new_chain.calc_spacings()),
            )
        )
        HW = (
            new_chain.guess_hessian_working.copy()
        )  #Comment this out to keep the hessian
        HP = new_chain.guess_hessian_plain.copy()
        c_hist = [new_chain]
        return (
            new_chain,
            Y,
            GW,
            GP,
            np.array(HW),
            np.array(HP),
            c_hist,
            respaced,
            Quality_old,
        )

    dE = new_chain.TotBandEnergy - old_chain.TotBandEnergy
    if dE > 0.0 and expect > 0.0 and dE > expect:
        Quality = (2 * expect - dE) / expect
    else:
        Quality = dE / expect
    GC = new_chain.CalcRMSCartGrad(GW)
    eGC = new_chain.CalcRMSCartGrad(expectG)
    GPC = new_chain.CalcRMSCartGrad(GW_prev)
    QualityG = 2.0 - GC / max(eGC, GPC / 2, params_avgg / 2)
    Quality = QualityG
    Describe = (
        "Good"
        if Quality > ThreHQ
        else ("Okay" if Quality > ThreLQ else ("Poor" if Quality > -1.0 else "Reject"))
    )
    print(
        "\n %13s %13s %13s %13s %11s %14s %13s"
        % (
            "GAvg(eV/Ang)",
            "GMax(eV/Ang)",
            "Length(Ang)",
            "DeltaE(kcal)",
            "RMSD(Ang)",
            "TrustRad(Ang)",
            "Step Quality",
        )
    )
    print(
        "@%13s %13s %13s %13s %11s  %8.4f (%s)  %13s"
        % (
            "% 8.4f  " % new_chain.avgg,
            "% 8.4f  " % new_chain.maxg,
            "% 8.4f  " % sum(new_chain.calc_spacings()),
            "% 8.4f  " % (au2kcal * dE / len(new_chain)),
            "% 8.4f  " % (ChainRMSD(new_chain, old_chain)),
            trust,
            trustprint,
            "% 6.2f (%s)" % (Quality, Describe),
        )
    )
    return (
        new_chain,
        Y,
        GW,
        GP,
        np.array(HW),
        np.array(HP),
        [old_chain],
        respaced,
        Quality,
    )


def converged(
    chain_maxg, chain_avgg, params_maxg, params_avgg, optCycle, params_maxcyc
):
    """
    This function was part of the 'while' loop in OptimizeChain(). Checking to see whether the chain is converged.
    """
    if chain_maxg < params_maxg and chain_avgg < params_avgg:
        print("--== Optimization Converged. ==--")
        return True
    if optCycle >= params_maxcyc:
        print("--== Maximum optimization cycles reached. ==--")
        return True
    return False


def takestep(
    chain,
    c_hist,
    params,
    optCycle,
    LastForce,
    ForceRebuild,
    trust,
    Y,
    GW,
    GP,
    HW,
    HP,
    result,
):
    """
    This function was part of the 'while' loop in OptimizeChain(). Take step to move the chain.
    """
    LastForce += ForceRebuild
    dy, expect, expectG, ForceRebuild = chain.CalcInternalStep(trust, HW, HP)
    # If ForceRebuild is True, the internal coordinate system
    # is rebuilt and the optimization skips a step
    if ForceRebuild:
        if LastForce == 0:
            pass
        elif LastForce == 1:
            print(
                "\x1b[1;91mFailed twice in a row to rebuild the coordinate system\x1b[0m"
            )
            print("\x1b[93mResetting Hessian\x1b[0m")
            params.reset = True
        elif LastForce == 2:
            print(
                "\x1b[1;91mFailed three times to rebuild the coordinate system\x1b[0m"
            )
            print("\x1b[93mContinuing in Cartesian coordinates\x1b[0m")
        else:
            raise RuntimeError("Coordinate system has failed too many times")
        CoordCounter = 0
        r0 = params.reset
        params.reset = True
        chain, Y, GW, GP, HW, HP = recover(
            c_hist, params, LastForce == 2, result
        )  # TODO: c_hist and params aren't being passed
        params.reset = r0
        print("\x1b[1;93mSkipping optimization step\x1b[0m")
        optCycle -= 1
    else:
        LastForce = 0
    # Before updating any of our variables, copy current variables to "previous"
    Y_prev = Y.copy()
    GW_prev = GW.copy()
    GP_prev = GP.copy()
    # Cartesian norm of the step
    # Whether the Cartesian norm comes close to the trust radius
    # Update the internal coordinates
    # Obtain a new chain with the step applied
    new_chain = chain.TakeStep(dy, printStep=False)
    respaced = new_chain.delete_insert(1.5)
    return (
        new_chain,
        chain,
        expect,
        expectG,
        ForceRebuild,
        LastForce,
        Y_prev,
        GW_prev,
        GP_prev,
        respaced,
        optCycle,
    )


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
        print("First, optimizing endpoint images.")
        chain.OptimizeEndpoints(params.maxg)
    print("Optimizing the chain.")
    chain.respace(0.01)
    chain.delete_insert(1.0)
    if params.align:
        print("Aligning Chain")
        chain.align()
    if params.nebew is not None:
        print("Energy weighted NEB will be performed.")
    chain.ComputeMetric()
    chain.ComputeChain(cyc=0)
    t0 = time.time()
    # chain.SetFlags(params.gtol, params.climb)
    # Obtain the guess Hessian matrix
    chain.ComputeGuessHessian(full=False, blank=isinstance(engine, Blank))
    # Print the status of the zeroth iteration
    print("-=# Chain optimization cycle 0 #=-")
    print("Spring Force: %.2f kcal/mol/Ang^2" % params.nebk)
    chain.PrintStatus()
    print("-= Chain Properties =-")
    print(
        "@%13s %13s %13s %13s %11s %13s %13s"
        % (
            "GAvg(eV/Ang)",
            "GMax(eV/Ang)",
            "Length(Ang)",
            "DeltaE(kcal)",
            "RMSD(Ang)",
            "TrustRad(Ang)",
            "Step Quality",
        )
    )
    print(
        "@%13s %13s %13s"
        % (
            "% 8.4f  " % chain.avgg,
            "% 8.4f  " % chain.maxg,
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
        print("-=# Chain optimization cycle %i #=-" % (optCycle))
        # =======================================#
        # |    Obtain an optimization step      |#
        # |    and update Cartesian coordinates |#
        # =======================================#
        print("Time since last ComputeChain: %.3f s" % (time.time() - t0))
        (
            chain,
            old_chain,
            expect,
            expectG,
            ForceRebuild,
            LastForce,
            Y_prev,
            GW_prev,
            GP_prev,
            respaced,
            optCycle,
        ) = takestep(
            chain,
            c_hist,
            params,
            optCycle,
            LastForce,
            ForceRebuild,
            trust,
            Y,
            GW,
            GP,
            HW,
            HP,
            None,
        )
        chain.ComputeChain(cyc=optCycle)
        t0 = time.time()
        # ----------------------------------------------------------
        chain, Y, GW, GP, HW, HP, c_hist, respaced, Quality = compare(
            old_chain,
            chain,
            ThreHQ,
            ThreLQ,
            GW,
            HW,
            HP,
            respaced,
            optCycle,
            expect,
            expectG,
            trust,
            trustprint,
            params.avgg,
            Quality,
        )
        if respaced:
            chain.SaveToDisk(fout="chain_%04i.xyz" % optCycle)
            continue
        chain.SaveToDisk(fout="chain_%04i.xyz" % optCycle)
        # =======================================#
        # |    Check convergence criteria       |#
        # =======================================#
        if converged(
            chain.maxg, chain.avgg, params.maxg, params.avgg, optCycle, params.maxcyc
        ):
            break
        # =======================================#
        # |  Adjust Trust Radius / Reject Step  |#
        # =======================================#
        chain, trust, trustprint, Y, GW, GP, good = qualitycheck(
            trust,
            chain,
            old_chain,
            Quality,
            ThreLQ,
            ThreRJ,
            ThreHQ,
            Y,
            GW,
            GP,
            Y_prev,
            GW_prev,
            GP_prev,
            params.tmax,
        )
        if not good:
            continue
        c_hist.append(chain)
        c_hist = c_hist[-params.history :]
        # =======================================#
        # |      Update the Hessian Matrix      |#
        # =======================================#
        chain, Y, GW, GP, HP, HW, Y_prev, GP_prev, GW_prev, _ = updatehessian(
            chain,
            old_chain,
            HP,
            HW,
            Y,
            Y_prev,
            GW,
            GW_prev,
            GP,
            GP_prev,
            LastForce,
            params,
            None,
        )


class nullengine(object):
    """
    Fake engine for QCFractal.
    """

    def __init__(self, charge, mult, elems, coords):
        self.elems = elems
        self.coords = np.array(coords).flatten()
        self.na = len(elems)
        self.nm = int(self.coords.size / (3 * self.na))
        M1 = Molecule()

        for i in range(self.nm):
            interval = 3 * self.na
            start = interval * i
            M2 = Molecule()
            M2.elem = self.elems
            M2.charge = charge
            M2.mult = mult
            M2.xyzs = [self.coords[start : start + self.na * 3].reshape(-1, 3)]
            if i == 0:
                M1 = M2
            else:
                M1 += M2
        self.M = M1


def chaintocoords(chain, ang=False):
    """
    Extracts Cartesian coordinates from an ElasticBand object.
    Parameters
    ----------
    chain: ElasticBand object
    ang: Bool
        True will return Cartesian coordinates in Ang (False in Bohr).

    Returns
    -------
    newcoords: list
        Cartesian coordinates in list. It is not numpy array because neb record socket in QCF can't process nparray.
    """
    newcoords = []
    if ang:
        factor = 1
    else:
        factor = bohr2ang
    for i in range(len(chain)):
        M_obj = chain.Structures[i].M
        coord = np.round(M_obj.xyzs[0] / factor, 8)
        newcoords.append(coord.tolist())
    return newcoords


def arrange(qcel_mols, align):
    """
    This function will align and respace a chain.
    Parameters
    ----------
    qcel_mols: [QCElemental Molecule object]
        QCElemental Molecule objects in a list that needs to be aligned.

    align: bool
        True will align the chain

    Returns
    -------
    aligned_chain: [QCElemental Molecule object]
        Aligned molecule objects
    """
    from qcelemental.models import Molecule as qcmol

    aligned_chain = []
    sym = qcel_mols[0].symbols.tolist()
    chg = qcel_mols[0].molecular_charge
    mult = qcel_mols[0].molecular_multiplicity
    M = Molecule()
    for i, mol in enumerate(qcel_mols):
        M1 = Molecule()
        xyzs = np.round(mol.geometry * bohr2ang, 8)
        M1.elem = sym
        M1.xyzs = [xyzs]
        if i == 0:
            M = M1
        else:
            M += M1

    opt_param = OptParams(**{"neb": True})
    chain = ElasticBand(
        M, engine=None, tmpdir="tmp", coordtype="cart", params=opt_param, plain=0
    )

    chain.respace(0.01)
    chain.delete_insert(1.0)
    if align:
        print("Aligning chain")
        chain.align(qcf=True)
    newcoords = chaintocoords(chain)
    for coords in newcoords:
        aligned_chain.append(
            qcmol(
                symbols=sym,
                geometry=coords,
                molecular_charge=chg,
                molecular_multiplicity=mult,
            )
        )

    return aligned_chain


def prepare(prev):
    """
    This function is for QCFractal. Takes a dictionary with parameters and prepare for the NEB calculation loops.
    """

    print("\n-=# Chain optimization cycle 0 #=-")

    coords_ang = np.array(prev.pop("geometry")) * bohr2ang
    args_dict = prev.get("params")
    elems = prev.get("elems")
    charge = prev.get("charge")
    mult = prev.get("mult")
    energies = prev.get("energies")
    gradients = prev.pop("gradients")

    params = {
        "neb": True,
        "images": args_dict.get("images"),
        "maxg": args_dict.get("maximum_force"),
        "avgg": args_dict.get("average_force"),
        "nebew": args_dict.get("energy_weighted"),
        "nebk": args_dict.get("spring_constant"),
        "maxcyc": args_dict.get("maximum_cycle"),
        "plain": args_dict.get("spring_type"),
        "reset": args_dict.get("hessian_reset"),
        "skip": not args_dict.get("hessian_reset"),
        "epsilon": args_dict.get("epsilon"),
        "coordsys": "cart",
    }

    print("Spring Force: %.2f kcal/mol/Ang^2" % params.get("nebk"))
    if params.get("nebew") is not None:
        print("Energey weighted NEB will be performed.")
    opt_param = OptParams(**params)
    opt_param.customengine = nullengine(charge, mult, elems, coords_ang)

    result = []
    for i in range(len(energies)):
        result.append({"energy": energies[i], "gradient": gradients[i]})

    M, engine = get_molecule_engine(**{"customengine": opt_param.customengine})
    tmpdir = "NEB.tmp"

    chain = ElasticBand(
        M,
        engine=engine,
        tmpdir=tmpdir,
        coordtype="cart",
        params=opt_param,
        plain=opt_param.plain,
    )

    trust = opt_param.trust
    chain.ComputeMetric()
    chain.ComputeChain(result=result)
    chain.ComputeGuessHessian(full=False, blank=isinstance(engine, Blank))
    chain.PrintStatus()

    print("-= Chain Properties =-")
    print(
        "@%13s %13s %13s %13s %11s %13s %13s"
        % (
            "GAvg(eV/Ang)",
            "GMax(eV/Ang)",
            "Length(Ang)",
            "DeltaE(kcal)",
            "RMSD(Ang)",
            "TrustRad(Ang)",
            "Step Quality",
        )
    )
    print(
        "@%13s %13s %13s"
        % (
            "% 8.4f  " % chain.avgg,
            "% 8.4f  " % chain.maxg,
            "% 8.4f  " % sum(chain.calc_spacings()),
        )
    )

    #Y = chain.get_internal_all()
    GW = chain.get_global_grad("total", "working")
    GP = chain.get_global_grad("total", "plain")
    HW = chain.guess_hessian_working.copy()
    HP = chain.guess_hessian_plain.copy()
    dy, expect, expectG, ForceRebuild = chain.CalcInternalStep(trust, HW, HP)
    new_chain = chain.TakeStep(dy, printStep=False)
    respaced = new_chain.delete_insert(1.5)
    newcoords = chaintocoords(new_chain)
    new_attrs = check_attr(new_chain)
    old_attrs = check_attr(chain)

    temp = {
        "GW": GW.tolist(),
        "GP": GP.tolist(),
        "HW": HW.tolist(),
        "HP": HP.tolist(),
        "new_attrs": new_attrs,
        "old_attrs": old_attrs,
        "trust": trust,
        "expect": expect,
        "expectG": expectG.tolist(),
        "respaced": respaced,
        "trustprint": "=",
        "frocerebuild": False,
        "lastforce": 0,
        "old_coord": chaintocoords(chain, True),
        "result": result,
    }
    prev.update(temp)
    return newcoords, prev


def switch(array, numpy=False):
    """
    Switches between numpy and list.
    """
    new = []
    if not numpy:
        for i in array:
            if i is not None:
                new.append(i.tolist())
            else:
                new.append(None)
    else:
        for i in array:
            if i is not None:
                new.append(np.array(i))
            else:
                new.append(None)
    return new


def add_attr(chain, attrs):
    """
    Add chain attributes to a given chain.
    """
    chain.TotBandEnergy = attrs.get("TotBandEnergy")
    if attrs.get("haveMetric", False):
        chain.haveMetric = True
        kmats = switch(attrs.get("kmats"), True)
        metrics = switch(attrs.get("metrics"), True)
        chain.kmats = kmats
        chain.metrics = metrics
    if attrs.get("climbSet", False):
        chain.climbSet = True
        chain.climbers = attrs.get("climbers")
        chain.locks = attrs.get('locks')
    return chain


def check_attr(chain):
    """
    Check a chain's attributes and extract them.
    """
    attrs = {}
    if chain.haveMetric:
        attrs["haveMetric"] = True
        kmats = switch(chain.kmats, False)
        metrics = switch(chain.metrics, False)
        attrs["kmats"] = kmats
        attrs["metrics"] = metrics
    if chain.climbSet:
        attrs["climbSet"] = True
        attrs["climbers"] = [int(i) for i in chain.climbers]
        attrs['locks'] = chain.locks
    attrs["TotBandEnergy"] = chain.TotBandEnergy

    return attrs

def dict_to_binary(the_dict):
    import msgpack
    bin = msgpack.dumps(the_dict)
    return bin

def nextchain(prev):
    """
    Generate a next chain's Cartesian coordinate for QCFractal.
    """
    coords_bohr = prev.pop("geometry")
    coords_ang = np.array(coords_bohr) * bohr2ang
    args_dict = prev.pop("params")
    elems = prev.get("elems")
    charge = prev.get("charge")
    mult = prev.get("mult")
    energies = prev.pop("energies")
    gradients = prev.pop("gradients")
    iteration = int(args_dict.get("iteration")) - 1

    result = []
    for i in range(len(energies)):
        result.append({"energy": energies[i], "gradient": gradients[i]})

    ThreLQ = 0.0
    ThreHQ = 0.5
    ThreRJ = 0.001

    print("\n-=# Chain optimization cycle %i #=-" % iteration)

    params_dict = {
        "neb": True,
        "images": args_dict.get("images"),
        "maxg": args_dict.get("maximum_force"),
        "avgg": args_dict.get("average_force"),
        "nebew": args_dict.get("energy_weighted"),
        "nebk": args_dict.get("spring_constant"),
        "maxcyc": args_dict.get("maximum_cycle"),
        "plain": args_dict.get("spring_type"),
        "reset": args_dict.get("hessian_reset"),
        "skip": not args_dict.get("hessian_reset"),
        "epsilon": args_dict.get("epsilon"),
        "coordsys": "cart",
    }

    params = OptParams(**params_dict)
    params.customengine = nullengine(charge, mult, elems, coords_ang)

    params2 = OptParams(**params_dict)
    params2.customengine = nullengine(
        charge, mult, elems, np.array(prev.get("old_coord"))
    )

    M_old, engine = get_molecule_engine(**{"customengine": params2.customengine})
    M, engine = get_molecule_engine(**{"customengine": params.customengine})

    tmpdir = "NEB.tmp"
    old_chain = ElasticBand(
        M_old,
        engine=engine,
        tmpdir=tmpdir,
        coordtype="cart",
        params=params2,
        plain=params2.plain,
    )
    chain = ElasticBand(
        M,
        engine=engine,
        tmpdir=tmpdir,
        coordtype="cart",
        params=params,
        plain=params.plain,
    )

    trust = prev.get("trust")
    trustprint = prev.get("trustprint", "=")
    Y = chain.get_internal_all()
    Y_prev = old_chain.get_internal_all()

    HW = prev.get("HW")
    HP = prev.get("HP")
    respaced = prev.get("respaced")
    expect = prev.get("expect")
    expectG = prev.get("expectG")
    Quality = prev.get("quality")
    LastForce = prev.get("lastforce", 0)
    ForceBuild = prev.get("forcerebuild", False)

    chain = add_attr(chain, prev.get("new_attrs"))
    old_chain = add_attr(old_chain, prev.get("old_attrs"))
    chain.ComputeGuessHessian(full=False, blank=isinstance(engine, Blank))
    old_chain.ComputeGuessHessian(full=False, blank=isinstance(engine, Blank))
    GW = np.array(prev.get("GW"))
    GP = np.array(prev.get("GP"))
    GW_prev = np.array(prev.get("GW_prev", GW.copy()))
    GP_prev = np.array(prev.get("GP_prev", GP.copy()))

    chain.ComputeChain(result=result)

    chain, Y, GW, GP, HW, HP, c_hist, respaced, Quality = compare(
        old_chain,
        chain,
        ThreHQ,
        ThreLQ,
        GW, #GW prev
        HW,
        HP,
        respaced,
        iteration,
        expect,
        expectG,
        trust,
        trustprint,
        params.avgg,
        Quality,
    )
    if respaced:
        (
            chain,
            old_chain,
            expect,
            expectG,
            ForceRebuild,
            LastForce,
            Y_prev,
            GW_prev,
            GP_prev,
            respaced,
            _,
        ) = takestep(
            chain,
            old_chain,
            params,
            iteration,
            LastForce,
            ForceBuild,
            trust,
            Y,
            GW,
            GP,
            HW,
            HP,
            prev.get('result', result),
        )
        new_attrs = check_attr(chain)
        old_attrs = check_attr(old_chain)
        newcoords = chaintocoords(chain)
        temp = {
            "GW": GW.tolist(),
            "GW_prev": GW_prev.tolist(),
            "GP": GP.tolist(),
            "GP_prev": GP_prev.tolist(),
            "HW": HW.tolist(),
            "HP": HP.tolist(),
            "new_attrs": new_attrs,
            "old_attrs": old_attrs,
            "expect": expect,
            "expectG": expectG.tolist(),
            "respaced": respaced,
            "forcerebuild": ForceRebuild,
            "lastforce": LastForce,
            "old_coord":chaintocoords(old_chain, True),
            "quality": Quality,
            "result": result,
        }
        prev.update(temp)
        return newcoords, prev

    if converged(
        chain.maxg, chain.avgg, params.maxg, params.avgg, iteration, params.maxcyc
    ):
        return None, {}

    # qualitycheck returns old_chain and smaller trust if quality is not good.
    chain, trust, trustprint, Y, GW, GP, good = qualitycheck(
        trust,
        chain,
        old_chain,
        Quality,
        ThreLQ,
        ThreRJ,
        ThreHQ,
        Y,
        GW,
        GP,
        Y_prev,
        GW_prev,
        GP_prev,
        params.tmax,
    )
    if not good:
        # chain here is old_chain
        chain.ComputeChain(result=prev.get("result"))
        (
            chain,
            old_chain,
            expect,
            expectG,
            ForceRebuild,
            LastForce,
            Y_prev,
            GW_prev,
            GP_prev,
            respaced,
            _,
        ) = takestep(
            chain,
            old_chain,
            params,
            iteration,
            LastForce,
            ForceBuild,
            trust,
            Y,
            GW,
            GP,
            HW,
            HP,
            prev.get("result"),
        )
        new_attrs = check_attr(chain)
        old_attrs = check_attr(old_chain)
        newcoords = chaintocoords(chain)
        # Here GW, GP and their previous values should be same.
        temp = {
            "GW": GW.tolist(),
            "GW_prev": GW_prev.tolist(),
            "GP": GP.tolist(),
            "GP_prev": GP_prev.tolist(),
            "new_attrs": new_attrs,
            "old_attrs": old_attrs,
            "trust": trust,
            "expect": expect,
            "expectG": expectG.tolist(),
            "quality": Quality,
            "respaced": respaced,
            "old_coord": chaintocoords(old_chain, True),
            "climbset": chain.climbSet,
            "result": result,
        }
        prev.update(temp)
        return newcoords, prev
    # If minimum eigenvalues of HW is lower than epsilon, GP, GW are same as their previous values.
    chain, Y, GW, GP, HP, HW, Y_prev, GP_prev, GW_prev, H_reset = updatehessian(
        chain,
        old_chain,
        HP,
        HW,
        Y,
        Y_prev,
        GW,
        GW_prev,
        GP,
        GP_prev,
        LastForce,
        params,
        prev.get("result", result),
    )
    if H_reset:
        result = prev.get("result", result)
    (
        chain,
        old_chain,
        expect,
        expectG,
        ForceRebuild,
        LastForce,
        Y_prev,
        GW_prev,
        GP_prev,
        respaced,
        _,
    ) = takestep(
        chain,
        old_chain,
        params,
        iteration,
        LastForce,
        ForceBuild,
        trust,
        Y,
        GW,
        GP,
        HW,
        HP,
        prev.get("result", result),
    )
    new_attrs = check_attr(chain)
    old_attrs = check_attr(old_chain)
    temp = {
        "GW": GW.tolist(),
        "GW_prev": GW_prev.tolist(),
        "GP": GP.tolist(),
        "GP_prev": GP_prev.tolist(),
        "HW": HW.tolist(),
        "HP": HP.tolist(),
        "new_attrs": new_attrs,
        "old_attrs": old_attrs,
        "trust": trust,
        "trustprint": trustprint,
        "expect": expect,
        "expectG": expectG.tolist(),
        "quality": Quality,
        "respaced": respaced,
        "old_coord": chaintocoords(old_chain, True),
        "lastforce": LastForce,
        "forcerebuild": ForceRebuild,
        "result": result,
    }
    newcoords = chaintocoords(chain)
    prev.update(temp)
    for k, v in prev.items():
        mem_size = sys.getsizeof(dict_to_binary({k: v}))*1e-9
        if mem_size > 1e-1:
            print('Size of %s : %f Gb' %(k, mem_size))
    return newcoords, prev


def main():

    args = parse_optimizer_args(sys.argv[1:])
    args["neb"] = True
    params = OptParams(**args)

    M, engine = get_molecule_engine(**args)

    if params.port != 0:
        createWorkQueue(params.port, debug=params.verbose)

    if params.prefix is None:
        tmpdir = os.path.splitext(args["input"])[0] + ".tmp"
    else:
        tmpdir = params.prefix + ".tmp"

    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    # Make the initial chain
    chain = ElasticBand(
        M,
        engine=engine,
        tmpdir=tmpdir,
        coordtype=params.coordsys,
        params=params,
        plain=params.plain,
    )
    OptimizeChain(chain, engine, params)


if __name__ == "__main__":
    main()
