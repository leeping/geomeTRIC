from __future__ import print_function
from __future__ import division

import os, sys, re, shutil, time
import argparse
import itertools
import numpy as np
# from guppy import hpy
from scipy.linalg import sqrtm
from .optimize import Optimize
from .params import OptParams
from .step import get_delta_prime_trm, brent_wiki, trust_step, calc_drms_dmax
from .engine import set_tcenv, load_tcin, TeraChem, Psi4, QChem, QCEngineAPI, Gromacs, Blank
from .internal import *
from .nifty import flat, row, col, pmat2d, printcool, createWorkQueue, getWorkQueue, wq_wait, ang2bohr, bohr2ang, kcal2au, au2kcal, au2evang
from .molecule import Molecule, EqualSpacing
from .errors import EngineError, CheckCoordError, Psi4EngineError, QChemEngineError, TeraChemEngineError, \
    ConicalIntersectionEngineError, OpenMMEngineError, GromacsEngineError, MolproEngineError, QCEngineAPIEngineError, GaussianEngineError, QCAIOptimizationError
from copy import deepcopy

def rms_gradient(gradx):
    """ Return the RMS of a Cartesian gradient. """
    atomgrad = np.sqrt(np.sum((gradx.reshape(-1,3))**2, axis=1))
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
    CoordSysDict = {'cart':(CartesianCoordinates, False, False),
                    'prim':(PrimitiveInternalCoordinates, True, False),
                    'dlc':(DelocalizedInternalCoordinates, True, False),
                    'hdlc':(DelocalizedInternalCoordinates, False, True),
                    'tric':(DelocalizedInternalCoordinates, False, False),
                    'trim':(PrimitiveInternalCoordinates, False, False)} # Primitive TRIC, i.e. not delocalized
    CoordClass, connect, addcart = CoordSysDict[coordtype]
    if CoordClass is DelocalizedInternalCoordinates:
        IC = CoordClass(M, build=True, connect=connect, addcart=addcart, cartesian=(coordtype == 'cart'), chain=chain, ic_displace=ic_displace, guessw=guessw)
    elif chain:
        IC = ChainCoordinates(M, connect=connect, addcart=addcart, cartesian=(coordtype == 'cart'), ic_displace=ic_displace, guessw=guessw)
    else:
        IC = CoordClass(M, build=True, connect=connect, addcart=addcart, chain=False, ic_displace=ic_displace)
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
            raise RuntimeError('Please pass in a Molecule object with just one frame')
        # Overwrite coordinates if we are passing in new ones
        if coords is not None:
            self.M.xyzs[0] = coords.reshape(-1,3)*bohr2ang
            self.M.build_topology()
        # Set initial Cartesian coordinates
        self.cartesians = self.M.xyzs[0].flatten()*ang2bohr
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
        self.qcfserver = False
        if type(self.engine).__name__ == "QCEngineAPI" and self.engine.client != False:       
            self.qcfserver = True
        
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
            raise RuntimeError('Please pass in a flat NumPy array in a.u.')
        if len(value.shape) != 1:
            raise RuntimeError('Please pass in a flat NumPy array in a.u.')
        if value.shape[0] != (3*self.M.na):
            raise RuntimeError('Input array dimensions should be 3x number of atoms')
        # Multiplying by bohr2ang copies it
        self.M.xyzs[0] = value.reshape(-1,3)*bohr2ang
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
        if xyz is None: xyz = self.cartesians
        Bmat = self.IC.wilsonB(xyz)
        Gx = np.dot(Bmat.T, np.array(gradq).T).flatten()
        return Gx

    def ConvertCartGradToIC(self, gradx, xyz=None):
        """ 
        Given a gradient in Cartesian coordinates, convert it back to an internal gradient. 
        Unfortunate "reversal" in the interface with respect to IC.calcGrad which takes xyz first!
        """
        if xyz is None: xyz = self.cartesians
        return self.IC.calcGrad(self.cartesians, gradx)
        
    def ComputeEnergyGradient(self):
        """ Compute energies and Cartesian gradients for the current structure. """
        if self.qcfserver == True:
            res = self.engine.calc_qcf(self.cartesians)
            self.engine.wait_qcf([res])
            result = self.engine.read_qcf(res)
        else:    
            result = self.engine.calc(self.cartesians, self.tmpdir)
        self.energy = result['energy'] 
        self.grad_cartesian = result['gradient'] 
        self.grad_internal = self.IC.calcGrad(self.cartesians, self.grad_cartesian)

    # def CopyOutput(self, cyc):
    #     """ Compute energies and Cartesian gradients for the current structure. """
    #     self.engine.number_output(self.tmpdir, cyc)
    #     # self.energy, self.grad_cartesian = self.engine.calc(self.cartesians, self.tmpdir)
    #     # self.grad_internal = self.IC.calcGrad(self.cartesians, self.grad_cartesian)

    def QueueEnergyGradient(self):
        self.engine.calc_wq(self.cartesians, self.tmpdir)
            
    def GetEnergyGradient(self):
        result = self.engine.read_wq(self.cartesians, self.tmpdir)
        self.energy = result['energy']
        self.grad_cartesian = result['gradient']
        self.grad_internal = self.IC.calcGrad(self.cartesians, self.grad_cartesian)

    def QCPortalEnergyGradient(self):
        resp_id = self.engine.calc_qcf(self.cartesians)
        return resp_id

    def GetQCEnergyGradient(self, cal_id):
        result = self.engine.read_qcf(cal_id)
        self.energy = result['energy']
        self.grad_cartesian = result['gradient']
        self.grad_internal = self.IC.calcGrad(self.cartesians, self.grad_cartesian)

    def OptimizeGeometry(self, gtol=None):
        """ Optimize the geometry of this Structure. """
        opt_params = OptParams()
        if gtol != None and opt_params.Convergence_grms * au2evang > gtol:
            opt_params.Convergence_grms = gtol / au2evang
            opt_params.Convergence_gmax = 1.5 * gtol / au2evang
        force_tric = True
        if force_tric:
            self.IC = CoordinateSystem(self.M, 'tric')
        else:
            self.IC = CoordinateSystem(self.M, self.coordtype)

        if self.qcfserver == False:  
            print("No QCFractal server, it will be carried locally.")
            optCoords = Optimize(self.cartesians, self.M, self.IC, self.engine, self.tmpdir, opt_params)#, xyzout=os.path.join(self.tmpdir,'optimize.xyz')) 
            self.cartesian = np.array(optCoords[-1].xyzs).flatten()*ang2bohr
        else:
            """
            Optimization procedure through QCAI
            """
            import time
            print("QCAI optimization started.")
            new_schema = deepcopy(self.engine.schema)
            new_schema['molecule']['geometry'] = self.cartesians.reshape(-1,3)
            qcel_mol = new_schema['molecule']
            model = new_schema['model']
            #1/10/2022 HP: Passing other keywords such as coordsys will be added later.
            opt_qcschema = { 
                    "keywords": None,
                    "qc_spec": {
                        "driver": "gradient",
                        "method": model["method"],
                        "basis": model["basis"],
                        "program": self.engine.program
                            }
                        }
            r=self.engine.client.add_procedure("optimization", "geometric",opt_qcschema, [qcel_mol]) #ComputeResponse
            proc_id = r.ids
            loop = 0 
            while True:
                proc = self.engine.client.query_procedures(id=proc_id)[0] #OptimizationRecord
                status = proc.status.split('.')[-1].upper().strip()
                if status == "INCOMPLETE":
                    time.sleep(50)
                    loop += 1
                elif status == "ERROR":
                    print("Error detected")
                    res = self.engine.client.modify_tasks("restart",proc.id)
                    print(res.n_updated,"ERROR status optimization resubmitted")
                    loop += 1
                elif status == "COMPLETE":
                    optCoords = proc.get_final_molecule().geometry     
                    print("QCAI optimization is done.")
                    break
        
                if loop > 100:
                    raise QCEngineAPIEngineError("Stuck in endpoint optimization procedure in NEB.")
            self.cartesian = np.array(optCoords).flatten()
        # Rebuild the internal coordinate system
        self.IC = CoordinateSystem(self.M, self.coordtype)
        self.CalcInternals()
        self.ComputeEnergyGradient()
       
class Chain(object):
    """ Class representing a chain of states. """
    def __init__(self, molecule, engine, tmpdir, coordtype, params, coords=None, ic_displace=False):
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
                coords = coords.reshape(len(self), 3*self.M.na)
            if coords.shape != (len(self), 3*self.M.na):
                raise RuntimeError('Coordinates do not have the right shape')
            for i in range(len(self)):
                self.M.xyzs[i] = coords[i,:].reshape(-1, 3)*bohr2ang
        self.na = self.M.na
        # The Structures are what store the individual Cartesian coordinates for each frame.
        self.Structures = [Structure(self.M[i], engine, os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i), self.coordtype) for i in range(len(self))]
        # The total number of variables being optimized
        # self.nvars = sum([self.Structures[n].nICs for n in range(1, len(self)-1)])
        # Locked images (those having met the convergence criteria) are not updated
        self.locks = [True] + [False for n in range(1, len(self)-1)] + [True]
        self.haveCalcs = False
        ### Test ###
        print("Starting Test")
        self.GlobalIC = CoordinateSystem(self.M, self.coordtype, chain=True, ic_displace=self.ic_displace, guessw=self.params.guessw)
        # xyz = np.array(self.M.xyzs)*ang2bohr
        # print xyz.shape
        # self.GlobalIC.checkFiniteDifference(self.get_cartesian_all(endpts=True))
        self.nvars = len(self.GlobalIC.Internals)
        # raw_input()
        self.qcfserver = False
        if type(self.engine).__name__ == "QCEngineAPI" and self.engine.client != False:       
            self.qcfserver = True

    def UpdateTempDir(self, iteration):
        self.tmpdir = os.path.join(os.path.split(self.tmpdir)[0], 'chain_%04i' % iteration)
        if not os.path.exists(self.tmpdir): os.makedirs(self.tmpdir)
        for i, s in enumerate(self.Structures):
            s.tmpdir = os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i)

    def __len__(self):
        """ Return the length of the chain. """
        return len(self.M)

    def ComputeEnergyGradient(self, cyc=None):
        """ Compute energies and gradients for each structure. """
        # This is the parallel point.
        wq = getWorkQueue()
        if wq is None and self.qcfserver==False: #If work queue and qcfractal aren't available, just run calculations locally.
            for i in range(len(self)):
                self.Structures[i].ComputeEnergyGradient()
        elif self.qcfserver==True: #Else if client is known, submit jobs to qcfractal server.
            ids=[]
            for i in range(len(self)):
                resp = self.Structures[i].QCPortalEnergyGradient()
                ids.append(resp)
            self.engine.wait_qcf(ids)
            for i, value in enumerate(ids):
                self.Structures[i].GetQCEnergyGradient(value)
        else: #If work queue is available, handle jobs with the work queue.
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
            self.Structures[i].grad_cartesian = other.Structures[i].grad_cartesian.copy()
            self.Structures[i].grad_internal = other.Structures[i].grad_internal.copy()
        self.haveCalcs = True

    def get_cartesian_all(self, endpts=False):
        """ Return the internal coordinates of images 1 .. N-2. """
        if endpts:
            return np.hstack(tuple([self.Structures[i].cartesians for i in range(len(self))]))
        else:
            return np.hstack(tuple([self.Structures[i].cartesians for i in range(1, len(self)-1)]))

    def get_internal_all_old(self):
        """ Return the internal coordinates of images 1 .. N-2. """
        return np.hstack(tuple([self.Structures[i].internals for i in range(1, len(self)-1)]))

    def get_internal_all(self):
        """ Return the internal coordinates of images 1 .. N-2. """
        return self.GlobalIC.calculate(np.hstack([self.Structures[i].cartesians for i in range(len(self))]).flatten())

    def getCartesianNorm_old(self, dy, verbose=False):
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
        currvar = 0
        rmsds = []
        maxds = []
        for n in range(1, len(self)-1):
            S = self.Structures[n]
            dy_i = dy[currvar:currvar+S.nICs]
            Xnew = S.IC.newCartesian(S.cartesians, dy_i, verbose=verbose)
            currvar += S.nICs
            rmsd, maxd = calc_drms_dmax(Xnew, S.cartesians, align=False)
            rmsds.append(rmsd)
            maxds.append(maxd)
        # The RMSD of the chain is tentatively defined as the maximum RMSD over all images.
        return np.max(np.array(rmsds))

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

    def anybork_old(self):
        return any([self.Structures[n].IC.bork for n in range(1, len(self)-1)])

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
            v0 = self.params.epsilon-Emin
        else:
            v0 = 0.0
        print("Hessian Eigenvalues (Working) :", " ".join(["% .4e" % i for i in Eig[:5]]), "...", " ".join(["% .4e" % i for i in Eig[-5:]]))
        if np.sum(np.array(Eig) < 0.0) > 5:
            print("%i Negative Eigenvalues" % (np.sum(np.array(Eig) < 0.0)))
        if Eig[0] != EigP[0]:
            print("Hessian Eigenvalues (Plain)   :", " ".join(["% .4e" % i for i in EigP[:5]]), "...", " ".join(["% .4e" % i for i in EigP[-5:]]))
            if np.sum(np.array(EigP) < 0.0) > 5:
                print("%i Negative Eigenvalues" % (np.sum(np.array(EigP) < 0.0)))
        if finish: return

        if Eig[0] < 0.0:
            dy, expect, _ = get_delta_prime_trm(0.0, X, G, np.eye(len(G)), None, False)
            print("\x1b[95mTaking steepest descent rather than Newton-Raphson step\x1b[0m")
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
        if self.params.verbose: print("dy(i): %.4f dy(c) -> target: %.4f -> %.4f" % (inorm, cnorm, trust))
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
                if self.params.verbose: print("\x1b[93mUsing stored solution at %.3e\x1b[0m" % froot.stored_val)
                iopt = froot.stored_arg
            elif self.anybork():
                # 2) If there is no stored solution, 
                # then reduce trust radius by 50% and try again
                # (up to three times)
                for i in range(3):
                    froot.target /= 2 
                    if self.params.verbose: print("\x1b[93mReducing target to %.3e\x1b[0m" % froot.target)
                    froot.above_flag = True
                    iopt = brent_wiki(froot.evaluate, 0.0, iopt, froot.target, cvg=0.1, verbose=self.params.verbose)
                    if not self.anybork(): break
            if self.anybork():
                print("\x1b[91mInverse iteration for Cartesians failed\x1b[0m")
                # This variable is added because IC.bork is unset later.
                ForceRebuild = True
            else:
                if self.params.verbose: print("\x1b[93mBrent algorithm requires %i evaluations\x1b[0m" % froot.counter)
            dy, expect = trust_step(iopt, v0, X, G, H, None, False, self.params.verbose)
        # Expected energy change should be calculated from PlainGrad
        GP = self.get_global_grad("total", "plain")
        expect = flat(0.5*np.dot(np.dot(row(dy),np.array(HP)),col(dy)))[0] + np.dot(dy,GP)
        expectG = flat(np.dot(np.array(H),col(dy))) + G
        return dy, expect, expectG, ForceRebuild

    def align(self):
        self.M.align() 
        self.Structures = [Structure(self.M[i], self.engine, os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i), self.coordtype) for i in range(len(self))]
        self.clearCalcs()
    
    def TakeStep_old(self, dy, verbose=False):
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
        for n in range(1, len(self)-1):
            S = self.Structures[n]
            dy_i = dy[currvar:currvar+S.nICs]
            Xnew = S.IC.newCartesian(S.cartesians, dy_i, verbose=verbose)
            Cnew.M.xyzs[n] = Xnew.reshape(-1, 3)*bohr2ang
            Cnew.Structures[n].cartesians = Xnew
            currvar += S.nICs
        Cnew.clearCalcs()
        Cnew.haveMetric = True
        return Cnew

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
        Xnew = self.GlobalIC.newCartesian(self.get_cartesian_all(endpts=True), dy, verbose=verbose)
        Xnew = Xnew.reshape(-1, 3*self.na)
        for n in range(1, len(self)-1):
            if not self.locks[n]:
                Cnew.M.xyzs[n] = Xnew[n].reshape(-1, 3)*bohr2ang
                Cnew.Structures[n].cartesians = Xnew[n]
        Cnew.clearCalcs(clearEngine=False)
        Cnew.haveMetric = True
        if printStep:
            Xold = self.get_cartesian_all(endpts=True)
            if hasattr(Cnew.GlobalIC, 'Prims'):
                plist = Cnew.GlobalIC.Prims.Internals
                prim = Cnew.GlobalIC.Prims
            else:
                plist = Cnew.GlobalIC.Internals
                prim = Cnew.GlobalIC
            icdiff = prim.calcDiff(Xnew, Xold)
            sorter = icdiff**2
            if hasattr(Cnew.GlobalIC, 'Prims'):
                dsort = np.argsort(dy**2)[::-1]
                print("Largest components of step (%i DLC total):" % len(dsort))
                print(' '.join(["%6i" % d for d in dsort[:10]]))
                print(' '.join(["%6.3f" % dy[d] for d in dsort[:10]]))
                for d in dsort[:3]:
                    print("Largest components of DLC %i (coeff % .3f):" % (d, dy[d]))
                    for i in np.argsort(np.array(self.GlobalIC.Vecs[:, d]).flatten()**2)[::-1][:5]:
                        p = plist[i]
                        print("%40s % .4f" % (p, self.GlobalIC.Vecs[i, d]))
            print("Largest Displacements:")
            for i in np.argsort(sorter)[::-1][:10]:
                p = plist[i]
                print("%40s % .3e % .3e % .3e" % (p, p.value(Xnew), p.value(Xold), icdiff[i]))
        return Cnew

    def respace(self, thresh):
        """
        Space-out NEB images that have merged.
        """
        respaced = False
        OldSpac = ' '.join(["%6.3f " % i for i in self.calc_spacings()])
        merge_images = []
        for i, spac in enumerate(self.calc_spacings()):
            if spac < thresh:
                merge_images.append(i)
                merge_images.append(i+1)
                if i > 0: merge_images.append(i-1)
                if i < len(self) - 2: merge_images.append(i+2)
        merge_images = sorted(list(set(merge_images)))
        in_segment = False
        merge_segments = []
        for i, im in enumerate(merge_images):
            if not in_segment:
                merge_left = im
                in_segment = True
            if in_segment:
                if im+1 in merge_images: continue
                merge_right = im
                in_segment = False
                merge_segments.append((merge_left, merge_right))
        for s in merge_segments:
            Mspac = deepcopy(self.Structures[s[0]].M)
            for i in range(s[0]+1, s[1]+1):
                Mspac.xyzs += self.Structures[i].M.xyzs
            Mspac_eq = EqualSpacing(Mspac, frames=len(Mspac), RMSD=True, align=False)
            for i in range(len(Mspac_eq)):
                self.Structures[s[0]+i] = Structure(Mspac_eq[i], self.engine, os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % (s[0]+i)), self.coordtype)
            print("Respaced images %s" % (list(range(s[0], s[1]+1))))
        if len(merge_segments) > 0:
            respaced = True
            self.clearCalcs(clearEngine=False)
            print("Image Number          :", ' '.join(["  %3i  " % i for i in range(len(self))]))
            print("Spacing (Ang)     Old :", " "*4, OldSpac)
            print("                  New :", " "*4, ' '.join(["%6.3f " % i for i in self.calc_spacings()]))
        return respaced

    def delete_insert(self, thresh):
        """
        Second algorithm for deleting images and inserting new ones.
        """
        respaced = False
        OldSpac = ' '.join(["%6.3f " % i for i in self.calc_spacings()])

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
            if thresh*spac_del < spac_ins:
                thresh = 1.0
                # The space (j) -- (j+1) is greater than (i) -- (i+2) times a threshold
                xyzs = [self.Structures[i].M.xyzs[0].copy() for i in range(len(self))]
                deli = left_del + 1
                insi = left_ins
                insj = left_ins + 1
                xavg = 0.5*(xyzs[insi]+xyzs[insj])
                xyzs.insert(insj, xavg)
                if insj > deli:
                    del xyzs[deli]
                else:
                    del xyzs[deli+1]
                for i in range(len(self)):
                    Mtmp = deepcopy(self.Structures[0].M)
                    Mtmp.xyzs = [xyzs[i]]
                    self.Structures[i] = Structure(Mtmp, self.engine, os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i), self.coordtype)
                print("Evening out spacings: Deleted image %2i and added a new image between %2i and %2i" % (deli, insi, insj))
                respaced = True
            else:
                break
            nloop += 1
            if nloop > len(self):
                raise RuntimeError("Stuck in a loop, bug likely!")
        if respaced:
            self.clearCalcs(clearEngine=False)
            print("Image Number          :", ' '.join(["  %3i  " % i for i in range(len(self))]))
            print("Spacing (Ang)     Old :", " "*4, OldSpac)
            print("                  New :", " "*4, ' '.join(["%6.3f " % i for i in self.calc_spacings()]))
        return respaced

    def SaveToDisk(self, fout='chain.xyz'):
        # Mout should be garbage-collected, right?
        Mout = deepcopy(self.Structures[0].M)
        for i in range(1, len(self)):
            Mout.xyzs += self.Structures[i].M.xyzs
        enes = np.array([s.energy for s in self.Structures])
        eneKcal = au2kcal*(enes - np.min(enes))
        # enes -= np.min(enes)
        # enes *= au2kcal
        Mout.comms = ["Image %i/%i, Energy = % 16.10f (%+.3f kcal/mol)" % (i, len(enes), enes[i], eneKcal[i]) for i in range(len(enes))]
        Mout.write(os.path.join(self.tmpdir, fout))


class ElasticBand(Chain):
    def __init__(self, molecule, engine, tmpdir, coordtype, params, coords=None, plain=False, ic_displace=False):
        super(ElasticBand, self).__init__(molecule, engine, tmpdir, coordtype, params, coords=coords, ic_displace=ic_displace)
        self.k = self.params.nebk*kcal2au/ang2bohr ** 2
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

    def RebuildIC(self, coordtype):
        Cnew = ElasticBand(self.M, self.engine, self.tmpdir, coordtype, self.params, None, plain=self.plain, ic_displace=self.ic_displace)
        Cnew.ComputeChain()
        return Cnew

    def set_tangent(self, i, value):
        if i < 1 or i > (len(self)-2):
            raise RuntimeError('Tangents are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)')
        self._tangents[i] = value

    def get_tangent(self, i):
        if i < 1 or i > (len(self)-2):
            raise RuntimeError('Tangents are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)')
        return self._tangents[i]

    def get_tangent_all(self):
        return np.hstack(tuple(self._tangents[1:len(self)-1]))

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
        if i < 1 or i > (len(self)-2):
            raise RuntimeError("Spring_Grads are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)")
        if value.shape[0] != self.Structures[i].nICs:
            raise RuntimeError("Dimensions of array being passed are wrong (%i ICs expected for image %i)" % (self.Structures[i].nICs, i))
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError("Please set the component argument to potential, spring, or total")
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError("Please set the projection argument to plain, projected, or working")
        self._grads.setdefault((component, projection), [None for j in range(len(self))])[i] = value.copy()
        self._grads[(component, projection)][i].flags.writeable = False

    def set_global_grad(self, value, component, projection):
        if value.ndim != 1:
            raise RuntimeError("Please pass a 1D array")
        if value.shape[0] != len(self.GlobalIC.Internals):
            raise RuntimeError("Dimensions of array being passed are wrong (%i ICs expected)" % (len(self.GlobalIC.Internals)))
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError("Please set the component argument to potential, spring, or total")
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError("Please set the projection argument to plain, projected, or working")
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
        if i < 1 or i > (len(self)-2):
            raise RuntimeError("Spring_Grads are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)")
        if value.shape[0] != self.Structures[i].nICs:
            raise RuntimeError("Dimensions of array being passed are wrong (%i ICs expected for image %i)" % (self.Structures[i].nICs, i))
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError("Please set the component argument to potential, spring, or total")
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError("Please set the projection argument to plain, projected, or working")
        if (component, projection) not in self._grads or self._grads[(component, projection)][i] is None:
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
        if i < 1 or i > (len(self)-2):
            raise RuntimeError('Spring_Grads are only defined for 1 .. N-2 (in a chain indexed from 0 .. N-1)')
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError("Please set the component argument to potential, spring, or total")
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError("Please set the projection argument to plain, projected, or working")
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
                    return self.get_grad(i, "potential", "plain") + self.get_grad(i, "spring", "plain")
                elif self.plain == 1:
                    return self.get_grad(i, "potential", "projected") + self.get_grad(i, "spring", "plain")
                elif self.plain == 0:
                    return self.get_grad(i, "potential", "projected") + self.get_grad(i, "spring", "projected")
        elif component == "total":
            return self.get_grad(i, "potential", projection) + self.get_grad(i, "spring", projection)
        if (component, projection) not in self._grads or self._grads[(component, projection)][i] is None:
            raise RuntimeError("Gradient has not been set")
        # print "Getting gradient for image", i, component, projection
        # LPW 2017-04-08: Removed copy operation, hope flags.writeable = False prevents unwanted edits
        return self._grads[(component, projection)][i]

    def get_global_grad(self, component, projection):
        if component not in ["potential", "spring", "total"]:
            raise RuntimeError("Please set the component argument to potential, spring, or total")
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError("Please set the projection argument to plain, projected, or working")
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
                    return self.get_global_grad("potential", "plain") + self.get_global_grad("spring", "plain")
                elif self.plain == 1:
                    return self.get_global_grad("potential", "projected") + self.get_global_grad("spring", "plain")
                elif self.plain == 0:
                    return self.get_global_grad("potential", "projected") + self.get_global_grad("spring", "projected")
        elif component == "total":
            return self.get_global_grad("potential", projection) + self.get_global_grad("spring", projection)
        if (component, projection) not in self._global_grads or self._global_grads[(component, projection)] is None:
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
            raise RuntimeError("Please set the component argument to potential, spring, or total")
        if projection not in ["plain", "projected", "working"]:
            raise RuntimeError("Please set the projection argument to plain, projected, or working")
        return np.hstack(tuple([self.get_grad(i, component, projection) for i in range(1, len(self)-1)]))

    def PrintStatus_old(self):
        enes = np.array([s.energy for s in self.Structures])
        enes -= np.min(enes)
        enes *= au2kcal
        taus = [self.get_tangent(i) for i in range(1, len(self)-1)]
        # This is for IC-tangents, experimental
        # taus = [self.Structures[i].ConvertICGradToCart(self.get_tangent(i)) for i in range(1, len(self)-1)]
        taus = [t / np.linalg.norm(t) for t in taus]
        symbols = ["(min)"]
        for i in range(1, len(self)-1):
            if enes[i-1] < enes[i]:
                if enes[i] < enes[i+1]:
                    symbols.append("---->")
                else:
                    symbols.append("(max)")
            else:
                if enes[i] > enes[i+1]:
                    symbols.append("<----")
                else:
                    symbols.append("(min)")
        symbols.append("(min)")
        symcolors = []
        for i in range(len(symbols)):
            if self.locks[i]:
                symcolors.append(("\x1b[94m", "\x1b[0m"))
            else:
                symcolors.append(("",""))
        print("Image Number          :", ' '.join(["  %3i  " % i for i in range(len(self))]))
        print("                       ", ' '.join(["%s%7s%s" % (symcolors[i][0], s, symcolors[i][1]) for i, s in enumerate(symbols)]))
        print("Energies  (kcal/mol)  :", end=' ')
        print(' '.join(["%7.3f" % n for n in enes]))
        rmsds = []
        for i in range(1, len(self)):
            dispVector = self.Structures[i].cartesians - self.Structures[i-1].cartesians
            dispVector = dispVector.reshape(-1,3)
            dispVector = dispVector ** 2
            dispVector = np.sum(dispVector, axis=1)
            dispVector = np.mean(dispVector)
            dispVector = np.sqrt(dispVector)
            rmsds.append(dispVector)
        print("RMSD (Ang)            :", end=' ')
        print(" "*5, ' '.join(["%5.2f  " % i for i in rmsds]))
        # print "Angles                :",
        # print " "*13, ' '.join(["%5.2f  " % (180.0/np.pi*np.arccos(np.dot(taus[n], taus[n+1]))) for n in range(len(taus)-1)])
        totGrad = ([self.Structures[0].grad_cartesian] + 
                   [self.Structures[i].ConvertICGradToCart(self.get_grad(i, "total", "working")) for i in range(1, len(self)-1)] +
                   [self.Structures[-1].grad_cartesian])

        # LPW print out force components
        vGrad = ([self.Structures[0].grad_cartesian] + 
                 [self.Structures[i].ConvertICGradToCart(self.get_grad(i, "potential", "working")) for i in range(1, len(self)-1)] +
                 [self.Structures[-1].grad_cartesian])

        spGrad = ([self.Structures[0].grad_cartesian*0.0] +
                  [self.Structures[i].ConvertICGradToCart(self.get_grad(i, "spring", "working")) for i in range(1, len(self)-1)] +
                  [self.Structures[-1].grad_cartesian*0.0])

        # The average gradient in eV/Angstrom
        avgg = np.mean([rms_gradient(totGrad[n]) for n in range(1, len(totGrad)-1)])*au2evang
        print("Gradients (eV/Ang)    :", end=' ') # % avgg
        print(' '.join(["%7.3f" % (rms_gradient(totGrad[n])*au2evang) for n in range(len(totGrad))]))
        # print "Potential Grad        :", # % avgg
        # print ' '.join(["%7.3f" % (rms_gradient(vGrad[n])*au2evang) for n in range(len(totGrad))])
        # print "Spring Grad           :", # % avgg
        # print ' '.join(["%7.3f" % (rms_gradient(spGrad[n])*au2evang) for n in range(len(totGrad))])
        print("Band Energy           : %.10f(Pot) + %.10f(Spr) = %.10f(Tot)" % (self.PotBandEnergy, self.SprBandEnergy, self.TotBandEnergy))
        self.avgg = avgg
        printDiffs = True
        if not printDiffs: return
        for n in range(1, len(self)-1):
            print("Image", n)
            if hasattr(self.Structures[n].IC, 'Prims'):
                ICP = self.Structures[n].IC.Prims
            else:
                ICP = self.Structures[n].IC
            drplus = self.Structures[n].IC.calcDiff(self.Structures[n+1].cartesians, self.Structures[n].cartesians)
            drminus = self.Structures[n].IC.calcDiff(self.Structures[n-1].cartesians, self.Structures[n].cartesians)
            print("Largest IC devs (next - curr); norm % 8.4f" % np.linalg.norm(drplus))
            for i in np.argsort(np.abs(drplus))[::-1][:5]:
                print("%30s % 8.4f" % (repr(ICP.Internals[i]), (drplus)[i]))
            print("Largest IC devs (prev - curr); norm % 8.4f" % np.linalg.norm(drminus))
            for i in np.argsort(np.abs(drminus))[::-1][:5]:
                print("%30s % 8.4f" % (repr(ICP.Internals[i]), (drminus)[i]))

    def calc_spacings(self):
        rmsds = []
        for i in range(1, len(self)):
            rmsd, maxd = calc_drms_dmax(self.Structures[i].cartesians, self.Structures[i-1].cartesians, align=False)
            rmsds.append(rmsd)
        return rmsds

    def calc_straightness(self, xyz0, analyze=False):
        xyz = xyz0.reshape(len(self), -1)
        xyz.flags.writeable = False
        straight = [1.0]
        for n in range(1, len(self)-1):
            if self.ic_displace:
                drplus = self.GlobalIC.calcDisplacement(xyz, n+1, n)
                drminus = self.GlobalIC.calcDisplacement(xyz, n-1, n)
                if analyze:
                    ndrplus = drplus / np.linalg.norm(drplus) 
                    ndrminus = drminus / np.linalg.norm(drminus)
                    vsum = ndrplus+ndrminus
                    dsum = drplus+drminus
                    dsort = np.argsort(vsum**2)[::-1]
                    if hasattr(self.GlobalIC,'Prims'):
                        plist = self.GlobalIC.Prims.ImageICs[n].Internals
                    else:
                        plist = self.GlobalIC.ImageICs[n].Internals
                    print("Image %i Kink:" % n)
                    for d in dsort[:5]:
                        print("%40s % .4f" % (plist[d], dsum[d]))
            else:
                drplus = xyz[n+1]-xyz[n]
                drminus = xyz[n-1]-xyz[n]
            drplus /= np.linalg.norm(drplus)
            drminus /= np.linalg.norm(drminus)
            straight.append(np.dot(drplus, -drminus))
        straight.append(1.0)
        return straight

    def SaveClimbingImages(self, cycle):
        if not self.climbSet: return
        enes = np.array([s.energy for s in self.Structures])
        eneKcal = au2kcal*(enes - np.min(enes))
        M = None
        for i, n in enumerate(self.climbers):
            if M is None:
                M = deepcopy(self.Structures[n].M)
            else:
                M += self.Structures[n].M
            grms = rms_gradient(self.Structures[n].grad_cartesian)*au2evang
            M.comms[i] = "Climbing Image - Chain %i Image %i Energy % 16.10f (%+.3f kcal/mol) RMSGrad %.3f eV/Ang" % (cycle, n, enes[n], eneKcal[n], grms)

        if self.params.prefix == None:
            #M.write("chains.tsClimb.xyz")
            M.write('.'.join(self.tmpdir.split('.')[:-1]) + ".tsClimb.xyz")
        else:
            M.write(self.params.prefix+".tsClimb.xyz")
    
    def PrintStatus(self):
        enes = np.array([s.energy for s in self.Structures])
        enes -= np.min(enes)
        enes *= au2kcal
        symbols = ["(min)"]
        for i in range(1, len(self)-1):
            if enes[i-1] == enes[i]:
                if enes[i] == enes[i+1]:
                    symbols.append("  =  ") # This may be used when all energies are zero
                elif enes[i] < enes[i+1]:
                    symbols.append("= -->") # This symbol should pretty much never be used
                else:
                    symbols.append("= <--") # This symbol should pretty much never be used
            elif enes[i-1] < enes[i]:
                if enes[i] == enes[i+1]:
                    symbols.append("--> =") # This symbol should pretty much never be used
                elif enes[i] < enes[i+1]:
                    symbols.append("---->")
                else:
                    if self.climbSet and i in self.climbers:
                        symbols.append("(^_^)")
                    else:
                        symbols.append("(max)")
            else:
                if enes[i] == enes[i+1]: 
                    symbols.append("<-- =") # This symbol should pretty much never be used
                elif enes[i] > enes[i+1]:
                    symbols.append("<----")
                else:
                    symbols.append("(min)")
        symbols.append("(min)")
        symcolors = []
        for i in range(len(symbols)):
            if self.locks[i]:
                symcolors.append(("\x1b[94m", "\x1b[0m"))
            else:
                symcolors.append(("",""))
        print("Image Number          :", ' '.join(["  %3i  " % i for i in range(len(self))]))
        print("                       ", ' '.join(["%s%7s%s" % (symcolors[i][0], s, symcolors[i][1]) for i, s in enumerate(symbols)]))
        print("Energies  (kcal/mol)  :", end=' ')
        print(' '.join(["%7.3f" % n for n in enes]))
        print("Spacing (Ang)         :", end=' ')
        print(" "*4, ' '.join(["%6.3f " % i for i in self.calc_spacings()]))
        # print "Angles                :",
        # print " "*13, ' '.join(["%5.2f  " % (180.0/np.pi*np.arccos(np.dot(taus[n], taus[n+1]))) for n in range(len(taus)-1)])

        xyz = self.get_cartesian_all(endpts=True)
        Bmat = self.GlobalIC.wilsonB(xyz)

        def GetCartesianGradient(component, projection):
            answer = np.dot(np.array(Bmat.T), np.array(self.get_global_grad(component, projection)).T).flatten()
            answer = answer.reshape(len(self), -1)
            if component in ['total', 'potential']:
                answer[0] = self.Structures[0].grad_cartesian
                answer[-1] = self.Structures[-1].grad_cartesian
            return answer

        totGrad = GetCartesianGradient("total", "working")
        vGrad = GetCartesianGradient("potential", "working")
        spGrad = GetCartesianGradient("spring", "working")
        straight = self.calc_straightness(xyz)#, analyze=True)

        # The average gradient in eV/Angstrom
        avgg = np.mean([rms_gradient(totGrad[n]) for n in range(1, len(totGrad)-1)])*au2evang
        maxg = np.max([rms_gradient(totGrad[n]) for n in range(1, len(totGrad)-1)])*au2evang
        print("Gradients (eV/Ang)    :", end=' ') # % avgg
        print(' '.join(["%7.3f" % (rms_gradient(totGrad[n])*au2evang) for n in range(len(totGrad))]))
        print("Straightness          :", end=' ') # % avgg
        print(' '.join(["%7.3f" % (straight[n]) for n in range(len(totGrad))]))
        self.avgg = avgg
        self.maxg = maxg
        printDiffs = False
        if not printDiffs: return
        for n in range(1, len(self)-1):
            ICP = self.GlobalIC.ImageICs[n].Internals
            drplus = self.GlobalIC.calcDisplacement(xyz, n+1, n)
            drminus = self.GlobalIC.calcDisplacement(xyz, n-1, n)
            print("Largest IC devs (%i - %i); norm % 8.4f" % (n+1, n, np.linalg.norm(drplus)))
            for i in np.argsort(np.abs(drplus))[::-1][:5]:
                print("%30s % 8.4f" % (repr(ICP[i]), (drplus)[i]))
            print("Largest IC devs (%i - %i); norm % 8.4f" % (n-1, n, np.linalg.norm(drminus)))
            for i in np.argsort(np.abs(drminus))[::-1][:5]:
                print("%30s % 8.4f" % (repr(ICP[i]), (drminus)[i]))

    def CalcRMSCartGrad(self, igrad):
        xyz = self.get_cartesian_all(endpts=True)
        Bmat = self.GlobalIC.wilsonB(xyz)
        cGrad = np.dot(np.array(Bmat.T), np.array(igrad).T).reshape(len(self), -1)
        # The average gradient in eV/Angstrom
        avgg = np.mean([rms_gradient(cGrad[n]) for n in range(1, len(cGrad)-1)])*au2evang
        return avgg
                
    def SetLocks_old(self, gtol):
        """ Set locks on images that are below the convergence tolerance 'gtol' and are connected to the ends through locked images. """
        totGrad = ([self.Structures[0].grad_cartesian] + 
                   [self.Structures[i].ConvertICGradToCart(self.get_grad(i, "total", "working")) for i in range(1, len(self)-1)] +
                   [self.Structures[-1].grad_cartesian])
        rmsGrad = [rms_gradient(totGrad[n])*au2evang for n in range(len(totGrad))]
        self.locks = [True] + [False for n in range(1, len(self)-1)] + [True]
        while True:
            newLocks = self.locks[:]
            for n in range(1, len(self)):
                if rmsGrad[n] < gtol:
                    if all(self.locks[:n]):
                        newLocks[n] = True
                    if all(self.locks[n+1:]):
                        newLocks[n] = True
            if newLocks == self.locks: break
            # print "oldLocks:", self.locks
            # print "newLocks:", newLocks
            self.locks = newLocks[:]

    def SetFlags(self):
        """ Set locks on images that are below the convergence tolerance 'gtol' and are connected to the ends through locked images. """
        xyz = self.get_cartesian_all(endpts=True)
        Bmat = self.GlobalIC.wilsonB(xyz)

        def GetCartesianGradient(component, projection):
            answer = np.dot(np.array(Bmat.T), np.array(self.get_global_grad(component, projection)).T).flatten()
            return answer.reshape(len(self), -1)

        totGrad = GetCartesianGradient("total", "working")

        rmsGrad = [rms_gradient(totGrad[n])*au2evang for n in range(len(totGrad))]

        avgg = np.mean(rmsGrad[1:-1])
        maxg = np.max(rmsGrad[1:-1])

        recompute = False
        if maxg < self.params.climb or self.climbSet:
            enes = np.array([s.energy for s in self.Structures])
            enes -= np.min(enes)
            enes *= au2kcal
            climbers = []
            cenes = []
            for i in range(1, len(self)-1):
                if enes[i-1] < enes[i] and enes[i] > enes[i+1]:
                    # Image "i" is a local maximum / climbing image
                    climbers.append(i)
                    cenes.append(enes[i])
            # Keep up to "ncimg" climbing images
            climbers = sorted(list(np.array(climbers)[np.argsort(np.array(cenes))[::-1]][:self.params.ncimg]))
            if self.climbSet and climbers != self.climbers:
                recompute = True
            if len(climbers) > 0:
                # Note: Climbers may turn on or off over the course of the optimization - look out
                if not self.climbSet or climbers != self.climbers:
                    recompute = True
                    print("--== Images set to Climbing Mode: %s ==--" % (','.join([str(i) for i in climbers])))
                self.climbSet = True
                self.climbers = climbers
            # Not sure if the Hessian also needs to be affected for the climbing image

        # Because the gradient depends on the neighboring images, a locked image may get unlocked if a neighboring image moves.
        # This is a bit problematic and we should revisit it in the future.
        new_scheme = True
        if new_scheme:
            newLocks = [True] + [False for n in range(1, len(self)-1)] + [True] #self.locks[:]
            for n in range(1, len(self)):
                if rmsGrad[n] < self.params.avgg:
                    newLocks[n] = True
            self.locks = newLocks[:]
        else:
            self.locks = [True] + [False for n in range(1, len(self)-1)] + [True]
            while True:
                newLocks = self.locks[:]
                for n in range(1, len(self)):
                    if rmsGrad[n] < self.params.avgg:
                        if all(self.locks[:n]):
                            newLocks[n] = True
                        if all(self.locks[n+1:]):
                            newLocks[n] = True
                if newLocks == self.locks: break
                # print "oldLocks:", self.locks
                # print "newLocks:", newLocks
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
        for n in range(1, len(self)-1):
            if self.ic_displace:
                drplus = self.GlobalIC.calcDisplacement(xyz, n+1, n)
                drminus = -self.GlobalIC.calcDisplacement(xyz, n-1, n)
            else:
                cc_next = self.Structures[n+1].cartesians
                cc_curr = self.Structures[n].cartesians
                cc_prev = self.Structures[n-1].cartesians
                drplus = cc_next - cc_curr
                drminus = cc_curr - cc_prev
            # Energy differences along the band
            dvplus = enes[n+1]-enes[n]
            dvminus = enes[n]-enes[n-1]
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
                if enes[n+1] > enes[n-1]:
                    # The energy of "next" exceeds "previous".
                    dr = drplus*absdvmax + drminus*absdvmin
                else:
                    # The energy of "previous" exceeds "next".
                    dr = drplus*absdvmin + drminus*absdvmax
            ndr = np.linalg.norm(dr)
            tau = dr / ndr 
            self.set_tangent(n, tau)

    def ComputeTangent_old(self):
        if not self.haveCalcs:
            raise RuntimeError("Calculate energies before tangents")
        enes = [s.energy for s in self.Structures]
        if self.ic_displace:
            grads = [s.grad_internal for s in self.Structures]
        else:
            grads = [s.grad_cartesian for s in self.Structures]
        xyz = self.get_cartesian_all(endpts=True)
        for n in range(1, len(self)-1):
            if self.ic_displace:
                drplus = self.GlobalIC.calcDisplacement(xyz, n+1, n)
                drminus = -self.GlobalIC.calcDisplacement(xyz, n-1, n)
            else:
                cc_next = self.Structures[n+1].cartesians
                cc_curr = self.Structures[n].cartesians
                cc_prev = self.Structures[n-1].cartesians
                drplus = cc_next - cc_curr
                drminus = cc_curr - cc_prev

            ndrplus = drplus / np.linalg.norm(drplus) 
            ndrminus = drminus / np.linalg.norm(drminus)
            tau /= np.linalg.norm(ndrplus+ndrminus)

            self.set_tangent(n, tau)

    def ComputeBandEnergy(self):
        self.SprBandEnergy = 0.0
        self.PotBandEnergy = 0.0
        self.TotBandEnergy = 0.0
        xyz = self.get_cartesian_all(endpts=True)
        energies = np.array([self.Structures[n].energy for n in range(len(self))])

        for n in range(1, len(self)-1):
            if self.ic_displace:
                drplus = self.GlobalIC.calcDisplacement(xyz, n+1, n)
                drminus = self.GlobalIC.calcDisplacement(xyz, n-1, n)
            else:
                cc_next = self.Structures[n+1].cartesians
                cc_curr = self.Structures[n].cartesians
                cc_prev = self.Structures[n-1].cartesians
                ndrplus = np.linalg.norm(cc_next - cc_curr)
                ndrminus = np.linalg.norm(cc_curr - cc_prev)
            # The spring constant connecting each pair of images
            # should have the same strength.  This rather confusing "if"
            # statement ensures the spring constant is not incorrectly
            # doubled for non-end springs
            fplus = 0.5 if n == (len(self)-2) else 0.25
            fminus = 0.5 if n == 1 else 0.25
            if self.params.ew:
                #HP_ew 
                E_i = energies[n]
                E_ref = min(energies[0], energies[-1]) # Reference energy can be either reactant or product. Lower energy is picked here.
                E_max = max(energies)
                k_max = self.k*2
                k_min = self.k/2
                a = (E_max - energies[n])/(E_max - E_ref)
                if E_i > E_ref:
                    k_new = (1-a)*k_max + a*k_min
                else:
                    k_new = k_min 
            else:
                k_new = self.k
            if self.ic_displace:
                self.SprBandEnergy += fplus*k_new*np.dot(drplus, drplus)
                self.SprBandEnergy += fminus*k_new*np.dot(drminus, drminus)
            else:
                self.SprBandEnergy += fplus*k_new*ndrplus ** 2
                self.SprBandEnergy += fminus*k_new*ndrminus ** 2
            self.PotBandEnergy += self.Structures[n].energy
        self.TotBandEnergy = self.SprBandEnergy + self.PotBandEnergy

    def ComputeProjectedGrad(self):
        if self.ic_displace:
            self.ComputeProjectedGrad_IC()
        else:
            self.ComputeProjectedGrad_CC()
        

    def ComputeProjectedGrad_IC_old(self):
        self._grads = OrderedDict()
        force_s_Carts = [np.zeros(3*self.na, dtype=float) for n in range(len(self))]
        for n in range(1, len(self)-1):
            tau = self.get_tangent(n)
            # Force from the potential
            fplus = 1.0 if n == (len(self)-2) else 0.5
            fminus = 1.0 if n == 1 else 0.5
            force_v = -self.Structures[n].grad_internal
            drplus = self.Structures[n].IC.calcDiff(self.Structures[n+1].cartesians, self.Structures[n].cartesians)
            drminus = self.Structures[n].IC.calcDiff(self.Structures[n-1].cartesians, self.Structures[n].cartesians)
            ndrplus = np.linalg.norm(drplus)
            ndrminus = np.linalg.norm(drminus)
            force_s_plus  = fplus*self.k*np.dot(self.kmats[n],drplus)
            force_s_minus = fminus*self.k*np.dot(self.kmats[n],drminus)
            force_s_plus_onNext_c = self.Structures[n].ConvertICGradToCart(-force_s_plus, xyz=self.Structures[n+1].cartesians)
            force_s_plus_onNext_i = self.Structures[n+1].ConvertCartGradToIC(force_s_plus_onNext_c)
            force_s_minus_onPrev_c = self.Structures[n].ConvertICGradToCart(-force_s_minus, xyz=self.Structures[n-1].cartesians)
            force_s_minus_onPrev_i = self.Structures[n-1].ConvertCartGradToIC(force_s_minus_onPrev_c)

            # Okay, this still needs some work.
            # Force from the spring in the tangent direction
            force_s_p = self.k*(ndrplus-ndrminus)*tau
            # Now get the perpendicular component of the force from the potential
            force_v_p = force_v - np.dot(force_v,tau)*tau
            # We should be working with gradients
            grad_s_p = -1.0*force_s_p
            grad_v = -1.0*force_v
            grad_v_p = -1.0*force_v_p
            grad_s = -1.0*(force_s_plus+force_s_minus)
            grad_s_onNext = -1.0*(force_s_plus_onNext_i)
            grad_s_onPrev = -1.0*(force_s_minus_onPrev_i)
            self.add_grad(n, grad_v, "potential", "plain")
            self.add_grad(n, grad_v_p, "potential", "projected")
            self.add_grad(n, grad_s_p, "spring", "projected")
            self.add_grad(n, grad_s, "spring", "plain")
            if n > 1:
                self.add_grad(n-1, grad_s_onPrev, "spring", "plain")
            if n < len(self)-2:
                self.add_grad(n+1, grad_s_onNext, "spring", "plain")

    def ComputeProjectedGrad_IC(self):
        xyz = self.get_cartesian_all(endpts=True).reshape(len(self),-1)
        grad_v_c = np.array([self.Structures[n].grad_cartesian for n in range(len(self))])
        energies = np.array([self.Structures[n].energy for n in range(len(self))])
        print("energies in Projected Grad",energies)
        grad_v_i = self.GlobalIC.calcGrad(xyz, grad_v_c.flatten())
        grad_v_p_c = np.zeros_like(grad_v_c)
        force_s_c = np.zeros_like(grad_v_c)
        force_s_p_c = np.zeros_like(grad_v_c)
        straight = self.calc_straightness(xyz)

        for n in range(1, len(self)-1):
            fplus = 1.0 if n == (len(self)-2) else 0.5
            fminus = 1.0 if n == 1 else 0.5
            drplus = self.GlobalIC.calcDisplacement(xyz, n+1, n)
            drminus = self.GlobalIC.calcDisplacement(xyz, n-1, n)
            if self.params.ew:
                #HP_ew 
                E_i = energies[n]
                E_ref = min(energies[0], energies[-1]) # Reference energy can be either reactant or product. Lower energy is picked here.
                E_max = max(energies)
                k_max = self.k*2
                k_min = self.k/2
                a = (E_max - energies[n])/(E_max - E_ref)
                if E_i > E_ref:
                    k_new = (1-a)*k_max + a*k_min
                else:
                    k_new = k_min 
            else:
                k_new = self.k
            force_s_Plus  = fplus*k_new*drplus
            force_s_Minus = fminus*k_new*drminus
            factor = 1.0 + 16*(1.0-straight[n])**2
            force_s_c[n] += self.GlobalIC.applyCartesianGrad(xyz, factor*(force_s_Plus+force_s_Minus), n, n)
            tau = self.get_tangent(n)
            # Force from the spring in the tangent direction
            ndrplus = np.linalg.norm(drplus)
            ndrminus = np.linalg.norm(drminus)
            force_s_p_c[n] = self.GlobalIC.applyCartesianGrad(xyz, k_new*(ndrplus-ndrminus)*tau, n, n)
            # Now get the perpendicular component of the force from the potential
            grad_v_im = self.GlobalIC.ImageICs[n].calcGrad(xyz[n], grad_v_c[n])
            grad_v_p_c[n] = self.GlobalIC.applyCartesianGrad(xyz, grad_v_im-np.dot(grad_v_im,tau)*tau, n, n)

            if self.climbSet and n in self.climbers:
                # The force in the direction of the tangent is reversed
                grad_v_p_c[n] = self.GlobalIC.applyCartesianGrad(xyz, grad_v_im-2*np.dot(grad_v_im,tau)*tau, n, n)

            if n > 1:
                force_s_c[n-1] -= self.GlobalIC.applyCartesianGrad(xyz, force_s_Minus, n-1, n)
            if n < len(self)-2:
                force_s_c[n+1] -= self.GlobalIC.applyCartesianGrad(xyz, force_s_Plus, n+1, n)

        for n in range(1, len(self)-1):
            if self.climbSet and n in self.climbers:
                # The climbing image feels no spring forces at all,
                # Note: We make the choice to change both the plain and the 
                # projected spring force.
                force_s_c[n] *= 0.0
                force_s_p_c[n] *= 0.0

        xyz = self.get_cartesian_all(endpts=True).reshape(len(self),-1)
        grad_v_c = np.array([self.Structures[n].grad_cartesian for n in range(len(self))])
        grad_v_i = self.GlobalIC.calcGrad(xyz, grad_v_c.flatten())
        grad_s_i = self.GlobalIC.calcGrad(xyz, -force_s_c.flatten())
        grad_v_p_i = self.GlobalIC.calcGrad(xyz, grad_v_p_c.flatten())
        grad_s_p_i = self.GlobalIC.calcGrad(xyz, -force_s_p_c.flatten())

        self.set_global_grad(grad_v_i, "potential", "plain")
        self.set_global_grad(grad_v_p_i, "potential", "projected")
        self.set_global_grad(grad_s_i, "spring", "plain")
        self.set_global_grad(grad_s_p_i, "spring", "projected")

    def ComputeProjectedGrad_CC(self):
        xyz = self.get_cartesian_all(endpts=True).reshape(len(self),-1)
        grad_v_c = np.array([self.Structures[n].grad_cartesian for n in range(len(self))])
        grad_v_i = self.GlobalIC.calcGrad(xyz, grad_v_c.flatten())
        grad_v_p_c = np.zeros_like(grad_v_c)
        energies = np.array([self.Structures[n].energy for n in range(len(self))])
        force_s_c = np.zeros_like(grad_v_c)
        force_s_p_c = np.zeros_like(grad_v_c)
        straight = self.calc_straightness(xyz)
        factor_out = "Factors:"
        for n in range(1, len(self)-1):
            tau = self.get_tangent(n)
            # Force from the potential
            fplus = 1.0 if n == (len(self)-2) else 0.5
            fminus = 1.0 if n == 1 else 0.5
            force_v = -self.Structures[n].grad_cartesian
            cc_next = self.Structures[n+1].cartesians
            cc_curr = self.Structures[n].cartesians
            cc_prev = self.Structures[n-1].cartesians
            ndrplus = np.linalg.norm(cc_next - cc_curr)
            ndrminus = np.linalg.norm(cc_curr - cc_prev)
            # Plain elastic band force

            if self.params.ew:
                #HP_ew 
                E_i = energies[n]
                E_ref = min(energies[0], energies[-1]) # Reference energy can be either reactant or product. Lower energy is picked here.
                E_max = max(energies)
                k_max = self.k*2
                k_min = self.k/2
                a = (E_max - energies[n])/(E_max - E_ref)
                if E_i > E_ref:
                    k_new = (1-a)*k_max + a*k_min
                else:
                    k_new = k_min 
            else:
                k_new = self.k

            force_s = k_new*(cc_prev + cc_next - 2*cc_curr)
            force_s_para = np.dot(force_s,tau)*tau
            force_s_ortho = force_s - force_s_para
            factor = 256*(1.0-straight[n])**4
            # Force from the spring in the tangent direction
            force_s_p = k_new*(ndrplus-ndrminus)*tau
            # Now get the perpendicular component of the force from the potential
            force_v_p = force_v - np.dot(force_v,tau)*tau
            if self.climbSet and n in self.climbers:
                # The climbing image feels no spring forces at all,
                # and the force in the direction of the tangent is reversed.
                # Note: We make the choice to change both the plain and the 
                # projected spring force.
                force_v_p = force_v - 2*np.dot(force_v,tau)*tau
                force_s_p *= 0.0
                force_s   *= 0.0
            # We should be working with gradients
            grad_s = -1.0*force_s
            grad_s_p = -1.0*force_s_p
            grad_v = -1.0*force_v
            grad_v_p = -1.0*force_v_p
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

    def FiniteDifferenceTest_old(self):
        self.ComputeChain()
        E = self.TotBandEnergy
        G = self.get_grad_all("total", "plain")
        ivar = 0
        h = 1e-6
        for n in range(1, len(self)-1):
            S = self.Structures[n]
            for iic in range(S.nICs):
                dy = np.zeros(self.nvars, dtype=float)
                dy[ivar] += h
                cplus = self.TakeStep(dy, verbose=False)
                cplus.ComputeChain(order=0)
                eplus = cplus.TotBandEnergy
                dy[ivar] -= 2*h
                cminus = self.TakeStep(dy, verbose=False)
                cminus.ComputeChain(order=0)
                eminus = cminus.TotBandEnergy
                fdiff = (eplus-eminus)/(2*h)
                print("\r%30s%5i : % .6e % .6e % .6e                   " % (repr(S.IC.Internals[iic]), ivar, G[ivar], fdiff, G[ivar]-fdiff))
                ivar += 1

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
            dy[i] -= 2*h
            cminus = self.TakeStep(dy, verbose=False)
            cminus.ComputeChain(order=0)
            eminus = cminus.TotBandEnergy
            fdiff = (eplus-eminus)/(2*h)
            print("\r%30s%5i : % .6e % .6e % .6e                   " % (repr(self.GlobalIC.Internals[i]), i, G[i], fdiff, G[i]-fdiff))

    def ComputeSpringHessian_old(self):
        # Compute both the plain and projected spring Hessian.
        self.spring_hessian_plain = np.zeros((self.nvars, self.nvars))
        self.spring_hessian_projected = np.zeros((self.nvars, self.nvars))
        currvar = 0
        h = 1e-5
        t0 = time.time()
        for n in range(1, len(self)-1):
            for j in range(self.Structures[n].nICs):
                self.Structures[n].internals[j] += h
                self.ComputeProjectedGrad()
                fPlus_plain = self.get_grad_all("spring", "plain")
                fPlus_projected = self.get_grad_all("spring", "projected")
                self.Structures[n].internals[j] -= 2*h
                self.ComputeProjectedGrad()
                fMinus_plain = self.get_grad_all("spring", "plain")
                fMinus_projected = self.get_grad_all("spring", "projected")
                self.Structures[n].internals[j] += h
                fDiff_plain = (fPlus_plain - fMinus_plain)/(2*h)
                fDiff_projected = (fPlus_projected - fMinus_projected)/(2*h) 
                self.spring_hessian_plain[:, currvar] = fDiff_plain
                self.spring_hessian_projected[:, currvar] = fDiff_projected
                currvar += 1
        # Reset gradient at the end.
        self.ComputeProjectedGrad()
        print("Spring Hessian took %.3f seconds" % (time.time() - t0))

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
            dy[i] -= 2*h
            cminus = self.TakeStep(dy, verbose=False)
            cminus.CopyEnergyGradient(self)
            cminus.ComputeTangent()
            cminus.ComputeProjectedGrad()
            gminus_plain = cminus.get_global_grad("spring", "plain")
            gminus_proj = cminus.get_global_grad("spring", "projected")
            self.spring_hessian_plain[i, :] = (gplus_plain-gminus_plain)/(2*h) 
            self.spring_hessian_projected[i, :] = (gplus_proj-gminus_proj)/(2*h)
        print("Spring Hessian took %.3f seconds" % (time.time() - t0))

    def ComputeGuessHessian_old(self):
        self.ComputeSpringHessian()
        guess_hessian_potential = np.zeros((self.nvars, self.nvars))
        currvar = 0
        for n in range(1, len(self)-1):
            guess_hessian_image = self.Structures[n].IC.guess_hessian(self.Structures[n].cartesians)
            nic = self.Structures[n].nICs
            guess_hessian_potential[currvar:currvar+nic, currvar:currvar+nic] = guess_hessian_image
            currvar += nic
        self.guess_hessian_plain = guess_hessian_potential + self.spring_hessian_plain
        self.guess_hessian_projected = guess_hessian_potential + self.spring_hessian_projected
        # Symmetrize
        self.guess_hessian_plain = 0.5 * (self.guess_hessian_plain + self.guess_hessian_plain.T)
        self.guess_hessian_projected = 0.5 * (self.guess_hessian_projected + self.guess_hessian_projected.T)
        self.guess_hessian_plain.flags.writeable = False
        self.guess_hessian_projected.flags.writeable = False
        # When plain is set to 1 or 2, we do not project out the perpendicular component of the spring force.
        if self.plain >= 1:
            self.guess_hessian_working = self.guess_hessian_plain
        else:
            self.guess_hessian_working = self.guess_hessian_projected

    def ComputeGuessHessian(self, full=False, blank=False):
        if full:
            # Compute both the plain and projected spring Hessian.
            self.guess_hessian_plain = np.zeros((self.nvars, self.nvars))
            self.guess_hessian_working = np.zeros((self.nvars, self.nvars))
            h = 1e-5
            t0 = time.time()
            for i in range(self.nvars):
                print("\rCoordinate %i/%i" % (i, self.nvars) + " "*100)
                dy = np.zeros(self.nvars, dtype=float)
                dy[i] += h
                cplus = self.TakeStep(dy, verbose=False)
                # An error will be thrown if grad_cartesian does not exist
                cplus.ComputeChain()
                gplus_plain = cplus.get_global_grad("total", "plain")
                gplus_work = cplus.get_global_grad("total", "working")
                dy[i] -= 2*h
                cminus = self.TakeStep(dy, verbose=False)
                cminus.ComputeChain()
                gminus_plain = cminus.get_global_grad("total", "plain")
                gminus_work = cminus.get_global_grad("total", "working")
                self.guess_hessian_plain[i, :] = (gplus_plain-gminus_plain)/(2*h) 
                self.guess_hessian_working[i, :] = (gplus_work-gminus_work)/(2*h)
            self.guess_hessian_plain = 0.5 * (self.guess_hessian_plain + self.guess_hessian_plain.T)
            self.guess_hessian_working = 0.5 * (self.guess_hessian_working + self.guess_hessian_working.T)
            print("Full Hessian took %.3f seconds" % (time.time() - t0))
        else:
            # self.ComputeSpringHessian()
            self.spring_hessian_plain = np.zeros((self.nvars, self.nvars), dtype=float)
            self.spring_hessian_projected = np.zeros((self.nvars, self.nvars), dtype=float)
            if not blank:
                guess_hessian_potential = self.GlobalIC.guess_hessian(self.get_cartesian_all(endpts=True), k=self.params.guessk)
            else:
            # guess_hessian_potential *= 0.0
                guess_hessian_potential = np.eye(self.spring_hessian_plain.shape[0]) * self.params.guessk
            self.guess_hessian_plain = guess_hessian_potential + self.spring_hessian_plain
            self.guess_hessian_projected = guess_hessian_potential + self.spring_hessian_projected
            # Symmetrize
            self.guess_hessian_plain = 0.5 * (self.guess_hessian_plain + self.guess_hessian_plain.T)
            self.guess_hessian_projected = 0.5 * (self.guess_hessian_projected + self.guess_hessian_projected.T)
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
            dy[i] -= 2*h
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

            self.guess_hessian_plain[i, :] = (gplus_plain-gminus_plain)/(2*h) 
            self.guess_hessian_projected[i, :] = (gplus_proj-gminus_proj)/(2*h)
            if self.plain == 0:
                self.guess_hessian_working[i, :] = (gplus_proj-gminus_proj)/(2*h)
            elif self.plain == 1:
                self.guess_hessian_working[i, :] = (vgrad_plus_proj+spgrad_plus_plain-vgrad_minus_proj-spgrad_minus_plain)/(2*h)
            elif self.plain == 2:
                self.guess_hessian_working[i, :] = (gplus_plain-gminus_plain)/(2*h)

        # Symmetrize
        self.guess_hessian_plain = 0.5 * (self.guess_hessian_plain + self.guess_hessian_plain.T)
        self.guess_hessian_projected = 0.5 * (self.guess_hessian_projected + self.guess_hessian_projected.T)
        self.guess_hessian_working = 0.5 * (self.guess_hessian_working + self.guess_hessian_working.T)

        print("Guess Hessian took %.3f seconds" % (time.time() - t0))

    def ComputeMetric(self):
        if self.haveMetric: return
        currvar = 0
        t0 = time.time()
        self.kmats = [None for i in range(len(self))]
        self.metrics = [None for i in range(len(self))]
        errs = []
        for n in range(1, len(self)-1):
            self.kmats[n] = np.eye(len(self.GlobalIC.ImageICs[n].Internals))
            self.metrics[n] = np.array(sqrtm(self.kmats[n]))
            errs.append(np.linalg.norm(np.abs(np.dot(self.metrics[n], self.metrics[n]) - self.kmats[n])))
        print("Metric completed in %.3f seconds, maxerr = %.3e" % (time.time() - t0, max(errs)))
        self.haveMetric = True

    def ComputeChain(self, order=1, cyc=None):
        if order >= 1:
            self.ComputeMetric()
        self.ComputeEnergyGradient(cyc=cyc)
        if order >= 1:
            self.ComputeTangent()
            self.ComputeProjectedGrad()
        if self.SetFlags():
            self.ComputeProjectedGrad()
        self.ComputeBandEnergy()

    def SaveToDisk(self, fout='chain.xyz'):
        super(ElasticBand, self).SaveToDisk(fout)

    def OptimizeEndpoints(self, gtol=None):
        self.Structures[0].OptimizeGeometry(gtol)
        self.Structures[-1].OptimizeGeometry(gtol)
        print("Optimizing End Points are done.")
        self.M.xyzs[0] = self.Structures[0].M.xyzs[0]
        self.M.xyzs[-1] = self.Structures[-1].M.xyzs[0]
        # The Structures are what store the individual Cartesian coordinates for each frame.
        self.Structures = [Structure(self.M[i], self.engine, os.path.join(self.tmpdir, "struct_%%0%ii" % len(str(len(self))) % i), self.coordtype) for i in range(len(self))]

def get_molecule_engine(args):
    # Read radii from the command line.
    if (len(args.radii) % 2) != 0:
        raise RuntimeError("Must have an even number of arguments for radii")
    nrad = int(len(args.radii)/2)
    radii = {}
    for i in range(nrad):
        radii[args.radii[2*i].capitalize()] = float(args.radii[2*i+1])
    # Create the Molecule object. The correct file to pass in depends on which engine is used,
    # so the command line interface could be improved at some point in the future
    if args.engine.lower() not in ['tera', 'psi4', 'qcengine', 'none', 'blank']:
        M = Molecule(args.input, radii=radii)
    elif args.coords is not None:
        M = Molecule(args.coords, radii=radii)[0]
    else:
        raise RuntimeError("With TeraChem/Psi4 engine, providing --coords is required.")
    # Read in the coordinates from the "--coords" command line option
    if args.coords is not None:
        Mxyz = Molecule(args.coords)
        if args.engine.lower() not in ['tera', 'psi4', 'qcengine', 'none', 'blank'] and M.elem != Mxyz.elem:
            raise RuntimeError("Atoms don't match for input file and coordinates file. Please add a single structure into the input")
        M.xyzs = Mxyz.xyzs
        M.comms = Mxyz.comms
        M.elem = Mxyz.elem
    # Select from the list of available engines
    if args.engine.lower() == 'qchem':
        Engine = QChem(M[0])
        Engine.set_nt(args.nt)
    elif args.engine.lower() == 'leps':
        if M.na != 3:
            raise RuntimeError("LEPS potential assumes three atoms")
        Engine = LEPS(M[0])
    elif args.engine.lower() == 'qcengine':
        Engine = QCEngineAPI(args.qcschema, args.qce_engine, args.client)   
    elif args.engine.lower() in ['none', 'blank']:
        Engine = Blank(M[0])
    else:
        # The Psi4 interface actually uses TeraChem input
        # LPW: This should be changed to match what Yudong has in optimize.py
        if args.engine.lower() == 'psi4':
            Psi4exe = shutil.which('psi4')
            if len(Psi4exe) == 0: raise RuntimeError("Please make sure psi4 executable is in your PATH")
        elif 'tera' in args.engine.lower():
            set_tcenv()
        else:
            raise RuntimeError('Engine not recognized or not specified (choose qchem, leps, terachem, psi4, reaxff)')
        tcin = load_tcin(args.input)
        M.charge = tcin['charge']
        M.mult = tcin['spinmult']
        if 'guess' in tcin:
            for f in tcin['guess'].split():
                if not os.path.exists(f):
                    raise RuntimeError("TeraChem input file specifies guess %s but it does not exist\nPlease include this file in the same folder as your input" % f)
        if len(args.tcguess) != 0:
            unrestricted = False
            if tcin['method'][0].lower() == 'u':
                unrestricted = True
            if unrestricted:
                if len(args.tcguess)%2 != 0:
                    raise RuntimeError("For unrestricted calcs, number of guess files must be an even number")
                guessfnms = list(itertools.chain(*[['ca%i' % (i+1), 'cb%i' % (i+1)] for i in range(len(args.tcguess)//2)]))
                if set(args.tcguess) != set(guessfnms):
                    raise RuntimeError("Please provide unrestricted guess files in pairs as ca1, cb1, ca2, cb2...")
            else:
                guessfnms = list(itertools.chain(*[['c%i' % (i+1)] for i in range(len(args.tcguess))]))
                if set(args.tcguess) != set(guessfnms):
                    raise RuntimeError("Please provide restricted guess files as c1, c2...")
            for f in guessfnms:
                if not os.path.exists(f):
                    raise RuntimeError("Please ensure alternative guess file %s exists in the current folder" % f)
            print("Running multiple SCF calculations using the following provided guess files: %s" % str(args.tcguess))
        else:
            guessfnms = []
        if args.engine.lower() == 'psi4':
            Engine = Psi4(M[0])
            psi4_in = "%s_2.in" %args.input.split(".")[0]
            if os.path.exists(psi4_in):
                os.remove(psi4_in)
            with open(psi4_in, 'a') as file_obj:
                file_obj.write("molecule {\n")
                file_obj.write("%i %i\n" %(M.charge, M.mult))
                coords = M.xyzs[0]
                for i, element in enumerate(M[0].elem):
                    if i == len(M[0].elem)-1:
                        file_obj.write("%-5s % 15.10f % 15.10f % 15.10f\n}\n" % (element, coords[i][0], coords[i][1], coords[i][2]))
                    else:
                        file_obj.write("%-5s % 15.10f % 15.10f % 15.10f\n" % (element, coords[i][0], coords[i][1], coords[i][2]))
                file_obj.write("\nset basis %s\n" %tcin['basis']) 
                file_obj.write("\ngradient(\'%s\')" %tcin['method'])
            
            Engine.load_psi4_input("%s_2.in" %args.input.split(".")[0])
        else:
            Engine = TeraChem(M[0], tcin)
            if args.nt != 1:
                raise RuntimeError("For TeraChem jobs, do not specify the number of threads; workers will decide which GPUs to run on, using CUDA_VISIBLE_DEVICES.")
    print("Input coordinates have %i frames. The following will be used to initialize NEB images:" % len(M))
    print(', '.join(["%i" % (int(round(i))) for i in np.linspace(0, len(M)-1, args.images)]))
    Msel = M[np.array([int(round(i)) for i in np.linspace(0, len(M)-1, args.images)])]
    return Msel, Engine

class ChainOptParams(object):
    """
    Container for optimization parameters.  
    The parameters used to be contained in the command-line "args", 
    but this was dropped in order to call Optimize() from another script.
    """
    def __init__(self, **kwargs):
        self.epsilon = kwargs.get('epsilon', 1e-5)
        self.check = kwargs.get('check', 0)
        self.verbose = kwargs.get('verbose', False)
        self.reset = kwargs.get('reset', False)
        self.trust = kwargs.get('trust', 0.1)
        self.tmax = kwargs.get('tmax', 0.3)
        self.maxg = kwargs.get('maxg', 0.05)
        self.avgg = kwargs.get('avgg', 0.025)
        self.align = kwargs.get('align', False)
        self.sepdir = kwargs.get('sepdir', False)
        self.ew = kwargs.get('ew', False)
        self.climb = kwargs.get('climb', 0.5)
        self.ncimg = kwargs.get('ncimg', 1)
        self.history = kwargs.get('history', 1)
        self.maxcyc = kwargs.get('maxcyc', 100)
        self.nebk = kwargs.get('nebk', 0.1)
        # Experimental feature that avoids resetting the Hessian
        self.skip = kwargs.get('skip', False)
        self.noopt = kwargs.get('noopt', False)
        # Experimental features (remind self later)
        self.guessk = kwargs.get('guessk', 0.05)
        self.guessw = kwargs.get('guessw', 0.1)
        self.prefix = kwargs.get('prefix', 'none')

def ChainRMSD(chain1, chain2):
    rmsds = []
    for n in range(1, len(chain1)-1):
        rmsd, maxd = calc_drms_dmax(chain1.Structures[n].cartesians, chain2.Structures[n].cartesians, align=False)
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
                dy, expect = trust_step(trial, v0, X, G, H, None, False, self.params.verbose)
                cnorm = chain.getCartesianNorm(dy, self.params.verbose)
                # Early "convergence"; this signals whether we have found a valid step that is
                # above the current target, but below the original trust radius. This happens
                # when the original trust radius fails, and we reduce the target step-length
                # as a contingency
                bork = chain.GlobalIC.bork
                # bork = any([chain.Structures[n].IC.bork for n in range(1, len(self.chain)-1)])
                self.from_above = (self.above_flag and not bork and cnorm < trust)
                self.stores[trial] = cnorm
                self.counter += 1
            # Store the largest trial value with cnorm below the target
            if cnorm-self.target < 0:
                if self.stored_val is None or cnorm > self.stored_val:
                    self.stored_arg = trial
                    self.stored_val = cnorm
            if self.params.verbose: print("dy(i): %.4f dy(c) -> target: %.4f -> %.4f%s" % (trial, cnorm, self.target, " (done)" if self.from_above else ""))
            return cnorm-self.target    

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
    for i in range(2, len(chain_hist)+1):
        rmsd = ChainRMSD(chain_hist[-i], chain_hist[-1])
        if rmsd > params.trust: break
        history += 1
    if history < 1:
        return
    print("Rebuilding Hessian using %i gradients" % history)

    y_seq = [c.get_internal_all() for c in chain_hist[-history-1:]]
    g_seq = [c.get_global_grad("total", projection) for c in chain_hist[-history-1:]]

    Yprev = y_seq[0]
    Gprev = g_seq[0]
    H0 = H.copy()
    for i in range(1, len(y_seq)):
        Y = y_seq[i]
        G = g_seq[i]
        Yprev = y_seq[i-1]
        Gprev = g_seq[i-1]
        Dy   = col(Y - Yprev)
        Dg   = col(G - Gprev)
        Mat1 = np.dot(Dg, Dg.T)/np.dot(Dg.T, Dy)[0,0]  
        Mat2 = np.dot(np.dot(np.array(H), Dy), np.dot(np.array(H), Dy).T)/np.dot((np.dot(Dy.T, np.array(H)),Dy))[0,0]
        print ("Mats in RebuildHessian", Mat1, Mat2)
        # Hstor = H.copy()
        H += Mat1-Mat2
    if np.min(np.linalg.eigh(H)[0]) < params.epsilon and params.reset:
        print("Eigenvalues below %.4e (%.4e) - returning guess" % (params.epsilon, np.min(np.linalg.eigh(H)[0])))
        H = H0.copy()

def recover(chain_hist, params, forceCart):
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
    newchain = chain_hist[-1].RebuildIC('cart' if forceCart else chain_hist[-1].coordtype)
    for ic, c in enumerate(chain_hist):
        # Copy operations here may allow old chains to be properly erased when dereferenced
        c.kmats = deepcopy(newchain.kmats)
        c.metrics = deepcopy(newchain.metrics)
        c.GlobalIC = deepcopy(newchain.GlobalIC)
        for i in range(len(newchain)):
            c.Structures[i].IC = deepcopy(newchain.Structures[i].IC)
        c.ComputeChain()
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
    if np.linalg.norm(Dg) < 1e-6 or np.linalg.norm(Dy) < 1e-6: return False
    Mat1 = np.dot(Dg,Dg.T)/np.dot(Dg.T,Dy)[0,0]
    Mat2 = np.dot(np.dot(H,Dy), np.dot(H,Dy).T)/np.dot(np.dot(Dy.T,H),Dy)[0,0]
    Eig = np.linalg.eigh(H)[0]
    Eig.sort()
    ndy = np.array(Dy).flatten()/np.linalg.norm(np.array(Dy))
    ndg = np.array(Dg).flatten()/np.linalg.norm(np.array(Dg))
    nhdy = np.array(H*Dy).flatten()/np.linalg.norm(np.array(H*Dy))
    if verbose: 
        print("Denoms: %.3e %.3e" % ((Dg.T*Dy)[0,0], (Dy.T*H*Dy)[0,0]), end=' ')
        print("Dots: %.3e %.3e" % (np.dot(ndg, ndy), np.dot(ndy, nhdy)), end=' ')
    H += Mat1-Mat2
    Eig1 = np.linalg.eigh(H)[0]
    Eig1.sort()
    if verbose:
        print("Eig-ratios: %.5e ... %.5e" % (np.min(Eig1)/np.min(Eig), np.max(Eig1)/np.max(Eig)))
    return Eig1
    # Then it's on to the next loop iteration!

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
    print("First, optimizing endpoint images.")
    chain.OptimizeEndpoints(params.maxg)
    print("Now optimizing the chain.")
    chain.respace(0.01)
    chain.delete_insert(1.0)
    if params.align: chain.align()
    if params.ew: print("Energy weighted NEB calculation.")
    chain.ComputeMetric()
    chain.ComputeChain(cyc=0)
    t0 = time.time()
    # chain.SetFlags(params.gtol, params.climb)
    # Obtain the guess Hessian matrix
    chain.ComputeGuessHessian(full=False, blank=isinstance(engine, Blank))
    # Print the status of the zeroth iteration
    print("-=# Optimization cycle 0 #=-")
    chain.PrintStatus()
    print("-= Chain Properties =-")
    print("@%13s %13s %13s %13s %11s %13s %13s" % ("GAvg(eV/Ang)", "GMax(eV/Ang)", "Length(Ang)", "DeltaE(kcal)", "RMSD(Ang)", "TrustRad(Ang)", "Step Quality"))
    print("@%13s %13s %13s" % ("% 8.4f  " % chain.avgg, "% 8.4f  " % chain.maxg, "% 8.4f  " % sum(chain.calc_spacings())))
    chain.SaveToDisk(fout='chain.xyz' if params.sepdir else 'chain_%04i.xyz' % 0)
    Y = chain.get_internal_all()
    GW = chain.get_global_grad("total", "working")
    GP = chain.get_global_grad("total", "plain")
    HW = chain.guess_hessian_working.copy()
    HP = chain.guess_hessian_plain.copy()
    #== Initialize some variables
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
    #== Enter the main optimization loop
    optCycle = 0
    LastRebuild=1
    respaced = False
    while True:
        #======================================================#
        #| At the top of the loop, our coordinates, energies, |#
        #| gradients, and Hessians are synchronized.          |#
        #======================================================#
        optCycle += 1
        print("-=# Optimization cycle %i #=-" % (optCycle))
        #=======================================#
        #|    Obtain an optimization step      |#
        #|    and update Cartesian coordinates |#
        #=======================================#
        LastForce += ForceRebuild
        dy, expect, expectG, ForceRebuild = chain.CalcInternalStep(trust, HW, HP)
        # If ForceRebuild is True, the internal coordinate system
        # is rebuilt and the optimization skips a step
        if ForceRebuild:
            if LastForce == 0:
                pass
            elif LastForce == 1:
                print("\x1b[1;91mFailed twice in a row to rebuild the coordinate system\x1b[0m")
                print("\x1b[93mResetting Hessian\x1b[0m")
                params.reset = True
            elif LastForce == 2:
                print("\x1b[1;91mFailed three times to rebuild the coordinate system\x1b[0m")
                print("\x1b[93mContinuing in Cartesian coordinates\x1b[0m")
            else:
                raise RuntimeError("Coordinate system has failed too many times")
            CoordCounter = 0
            r0 = params.reset
            params.reset = True
            chain, Y, GW, GP, HW, HP = recover(c_hist, params, LastForce==2)
            params.reset = r0
            print("\x1b[1;93mSkipping optimization step\x1b[0m")
            optCycle -= 1
            continue
        else:
            LastForce = 0
        # Before updating any of our variables, copy current variables to "previous"
        Y_prev = Y.copy()
        GW_prev = GW.copy()
        GP_prev = GP.copy()
        # Cartesian norm of the step
        cnorm = chain.getCartesianNorm(dy, params.verbose)
        # Whether the Cartesian norm comes close to the trust radius
        bump = cnorm > 0.8 * trust
        # Update the internal coordinates
        # Obtain a new chain with the step applied
        chain = chain.TakeStep(dy, printStep=False)
        respaced = chain.delete_insert(1.5)
        if params.align: chain.align()
        if params.sepdir: chain.UpdateTempDir(optCycle)
        print("Time since last ComputeChain: %.3f s" % (time.time() - t0))
        chain.ComputeChain(cyc=optCycle)
        t0 = time.time()
        Y1 = Y + dy
        Y = chain.get_internal_all()
        GW = chain.get_global_grad("total", "working")
        GP = chain.get_global_grad("total", "plain")
        # Print Optimization Status
        chain.PrintStatus()
        chain.SaveClimbingImages(optCycle)
        if respaced:
            print("Respaced images - resetting Hessian and skipping trust radius update")
            print("@%13s %13s %13s %13s %11s %13s %13s" % ("GAvg(eV/Ang)", "GMax(eV/Ang)", "Length(Ang)", "DeltaE(kcal)", "RMSD(Ang)", "TrustRad(Ang)", "Step Quality"))
            print("@%13s %13s %13s" % ("% 8.4f  " % chain.avgg, "% 8.4f  " % chain.maxg, "% 8.4f  " % sum(chain.calc_spacings())))
            HW = chain.guess_hessian_working.copy()
            HP = chain.guess_hessian_plain.copy()
            c_hist = [chain]
            chain.SaveToDisk(fout='chain.xyz' if params.sepdir else 'chain_%04i.xyz' % optCycle)
            continue
        dE = chain.TotBandEnergy - c_hist[-1].TotBandEnergy
        dE = chain.TotBandEnergy - c_hist[-1].TotBandEnergy
        if dE > 0.0 and expect > 0.0 and dE > expect:
            Quality = (2*expect-dE)/expect 
        else:
            Quality = dE/expect
        GC  = chain.CalcRMSCartGrad(GW)
        eGC = chain.CalcRMSCartGrad(expectG)
        GPC = chain.CalcRMSCartGrad(GW_prev)
        QualityG = 2.0 - GC/max(eGC, GPC/2, params.avgg/2) 
        Quality = max(Quality, QualityG)
        Quality = QualityG
        Describe = "Good" if Quality > ThreHQ else ("Okay" if Quality > ThreLQ else ("Poor" if Quality > -1.0 else "Reject"))
        print()
        print(" %13s %13s %13s %13s %11s %14s %13s" % ("GAvg(eV/Ang)", "GMax(eV/Ang)", "Length(Ang)", "DeltaE(kcal)", "RMSD(Ang)", "TrustRad(Ang)", "Step Quality"))
        print("@%13s %13s %13s %13s %11s  %8.4f (%s)  %13s" % ("% 8.4f  " % chain.avgg, "% 8.4f  " % chain.maxg, "% 8.4f  " % sum(chain.calc_spacings()), "% 8.4f  " % (au2kcal*dE/len(chain)), 
                                                               "% 8.4f  " % (ChainRMSD(chain, c_hist[-1])), trust, trustprint, "% 6.2f (%s)" % (Quality, Describe)))
        chain.SaveToDisk(fout='chain.xyz' if params.sepdir else 'chain_%04i.xyz' % optCycle)
        #=======================================#
        #|    Check convergence criteria       |#
        #=======================================#
        # if chain.avgg < params.gtol:
        if chain.maxg < params.maxg and chain.avgg < params.avgg:
            print("--== Optimization Converged. ==--")
            break
        if optCycle >= params.maxcyc:
            print("--== Maximum optimization cycles reached. ==--")
            break
        #=======================================#
        #|  Adjust Trust Radius / Reject Step  |#
        #=======================================#
        ### Adjust Trust Radius and/or Reject Step ###
        # If the trust radius is under ThreRJ then do not reject.
        rejectOk = (trust > ThreRJ and chain.TotBandEnergy - c_hist[-1].TotBandEnergy)
        if Quality <= ThreLQ:
            # For bad steps, the trust radius is reduced
            trust = max(ThreRJ/10, min(ChainRMSD(chain, c_hist[-1]), trust)/2) #Division
            trustprint = "\x1b[91m-\x1b[0m"
        elif Quality >= ThreHQ:
            if trust < params.tmax:
                # For good steps, the trust radius is increased
                trust = min(np.sqrt(2)*trust, params.tmax)
                trustprint = "\x1b[92m+\x1b[0m"
            else:
                trustprint = "="
        else:
            trustprint = "="
        # LP-Experiment: Trust radius should be smaller than half of chain spacing
        # Otherwise kinks can (and do) appear!
        trust = min(trust, min(chain.calc_spacings()))

        if Quality < -1 and rejectOk:
            # Reject the step and take a smaller one from the previous iteration
            Y = Y_prev.copy()
            GW = GW_prev.copy()
            GP = GP_prev.copy()
            # LPW 2017-04-08: Removed deepcopy to save memory.
            # If unexpected behavior appears, check here.
            chain = c_hist[-1]
            print("Reducing trust radius to %.1e and rejecting step" % trust)
            continue

        # Append chain to history
        # First delete the older chains from memory
        # if len(c_hist) >= params.history:
        #     cdel = c_hist.pop(0)
        #     del cdel
        c_hist.append(chain)
        c_hist = c_hist[-params.history:]
        #=======================================#
        #|      Update the Hessian Matrix      |#
        #=======================================#
        # Return true if reset is forced.
        HP_bak = HP.copy()
        HW_bak = HW.copy()
        BFGSUpdate(Y, Y_prev, GP, GP_prev, HP, params)
        Eig1 = BFGSUpdate(Y, Y_prev, GW, GW_prev, HW, params)
        if np.min(Eig1) <= params.epsilon:
            if params.reset:
                print("Eigenvalues below %.4e (%.4e) - will reset the Hessian" % (params.epsilon, np.min(Eig1)))
                chain, Y, GW, GP, HW, HP = recover([c_hist[-1]], params, LastForce)
                Y_prev = Y.copy()
                GP_prev = GP.copy()
                GW_prev = GW.copy()
            elif params.skip:
                print("Eigenvalues below %.4e (%.4e) - skipping Hessian update" % (params.epsilon, np.min(Eig1)))
                Y_prev = Y.copy()
                GP_prev = GP.copy()
                GW_prev = GW.copy()
                HP = HP_bak.copy()
                HW = HW_bak.copy()
        del HP_bak
        del HW_bak

def plot_matrix(mat):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    fig.set_size_inches(5,5)
    logabs = np.log(np.abs(mat))
    im = ax.imshow(logabs.T, interpolation='nearest', origin='upper')
    fig.colorbar(im)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9)
    fig.savefig('plot.png')

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordsys', type=str, default='cart', help='Coordinate system: "cart" for Cartesian, "prim" for Primitive (a.k.a redundant), '
                   '"dlc" for Delocalized Internal Coordinates, "hdlc" for Hybrid Delocalized Internal Coordinates, "tric" for Translation-Rotation'
                   'Internal Coordinates (default). Currently only cart and prim are supported.')
    parser.add_argument('--engine', type=str, default='none', help='Choose how to calculate the energy (qchem, leps, reaxff).')
    parser.add_argument('--prefix', type=str, default=None, help='Specify a prefix for output file and temporary directory.')
    parser.add_argument('--port', type=int, default=0, help='Specify a port for Work Queue when running in parallel.')
    parser.add_argument('--maxg', type=float, default=0.05, help='Converge when maximum RMS-gradient for any image falls below this threshold.')
    parser.add_argument('--avgg', type=float, default=0.025, help='Converge when average RMS-gradient falls below this threshold.')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Small eigenvalue threshold.')
    parser.add_argument('--guessk', type=float, default=0.05, help='Guess Hessian eigenvalue for displacements.')
    parser.add_argument('--guessw', type=float, default=0.1, help='Guess weight for chain coordinates.')
    parser.add_argument('--tcguess', type=str, default=[], nargs="+", help='Provide MO guess files for TC as c0, c1, .. or ca0, cb0, ca1, cb1 ..')
    parser.add_argument('--nogenguess', action='store_true', help='When MO guess files are provided, skip calculation that generates the guess')
    parser.add_argument('--verbose', action='store_true', help='Write out extra information.')
    parser.add_argument('--reset', action='store_true', help='Reset Hessian when eigenvalues are under epsilon.')
    parser.add_argument('--align', action='store_true', help='Align images (experimental).')
    parser.add_argument('--fdcheckg', action='store_true', help='Finite-difference gradient test (do not optimize).')
    parser.add_argument('--trust', type=float, default=0.1, help='Starting trust radius.')
    parser.add_argument('--tmax', type=float, default=0.3, help='Maximum trust radius.')
    parser.add_argument('--radii', type=str, nargs="+", default=["Na","0.0"], help='List of atomic radii for coordinate system.')
    parser.add_argument('--coords', type=str, help='Provide coordinates (overwrites what you have in quantum chemistry input files).')
    parser.add_argument('--images', type=int, default=11, help='Number of NEB images to use.')
    parser.add_argument('--icdisp', action='store_true', help='Compute displacements using internal coordinates.')
    parser.add_argument('--sepdir', action='store_true', help='Store each chain in a separate folder.')
    parser.add_argument('--skip', action='store_true', help='Skip Hessian updates that would introduce negative eigenvalues.')
    parser.add_argument('--plain', type=int, default=0, help='1: Use plain elastic band for spring force. 2: Use plain elastic band for spring AND potential.')
    parser.add_argument('--nebk', type=float, default=1, help='NEB spring constant in units of kcal/mol/Ang^2.')
    parser.add_argument('--ew', action='store_true', help='Energy weighted NEB calculation (k range is nebk - nebk/10)')
    parser.add_argument('--history', type=int, default=1, help='Chain history to keep in memory; note chains are very memory intensive, >1 GB each')
    parser.add_argument('--maxcyc', type=int, default=100, help='Maximum number of chain optimization cycles to perform')
    parser.add_argument('--climb', type=float, default=0.5, help='Activate climbing image for max-energy points when max gradient falls below this threshold.')
    parser.add_argument('--ncimg', type=int, default=1, help='Number of climbing images to expect.')
    parser.add_argument('--nt', type=int, default=1, help='Specify number of threads for running in parallel (for TeraChem this should be number of GPUs)')
    parser.add_argument('--input', type=str, help='TeraChem or Q-Chem input file')

    return parser

def main():
    parser = build_args() 

    print(' '.join(sys.argv))

    args = parser.parse_args(sys.argv[1:])

    M, engine = get_molecule_engine(args)
    M.align()

    if args.port != 0:
        createWorkQueue(args.port, debug=args.verbose)
    if args.prefix is None:
        tmpdir = os.path.splitext(args.input)[0]+".tmp"
    else:
        tmpdir = args.prefix+".tmp"
    # Make the initial chain
    if args.sepdir:
        tmpdir = os.path.join(tmpdir, 'chain_%04i' % 0)
    params = ChainOptParams(**vars(args))
    chain = ElasticBand(M, engine=engine, tmpdir=tmpdir, coordtype=args.coordsys, params=params, plain=args.plain, ic_displace=args.icdisp)
    if args.fdcheckg:
        chain.FiniteDifferenceTest()
        chain.SaveToDisk()
    else:
        OptimizeChain(chain, engine, params)
    
if __name__ == "__main__":
    main()
