"""
ASE-engine, perform energy and force calculations with ASE-compatible calculators

Copyright 2021 Tamas K. Stenczel
"""

# python 2 madness
try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError

import logging
import os
import tempfile

try:
    from ase import Atoms
    from ase.optimize.optimize import Optimizer
except (ModuleNotFoundError, ImportError):
    Atoms = None
    Optimizer = None

import geometric
from .optimize import OptParams, set_up_coordinate_system, ang2bohr, OPT_STATE
from .ase_engine import EngineASE
from .molecule import Molecule


class ExtMolecule(Molecule):

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms):
        # initialise from ASE atoms -- hack with a temporary file

        # with tempfile.TemporaryFile("w", suffix=".xyz") as file:
        #     ase.io.write(file, atoms)
        #     mol = cls(file.name, ftype="xyz")

        mol = cls()

        mol.Data["elem"] = atoms.get_chemical_symbols()
        mol.Data["xyzs"] = [atoms.get_positions()]

        return mol


class GeomeTRIC(Optimizer):
    """Optimizer wrapper for GeomeTRIC.

    limitations:
    - fmax in self.run() is not understood, uses the internal convergence thresholds or any set on construction
    - GeomeTRIC's constraints are not implemented, if ASE is modifying the forces then it may work, the plan is
    to pull the constraints from ASE and convert them to the GeomeTRIC internal ones, or just use the constraint
    input of GeomeTRIC and raise errors if the Atoms object has any constraints on it
    - logging is not implemented, the default one is called which is incorrect

    """

    def __init__(
            self,
            atoms,
            restart=None,
            logfile="-",
            trajectory=None,
            master=None,
            append_trajectory=False,
            force_consistent=False,
            **kwargs
    ):
        super(GeomeTRIC, self).__init__(atoms, restart, logfile, trajectory,
                                        master=master, append_trajectory=append_trajectory,
                                        force_consistent=force_consistent)

        self.geometric_kwargs = kwargs

        self.opt_params = OptParams(**kwargs)

        # molecule
        self.molecule = ExtMolecule.from_ase_atoms(atoms)

        # engine -- uses the calculator of the atoms object
        self.engine = EngineASE(self.molecule, self.atoms.calc)

        # constraints
        self.constraint_coordinates = None
        self.constraint_values = None
        self.handle_constraints(kwargs.get('constraints', None))

        # IC
        self.IC = set_up_coordinate_system(self.molecule, coordsys=kwargs.get('coordsys', 'tric'),
                                           conmethod=self.opt_params.conmethod,
                                           CVals=self.constraint_values, Cons=self.constraint_coordinates)

        # auxiliary things
        # Input file for optimization; QC input file or OpenMM .xml file
        inputf = kwargs.get('input')
        # arg_prefix = kwargs.get('prefix', None)  # prefix for output file and temporary directory
        # prefix = arg_prefix if arg_prefix is not None else os.path.splitext(inputf)[0]
        prefix = "prefix"
        logger = logging.getLogger(__name__)
        self.opt_params.xyzout = prefix + "_optim.xyz"

        # the GeomeTRIC optimizer object
        self.temporary_directory = tempfile.mkdtemp()
        self.optimizer = geometric.optimize.Optimizer(self.molecule.xyzs[0].flatten() * ang2bohr, self.molecule,
                                                      self.IC, self.engine, self.temporary_directory, self.opt_params)

        # first step
        self.first_step_done = False

    def cleanup(self):
        # cleans the temporary directory created
        if os.path.isdir(self.temporary_directory):
            os.rmdir(self.temporary_directory)

    def handle_constraints(self, constraints_file):
        # fixme: change this to ASE's constraints parsed

        # # Read in the constraints
        # constraints = kwargs.get('constraints', None)  # Constraint input file (optional)
        #
        # if constraints is not None:
        #     Cons, CVals = parse_constraints(self.molecule, open(constraints).read())
        # else:
        #     Cons = None
        #     CVals = None

        self.constraint_coordinates = None
        self.constraint_values = None

    def converged(self, forces=None):
        return self.optimizer.state == OPT_STATE.CONVERGED

    def first_step(self):

        if not self.first_step_done:
            self.optimizer.calcEnergyForce()
            self.optimizer.prepareFirstStep()

        self.first_step_done = True

    def run(self, fmax=0.05, steps=None):
        """ call Dynamics.run and use the optimiser to say if it is converged"""

        # do the first step
        self.first_step()

        # run the optimisation
        return super(GeomeTRIC, self).run(self)

    def step(self):
        # GeomeTRIC's step
        self.optimizer.step()
        if self.optimizer.state == OPT_STATE.NEEDS_EVALUATION:
            self.optimizer.calcEnergyForce()
            self.optimizer.evaluateStep()

        # update atoms object
        self.atoms.set_positions(self.molecule.xyzs[0])
