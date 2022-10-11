"""
ASE-engine, perform energy and force calculations with ASE-compatible calculators

Copyright 2021-2022 Tamas K. Stenczel
"""

# python 2 madness
try:
    ModuleNotFoundError
except NameError: # pragma: no cover
    ModuleNotFoundError = ImportError

try:
    from ase.calculators.calculator import Calculator
    from ase import Atoms
    from ase import units
except (ModuleNotFoundError, ImportError): # pragma: no cover
    Calculator = None
    Atoms = None
    units = None

import importlib

from .engine import Engine, EngineError
from .molecule import Molecule


class EngineASE(Engine):
    def __init__(self, molecule: Molecule, calculator: Calculator):
        super().__init__(molecule)

        self.calculator = calculator
        self.ase_atoms = Atoms(self.M.elem, positions=self.M.Data.get("xyzs")[0])
        self.ase_atoms.calc = self.calculator

    @classmethod
    def from_calculator_constructor(cls, molecule: Molecule, calculator, *args, **kwargs):
        return cls(molecule, calculator(*args, **kwargs))

    @classmethod
    def from_calculator_string(cls, molecule: Molecule, calculator_import: str, *args, **kwargs):
        # this imports the calculator
        module_name = ".".join(calculator_import.split(".")[:-1])
        class_name = calculator_import.split(".")[-1]

        # import the module of the calculator
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise EngineError("ASE-calculator's module is not found: {}".format(class_name))

        # class of the calculator
        if hasattr(module, class_name):
            calc_class = getattr(module, class_name)
        else:
            raise EngineError("ASE-calculator's class ({}) not found in module {}".format(class_name, module_name))

        if not issubclass(calc_class, Calculator):
            raise EngineError("Not an ASE calculator class ({}) found in module ({})".format(class_name, module_name))

        # construct from the constructor
        return cls.from_calculator_constructor(molecule, calc_class, *args, **kwargs)

    def update_atoms(self, coords):
        # sets the positions, given in Bohr
        self.ase_atoms.set_positions(coords.reshape(-1, 3) * units.Bohr)

    def calc_new(self, coords, dirname):
        """
        Top-level method for a single-point calculation.
        Calculation will be skipped if results are contained in the hash table,
        and optionally, can skip calculation if output exists on disk (which is
        useful in the case of restarting a crashed Hessian calculation)

        Parameters
        ----------
        coords : np.array
            1-dimensional array of shape (3*N_atoms) containing atomic coordinates in Bohr
        dirname : str
            Relative path containing calculation files

        Returns
        -------
        result : dict
            Dictionary containing results:
            result['energy'] = float
                Energy in atomic units
            result['gradient'] = np.array
                1-dimensional array of same shape as coords, containing nuclear gradients in a.u.
            result['s2'] = float
                Optional output containing expectation value of <S^2> operator, used in
                crossing point optimizations
        """

        self.update_atoms(coords)

        # the calculation
        forces = self.calculator.get_forces(self.ase_atoms)
        energy = self.calculator.get_potential_energy(self.ase_atoms)

        return {
            "energy": energy / units.Hartree,  # eV -> Ha
            "gradient": - forces.flatten() / units.Hartree * units.Bohr  # eV/A -> Ha/Bohr
        }

    def calc_wq_new(self, coords, dirname):
        raise NotImplementedError

    def copy_scratch(self, src, dest):
        # this does nothing for ASE for now
        return
