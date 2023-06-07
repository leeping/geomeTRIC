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

import os,sys
import importlib
import json
import numpy as np

from .engine import Engine, EngineError
from .errors import CheckCoordError
from .molecule import Molecule
from .nifty import ang2bohr, bohr2ang, getWorkQueue, queue_up_src_dest

class EngineASE(Engine):
    def __init__(self, molecule: Molecule, calculator: Calculator):
        super().__init__(molecule)

        self.calculator = calculator
        self.ase_atoms = Atoms(self.M.elem, positions=self.M.Data.get("xyzs")[0])
        self.ase_atoms.calc = self.calculator

    @classmethod
    def from_calculator_constructor(cls, molecule: Molecule, calculator, *args, **kwargs):
        obj = cls(molecule, calculator(*args, **kwargs))
        # A workaround to set the charge and spin multiplicity.
        charge = kwargs.get("charge", 0)
        initial_charges = np.zeros(len(obj.ase_atoms))
        initial_charges[0] = charge
        obj.ase_atoms.set_initial_charges(initial_charges)
        mult = kwargs.get("mult", 1)
        initial_spins = np.zeros(len(obj.ase_atoms))
        initial_spins[0] = mult-1
        obj.ase_atoms.set_initial_magnetic_moments(initial_spins)
        # This stores the needed information to re-create the Engine from strings (for example when using Work Queue)
        obj.calculator_import_path = calculator.__module__+'.'+calculator.__name__
        obj.calculator_kwargs = kwargs
        return obj

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
        # Set up Work Queue object
        wq = getWorkQueue()
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        self.M[0].write(os.path.join(dirname, "start.xyz"))
        # Specify WQ input and output files
        in_files = [('%s/start.xyz' % dirname, 'start.xyz')]
        out_files = [('%s/ase_energy.txt' % dirname, 'ase_energy.txt'),
                     ('%s/ase_gradient.txt' % dirname, 'ase_gradient.txt')]
        cmd="run-ase --nt 1 --ase-class=%s --ase-kwargs='%s' start.xyz" % (self.calculator_import_path, json.dumps(self.calculator_kwargs))
        queue_up_src_dest(wq, cmd, in_files, out_files, verbose=False, print_time=600)

    def read_result(self, dirname, check_coord=None):
        """ Read ASE calculation output. """
        if check_coord is not None:
            read_xyz_success = False
            if os.path.exists(os.path.join(dirname, "start.xyz")):
                try:
                    read_xyz = Molecule(os.path.join(dirname, "start.xyz"), build_topology=False).xyzs[0].flatten()/bohr2ang
                    # print("Successfully read xyz coordinates from", os.path.join(dirname, "start.xyz"))
                    read_xyz_success = True
                except: pass
            if not read_xyz_success or np.linalg.norm(check_coord - read_xyz) > 1e-8:
                # print("Raising CheckCoordError")
                raise CheckCoordError
        result = {}
        result["energy"] = float(open(os.path.join(dirname, "ase_energy.txt")).readlines()[0].strip())
        result["gradient"] = np.loadtxt(os.path.join(dirname, "ase_gradient.txt")).flatten()
        return result

    def copy_scratch(self, src, dest):
        # this does nothing for ASE for now
        return

def parse_args(*args):
    import argparse
    parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter, fromfile_prefix_chars='@')
    grp_univ = parser.add_argument_group('universal', 'Relevant to every job')
    grp_univ.add_argument('input', type=str, help='REQUIRED positional argument: Quantum chemistry or MM input file for calculation\n ')
    grp_univ.add_argument('--nt', type=int, default=1, help='Specify number of threads for running in parallel\n(for TeraChem this should be number of GPUs)')
    grp_software = parser.add_argument_group('software', 'Options specific for certain software packages')
    grp_software.add_argument('--ase-class', type=str, default='xtb.ase.calculator.XTB', help='ASE calculator import path, eg. "ase.calculators.lj.LennardJones"')
    grp_software.add_argument('--ase-kwargs', type=str, default='{"method":"GFN2-xTB"}', help='ASE calculator keyword args, as JSON dictionary, eg. {"param_filename":"path/to/file.xml"}')
    grp_help = parser.add_argument_group('help', 'Get help')
    grp_help.add_argument('-h', '--help', action='help', help='Show this help message and exit')
    args_dict = {}
    for k, v in vars(parser.parse_args(*args)).items():
        if v is not None:
            args_dict[k] = v
    return args_dict

def main():
    args = parse_args(sys.argv[1:])
    M = Molecule(args["input"])[0]
    ase_class_name = args["ase_class"]
    ase_kwargs = args["ase_kwargs"]
    os.environ["OMP_NUM_THREADS"] = "%i" % args["nt"]
    engine = EngineASE.from_calculator_string(M, ase_class_name, **json.loads(ase_kwargs))
    coords = M.xyzs[0].flatten()*ang2bohr
    result = engine.calc_new(coords, '.')
    with open("ase_energy.txt", "w") as f:
        print("% 18.12e\n" % result['energy'], file=f)
    np.savetxt("ase_gradient.txt", result['gradient'].reshape(-1, 3), fmt="% 18.12e")
        
if __name__ == '__main__':
    main()
