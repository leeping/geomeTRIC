"""
params.py: Optimization parameters and user options

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

from __future__ import division
import os, argparse
import numpy as np
from .errors import ParamError
from .nifty import logger

class OptParams(object):
    """
    Container for optimization parameters.
    The parameters used to be contained in the command-line "args",
    but this was dropped in order to call Optimize() from another script.
    """
    def __init__(self, **kwargs):
        # Whether we are optimizing for a transition state. This changes a number of default parameters.
        self.transition = kwargs.get('transition', False)
        # Intrinsic Reaction Coordinate method. This changes a number of default parameters.
        self.irc = kwargs.get('irc', False)
        # CI optimizations sometimes require tiny steps
        self.meci = kwargs.get('meci', False)
        # Handle convergence criteria; this edits the kwargs
        self.convergence_criteria(**kwargs)
        # Threshold (in a.u. / rad) for activating alternative algorithm that enforces precise constraint satisfaction
        self.enforce = kwargs.get('enforce', 0.0)
        # Small eigenvalue threshold
        self.epsilon = kwargs.get('epsilon', 1e-5)
        # Interval for checking the coordinate system for changes
        self.check = kwargs.get('check', 0)
        # More verbose printout
        self.verbose = kwargs.get('verbose', False)
        # Starting value of the trust radius
        # Because TS optimization is experimental, use conservative trust radii
        self.trust = kwargs.get('trust', 0.01 if self.transition else 0.1)
        # Maximum value of trust radius
        self.tmax = kwargs.get('tmax', 0.03 if self.transition else self.trust if self.irc else 0.3)
        # Minimum value of the trust radius
        # Also sets the maximum step size that can be rejected
        # LPW: Add to documentation later:
        # This parameter is complicated.  It represents the length scale below which the PES is expected to be smooth.
        # If set too small, the optimization could get "stuck" in a small divot on the PES and be unable to escape.
        # This could happen for example if the potential energy surface contains a "step", which occurs infrequently;
        # we don't want the optimization to stop there completely, but rather we accept a bad step and keep going.
        # If set too large, then the rejection algorithm becomes ineffective as too-large low-quality steps will be kept.
        # It should also not be smaller than Convergence_drms because that could artifically cause convergence.
        # PS: If the PES is inherently rough on extremely small length scales,
        # then the optimization is not expected to converge regardless of tmin.
        # Note 2023-10-29: Testing has shown that DFT calculations can have gradient errors on the order of 1e-4 
        # in various engines when the default grid is used. This could lead to failures if the attempted 
        # optimization step is larger than Convergence_drms, the step is rejected, then (almost) the same step
        # is attempted again. Therefore, the minimum trust radius should be set smaller than the convergence
        # criterion to avoid this infinite loop scenario.
        self.tmin = kwargs.get('tmin', min(1.0e-4, self.Convergence_drms*0.1))
        # Use maximum component instead of RMS displacement when applying trust radius.
        self.usedmax = kwargs.get('usedmax', False)
        # Sanity checks on trust radius
        if self.tmax < self.tmin:
            raise ParamError("Max trust radius must be larger than min")
        # The trust radius should not be outside (tmin, tmax)
        self.trust = min(self.tmax, self.trust)
        self.trust = max(self.tmin, self.trust)
        # Maximum number of optimization cycles
        self.maxiter = kwargs.get('maxiter', 300)
        # Write Hessian matrix at optimized structure to text file
        self.write_cart_hess = kwargs.get('write_cart_hess', None)
        # Output .xyz is deliberately not set here in order to give run_optimizer()
        # and geometric_run_json() more control over the default value of this variable.
        self.xyzout = kwargs.get('xyzout', None)
        # Name of the qdata.txt file to be written.
        # The CLI is designed so the user passes true/false instead of the file name.
        self.qdata = 'qdata.txt' if kwargs.get('qdata', False) else None
        # Bond order threshold parameter, when using bond orders to build bonds.
        # Turned on by default for TS calculations; with sufficient testing could turn on for minimization.
        self.bothre = kwargs.get('bothre', 0.6 if self.transition else 0.0)
        # Whether to calculate or read a Hessian matrix.
        self.hessian = kwargs.get('hessian', None)
        # Whether to use BigChem to carry the Hessian and NEB calculations.
        self.bigchem = kwargs.get('bigchem', False)

        if self.hessian is None:
            # Default is to calculate Hessian in the first step if searching for a transition state.
            # Otherwise the default is to never calculate the Hessian.
            if self.transition or self.irc: self.hessian = 'first'
            else: self.hessian = 'never'
        elif self.hessian.startswith('file:'):
            if os.path.exists(self.hessian[5:]):
                # If a path is provided for reading a Hessian file, read it now.
                self.hess_data = np.loadtxt(self.hessian[5:])
            else:
                raise IOError("No Hessian data file found at %s" % self.hessian)
        elif self.hessian.startswith('file+last:'):
            if os.path.exists(self.hessian[10:]):
                # If a path is provided for reading a Hessian file, read it now.
                self.hess_data = np.loadtxt(self.hessian[10:])
                self.hessian = 'last'
            else:
                raise IOError("No Hessian data file found at %s" % self.hessian)
        elif self.hessian.lower() in ['never', 'first', 'each', 'stop', 'last', 'first+last']:
            self.hessian = self.hessian.lower()
        else:
            raise RuntimeError("Hessian command line argument can only be never, first, last, first+last, file+last, each, stop, or file:<path>")
        # Perform a frequency analysis whenever a cartesian Hessian is computed
        self.frequency = kwargs.get('frequency', None)
        if self.frequency is None: self.frequency = True
        # Temperature and pressure for harmonic free energy
        self.temperature, self.pressure = kwargs.get('thermo', [300.0, 1.0])
        # Number of desired samples from Wigner distribution
        self.wigner = kwargs.get('wigner', 0)
        if self.wigner and not self.frequency:
            raise ParamError('Wigner sampling requires frequency analysis')
        # Ignore N lowest force constants when computing free energy
        # (may be used when comparing two free energies when some of the modes are imaginary)
        self.ignore_modes = kwargs.get('ignore_modes', 0)
        # Reset Hessian to guess whenever eigenvalues drop below epsilon
        self.reset = kwargs.get('reset', None)
        if self.reset is None: self.reset = not (self.transition or self.meci or self.hessian == 'each')
        # Subtract net force and torque components from the gradient.
        # In DFT calcs, there is often a small nonzero torque that is
        # inconsistent with the energy change (W. Swope, personal communication).
        # This torque component may cause geomeTRIC to fail to converge because
        # it induces a gradual rotation of the structure that does not reduce the energy.
        # Enabling this option eliminates the net force/torque component, which
        # should speed convergence in these cases.
        # In other cases (e.g. Fe4N catalyst), enabling this option can slow convergence.
        # 0 = never project; 1 = auto-detect (default); 2 = always project
        self.subfrctor = kwargs.get('subfrctor', 1)

    def convergence_criteria(self, **kwargs):
        criteria = kwargs.get('converge', [])
        # Whether to converge successfully on reaching maximum number of iterations
        self.Converge_maxiter = False
        # Iterate backward through the list, in case someone lists maxiter twice.
        for i in list(range(len(criteria)))[::-1]:
            if criteria[i].lower() == 'maxiter':
                criteria.pop(i)
                self.Converge_maxiter = True
        if len(criteria)%2 != 0:
            raise RuntimeError('Please pass an even number of options to --converge (excluding maxiter)')
        for i in range(int(len(criteria)/2)):
            key = 'convergence_' + criteria[2*i].lower()
            try:
                val = float(criteria[2*i+1])
                logger.info('Using convergence criteria: %s %.2e\n' % (key, val))
            except ValueError:
                # This must be a set
                val = str(criteria[2*i+1])
                logger.info('Using convergence criteria set: %s %s\n' % (key, val))
            kwargs[key] = val
        # convergence dictionary to store criteria stored in order of energy, grms, gmax, drms, dmax
        # 'GAU' contains the default convergence criteria that are used when nothing is passed.
        convergence_sets = {'GAU': [1e-6, 3e-4, 4.5e-4, 1.2e-3, 1.8e-3],
                            'NWCHEM_LOOSE': [1e-6, 3e-3, 4.5e-3, 3.6e-3, 5.4e-3],
                            'GAU_LOOSE': [1e-6, 1.7e-3, 2.5e-3, 6.7e-3, 1e-2],
                            'TURBOMOLE': [1e-6, 5e-4, 1e-3, 5.0e-4, 1e-3],
                            'INTERFRAG_TIGHT': [1e-6, 1e-5, 1.5e-5, 4.0e-4, 6.0e-4],
                            'GAU_TIGHT': [1e-6, 1e-5, 1.5e-5, 4e-5, 6e-5],
                            'GAU_VERYTIGHT': [1e-6, 1e-6, 2e-6, 4e-6, 6e-6]}
        # Q-Chem style convergence criteria (i.e. gradient and either energy or displacement)
        self.qccnv = kwargs.get('qccnv', False)
        # Molpro style convergence criteria (i.e. gradient and either energy or displacement, with different defaults)
        self.molcnv = kwargs.get('molcnv', False)
        # Check if there is a convergence set passed else use the default
        set_name = kwargs.get('convergence_set', 'GAU').upper()
        # If we have extra keywords apply them here else use the set
        # Convergence criteria in a.u. and Angstrom
        self.Convergence_energy = kwargs.get('convergence_energy', convergence_sets[set_name][0])
        self.Convergence_grms = kwargs.get('convergence_grms', convergence_sets[set_name][1])
        self.Convergence_gmax = kwargs.get('convergence_gmax', convergence_sets[set_name][2])
        self.Convergence_drms = kwargs.get('convergence_drms', convergence_sets[set_name][3])
        self.Convergence_dmax = kwargs.get('convergence_dmax', convergence_sets[set_name][4])
        # Convergence criteria that are only used if molconv is set to True
        self.Convergence_molpro_gmax = kwargs.get('convergence_molpro_gmax', 3e-4)
        self.Convergence_molpro_dmax = kwargs.get('convergence_molpro_dmax', 1.2e-3)
        # Convergence criteria for constraint violation
        self.Convergence_cmax = kwargs.get('convergence_cmax', 1.0e-2)

    def printInfo(self):
        if self.subfrctor == 2:
            logger.info(' Net force and torque will be projected out of gradient.\n')
        if self.transition:
            logger.info(' Transition state optimization requested.\n')
        if self.irc:
            logger.info(' The IRC method requested.\n')
        if self.hessian == 'first':
            logger.info(' Hessian will be computed on the first step.\n')
        elif self.hessian == 'each':
            logger.info(' Hessian will be computed for each step.\n')
        elif self.hessian == 'stop':
            logger.info(' Hessian will be computed for first step, then program will stop.\n')
        elif self.hessian == 'last':
            logger.info(' Hessian will be computed for last step.\n')
        elif self.hessian == 'first+last':
            logger.info(' Hessian will be computed for both first and last step.\n')
        elif self.hessian.startswith('file:'):
            logger.info(' Hessian data will be read from file: %s\n' % self.hessian[5:])
        elif self.hessian.startswith('file+last:'):
            logger.info(' Hessian data will be read from file: %s, then computed for the last step.\n' % self.hessian[5:])

class NEBParams(object):
    """
    Container for optimization parameters.
    """
    def __init__(self, **kwargs):
        self.prefix = kwargs.get('prefix', None)
        self.images = kwargs.get('images', 11)
        self.plain = kwargs.get('plain', 0)
        self.maxg = kwargs.get('maxg', 0.05)
        self.avgg = kwargs.get('avgg', 0.025)
        self.guessk = kwargs.get('guessk', 0.05)
        self.guessw = kwargs.get('guessw', 0.1)
        self.nebk = kwargs.get('nebk', 1.0)
        self.history = kwargs.get('neb_history', 1)
        self.maxcyc = kwargs.get('neb_maxcyc', 100)
        self.climb = kwargs.get('climb', 0.5)
        self.ncimg = kwargs.get('ncimg', 1)
        self.optep = kwargs.get('optep', False)
        self.align = kwargs.get('align', True)
        self.epsilon = kwargs.get('epsilon', 1e-5)
        self.verbose = kwargs.get('verbose', False)
        self.trust = kwargs.get('trust', 0.1)
        self.tmax = kwargs.get('tmax', 0.3)
        self.tmin = kwargs.get('tmin', 1.2e-3)
        self.skip = kwargs.get('skip', False)
        self.bigchem = kwargs.get('bigchem', False)

        # Sanity checks on trust radius
        if self.tmax < self.tmin:
            raise ParamError("Max trust radius must be larger than min")
        # The trust radius should not be outside (tmin, tmax)
        self.trust = min(self.tmax, self.trust)
        self.trust = max(self.tmin, self.trust)
        self.xyzout = kwargs.get('xyzout', None)

def str2bool(v):
    """ Allows command line options such as "yes" and "True" to be converted into Booleans. """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'on', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'off', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ArgumentParserWithFile(argparse.ArgumentParser):
    """
    Customized argument parser which can read a constraints file and
    parse text in between $options and $<anything_else> flags as
    command line arguments, to supplement any arguments not provided
    on the command line.

    The syntax in the constraints file can come after  look like:
    $options

    $end
    """
    def __init__(self, *args, **kwargs):
        self.in_options = False
        super(ArgumentParserWithFile, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        line = line.split("#")[0].strip() # Don't lower-case because may be case sensitive
        if '$options' in line:
            self.in_options = True
        elif '$' in line:
            self.in_options = False
        elif self.in_options:
            s = line.split()
            s[0] = '--'+s[0]
            return s
        return []

def parse_optimizer_args(*args):

    """
    Read user input from the command line interface.
    Designed to be called by optimize.main() passing in sys.argv[1:]

    Avoid setting default values for variables here. The default values of certain variables
    depends on the values of other variables. The OptParams() and get_molecule_engine()
    functions sets the default values of variables. This also ensures compatibility with
    the JSON API.
    """

    parser = ArgumentParserWithFile(add_help=False, formatter_class=argparse.RawTextHelpFormatter, fromfile_prefix_chars='@')

    grp_univ = parser.add_argument_group('universal', 'Relevant to every job')
    grp_univ.add_argument('input', type=str, help='REQUIRED positional argument: Quantum chemistry or MM input file for calculation\n ')
    grp_univ.add_argument('constraints', type=str, nargs='?', help='OPTIONAL positional argument: File containing constraint specifications and/or additional options\n ')
    grp_univ.add_argument('--coordsys', type=str, help='Coordinate system:\n'
                          '"tric" for Translation-Rotation Internal Coordinates (default)\n'
                          '"cart" = Cartesian coordinate system\n'
                          '"prim" = Primitive (a.k.a redundant internal coordinates)\n '
                          '"dlc" = Delocalized Internal Coordinates,\n'
                          '"hdlc" = Hybrid Delocalized Internal Coordinates\n'
                          '"tric-p" for primitive Translation-Rotation Internal Coordinates (no delocalization)\n ')
    # TeraChem as a default option is only for the command line interface.
    grp_univ.add_argument('--engine', type=str, help='Specify engine for computing energies and gradients.\n'
                          '"tera" = TeraChem (default)         "qchem" = Q-Chem\n'
                          '"psi4" = Psi4                       "openmm" = OpenMM (pass a force field or XML input file)\n'
                          '"molpro" = Molpro                   "gmx" = Gromacs (pass conf.gro; requires topol.top and shot.mdp\n '
                          '"gaussian" = Gaussian09/16          "ase" = ASE calculator, use --ase-class/--ase-kwargs\n '
                          '"quick" = QUICK\n')
    grp_univ.add_argument('--nt', type=int, help='Specify number of threads for running in parallel\n(for TeraChem this should be number of GPUs)')

    grp_jobtype = parser.add_argument_group('jobtype', 'Control the type of optimization job')
    grp_jobtype.add_argument('--transition', type=str2bool, help='Provide "yes" to Search for a first order saddle point / transition state.\n ')
    grp_jobtype.add_argument('--irc', type=str2bool, help='Provide "yes" to perform the IRC method.\n ')
    grp_jobtype.add_argument('--meci', type=str, nargs="+", help='Provide second input file and search for minimum-energy conical\n '
                             'intersection or crossing point between two SCF solutions (TeraChem and Q-Chem supported).\n'
                             'Or, provide "engine" if the engine directly provides the MECI objective function and gradient.\n')
    grp_jobtype.add_argument('--meci_sigma', type=float, help='Sigma parameter for MECI penalty function (default 3.5).\n'
                            'Not used if the engine computes the MECI objective function directly.\n ')
    grp_jobtype.add_argument('--meci_alpha', type=float, help='Alpha parameter for MECI penalty function (default 0.025).\n'
                             'Not used if the engine computes the MECI objective function directly.\n ')
    grp_jobtype.add_argument('--rigid', type=str2bool, help='Provide "yes" to keep molecules rigid during optimization (only with TRIC)')

    grp_hessian = parser.add_argument_group('hessian', 'Control the calculation of Hessian (force constant) matrices and derived quantities')
    grp_hessian.add_argument('--hessian', type=str, help='Specify when to calculate Cartesian Hessian using finite difference of gradient.\n'
                             '"never" : Do not calculate or read Hessian data. (default for minimization)\n'
                             '"first" : Calculate Hessian for the initial structure. (default for TS optimization)\n'
                             '"last" : Calculate Hessian at conclusion of optimization.\n'
                             '"first+last" : Calculate Hessian for both the first and last structure.\n'
                             '"each" : Calculate for each step in the optimization (costly).\n'
                             '"stop" : Calculate Hessian for initial structure, then exit.\n'
                             'file:<path> : Read initial Hessian data in NumPy format from path, e.g. file:folder/hessian.txt\n'
                             'file+last:<path> : Read initial Hessian data as above, then compute for the last structure.\n ')
    grp_hessian.add_argument('--wqport', type=int, help='Work Queue port used to distribute Hessian calculations. Workers must be started separately. \n ')
    grp_hessian.add_argument('--bigchem', type=str2bool, help='Provide "Yes" to use BigChem for performing the Hessian calculation in parallel. \n'
                                                              'Please ensure that BigChem is running with workers properly. \n ')
    grp_hessian.add_argument('--frequency', type=str2bool, help='Perform frequency analysis whenever Hessian is calculated, default is yes/true. \n')
    grp_hessian.add_argument('--thermo', type=float, nargs=2, help='Temperature (K) and pressure (bar) for harmonic free energy\n'
                             'following frequency analysis, default is 300 K and 1.0 bar.\n ')
    grp_hessian.add_argument('--wigner', type=int, help='Number of desired samples from Wigner distribution after frequency analysis.\n'
                             'Provide negative number to overwrite any existing samples.\n ')
    grp_hessian.add_argument('--ignore_modes', type=int, help='Number of modes to ignore when computing harmonic free energy (default 0).\n'
                             'The lowest/most negative force constants are ignored first.\n ')

    grp_optparam = parser.add_argument_group('optparam', 'Control various aspects of the optimization algorithm')
    grp_optparam.add_argument('--maxiter', type=int, help='Maximum number of optimization steps, default 300.\n ')
    grp_optparam.add_argument('--converge', type=str, nargs="+", help='Custom convergence criteria as key/value pairs.\n'
                              'Provide the name of a criteria set as "set GAU_LOOSE" or "set TURBOMOLE",\n'
                              'and/or set specific criteria using key/value pairs e.g. "energy 1e-5 grms 1e-3"\n '
                              'and/or add the MAXITER keyword to enable successful convergence on maximum iterations reached')
    grp_optparam.add_argument('--trust', type=float, help='Starting trust radius, defaults to 0.1 (energy minimization) or 0.01 (TS optimization).\n ')
    grp_optparam.add_argument('--tmax', type=float, help='Maximum trust radius, defaults to 0.3 (energy minimization) or 0.03 (TS optimization).\n ')
    grp_optparam.add_argument('--tmin', type=float, help='Minimum trust radius, do not reject steps trust radius is below this threshold (method-dependent).\n ')
    grp_optparam.add_argument('--usedmax', type=str2bool, help='Use maximum component instead of RMS displacement when applying trust radius.\n ')
    grp_optparam.add_argument('--enforce', type=float, help='Enforce exact constraints when within provided tolerance (in a.u./radian, default 0.0)\n ')
    grp_optparam.add_argument('--conmethod', type=int, help='Set to 1 to enable alternate constraint algorithm (default 0).\n ')
    grp_optparam.add_argument('--reset', type=str2bool, help='Reset approximate Hessian to guess when eigenvalues are under epsilon.\n '
                              'Defaults to True for minimization and False for transition states.\n ')
    grp_optparam.add_argument('--epsilon', type=float, help='Small eigenvalue threshold for resetting Hessian, default 1e-5.\n ')
    grp_optparam.add_argument('--check', type=int, help='Check coordinates every <N> steps and rebuild coordinate system, disabled by default.\n')
    grp_optparam.add_argument('--subfrctor', type=int, help='Project out net force and torque components from nuclear gradient.\n'
                              '0 = never project; 1 = auto-detect (default); 2 = always project.')

    grp_modify = parser.add_argument_group('structure', 'Modify the starting molecular structure or connectivity')
    grp_modify.add_argument('--radii', type=str, nargs="+", help='List of atomic radii for construction of coordinate system.\n '
                            'Provide pairs of symbol/radius values such as Na 0.0 Fe 1.5\n ')
    grp_modify.add_argument('--pdb', type=str, help='PDB file name with coordinates and resids. TRIC will add T+R coordinates for each residue.\n ')
    grp_modify.add_argument('--coords', type=str, help='Coordinate file to override the QM input file / PDB file. The LAST frame will be used.\n ')
    grp_modify.add_argument('--frag', type=str2bool, help='Provide "yes" to delete bonds between residues, producing\n'
                            'separate fragments in the internal coordinate system.')
    grp_modify.add_argument('--bothre', type=float, help='Set the bond order threshold for building bonds in transition state calculations (Q-Chem, TeraChem only). Set 0.0 to disable.\n ')

    grp_output = parser.add_argument_group('output', 'Control the format and amount of the output')
    grp_output.add_argument('--prefix', type=str, help='Specify a prefix for log file and temporary directory.\n'
                            'Defaults to the input file path (incl. file name with extension removed).\n ')
    grp_output.add_argument('--verbose', type=int, help='Set to positive for more verbose printout.\n'
                            '0 = Default print level.     1 = Basic info about optimization step.\n'
                            '2 = Include microiterations. 3 = Lots of printout from low-level functions.\n ')
    grp_output.add_argument('--qdata', type=str2bool, help='Provide "yes" to write qdata.txt containing coordinates, energies, gradients for each structure.\n ')
    grp_output.add_argument('--logINI',  type=str, dest='logIni', help='.ini file for customizing logging output.\n ')
    grp_output.add_argument('--write_cart_hess', type=str, help='Write approximate cartesian Hessian at optimized geometry to specified file.\n ')

    grp_software = parser.add_argument_group('software', 'Options specific for certain software packages')
    grp_software.add_argument('--molproexe', type=str, help='Specify absolute path of Molpro executable.\n ')
    grp_software.add_argument('--molcnv', type=str2bool, help='Provide "yes" to use Molpro style convergence criteria instead of the default.\n ')
    grp_software.add_argument('--qcdir', type=str, help='Provide an initial Q-Chem scratch folder (e.g. supplied initial guess).\n ')
    grp_software.add_argument('--qccnv', type=str2bool, help='Provide "yes" to Use Q-Chem style convergence criteria instead of the default.\n ')

    grp_software.add_argument(
        '--ase-class',
        type=str,
        help='ASE calculator import path, eg. "ase.calculators.lj.LennardJones"')
    grp_software.add_argument(
        '--ase-kwargs',
        type=str,
        help='ASE calculator keyword args, as JSON dictionary, eg. {"param_filename":"path/to/file.xml"}')

    grp_debug = parser.add_argument_group('debug', 'Relevant for development and debugging')
    grp_debug.add_argument('--displace', type=str2bool, help='Provide "yes" to write out displacements of the internal coordinates.\n ')
    grp_debug.add_argument('--fdcheck', type=str2bool, help='Provide "yes" to check internal coordinate gradients using finite difference.\n ')

    grp_help = parser.add_argument_group('help', 'Get help')
    grp_help.add_argument('-h', '--help', action='help', help='Show this help message and exit')

    # Keep all arguments whose values are not None, so that the setting of default values
    # in OptParams() and get_molecule_engine() will work properly.
    args_dict = {}
    for k, v in vars(parser.parse_args(*args)).items():
        if v is not None:
            args_dict[k] = v

    # Check that the input file exists
    # OpenMM .xml files don't have to be in the current folder.
    if not args_dict['input'].endswith('.xml') and not os.path.exists(args_dict['input']):
        raise RuntimeError("Input file does not exist")

    # Parse the constraints file for additional command line options to be added
    if 'constraints' in args_dict:
        if not os.path.exists(args_dict['constraints']):
            raise RuntimeError("Constraints / options file does not exist")
        args2 = (['_', '__', '@'+args_dict['constraints']],)
        for k, v in vars(parser.parse_args(*args2)).items():
            if v is None: continue
            if k in ['input', 'constraints']: continue
            if k not in args_dict:
                args_dict[k] = v
            elif k in args_dict and v != args_dict[k]:
                raise RuntimeError("Command line argument %s conflicts with provided value in %s" % (k, args_dict['constraints']))

    # Set any defaults that are neither provided on the command line nor in the options file
    if 'engine' not in args_dict:
        args_dict['engine'] = 'tera'

    return args_dict

def parse_neb_args(*args):

    """
    Read user input from the command line interface.
    Designed to be called by neb.main() passing in sys.argv[1:]
    """

    parser = ArgumentParserWithFile(add_help=False, formatter_class=argparse.RawTextHelpFormatter, fromfile_prefix_chars='@')

    grp_univ = parser.add_argument_group('universal', 'Relevant to every job')
    grp_univ.add_argument('input', type=str, help='REQUIRED positional argument: Quantum chemistry or MM input file for calculation\n ')
    grp_univ.add_argument('chain_coords', type=str, help='REQUIRED positional argument: Coordinate file containing multiple frames for NEB\n ')
    # TeraChem as a default option is only for the command line interface.
    grp_univ.add_argument('--engine', type=str, help='Specify engine for computing energies and gradients.\n'
                          '"tera" = TeraChem (default)         "qchem" = Q-Chem\n'
                          '"psi4" = Psi4                       "openmm" = OpenMM (pass a force field or XML input file)\n'
                          '"molpro" = Molpro                   "gmx" = Gromacs (pass conf.gro; requires topol.top and shot.mdp\n '
                          '"gaussian" = Gaussian09/16          "ase" = ASE calculator, use --ase-class/--ase-kwargs\n '
                          '"quick" = QUICK\n')
    grp_univ.add_argument('--nt', type=int, help='Specify number of threads for running in parallel\n(for TeraChem this should be number of GPUs)')

    grp_nebparam = parser.add_argument_group('nebparam', 'Control the NEB calculation')
    grp_nebparam.add_argument('--maxg', type=float, help='Converge when the maximum RMS-gradient of all images falls below this threshold (default 0.05 ev/Ang).\n ')
    grp_nebparam.add_argument('--avgg', type=float, help='Converge when the average RMS-gradient of all images falls below this threshold (default 0.025 ev/Ang).\n ')
    grp_nebparam.add_argument('--guessk', type=float, help='Guess the initial Hessian eigenvalue for displacements (default 0.05).\n ')
    #HP 5/10/2024: guessw will be enabled once IC NEB is implemented.
    #grp_nebparam.add_argument('--guessw', type=float, help='Guess weight for chain coordinates (default 0.1).\n ')
    grp_nebparam.add_argument('--nebk', type=float, help='NEB spring constant in units of kcal/mol/Ang^2 (default 1.0).\n ')
    #HP 1/16/2025: neb_history is commented out because rebuilding the Hessian based on changes in IC isn't available.
    #grp_nebparam.add_argument('--neb_history', type=int, help='Chain history to keep in memory; note chains are very memory intensive, >1 GB each (default 1).\n ')
    grp_nebparam.add_argument('--neb_maxcyc', type=int, help='Maximum number of chain optimization cycles to perform (default 100).\n ')
    grp_nebparam.add_argument('--climb', type=float, help='Activate climbing image for max-energy points when max RMS-gradient falls below this threshold (default 0.5).\n ')
    grp_nebparam.add_argument('--ncimg', type=int, help='Number of climbing images to expect (default 1).\n ')
    grp_nebparam.add_argument('--images', type=int, help='Number of NEB images to use (default 11).\n ')
    grp_nebparam.add_argument('--plain', type=int, help='1: Use plain elastic band for spring force. 2: Use plain elastic band for spring AND potential (default 0).\n ')
    grp_nebparam.add_argument('--optep', type=str2bool, help='Provide "yes" to optimize two end points of the initial input chain.\n ')
    grp_nebparam.add_argument('--align', type=str2bool, help='Align images before starting the NEB method (default yes). If "--optep" is set to "yes", the images will be aligned after optimizing the end points.\n ')
    grp_nebparam.add_argument('--trust', type=float, help='Starting trust radius (default 0.1). \n ')
    grp_nebparam.add_argument('--tmax', type=float, help='Maximum trust radius (default 0.3).\n ')
    grp_nebparam.add_argument('--tmin', type=float, help='Minimum trust radius, do not reject steps trust radius is below this threshold.\n ')
    grp_nebparam.add_argument('--skip', type=str2bool, help='Setting it to ``yes`` will skip Hessian updates that would introduce negative eigenvalues instead of resetting it. By default, it will reset the Hessian when negative Hessian eigenvalues are detected.\n ')
    grp_nebparam.add_argument('--epsilon', type=float, help='Small eigenvalue threshold for resetting Hessian, default 1e-5.\n ')
    grp_nebparam.add_argument('--wqport', type=int, help='Work Queue port used to distribute singlepoint calculations. Workers must be started separately.\n ')
    grp_nebparam.add_argument('--bigchem', type=str2bool, help='Provide "Yes" to use BigChem for performing the NEB calculation in parallel. \n'
                                                               'Please ensure that BigChem is running with workers properly. \n')
    grp_output = parser.add_argument_group('output', 'Control the format and amount of the output')
    grp_output.add_argument('--prefix', type=str, help='Specify a prefix for log file and temporary directory.\n'
                            'Defaults to the input file path (incl. file name with extension removed).\n ')
    grp_output.add_argument('--verbose', type=int, help='Set to positive for more verbose printout.\n'
                            '0 = Default print level.     1 = Basic info about optimization step.\n'
                            '2 = Include microiterations. 3 = Lots of printout from low-level functions.\n ')
    grp_help = parser.add_argument_group('help', 'Get help')
    grp_help.add_argument('-h', '--help', action='help', help='Show this help message and exit')

    # Keep all arguments whose values are not None, so that the setting of default values
    # in OptParams() and get_molecule_engine() will work properly.
    args_dict = {}
    for k, v in vars(parser.parse_args(*args)).items():
        if v is not None:
            args_dict[k] = v

    # Check that the input file exists
    # OpenMM .xml files don't have to be in the current folder.
    if not args_dict['input'].endswith('.xml') and not os.path.exists(args_dict['input']):
        raise RuntimeError("Input file does not exist")

    # Set any defaults that are neither provided on the command line nor in the options file
    if 'engine' not in args_dict:
        args_dict['engine'] = 'tera'

    return args_dict
