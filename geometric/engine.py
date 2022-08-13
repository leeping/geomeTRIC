"""
engine.py: Communicate with QM or MM software packages for energy/gradient info

Copyright 2016-2020 Regents of the University of California and the Authors

Authors: Lee-Ping Wang, Chenchen Song

Contributors: Yudong Qiu, Daniel G. A. Smith, Sebastian Lee, Qiming Sun,
Alberto Gobbi, Josh Horton

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

import shutil
import subprocess
from collections import OrderedDict
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
import re
import os

from .molecule import Molecule
from .nifty import bak, au2ev, eqcgmx, fqcgmx, bohr2ang, logger, getWorkQueue, queue_up_src_dest, rootdir, splitall, copy_tree_over
from .errors import EngineError, CheckCoordError, Psi4EngineError, QChemEngineError, TeraChemEngineError, \
    ConicalIntersectionEngineError, OpenMMEngineError, GromacsEngineError, MolproEngineError, QCEngineAPIEngineError, GaussianEngineError

# Strings matching common DFT functionals
# exclude "pw", "scan" because they might cause false positives
dft_strings = ["lda", "svwn", "lyp", "b88", "p86", "b97", "hcth", "tpss", "hse", 
               "hjs", "pbe", "m05", "m06", "m08", "m11", "m12", "m15", "gga"]

#=============================#
#| Useful TeraChem functions |#
#=============================#

def edit_tcin(fin=None, fout=None, options=None, defaults=None, reqxyz=True, ignore_sections=True):
    """
    Parse, modify, and/or create a TeraChem input file.

    Parameters
    ----------
    fin : str, optional
        Name of the TeraChem input file to be read
    fout : str, optional
        Name of the TeraChem output file to be written, if desired
    options : dict, optional
        Dictionary of options to overrule TeraChem input file. Pass None as value to delete a key.
    defaults : dict, optional
        Dictionary of options to add to the end
    reqxyz : bool, optional
        Require .xyz file to be present in the current folder
    ignore_sections : bool, optional
        Do not parse any blocks delimited by dollar signs (not copied to output and not returned)

    Returns
    -------
    dictionary
        Keys mapped to values as strings.  Certain keys will be changed to integers (e.g. charge, spinmult).
        Keys are standardized to lowercase.
    """
    if defaults is None:
        defaults = {}
    if options is None:
        options = {}
    if not ignore_sections:
        raise RuntimeError("Currently only ignore_constraints=True is supported")
    intkeys = ['charge', 'spinmult']
    Answer = OrderedDict()
    # Read from the input if provided
    if fin is not None:
        tcin_dirname = os.path.dirname(os.path.abspath(fin))
        section_mode = False
        for line in open(fin).readlines():
            line = line.split("#")[0].strip()
            if len(line) == 0: continue
            if line == '$end':
                section_mode = False
                continue
            elif line.startswith("$"):
                section_mode = True
            if section_mode : continue
            if line == 'end': break
            s = line.split(' ', 1)
            k = s[0].lower()
            try:
                v = s[1].strip()
            except IndexError:
                raise RuntimeError("%s contains an error on the following line:\n%s" % (fin, line))
            if k == 'coordinates' and reqxyz:
                if not os.path.exists(os.path.join(tcin_dirname, v.strip())):
                    raise RuntimeError("TeraChem coordinate file does not exist")
            if k in intkeys:
                v = int(v)
            if k in Answer:
                raise RuntimeError("Found duplicate key in TeraChem input file: %s" % k)
            Answer[k] = v
    # Replace existing keys with ones from options
    for k, v in options.items():
        Answer[k] = v
    # Append defaults to the end
    Answer2 = Answer.copy()
    for k, v in defaults.items():
        if k not in Answer.keys():
            Answer2[k] = v
    Answer = Answer2.copy()
    for k, v in Answer.items():
        if v is None:
            del Answer2[k]
    Answer = Answer2.copy()
    # Print to the output if provided
    havekeys = []
    if fout is not None:
        with open(fout, 'w') as f:
            # If input file is provided, try to preserve the formatting
            if fin is not None:
                for line in open(fin).readlines():
                    # Find if the line contains a key
                    haveKey = False
                    uncomm = line.split("#", 1)[0].strip()
                    # Don't keep anything past the 'end' keyword
                    if uncomm.lower() == 'end': break
                    if len(uncomm) > 0:
                        haveKey = True
                        comm = line.split("#", 1)[1].replace('\n','') if len(line.split("#", 1)) == 2 else ''
                        s = line.split(' ', 1)
                        w = re.findall('[ ]+',uncomm)[0]
                        k = s[0].lower()
                        if k in Answer:
                            line_out = k + w + str(Answer[k]) + comm
                            havekeys.append(k)
                        else:
                            line_out = line.replace('\n', '')
                    else:
                        line_out = line.replace('\n', '')
                    print(line_out, file=f)
            for k, v in Answer.items():
                if k not in havekeys:
                    print("%-25s     %s" % (k, str(v)), file=f)
    return Answer

def set_tcenv():
    if 'TeraChem' not in os.environ:
        raise RuntimeError('Please set TeraChem environment variable')
    TCHome = os.environ['TeraChem']
    os.environ['PATH'] = os.path.join(TCHome,'bin')+":"+os.environ['PATH']
    os.environ['LD_LIBRARY_PATH'] = os.path.join(TCHome,'lib')+":"+os.environ['LD_LIBRARY_PATH']

def load_tcin(f_tcin):
    tcdef = OrderedDict()
    tcdef['convthre'] = "3.0e-6"
    tcdef['threall'] = "1.0e-13"
    tcdef['scf'] = "diis+a"
    tcdef['maxit'] = "50"
    # tcdef['dftgrid'] = "1"
    # tcdef['precision'] = "mixed"
    # tcdef['threspdp'] = "1.0e-8"
    tcin = edit_tcin(fin=f_tcin, options={'run':'gradient'}, defaults=tcdef)
    return tcin

#====================================#
#| Classes for external codes used  |#
#| to calculate energy and gradient |#
#====================================#

class Engine(object):
    def __init__(self, molecule):
        if len(molecule) != 1:
            raise RuntimeError('Please pass only length-1 molecule objects to engine creation')
        self.M = deepcopy(molecule)
        self.stored_calcs = OrderedDict()

    # def __deepcopy__(self, memo):
    #     return copy(self)

    def calc(self, coords, dirname, read_data=False, copydir=None):
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
        read_data : bool, default=False
            If valid calculation output files exist in dirname, read the results instead of
            running a new calculation
        copydir : str, default=None
            If provided, the contents of this folder will be copied to the scratch folder
            prior to starting a calculation (e.g. when calculating the Hessian we want to use SCF
            guess of the midpoint)

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
        coord_hash = hash(coords.tostring())
        if coord_hash in self.stored_calcs:
            result = self.stored_calcs[coord_hash]['result']
        else:
            # If the read_data flag is set to True, then attempt to read the
            # result from the temp-folder, then skip the calculation if successful.
            read_success = False
            if read_data and os.path.exists(dirname) and hasattr(self, 'read_result'):
                try:
                    result = self.read_result(dirname, check_coord=coords)
                    read_success = True
                    logger.info("Successfully read existing single-point result from %s\n" % dirname)
                except (EngineError, CheckCoordError): pass
            if not read_success:
                if copydir:
                    self.copy_scratch(copydir, dirname)
                elif not os.path.exists(dirname): os.makedirs(dirname)
                result = self.calc_new(coords, dirname)
            self.stored_calcs[coord_hash] = {'coords':coords, 'result':result}
        return result

    def clearCalcs(self):
        self.stored_calcs = OrderedDict()

    def calc_new(self, coords, dirname):
        raise NotImplementedError("Not implemented for the base class")

    def calc_wq(self, coords, dirname, read_data=False, copydir=None):
        """
        Top-level method for submitting a single-point calculation using Work Queue. 
        Different from calc(), this method does not return results, because the control
        flow involves submitting calculations to WQ and gathering data after calculations
        are complete.
        
        Calculation will be skipped if results are contained in the hash table, 
        and optionally, can skip calculation if output exists on disk (which is 
        useful in the case of restarting a crashed Hessian calculation)
        
        Parameters
        ----------
        coords : np.array
            1-dimensional array of shape (3*N_atoms) containing atomic coordinates in Bohr
        dirname : str
            Relative path containing calculation files
        read_data : bool, default=False
            If valid calculation output files exist in dirname, read the results instead of
            running a new calculation
        copydir : str, default=None
            If provided, the contents of this folder will be copied to the scratch folder
            prior to starting a calculation (e.g. when calculating the Hessian we want to use SCF
            guess of the midpoint)
        """
        coord_hash = hash(coords.tostring())
        if coord_hash in self.stored_calcs:
            return
        else:
            # If the read_data flag is set to True, then attempt to read the
            # result from the temp-folder, then skip the calculation if successful.
            read_success = False
            if read_data and os.path.exists(dirname) and hasattr(self, 'read_result'):
                try:
                    result = self.read_result(dirname, check_coord=coords)
                    read_success = True
                except (EngineError, CheckCoordError): pass
            if not read_success:
                if copydir:
                    self.copy_scratch(copydir, dirname)
                self.calc_wq_new(coords, dirname)

    def calc_wq_new(self, coords, dirname):
        raise NotImplementedError("Work Queue is not implemented for this class")

    def read_wq(self, coords, dirname):
        """
        Read Work Queue results after all jobs are completed.

        Parameters
        ----------
        coords : np.array
            1-dimensional array of shape (3*N_atoms) containing atomic coordinates in Bohr.
            Used for retrieving results from hash table (in which case calc was not submitted)
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
        coord_hash = hash(coords.tostring())
        if coord_hash in self.stored_calcs:
            result = self.stored_calcs[coord_hash]['result']
        else:
            if not os.path.exists(dirname):
                raise RuntimeError("In read_wq, %s doesn't exist" % dirname)
            result = self.read_result(dirname)
            self.stored_calcs[coord_hash] = {'coords':coords, 'result':result}
        return result

    def number_output(self, dirname, calcNum):
        return

    def copy_scratch(self, src, dest):
        logger.warning("copy_scratch not implemented for this engine\n")
        return

    def save_guess_files(self, dirname):
        return

    def load_guess_files(self, dirname):
        return

    def detect_dft(self):
        return False

class Blank(Engine):
    """
    Always return zero energy and gradient.
    """
    def __init__(self, molecule):
        super(Blank, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        energy = 0.0
        gradient = np.zeros(len(coords), dtype=float)
        return {'energy':energy, 'gradient':gradient}

class TeraChem(Engine): # pragma: no cover
    """
    Run a TeraChem energy and gradient calculation.
    """
    def __init__(self, molecule, tcin, dirname=None):
        self.tcin = tcin.copy()
        # Scratch folder
        if 'scrdir' in self.tcin:
            self.scr = self.tcin['scrdir']
        else:
            self.scr = 'scr'
        # A few notes about the electronic structure method
        self.casscf = self.tcin.get('casscf', 'no').lower() == 'yes'
        self.unrestricted = (self.tcin['method'][0] == 'u')
        # Build a list of guess files
        if self.casscf and 'casguess' in self.tcin:
            # CASSCF guess uses key 'casguess' and skips SCF entirely
            guessVal = self.tcin['casguess'].split()
            self.initguess_mode = 'file'
            self.initguess_files = guessVal
        elif 'guess' in self.tcin:
            guessVal = self.tcin['guess'].split()
            if guessVal[0] in ['frag', 'sad', 'sadlp', 'exciton', 'frag_scf']:
                self.initguess_mode = guessVal[0]
                self.initguess_files = guessVal[1:]
            elif guessVal[0] in ['hcore', 'generate']:
                self.initguess_mode = guessVal[0]
                self.initguess_files = []
            else:
                self.initguess_mode = 'file'
                self.initguess_files = guessVal
        else:
            self.initguess_mode = 'none'
            self.initguess_files = []
        # Check that all starting guess files exist
        for f in self.initguess_files:
            if not os.path.exists(f):
                raise TeraChemEngineError('%s guess file is missing' % f)
        # Management of QM/MM: Read qmindices and 
        # store locations of prmtop and qmindices files,
        self.qmmm = 'qmindices' in tcin
        if self.qmmm:
            if not os.path.exists(tcin['coordinates']):
                raise RuntimeError("TeraChem QM/MM coordinate file does not exist")
            if not os.path.exists(tcin['prmtop']):
                raise RuntimeError("TeraChem QM/MM prmtop file does not exist")
            if not os.path.exists(tcin['qmindices']):
                raise RuntimeError("TeraChem QM/MM qmindices file does not exist")
            self.qmindices_name = os.path.abspath(tcin['qmindices'])
            self.prmtop_name = os.path.abspath(tcin['prmtop'])
            self.qmindices = [int(i.split()[0]) for i in open(self.qmindices_name).readlines()]
            self.M_full = Molecule(tcin['coordinates'], ftype='inpcrd', build_topology=False)
        super(TeraChem, self).__init__(molecule)

    def orbital_filenames(self):
        """
        Names of orbital files generated by TeraChem calculations.
        """
        orbfnms = []
        if self.casscf:
            orbfnms.append('c0.casscf')
        elif self.unrestricted:
            orbfnms.append('ca0')
            orbfnms.append('cb0')
        else:
            orbfnms.append('c0')
        return orbfnms

    def copy_guess_files(self, dirname):
        """
        Prior to running a TeraChem gradient calculation, 
        copy guess files to expected locations and make edits
        to the TeraChem input file to use these files.

        Guess files are used in the following priority:
        1) If default orbital filenames exist in dirname/scr e.g. from a previous calculation, 
           they will supersede the user-provided initial guess for the current calculation
        2) Otherwise, the user-provided initial guess will be used.
        
        These files are copied to "dirname", either from <root>dirname/scr in the former case,
        or from <root> in the latter case. (<root> is the folder in which the calculation is run.)

        Returns
        -------
        copied_files : list
            The list of guess files that the TeraChem calculation will use.
        """
        # Default scratch file names written by TeraChem to run.tmp/scr
        orbital_files = self.orbital_filenames()
        # Names of scratch files that were actually used (all in the run.tmp folder)
        copied_files = []
        if all([os.path.exists(os.path.join(dirname, self.scr, f)) for f in orbital_files]):
            # If all scratch files (with default names) exist in run.tmp/scr folder, e.g. from a previous calc,
            # then copy it to the run.tmp folder and use it as the guess in the current calculation.
            for f in orbital_files:
                shutil.copy2(os.path.join(dirname, self.scr, f), os.path.join(dirname, f))
                copied_files.append(f)
            if 'purify' not in self.tcin: 
                self.tcin['purify'] = 'no'
            if 'mixguess' not in self.tcin: 
                self.tcin['mixguess'] = "0.0"
            if self.casscf:
                self.tcin['scf'] = 'diis'
            self.tcin['casguess' if self.casscf else 'guess'] = ' '.join(orbital_files)
        elif self.initguess_mode != 'none':
            # If scratch files from previous calc do not exist, then copy initial guess files
            for f in self.initguess_files:
                if os.path.exists(f):
                    shutil.copy2(f, os.path.join(dirname, f))
                else:
                    raise TeraChemEngineError("%s guess file is missing and this code shouldn't be called" % f)
            if self.initguess_mode == 'file':
                self.tcin['casguess' if self.casscf else 'guess'] = ' '.join(self.initguess_files)
                if self.casscf: self.tcin['scf'] = 'diis'
            elif self.initguess_mode != 'none':
                self.tcin['guess'] = ' '.join([self.initguess_mode] + self.initguess_files)
            copied_files = self.initguess_files[:]
        return copied_files

    def save_guess_files(self, dirname):
        for f in self.orbital_filenames():
            shutil.copy2(os.path.join(dirname, self.scr, f), os.path.join(dirname, self.scr, f+".sav"))
        
    def load_guess_files(self, dirname):
        for f in self.orbital_filenames():
            if os.path.exists(os.path.join(dirname, self.scr, f+".sav")):
                logger.info("Restoring guess file from %s\n" % os.path.join(dirname, self.scr, f+".sav"))
                shutil.copy2(os.path.join(dirname, self.scr, f+".sav"), os.path.join(dirname, self.scr, f))

    def calc_new(self, coords, dirname):
        # Ensure guess files are in the correct locations
        self.copy_guess_files(dirname)
        # Set coordinate file name
        start_xyz = 'start.rst7' if self.qmmm else 'start.xyz'
        self.tcin['coordinates'] = start_xyz
        self.tcin['run'] = 'gradient'
        # Write the TeraChem input file
        edit_tcin(fout="%s/run.in" % dirname, options=self.tcin)
        # Back up any existing output files
        # Commented out (should be enabled during debuggin')
        bak('run.out', cwd=dirname, start=0)
        bak(start_xyz, cwd=dirname, start=0)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        if self.qmmm:
            # Copy QMMM files to the correct locations and set positions in inpcrd/rst7 file
            shutil.copy2(self.qmindices_name, dirname)
            shutil.copy2(self.prmtop_name, dirname)
            self.M_full.xyzs[0][self.qmindices, :] = self.M.xyzs[0]
            self.M_full[0].write(os.path.join(dirname, start_xyz), ftype='inpcrd')
        else:
            self.M[0].write(os.path.join(dirname, start_xyz))
        # Run TeraChem
        subprocess.check_call('terachem run.in > run.out', cwd=dirname, shell=True)
        # Extract energy and gradient
        result = self.read_result(dirname)
        return result

    def calc_bondorder(self, coords, dirname):
        self.tcin['bond_order_mat'] = 'yes'
        self.calc_new(coords, dirname)
        bo_mat = []
        for ln, line in enumerate(open(os.path.join(dirname, self.scr, 'bond_order.mat')).readlines()):
            if ln >= 2:
                bo_mat.append([float(i) for i in line.split()[1:]])
        del self.tcin['bond_order_mat']
        return np.array(bo_mat)

    def calc_wq_new(self, coords, dirname):
        # Set up Work Queue object
        wq = getWorkQueue()
        scrdir = os.path.join(dirname, self.scr)
        if not os.path.exists(dirname): os.makedirs(dirname)
        if not os.path.exists(scrdir): os.makedirs(scrdir)
        # Ensure guess files are in the correct locations
        guessfnms = self.copy_guess_files(dirname)
        # Set coordinate file name
        start_xyz = 'start.rst7' if self.qmmm else 'start.xyz'
        self.tcin['coordinates'] = start_xyz
        self.tcin['run'] = 'gradient'
        # For queueing up jobs, delete GPU key and let the worker decide
        self.tcin['gpus'] = None
        # Write the TeraChem input file
        edit_tcin(fout="%s/run.in" % dirname, options=self.tcin)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        if self.qmmm:
            # Copy QMMM files to the correct locations and set positions in inpcrd/rst7 file
            shutil.copy2(self.qmindices_name, dirname)
            shutil.copy2(self.prmtop_name, dirname)
            self.M_full.xyzs[0][self.qmindices, :] = self.M.xyzs[0]
            self.M_full[0].write(os.path.join(dirname, start_xyz), ftype='inpcrd')
        else:
            self.M[0].write(os.path.join(dirname, start_xyz))
        # Specify WQ input and output files
        in_files = [('%s/run.in' % dirname, 'run.in'), ('%s/%s' % (dirname, start_xyz), start_xyz)]
        if self.qmmm:
            qmindices_filename = os.path.split(self.qmindices_name)[1]
            prmtop_filename = os.path.split(self.prmtop_name)[1]
            in_files += [("%s/%s" % (dirname, qmindices_filename), qmindices_filename)]
            in_files += [("%s/%s" % (dirname, prmtop_filename), prmtop_filename)]
        for f in guessfnms:
            in_files.append((os.path.join(dirname, f), f))
        out_files = [('%s/run.out' % dirname, 'run.out')]
        out_scr = self.orbital_filenames()
        out_scr += ['grad.xyz', 'mullpop']
        for f in out_scr:
            out_files.append((os.path.join(dirname, self.scr, f), os.path.join(self.scr, f)))
        queue_up_src_dest(wq, "terachem run.in &> run.out", in_files, out_files, verbose=False, print_time=600)

    def number_output(self, dirname, calcNum):
        if not os.path.exists(os.path.join(dirname, 'run.out')):
            raise RuntimeError('run.out does not exist')
        shutil.copy2(os.path.join(dirname,start_xyz), os.path.join(dirname,'start_%03i.%s' % (calcNum, os.path.splitext(start_xyz)[1])))
        shutil.copy2(os.path.join(dirname,'run.out'), os.path.join(dirname,'run_%03i.out' % calcNum))

    def read_result(self, dirname, check_coord=None):
        if check_coord is not None:
            read_xyz_success = False
            start_xyz = 'start.rst7' if self.qmmm else 'start.xyz'
            if os.path.exists(os.path.join(dirname, start_xyz)):
                try:
                    M = Molecule(os.path.join(dirname, start_xyz))
                    read_xyz = M.xyzs[0][self.qmindices] if self.qmmm else M.xyzs[0]
                    read_xyz = read_xyz.flatten() / bohr2ang
                    read_xyz_success = True
                except: pass
            if not read_xyz_success or np.linalg.norm(check_coord - read_xyz) > 1e-8:
                # If the upcoming calculation is for a different geometry than the existing one,
                # then delete the guess files to prevent landing in the wrong state by accident.
                for f in self.orbital_filenames():
                    if os.path.exists(os.path.join(dirname, f)):
                        os.remove(os.path.join(dirname, f))
                    if os.path.exists(os.path.join(dirname, self.scr, f)):
                        os.remove(os.path.join(dirname, self.scr, f))
                raise CheckCoordError
        # Extract energy and gradient
        try:
            # LPW note: Using python to call awk is not ideal and would take a bit of elbow grease to fix in the future.
            subprocess.call("awk '/FINAL ENERGY/ {p=$3} /Correlation Energy/ {p+=$5} /FINAL Target State Energy/ {p=$5} END {printf \"%.10f\\n\", p}' run.out > energy.txt", cwd=dirname, shell=True)
            subprocess.call("awk '/Gradient units are Hartree/,/Net gradient|Point charge part/ {if ($1 ~ /^-?[0-9]/) {print}}' run.out > grad.txt", cwd=dirname, shell=True)
            subprocess.call("awk 'BEGIN {s=0} /SPIN S-SQUARED/ {s=$3} END {printf \"%.6f\\n\", s}' run.out > s-squared.txt", cwd=dirname, shell=True)
            energy = float(open(os.path.join(dirname,'energy.txt')).readlines()[0].strip())
            na = len(self.qmindices) if self.qmmm else self.M.na
            gradient = np.loadtxt(os.path.join(dirname,'grad.txt'))[:na].flatten()
            s2 = float(open(os.path.join(dirname,'s-squared.txt')).readlines()[0].strip())
            assert gradient.shape[0] == self.M.na*3
        except (OSError, IOError, IndexError, RuntimeError, AssertionError, subprocess.CalledProcessError):
            raise TeraChemEngineError
        return {'energy':energy, 'gradient':gradient, 's2':s2}

    def copy_scratch(self, src, dest):
        if os.path.split(self.scr)[0]:
            raise TeraChemEngineError("copy_scratch cannot be used because %s contains subfolders" % self.scr)
        if not os.path.exists(dest): os.makedirs(dest)
        if not os.path.exists(os.path.join(src, self.scr)):
            raise TeraChemEngineError("Trying to copy %s but it does not exist" % os.path.join(src, self.scr))
        copy_tree_over(os.path.join(src, self.scr), os.path.join(dest, self.scr))

    def detect_dft(self):
        for i in dft_strings:
            if i.lower() in self.tcin['method'].lower():
                return True
        return False

class OpenMM(Engine):
    """
    Run a OpenMM energy and gradient calculation.
    """
    def __init__(self, molecule, pdb, xml):
        try:
            import simtk.openmm.app as app
            import simtk.openmm as mm
            import simtk.unit as u
        except ImportError:
            raise ImportError("OpenMM computation object requires the 'simtk' package. Please pip or conda install 'openmm' from omnia channel.")
        pdb = app.PDBFile(pdb)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        xmlSystem = False
        self.combination = None
        self.n_virtual_sites = 0
        if os.path.exists(xml):
            xmlStr = open(xml).read()
            # check if we have opls combination rules if the xml is present
            try:
                self.combination = ET.fromstring(xmlStr).find('NonbondedForce').attrib['combination']
            except (AttributeError, KeyError):
                pass
            try:
                # If the user has provided an OpenMM system, we can use it directly
                system = mm.XmlSerializer.deserialize(xmlStr)
                xmlSystem = True
                logger.info("Treating the provided xml as a system XML file\n")
            except ValueError:
                logger.info("Treating the provided xml as a force field XML file\n")
        else:
            logger.info("xml file not in the current folder, treating as a force field XML file and setting up in gas phase.\n")
        if not xmlSystem:
            try:
                forcefield = app.ForceField(xml)
            except ValueError:
                raise OpenMMEngineError('Provided input file is not an installed force field XML file')
            try:
                system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
            except ValueError:
                modeller.addExtraParticles(forcefield)
                system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
        # apply opls combination rule if we are using it
        if self.combination == 'opls':
            logger.info("\nUsing geometric combination rules\n")
            system = self.opls(system)
        integrator = mm.VerletIntegrator(1.0*u.femtoseconds)
        platform = mm.Platform.getPlatformByName('Reference')
        self.n_virtual_sites = sum([system.isVirtualSite(particle) for particle in range(system.getNumParticles())])
        self.simulation = app.Simulation(pdb.topology, system, integrator, platform)
        super(OpenMM, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        from simtk.openmm import Vec3
        import simtk.unit as u
        try:
            self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
            pos = [Vec3(self.M.xyzs[0][i, 0]/10, self.M.xyzs[0][i, 1]/10, self.M.xyzs[0][i, 2]/10) for i in range(self.M.na)]*u.nanometer
            for _ in range(self.n_virtual_sites):
                pos.extend([Vec3(0, 0, 0)]*u.nanometer)
            self.simulation.context.setPositions(pos)
            if self.n_virtual_sites:
                self.simulation.context.computeVirtualSites()
            state = self.simulation.context.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / eqcgmx
            gradient = state.getForces(asNumpy=True).flatten() / fqcgmx
            if self.n_virtual_sites:
                gradient = gradient[:-self.n_virtual_sites*3]
        except:
            raise OpenMMEngineError
        return {'energy': energy, 'gradient': gradient}

    @staticmethod
    def opls(system):
        """Apply the opls combination rule to the system."""

        from numpy import sqrt
        import simtk.openmm as mm

        # get system information from the openmm system
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in
                  range(system.getNumForces())}
        # use the nondonded_force tp get the same rules
        nonbonded_force = forces['NonbondedForce']
        lorentz = mm.CustomNonbondedForce(
            'epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)*4.0')
        lorentz.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
        lorentz.addPerParticleParameter('sigma')
        lorentz.addPerParticleParameter('epsilon')
        lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
        system.addForce(lorentz)
        ljset = {}
        # Now for each particle calculate the combination list again
        for index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            # print(nonbonded_force.getParticleParameters(index))
            ljset[index] = (sigma, epsilon)
            lorentz.addParticle([sigma, epsilon])
            nonbonded_force.setParticleParameters(
                index, charge, 0, 0)
        for i in range(nonbonded_force.getNumExceptions()):
            (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
            # ALL THE 12,13 interactions are EXCLUDED FROM CUSTOM NONBONDED FORCE
            # All 1,4 are scaled by the amount in the xml file
            lorentz.addExclusion(p1, p2)
            if eps._value != 0.0:
                # combine sigma using the geometric combination rule
                sig14 = sqrt(ljset[p1][0] * ljset[p2][0])
                nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
        return system

    def copy_scratch(self, src, dest):
        return

class Gaussian(Engine):
    """
    Run a Gaussian energy and gradient calculation.
    """
    def __init__(self, molecule, exe=None, threads=None):
        super(Gaussian, self).__init__(molecule)
        self.threads = threads
        if exe.lower() in ("g16", "g09"):
            self.gaussian_exe = exe.lower()
        else:
            raise ValueError("Only g16 and g09 are supported.")

    def load_gaussian_input(self, gaussian_input):
        """
        We can read the .com files using molecule but we can not write them so use the template method.

        Note only Gaussian cartesian coordinates are supported.
        We also edit the checkpoint file name and change the processors flag if not present.

        Example input file:

        %Mem=6GB
        %NProcShared=2
        %Chk=lig
        # B3LYP/6-31G(d) Force=NoStep

        water energy

        0   1
        O  -0.464   0.177   0.0
        H  -0.464   1.137   0.0
        H   0.441  -0.143   0.0


        """
        reading_molecule, found_geo, found_force = False, False, False
        gauss_temp = []  # store a template of the input file for generating new ones
        with open(gaussian_input) as gauss_in:
            for line in gauss_in:
                match = re.search(r"^ *[A-Z][a-z]?(.*[-+]?([0-9]*\.)?[0-9]+){3}$", line)
                if match is not None:
                    reading_molecule = True
                    if not found_geo:
                        found_geo = True
                        gauss_temp.append("$!geometry@here")

                elif reading_molecule:
                    if line.strip() == '':
                        reading_molecule = False
                        gauss_temp.append(line)

                elif "%nprocshared" in line.lower():
                    # we should replace the line with our threads value
                    if self.threads is not None:
                        gauss_temp.append("%NProcShared=" + str(self.threads) + "\n")
                    else:
                        gauss_temp.append(line)

                elif "%chk" in line.lower():
                    # we should replace it with what we want
                    gauss_temp.append("%Chk=ligand\n")

                elif line.startswith('#'):
                    self.route_line = line

                    if "force=nostep" in line.lower():
                        found_force = True
                    gauss_temp.append("$!route@here")

                else:
                    gauss_temp.append(line)

        if not found_force:
            raise RuntimeError("Gaussian inputfile %s should have force=nostep in route line." % gaussian_input)

        # now we need to make sure the chk point file and threads were set
        if not any("%chk" in command.lower() for command in gauss_temp):
            # insert at the top
            gauss_temp.insert(0, "%Chk=ligand\n")
        if not any("%nprocshared" in command.lower() for command in gauss_temp):
            # if threads is not none set it else use 1 thread
            thread_str = "%NProcShared=" + str(self.threads or 1)
            gauss_temp.insert(0, thread_str + "\n")

        self.gauss_temp = gauss_temp

    def calc_new(self, coords, dirname):
        """
        Run the gaussian single point calculation using the given exe.
        """
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        # Write Gaussian com file
        with open(os.path.join(dirname, 'gaussian.com'), 'w') as outfile:
            for line in self.gauss_temp:
                if line == '$!route@here':
                    if 'guess' not in self.route_line.lower() and os.path.exists(os.path.join(dirname, 'ligand.chk')):
                        outfile.write(self.route_line.replace('\n','') + " Guess=Read\n")
                    else:
                        outfile.write(self.route_line)
                elif line == '$!geometry@here':
                    for i, (e, c) in enumerate(zip(self.M.elem, self.M.xyzs[0])):
                        outfile.write("%-7s %13.7f %13.7f %13.7f\n" % (e, c[0], c[1], c[2]))
                else:
                    outfile.write(line)
        try:
            # Run Gaussian
            subprocess.check_call('%s < gaussian.com > gaussian.log && formchk ligand.chk ligand.fchk > form_log.txt' % self.gaussian_exe, cwd=dirname, shell=True)
            # Read energy and gradients from Gaussian output
            result = self.read_result(dirname)
        except (OSError, IOError, RuntimeError, subprocess.CalledProcessError):
            raise GaussianEngineError
        return result

    def read_result(self, dirname, check_coord=None):
        """
        Read the result of the output file to get the gradient and energy.
        """
        if check_coord is not None:
            raise CheckCoordError("Coordinate checking not implemented")
        energy, gradient = None, None
        # first get the energy from the formatted checkpoint file, works for all methods
        fchk_out = os.path.join(dirname, "ligand.fchk")
        with open(fchk_out) as fchk:
            for line in fchk:
                if "Total Energy" in line:
                    energy = float(line.split()[-1])
                    break
        # now we get the gradient from the output in Hartrees/Bohr
        gaussian_out = os.path.join(dirname, 'gaussian.log')
        with open(gaussian_out) as outfile:
            found_grad = False
            for line in outfile:
                line_strip = line.strip()
                if " Forces (Hartrees/Bohr)" in line_strip:
                    found_grad = True
                    gradient = []
                elif found_grad:
                    ls = line_strip.split()
                    if len(ls) == 5 and ls[0].isdigit() and ls[1].isdigit():
                        gradient.append([-float(g) for g in ls[2:]])
                    elif "Cartesian Forces:  Max" in line:
                        found_grad = False
        if energy is None:
            raise RuntimeError("Gaussian energy is not found in %s, please check." % fchk_out)
        if gradient is None:
            raise RuntimeError("Gaussian gradient is not found in %s, please check." % gaussian_out)
        gradient = np.array(gradient, dtype=np.float64).ravel()
        return {'energy':energy, 'gradient':gradient}

    def detect_dft(self):
        for i in dft_strings:
            if i.lower() in self.route_line.lower():
                return True
        return False

class Psi4(Engine):
    """
    Run a Psi4 energy and gradient calculation.
    """
    def __init__(self, molecule=None, threads=None):
        # molecule.py can not parse psi4 input yet, so we use self.load_psi4_input() as a walk around
        if molecule is None:
            # create a fake molecule
            molecule = Molecule()
            molecule.elem = ['H']
            molecule.xyzs = [[[0,0,0]]]
        super(Psi4, self).__init__(molecule)
        self.threads = threads

    def nt(self):
        if self.threads is not None:
            return " -n %i" % self.threads
        else:
            return ""

    def load_psi4_input(self, psi4in):
        """ Psi4 input file parser, only support xyz coordinates for now """
        coords = []
        elems = []
        fragn = []
        found_molecule, found_geo, found_gradient, found_symmetry, found_no_reorient, found_no_com = False, False, False, False, False, False
        psi4_temp = [] # store a template of the input file for generating new ones
        for line in open(psi4in):
            if 'molecule' in line:
                found_molecule = True
                psi4_temp.append(line)
            elif found_molecule is True:
                ls = line.split()
                if len(ls) == 4:
                    if found_geo == False:
                        found_geo = True
                        psi4_temp.append("$!geometry@here")
                    # parse the xyz format
                    elems.append(ls[0])
                    coords.append(ls[1:4])
                elif '--' in line:
                    fragn.append(len(elems))
                elif 'symmetry' in line:
                    found_symmetry = True
                    if line.split()[1].lower() != 'c1':
                        raise Psi4EngineError("Symmetry must be set to c1 to prevent rotations of the coordinate frame.")
                elif 'no_reorient' in line or 'noreorient' in line:
                    found_no_reorient = True
                elif 'no_com' in line or 'nocom' in line:
                    found_no_com = True
                elif 'units' in line:
                    if line.split()[1].lower()[:3] != 'ang':
                        raise Psi4EngineError("Must use Angstroms as coordinate input.")
                else:
                    if '}' in line:
                        found_molecule = False
                        if not found_no_com:
                            psi4_temp.append("no_com\n")
                        if not found_no_reorient:
                            psi4_temp.append("no_reorient\n")
                        if not found_symmetry:
                            psi4_temp.append("symmetry c1\n")
                    psi4_temp.append(line)
            else:
                psi4_temp.append(line)
            if "gradient(" in line:
                found_gradient = True
        if found_gradient == False:
            raise RuntimeError("Psi4 inputfile %s should have gradient() command." % psi4in)
        self.M = Molecule()
        self.M.elem = elems
        self.M.xyzs = [np.array(coords, dtype=np.float64)]
        self.psi4_temp = psi4_temp
        self.fragn = fragn

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        # Write Psi4 input.dat
        with open(os.path.join(dirname, 'input.dat'), 'w') as outfile:
            for line in self.psi4_temp:
                if line == '$!geometry@here':
                    for i, (e, c) in enumerate(zip(self.M.elem, self.M.xyzs[0])):
                        if i in self.fragn:
                            outfile.write('--\n')
                        outfile.write("%-7s %13.7f %13.7f %13.7f\n" % (e, c[0], c[1], c[2]))
                else:
                    outfile.write(line)
        try:
            # Run Psi4
            subprocess.check_call('psi4%s input.dat' % self.nt(), cwd=dirname, shell=True)
            # Read energy and gradients from Psi4 output
            result = self.read_result(dirname)
        except (OSError, IOError, RuntimeError, subprocess.CalledProcessError):
            raise Psi4EngineError
        return result
    
    def read_result(self, dirname, check_coord=None):
        """ Read Psi4 calculation output. """
        if check_coord is not None:
            raise CheckCoordError("Coordinate checking not implemented")
        energy, gradient = None, None
        psi4out = os.path.join(dirname, 'output.dat')
        with open(psi4out) as outfile:
            found_grad = False
            found_num_grad = False
            for line in outfile:
                line_strip = line.strip()
                if line_strip.startswith('*'):
                    # this works for CCSD and CCSD(T) total energy
                    ls = line_strip.split()
                    if len(ls) > 4 and ls[2] == 'total' and ls[3] == 'energy':
                        energy = float(ls[-1])
                elif line_strip.startswith('Total Energy'):
                    # this works for DF-MP2 total energy
                    ls = line_strip.split()
                    if ls[-1] == '[Eh]':
                        energy = float(ls[-2])
                    else:
                        # this works for HF and DFT total energy
                        try:
                            energy = float(ls[-1])
                        except:
                            pass
                elif line_strip == '-Total Gradient:' or line_strip == '-Total gradient:':
                    # this works for most of the analytic gradients
                    found_grad = True
                    gradient = []
                elif found_grad is True:
                    ls = line_strip.split()
                    if len(ls) == 4:
                        if ls[0].isdigit():
                            gradient.append([float(g) for g in ls[1:4]])
                    else:
                        found_grad = False
                        found_num_grad = False
                elif line_strip == 'Gradient written.':
                    # this works for CCSD(T) gradients computed by numerical displacements
                    found_num_grad = True
                    logger.info("found num grad\n")
                elif found_num_grad is True and line_strip.startswith('------------------------------'):
                    for _ in range(4):
                        line = next(outfile)
                    found_grad = True
                    gradient = []
        if energy is None:
            raise RuntimeError("Psi4 energy is not found in %s, please check." % psi4out)
        if gradient is None:
            raise RuntimeError("Psi4 gradient is not found in %s, please check." % psi4out)
        gradient = np.array(gradient, dtype=np.float64).ravel()
        return {'energy':energy, 'gradient':gradient}
    
    def copy_scratch(self, src, dest):
        # Psi4 scratch file handling is complicated and depends on the type of job being run,
        # so we will opt not to store and retrieve scratch files for now.
        return

    def detect_dft(self):
        for line in self.psi4_temp:
            if "gradient(" in line:
                for i in dft_strings:
                    if i.lower() in line.lower():
                        return True
        return False

class QChem(Engine): # pragma: no cover
    def __init__(self, molecule, dirname=None, qcdir=None, threads=None):
        super(QChem, self).__init__(molecule)
        self.threads = threads
        self.prep_temp_folder(dirname, qcdir)

    def prep_temp_folder(self, dirname, qcdir):
        # Provide an initial qchem scratch folder (e.g. supplied initial guess)
        if not qcdir:
            self.qcdir = False
            return
        elif not os.path.exists(qcdir):
            raise QChemEngineError("qcdir points to a folder that doesn't exist")
        if not dirname:
            raise QChemEngineError("If qcdir is provided, dirname must also be provided")
        elif not os.path.exists(dirname):
            os.makedirs(dirname)
        copy_tree_over(qcdir, os.path.join(dirname, "run.d"))
        self.M.edit_qcrems({'scf_guess':'read'})
        self.qcdir = True

    def nt(self):
        if self.threads is not None:
            return " -nt %i" % self.threads
        else:
            return ""

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        self.M.edit_qcrems({'jobtype':'force'})
        # LPW 2019-12-17: In some cases (such as running a FD Hessian calculation)
        # we may use multiple folders for running different calcs using the same engine.
        # When changing to a new folder without run.d, the calculation will crash if scf_guess is set to read.
        # So we check if run.d actually exists in the folder. If not, do not read the guess.
        # This is a hack and at some point we may want some more flexibility in which qcdir we use.
        if self.qcdir and not os.path.exists(os.path.join(dirname, 'run.d')):
            self.qcdir = False
            self.M.edit_qcrems({'scf_guess':None})
        # If symmetry is enabled by default in Q-Chem, it will mess up finite difference Hessian computations and possibly other things.
        self.M.edit_qcrems({'symmetry':'off', 'sym_ignore':'true'})
        self.M[0].write(os.path.join(dirname, 'run.in'))
        try:
            # Run Q-Chem
            if self.qcdir:
                subprocess.check_call('qchem%s run.in run.out run.d > run.log 2>&1' % self.nt(), cwd=dirname, shell=True)
            else:
                subprocess.check_call('qchem%s run.in run.out run.d > run.log 2>&1' % self.nt(), cwd=dirname, shell=True)
                # Assume reading the SCF guess is desirable
                self.qcdir = True
                self.M.edit_qcrems({'scf_guess':'read'})
            result = self.read_result(dirname)
        except (OSError, IOError, RuntimeError, subprocess.CalledProcessError):
            raise QChemEngineError
        return result

    def calc_bondorder(self, coords, dirname):
        # Make a copy of the 'unmodified' molecule object
        M_bak = deepcopy(self.M)
        # Set scf_final_print 1 to get the Mayer bond order
        self.M.edit_qcrems({'scf_final_print':'1'})
        # Actually run the Q-Chem calculation
        self.calc_new(coords, dirname)
        # Read the bond order from the Q-Chem output file
        M_qcout = Molecule(os.path.join(dirname, 'run.out'), build_topology=False)
        # Restore the 'old' molecule object
        self.M = M_bak
        # Return the Mayer bond order as a matrix
        return M_qcout.qm_bondorder[-1]

    def calc_wq_new(self, coords, dirname):
        wq = getWorkQueue()
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file<
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        self.M.edit_qcrems({'jobtype':'force'})
        self.M[0].write(os.path.join(dirname, 'run.in'))
        in_files = [('%s/run.in' % dirname, 'run.in')]
        out_files = [('%s/run.out' % dirname, 'run.out'), ('%s/run.log' % dirname, 'run.log')]
        if self.qcdir:
            raise RuntimeError("--qcdir currently not supported with Work Queue")
        queue_up_src_dest(wq, "qchem%s run.in run.out &> run.log" % self.nt(), in_files, out_files, verbose=False)

    def number_output(self, dirname, calcNum):
        if not os.path.exists(os.path.join(dirname, 'run.out')):
            raise RuntimeError('run.out does not exist')
        shutil.copy2(os.path.join(dirname,'run.out'), os.path.join(dirname,'run_%03i.out' % calcNum))

    def read_result(self, dirname, check_coord=None):
        if check_coord is not None:
            read_xyz_success = False
            if os.path.exists('%s/run.out' % dirname): 
                try:
                    M1 = Molecule('%s/run.out' % dirname, build_topology=False)
                    read_xyz = M1.xyzs[0].flatten() / bohr2ang
                    read_xyz_success = True
                except: pass
            if not read_xyz_success or np.linalg.norm(check_coord - read_xyz) > 1e-8:
                raise CheckCoordError
        M1 = Molecule('%s/run.out' % dirname, build_topology=False)
        # In the case of multi-stage jobs, the last energy and gradient is what we want.
        energy = M1.qm_energies[-1]
        # Parse gradient from Q-Chem binary file. (Written by default without -save)
        gradient = np.fromfile('%s/run.d/131.0' % dirname)
        # Assume that the last occurence of "S^2" is what we want.
        s2 = 0.0
        # The 'iso-8859-1' prevents some strange errors that show up when reading the Archival summary line
        for line in open('%s/run.out' % dirname, encoding='iso-8859-1'):
            if "<S^2>" in line:
                s2 = float(line.split()[-1])
        return {'energy':energy, 'gradient':gradient, 's2':s2}

    def copy_scratch(self, src, dest):
        if not os.path.exists(dest): os.makedirs(dest)
        if not os.path.exists(os.path.join(src, 'run.d')):
            raise QChemEngineError("Trying to copy %s but it does not exist" % os.path.join(src, 'run.d'))
        copy_tree_over(os.path.join(src, 'run.d'), os.path.join(dest, 'run.d'))

    def detect_dft(self):
        for qcrem in self.qcrems:
            for key, val in qcrem.items():
                if key.lower() in ['method', 'exchange', 'correlation']:
                    if any([i.lower() in val.lower() for i in dft_strings]):
                        return True
        return False
    
class Gromacs(Engine):
    def __init__(self, molecule):
        super(Gromacs, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        try:
            from forcebalance.gmxio import GMX
        except ImportError:
            raise ImportError("ForceBalance is needed to compute energies and gradients using Gromacs.")
        if not os.path.exists(dirname): os.makedirs(dirname)
        try:
            Gro = Molecule("conf.gro")
            Gro.xyzs[0] = coords.reshape(-1,3) * bohr2ang
            cwd = os.getcwd()
            shutil.copy2("topol.top", dirname)
            shutil.copy2("shot.mdp", dirname)
            os.chdir(dirname)
            Gro.write("coords.gro")
            G = GMX(coords="coords.gro", gmx_top="topol.top", gmx_mdp="shot.mdp")
            EF = G.energy_force()
            Energy = EF[0, 0] / eqcgmx
            Gradient = EF[0, 1:] / fqcgmx
            os.chdir(cwd)
        except (OSError, IOError, RuntimeError, subprocess.CalledProcessError):
            raise GromacsEngineError
        return {'energy':Energy, 'gradient':Gradient}

    def copy_scratch(self, src, dest):
        return

class Molpro(Engine):
    """
    Run a Molpro energy and gradient calculation.
    """
    def __init__(self, molecule=None, threads=None):
        # molecule.py can not parse molpro input yet, so we use self.load_molpro_input() as a walk around
        if molecule is None:
            # create a fake molecule
            molecule = Molecule()
            molecule.elem = ['H']
            molecule.xyzs = [[[0,0,0]]]
        super(Molpro, self).__init__(molecule)
        self.threads = threads
        self.molproExePath = None

    def molproExe(self):
        if self.molproExePath is not None:
            return self.molproExePath
        else:
            return "molpro"

    def set_molproexe(self, molproExePath):
        self.molproExePath = molproExePath

    def nt(self):
        if self.threads is not None:
            return " -n %i" % self.threads
        else:
            return ""

    def load_molpro_input(self, molproin):
        """ Molpro input file parser, only support xyz coordinates for now """
        coords = []
        elems = []
        labels = []
        found_molecule, found_geo, found_gradient = False, False, False
        molpro_temp = [] # store a template of the input file for generating new ones
        for line in open(molproin):
            if 'geometry' in line:
                found_molecule = True
                molpro_temp.append(line)
            elif found_molecule is True:
                ls = line.split()
                if len(ls) == 4:
                    if found_geo == False:
                        found_geo = True
                        molpro_temp.append("$!geometry@here")
                    # parse the xyz format
                    elem = re.search('[A-Z][a-z]*',ls[0]).group(0)
                    elems.append( elem ) # grabs the element
                    labels.append( ls[0].split(elem)[-1] ) # grabs label after element specification
                    coords.append(ls[1:4]) # grabs the coordinates
                else:
                    molpro_temp.append(line)
                    if '}' in line:
                        found_molecule = False
            else:
                molpro_temp.append(line)
            if "force" in line:
                found_gradient = True
        if found_gradient == False:
            raise RuntimeError("Molpro inputfile %s should have force command." % molproin)
        self.M = Molecule()
        self.M.elem = elems
        self.M.xyzs = [np.array(coords, dtype=np.float64)]
        self.labels = labels
        self.molpro_temp = molpro_temp

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        # Write Molpro run.mol
        with open(os.path.join(dirname, 'run.mol'), 'w') as outfile:
            for line in self.molpro_temp:
                if line == '$!geometry@here':
                    for e, lab, c in zip(self.M.elem, self.labels, self.M.xyzs[0]):
                        outfile.write("%s%-7s %13.7f %13.7f %13.7f\n" % (e, lab, c[0], c[1], c[2]))
                else:
                    outfile.write(line)
        try:
            # Run Molpro
            subprocess.check_call('%s%s run.mol' % (self.molproExe(), self.nt()), cwd=dirname, shell=True)
            # Read energy and gradients from Molpro output
            result = self.read_result(dirname)
        except (OSError, IOError, RuntimeError, subprocess.CalledProcessError):
            raise MolproEngineError
        return result

    def number_output(self, dirname, calcNum):
        if not os.path.exists(os.path.join(dirname, 'run.out')):
            raise RuntimeError('run.out does not exist')
        shutil.copy2(os.path.join(dirname,'run.out'), os.path.join(dirname,'run_%03i.out' % calcNum))

    def read_result(self, dirname, check_coord=None):
        """ read an output file from Molpro"""
        if check_coord is not None:
            raise CheckCoordError("Coordinate check not implemented")
        energy, gradient = None, None
        molpro_out = os.path.join(dirname, 'run.out')
        with open(molpro_out) as outfile:
            found_grad = False
            for line in outfile:
                line_strip = line.strip()
                fields = line_strip.split()
                if line_strip.startswith('!'):
                    # This works for RHF and RKS
                    if len(fields) == 5 and fields[-2] == 'Energy':
                        energy = float(fields[-1])
                    # This works for MP2, CCSD and CCSD(T) total energy
                    elif len(fields) == 4 and fields[1] == 'total' and fields[2] == 'energy:':
                        energy = float(fields[-1])
                elif len(fields) > 4 and fields[-4] == 'GRADIENT' and fields[-3] == 'FOR' and fields[-2] == 'STATE':
                    # this works for most of the analytic gradients
                    found_grad = True
                    gradient = []
                    # Skip three lines of header
                    next(outfile)
                    next(outfile)
                    next(outfile)
                elif found_grad is True:
                    if len(fields) == 4:
                        if fields[0].isdigit():
                            gradient.append([float(g) for g in fields[1:4]])
                    elif "Nuclear force contribution to virial" in line:
                        found_grad = False
                    else:
                        continue
        if energy is None:
            raise RuntimeError("Molpro energy is not found in %s, please check." % molpro_out)
        if gradient is None:
            raise RuntimeError("Molpro gradient is not found in %s, please check." % molpro_out)
        gradient = np.array(gradient, dtype=np.float64).ravel()
        return {'energy':energy, 'gradient':gradient}

    def detect_dft(self):
        for line in self.molpro_temp:
            for keyword in ["ks,", "ks;", "ks}"]:
                if keyword in line.lower() or line.lower().strip().endswith("ks"):
                    return True
        return False

class QCEngineAPI(Engine):
    def __init__(self, schema, program):
        try:
            import qcengine
        except ImportError:
            raise ImportError("QCEngine computation object requires the 'qcengine' package. Please pip or conda install 'qcengine'.")

        self.schema = schema
        self.program = program
        self.schema["driver"] = "gradient"

        self.M = Molecule()
        self.M.elem = list(schema["molecule"]["symbols"])

        # Geometry in (-1, 3) array in angstroms
        geom = np.array(schema["molecule"]["geometry"], dtype=np.float64).reshape(-1, 3) * bohr2ang
        self.M.xyzs = [geom]

        # Use or build connectivity
        if schema["molecule"].get("connectivity", None) is not None:
            self.M.Data["bonds"] = sorted((x[0], x[1]) for x in schema["molecule"]["connectivity"])
            self.M.built_bonds = True
        else:
            self.M.build_bonds()
        # one additional attribute to store each schema on the opt trajectory
        self.schema_traj = []

    def calc_new(self, coords, dirname):
        import qcengine
        new_schema = deepcopy(self.schema)
        new_schema["molecule"]["geometry"] = coords.tolist()
        new_schema.pop("program", None)
        ret = qcengine.compute(new_schema, self.program, return_dict=True)
        # store the schema_traj for run_json to pick up
        self.schema_traj.append(ret)
        if ret["success"] is False:
            raise QCEngineAPIEngineError("QCEngineAPI computation did not execute correctly. Message: " + ret["error"]["error_message"])
        # Unpack the energy and gradient
        energy = ret["properties"]["return_energy"]
        gradient = np.array(ret["return_result"])
        return {'energy':energy, 'gradient':gradient}

    def calc(self, coords, dirname, **kwargs):
        # overwrites the calc method of base class to skip caching and creating folders
        # **kwargs: for throwing away other arguments such as read_data and copyfiles.
        return self.calc_new(coords, dirname)

    def detect_dft(self):
        return any([i.lower() in self.schema["model"]["method"].lower() for i in dft_strings])

class ConicalIntersection(Engine):
    """
    Compute conical intersection objective function with penalty constraint.
    Implements the theory from Levine, Coe and Martinez, J. Phys. Chem. B 2008.
    """
    def __init__(self, molecule, engines, sigma, alpha):
        self.engines = deepcopy(engines)
        self.sigma = sigma
        self.alpha = alpha
        super(ConicalIntersection, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        n_states = len(self.engines)
        n_states2 = n_states * (n_states - 1) / 2
        E_states = []
        G_states = []
        S_states = []
        for istate in range(n_states):
            state_dnm = os.path.join(dirname, 'state_%i' % istate)
            if not os.path.exists(state_dnm): os.makedirs(state_dnm)
            try:
                spcalc = self.engines[istate].calc(coords, state_dnm)
            except EngineError:
                raise ConicalIntersectionEngineError
            E_states.append(spcalc['energy'])
            G_states.append(spcalc['gradient'])
            S_states.append(spcalc.get('s2', 0.0))
        E_states = np.array(E_states)
        E_order = np.argsort(E_states)
        EAvg = 0.0
        GAvg = np.zeros_like(G_states[0])
        EPen = 0.0
        GPen = np.zeros_like(G_states[0])

        report_blk1 = []
        report_blk2 = []
        report_blk3 = []

        for i in range(n_states):
            I = E_order[i]
            EAvg += E_states[I] / n_states
            GAvg += G_states[I] / n_states
            atomgrad = np.sqrt(np.sum((G_states[I].reshape(-1,3))**2, axis=1))
            rms_gradient = np.sqrt(np.mean(atomgrad**2))
            max_gradient = np.max(atomgrad)
            report_blk1.append("%5i % 18.10f %7.4f %9.3e %9.3e   " % (I, E_states[I], S_states[I], rms_gradient, max_gradient))
            report_blk2_str = ""
            report_blk3_str = ""
            for j in range(i+1, n_states):
                J = E_order[j]
                EDif = E_states[J] - E_states[I]
                GDif = G_states[J] - G_states[I]
                EPen += self.sigma * EDif**2 / ((EDif + self.alpha) * n_states2)
                GPen += self.sigma * (EDif**2 + 2*self.alpha*EDif)/((EDif+self.alpha)**2 * n_states2) * GDif
                GAng = np.dot(G_states[I], G_states[J])/(np.linalg.norm(G_states[I])*np.linalg.norm(G_states[J]))
                report_blk2_str += " %8.5f" % (EDif * au2ev)
                report_blk3_str += " % 8.4f" % GAng
            report_blk2.append(report_blk2_str)
            report_blk3.append(report_blk3_str)

        width1 = max([len(line) for line in report_blk1])
        width2 = max([len(line) for line in report_blk2])
        width3 = max([len(line) for line in report_blk3])


                # logger.info("E[%i]= % .7f E[%i]= % .7f S2[%i]= %.4f S2[%i]= %.4f Gap=%.7f CosGrad= % .4f\n"
                #             % (I, E_states[I], J, E_states[J], I, S_states[I], J, S_states[J], EDif, GAng))

        Obj = EAvg + EPen
        ObjGrad = GAvg + GPen
        
        logger.info(">>> MECI Report: <E> = % 18.10f Penalty = %15.10f <<<\n" % (EAvg, EPen))
        logger.info("%5s %18s %7s %9s %9s   %%%is   %%%is\n" % ("State", "Energy (a.u.)", "<S^2>", "G_rms", "G_max", width2, width3) % ("Gaps (eV)", "Cos(^Gi,^Gj)"))
        for ln in range(len(report_blk1)):
            logger.info("%%%is%%%is   %%%is\n" % (width1, width2, width3) % (report_blk1[ln], report_blk2[ln], report_blk3[ln]))
        return {'energy':Obj, 'gradient':ObjGrad}

    def number_output(self, dirname, calcNum):
        for istate in range(len(self.engines)):
            state_dnm = os.path.join(dirname, 'state_%i' % istate)
            self.engines[istate].number_output(state_dnm, calcNum)

    def detect_dft(self):
        for istate in range(len(self.engines)):
            if self.engines[istate].detect_dft():
                return True
        else:
            return False
