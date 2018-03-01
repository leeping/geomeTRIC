#!/usr/bin/env python

from __future__ import print_function, division
import os, sys, shutil
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from forcebalance.gmxio import GMX
from forcebalance.molecule import Molecule
from forcebalance.nifty import eqcgmx, fqcgmx, getWorkQueue, queue_up_src_dest
from geometric.global_vars import *
from copy import copy
import subprocess

#=============================#
#| Useful TeraChem functions |#
#=============================#

def edit_tcin(fin=None, fout=None, options={}, defaults={}):
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

    Returns
    -------
    dictionary
        Keys mapped to values as strings.  Certain keys will be changed to integers (e.g. charge, spinmult).
        Keys are standardized to lowercase.
    """
    intkeys = ['charge', 'spinmult']
    Answer = OrderedDict()
    # Read from the input if provided
    if fin is not None:
        for line in open(fin).readlines():
            line = line.split("#")[0].strip()
            if len(line) == 0: continue
            if line == 'end': break
            s = line.split(' ', 1)
            k = s[0].lower()
            v = s[1].strip()
            if k == 'coordinates':
                if not os.path.exists(v.strip()):
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
    for k, v in defaults.items():
        if k not in Answer.keys():
            Answer[k] = v
    for k, v in Answer.items():
        if v is None:
            del Answer[k]
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
                    print("%-15s %s" % (k, str(v)), file=f)
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
stored_calcs = OrderedDict()

class Engine(object):
    def __init__(self, molecule):
        if len(molecule) != 1:
            raise RuntimeError('Please pass only length-1 molecule objects to engine creation')
        self.M = deepcopy(molecule)
        # self.stored_calcs = OrderedDict()

    # def __deepcopy__(self, memo):
    #     return copy(self)

    def calc(self, coords, dirname):
        global stored_calcs
        coord_hash = hash(coords.tostring())
        if coord_hash in stored_calcs:
            energy = stored_calcs[coord_hash]['energy']
            gradient = stored_calcs[coord_hash]['gradient']
        else:
            if not os.path.exists(dirname): os.makedirs(dirname)
            energy, gradient = self.calc_new(coords, dirname)
            stored_calcs[coord_hash] = {'coords':coords,'energy':energy,'gradient':gradient}
        return energy, gradient

    def clearCalcs(self):
        global stored_calcs
        stored_calcs = OrderedDict()

    def calc_new(self, coords, dirname):
        raise NotImplementedError("Not implemented for the base class")

    def calc_wq(self, coords, dirname):
        coord_hash = hash(coords.tostring())
        if coord_hash in stored_calcs:
            return
        else:
            self.calc_wq_new(coords, dirname)

    def calc_wq_new(self, coords, dirname):
        raise NotImplementedError("Work Queue is not implemented for this class")

    def read_wq(self, coords, dirname):
        global stored_calcs
        coord_hash = hash(coords.tostring())
        if coord_hash in stored_calcs:
            energy = stored_calcs[coord_hash]['energy']
            gradient = stored_calcs[coord_hash]['gradient']
        else:
            if not os.path.exists(dirname):
                raise RuntimeError("In read_wq, %s doesn't exist" % dirname)
            energy, gradient = self.read_wq_new(coords, dirname)
            stored_calcs[coord_hash] = {'coords':coords,'energy':energy,'gradient':gradient}
        return energy, gradient

    def read_wq_new(self, coords, dirname):
        raise NotImplementedError("Work Queue is not implemented for this class")

    def number_output(self, dirname, calcNum):
        return

class Blank(Engine):
    """
    Always return zero energy and gradient.
    """
    def __init__(self, molecule):
        super(Blank, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        energy = 0.0
        gradient = np.zeros(len(coords), dtype=float)
        return energy, gradient

class TeraChem(Engine):
    """
    Run a TeraChem energy and gradient calculation.
    """
    def __init__(self, molecule, tcin):
        self.tcin = tcin.copy()
        super(TeraChem, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        guesses = []
        have_guess = False
        for f in ['c0', 'ca0', 'cb0']:
            if os.path.exists(os.path.join(dirname, 'scr', f)):
                shutil.copy2(os.path.join(dirname, 'scr', f), os.path.join(dirname, f))
                guesses.append(f)
                have_guess = True
        # This is for when we start geometry optimizations
        # and we have a guess prepped and ready to go.
        if not have_guess and 'guess' in self.tcin:
            for f in self.tcin['guess'].split():
                if os.path.exists(f):
                    shutil.copy2(f, dirname)
                    guesses.append(f)
                    have_guess = True
                else:
                    del self.tcin['guess']
                    have_guess = False
                    break
        self.tcin['coordinates'] = 'start.xyz'
        self.tcin['run'] = 'gradient'
        if have_guess:
            self.tcin['guess'] = ' '.join(guesses)
            self.tcin['purify'] = 'no'
            self.tcin['mixguess'] = "0.0"
        edit_tcin(fout="%s/run.in" % dirname, options=self.tcin)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * 0.529177
        self.M[0].write(os.path.join(dirname, 'start.xyz'))
        # Run TeraChem
        subprocess.call('terachem run.in > run.out', cwd=dirname, shell=True)
        # Extract energy and gradient
        subprocess.call("awk '/FINAL ENERGY/ {p=$3} /Correlation Energy/ {p+=$5} END {printf \"%.10f\\n\", p}' run.out > energy.txt", cwd=dirname, shell=True)
        subprocess.call("awk '/Gradient units are Hartree/,/Net gradient/ {if ($1 ~ /^-?[0-9]/) {print}}' run.out > grad.txt", cwd=dirname, shell=True)
        energy = float(open(os.path.join(dirname,'energy.txt')).readlines()[0].strip())
        gradient = np.loadtxt(os.path.join(dirname,'grad.txt')).flatten()
        return energy, gradient

    def calc_wq_new(self, coords, dirname):
        # Run TeraChem
        wq = getWorkQueue()
        if not os.path.exists(dirname): os.makedirs(dirname)
        scrdir = os.path.join(dirname, 'scr')
        if not os.path.exists(scrdir): os.makedirs(scrdir)
        guesses = []
        have_guess = False
        unrestricted = self.tcin['method'][0] == 'u'
        if unrestricted:
            guessfnms = ['ca0', 'cb0']
        else:
            guessfnms = ['c0']
        for f in ['c0', 'ca0', 'cb0']:
            if f not in guessfnms: continue
            if os.path.exists(os.path.join(dirname, 'scr', f)):
                shutil.move(os.path.join(dirname, 'scr', f), os.path.join(dirname, f))
                guesses.append(f)
            if os.path.exists(os.path.join(dirname, f)):
                if f not in guesses: guesses.append(f)
        # Check if all the appropriate guess files have been found
        # and moved to "dirname"
        have_guess = (guesses == guessfnms)
        # This is for when we start geometry optimizations
        # and we have a guess prepped and ready to go.
        if not have_guess and 'guess' in self.tcin:
            for f in self.tcin['guess'].split():
                if os.path.exists(f):
                    shutil.copy2(f, dirname)
                    guesses.append(f)
                    have_guess = True
                else:
                    del self.tcin['guess']
                    have_guess = False
                    break
        self.tcin['coordinates'] = 'start.xyz'
        self.tcin['run'] = 'gradient'
        # For queueing up jobs, delete GPU key and let the worker decide
        self.tcin['gpus'] = None
        if have_guess:
            self.tcin['guess'] = ' '.join(guesses)
            self.tcin['purify'] = 'no'
            self.tcin['mixguess'] = "0.0"
        tcopts = edit_tcin(fout="%s/run.in" % dirname, options=self.tcin)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * 0.529177
        self.M[0].write(os.path.join(dirname, 'start.xyz'))
        in_files = [('%s/run.in' % dirname, 'run.in'), ('%s/start.xyz' % dirname, 'start.xyz')]
        out_files = [('%s/run.out' % dirname, 'run.out')]
        if have_guess:
            for g in guesses:
                in_files.append((os.path.join(dirname, g), g))
        for g in guessfnms:
            out_files.append((os.path.join(dirname, 'scr', g), os.path.join('scr', g)))
        queue_up_src_dest(wq, "%s/runtc run.in &> run.out" % rootdir, in_files, out_files, verbose=False)

    def number_output(self, dirname, calcNum):
        if not os.path.exists(os.path.join(dirname, 'run.out')):
            raise RuntimeError('run.out does not exist')
        shutil.copy2(os.path.join(dirname,'start.xyz'), os.path.join(dirname,'start_%03i.xyz' % calcNum))
        shutil.copy2(os.path.join(dirname,'run.out'), os.path.join(dirname,'run_%03i.out' % calcNum))

    def read_wq_new(self, coords, dirname):
        # Extract energy and gradient
        subprocess.call("awk '/FINAL ENERGY/ {p=$3} /Correlation Energy/ {p+=$5} END {printf \"%.10f\\n\", p}' run.out > energy.txt", cwd=dirname, shell=True)
        subprocess.call("awk '/Gradient units are Hartree/,/Net gradient/ {if ($1 ~ /^-?[0-9]/) {print}}' run.out > grad.txt", cwd=dirname, shell=True)
        energy = float(open(os.path.join(dirname,'energy.txt')).readlines()[0].strip())
        gradient = np.loadtxt(os.path.join(dirname,'grad.txt')).flatten()
        return energy, gradient

class Psi4(Engine):
    """
    Run a Psi4 energy and gradient calculation.
    """
    def __init__(self, molecule=None):
        # molecule.py can not parse psi4 input yet, so we use self.load_psi4_input() as a walk around
        if molecule is None:
            # create a fake molecule
            molecule = Molecule()
            molecule.elem = ['H']
            molecule.xyzs = [[[0,0,0]]]
        super(Psi4, self).__init__(molecule)
        self.threads = None

    def nt(self):
        if self.threads is not None:
            return " -n %i" % self.threads
        else:
            return ""

    def set_nt(self, threads):
        self.threads = threads

    def load_psi4_input(self, psi4in):
        """ Psi4 input file parser, only support xyz coordinates for now """
        coords = []
        elems = []
        found_molecule, found_geo, found_gradient = False, False, False
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
                else:
                    psi4_temp.append(line)
                    if '}' in line:
                        found_molecule = False
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

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        # Write Psi4 input.dat
        with open(os.path.join(dirname, 'input.dat'), 'w') as outfile:
            for line in self.psi4_temp:
                if line == '$!geometry@here':
                    for e, c in zip(self.M.elem, self.M.xyzs[0]):
                        outfile.write("%-7s %13.7f %13.7f %13.7f\n" % (e, c[0], c[1], c[2]))
                else:
                    outfile.write(line)
        # Run Psi4
        subprocess.call('psi4%s input.dat' % self.nt(), cwd=dirname, shell=True)
        # Read energy and gradients from Psi4 output
        energy, gradient = self.parse_psi4_output(os.path.join(dirname, 'output.dat'))
        return energy, gradient

    def parse_psi4_output(self, psi4out):
        """ read an output file from Psi4 """
        energy, gradient = None, None
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
                    print("found num grad")
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
        return energy, gradient



class QChem(Engine):
    def __init__(self, molecule):
        super(QChem, self).__init__(molecule)
        self.qcdir = False
        self.threads = None

    def nt(self):
        if self.threads is not None:
            return " -nt %i" % self.threads
        else:
            return ""

    def set_nt(self, threads):
        self.threads = threads

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname): os.makedirs(dirname)
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang
        self.M.edit_qcrems({'jobtype':'force'})
        self.M[0].write(os.path.join(dirname, 'run.in'))
        # Run Qchem
        if self.qcdir:
            subprocess.call('qchem%s run.in run.out run.d &> run.log' % self.nt(), cwd=dirname, shell=True)
        else:
            subprocess.call('qchem%s run.in run.out run.d &> run.log' % self.nt(), cwd=dirname, shell=True)
            # Assume reading the SCF guess is desirable
            self.qcdir = True
            self.M.edit_qcrems({'scf_guess':'read'})
        M1 = Molecule('%s/run.out' % dirname)
        energy = M1.qm_energies[0]
        gradient = M1.qm_grads[0].flatten()
        return energy, gradient

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

    def read_wq_new(self, coords, dirname):
        M1 = Molecule('%s/run.out' % dirname)
        energy = M1.qm_energies[0]
        gradient = M1.qm_grads[0].flatten()
        return energy, gradient

class Gromacs(Engine):
    def __init__(self, molecule):
        super(Gromacs, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        if not os.path.exists(dirname): os.makedirs(dirname)
        Gro = Molecule("conf.gro")
        Gro.xyzs[0] = coords.reshape(-1,3) * 0.529
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
        return Energy, Gradient
