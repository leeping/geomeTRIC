"""
A set of tests for energy minimization
"""

import copy
import numpy as np
import json, os, shutil
from . import addons
import geometric
import pytest
import itertools
import time
import subprocess

localizer = addons.in_folder
datad = addons.datad
exampled = addons.exampled

@addons.using_psi4
def test_hcccn_minimize_psi4(localizer):
    """
    Optimize a linear HCCCN molecule
    """
    shutil.copy2(os.path.join(datad, 'hcccn.psi4in'), os.path.join(os.getcwd(), 'hcccn.psi4in'))
    progress = geometric.optimize.run_optimizer(engine='psi4', input='hcccn.psi4in', converge=['gmax', '1.0e-5'], 
                                                nt=4, reset=False, trust=0.1, tmax=0.3)
    e_ref = -167.6136203991
    assert progress.qm_energies[-1] < (e_ref + 1e-5)
    # Check that the optimization converged in less than 10 steps
    assert len(progress) < 10

@addons.using_quick
def test_water2_minimize_quick(localizer):
    """
    Optimize a water dimer 
    """
    shutil.copy2(os.path.join(exampled, "1-simple-examples","water2_quick","Water2.qkin"), os.path.join(os.getcwd(), "Water2.qkin"))
    progress = geometric.optimize.run_optimizer(engine='quick', input='Water2.qkin', converge=['gmax', '1.0e-5'],
                                                nt=4, reset=False, trust=0.1, tmax=0.3)
    e_ref = -149.9412443430
    assert progress.qm_energies[-1] < (e_ref + 1e-5)
    # Check that the optimization converged in less than 25 steps (took 20 steps in LPW's local test.)
    assert len(progress) < 25

@addons.using_quick
def test_water2_minimize_quick_converge_maxiter(localizer):
    """
    Optimize a water dimer, but "converge" when maximum number of iterations (5) is reached.
    """
    shutil.copy2(os.path.join(exampled, "1-simple-examples","water2_quick","Water2.qkin"), os.path.join(os.getcwd(), "Water2.qkin"))
    progress = geometric.optimize.run_optimizer(engine='quick', input='Water2.qkin', converge=['gmax', '1.0e-5', 'maxiter'],
                                                nt=4, reset=False, trust=0.1, tmax=0.3, maxiter=5)
    assert len(progress) == 6
    # e_ref = -149.9412443430
    # assert progress.qm_energies[-1] < (e_ref + 1e-5)
    # # Check that the optimization converged in less than 25 steps (took 20 steps in LPW's local test.)
    # assert len(progress) < 25
