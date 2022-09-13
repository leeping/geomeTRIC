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
