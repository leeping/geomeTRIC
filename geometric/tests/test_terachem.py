"""
A set of tests for using the OpenMM engine
"""

import copy
import numpy as np
import json, os, shutil
from . import addons
import geometric
import pytest
import itertools

localizer = addons.in_folder
datad = addons.datad
exampled = addons.exampled

@addons.using_terachem
def test_meci_qmmm_terachem(localizer):
    """
    Find the MECI of a retinal protonated Schiff base model with static QM/MM water molecules.
    """
    for fnm in ['x.prmtop', 'qmindices.txt', 'run.tcin', 'run1.tcin']:
        shutil.copy(os.path.join(exampled, '0-regression-tests', 'psb3-qmmm-meci', fnm), os.getcwd())
    shutil.copy(os.path.join(exampled, '0-regression-tests', 'psb3-qmmm-meci', 'converged.inpcrd'), 
                os.path.join(os.getcwd(), 'x.inpcrd'))
    progress = geometric.optimize.run_optimizer(engine='terachem', input='run.tcin', meci=['run1.tcin'], meci_alpha=0.03)
    e_ref = -251.9957148967
    assert progress.qm_energies[-1] < (e_ref + 0.001)
    
