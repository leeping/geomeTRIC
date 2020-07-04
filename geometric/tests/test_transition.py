"""
A set of tests for transition state optimization
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

@addons.using_psi4
def test_transition_hcn_psi4(localizer):
    """
    Optimize the transition state of the HCN <-> HNC isomerization.
    """
    shutil.copy2(os.path.join(datad, 'hcn.psi4in'), os.getcwd())
    progress = geometric.optimize.run_optimizer(engine='psi4', transition=True, input='hcn.psi4in',
                                                converge=['gmax', '1.0e-5'])
    # The results here are in Angstrom
    #
    ref = geometric.molecule.Molecule(os.path.join(datad, 'hcn_ts_optim.xyz')).xyzs[0]
    e_ref = -92.2460196061
    xdiff = (progress.xyzs[-1] - ref).flatten()
    rmsd, maxd = geometric.optimize.calc_drms_dmax(progress.xyzs[-1], ref, align=True)
    print("Energy difference from reference:", progress.qm_energies[-1]-e_ref)
    print("RMS / Max displacement from reference:", rmsd, maxd)
    # Check that the energy is 0.0001 a.u. above reference. 
    assert progress.qm_energies[-1] < (e_ref + 0.0001)
    # Check that the optimization converged in less than 20 steps
    assert len(progress) < 20
    # Check that the geometry matches the reference to within 0.01 RMS 0.02 max displacement
    assert rmsd < 0.01
    assert maxd < 0.02

