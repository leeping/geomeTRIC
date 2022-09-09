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
import time

localizer = addons.in_folder
datad = addons.datad

@addons.using_psi4
def test_transition_hcn_psi4(localizer):
    """
    Optimize the transition state of the HCN <-> HNC isomerization.
    """
    shutil.copy2(os.path.join(datad, 'hcn.psi4in'), os.getcwd())
    if os.path.exists(os.path.join(os.getcwd(), 'hcn.tmp')):
        shutil.rmtree(os.path.join(os.getcwd(), 'hcn.tmp'))
    progress = geometric.optimize.run_optimizer(engine='psi4', transition=True, input='hcn.psi4in',
                                                converge=['gmax', '1.0e-5'], trust=0.1, tmax=0.3, hessian='first+last')
    # The results here are in Angstrom
    #
    molecule = geometric.molecule.Molecule(os.path.join(datad, 'hcn_ts_optim.xyz'))
    ref = molecule.xyzs[0]
    e_ref = -92.2460196061
    rmsd, maxd = geometric.optimize.calc_drms_dmax(progress.xyzs[-1], ref, align=True)
    # Check that the energy is 0.0001 a.u. above reference. 
    assert progress.qm_energies[-1] < (e_ref + 0.0001)
    # Check that the optimization converged in less than 20 steps
    assert len(progress) < 20
    # Check that the geometry matches the reference to within 0.01 RMS 0.02 max displacement
    assert rmsd < 0.001
    assert maxd < 0.002
    coords = progress.xyzs[-1].flatten()*geometric.nifty.ang2bohr
    t0 = time.time()
    hessian = geometric.normal_modes.calc_cartesian_hessian(coords, progress[-1], None, 'hcn.tmp')
    # The latest Hessian should have been read in from file.
    assert (time.time() - t0) < 1
    # Todo: Add tests for correctness of Wigner sampling
    freqs, modes, G = geometric.normal_modes.frequency_analysis(coords, hessian, elem=progress.elem, energy=progress.qm_energies[-1], wigner=(-10, 'hcn.wigner'))
    np.testing.assert_almost_equal(G, -92.25677301, decimal=5)
    np.testing.assert_almost_equal(freqs[0]/10, -121.5855, decimal=0)
