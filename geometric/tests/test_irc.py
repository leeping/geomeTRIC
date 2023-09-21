"""
A set of tests for IRC
"""

import os, shutil
from . import addons
import geometric
import numpy as np

localizer = addons.in_folder
datad = addons.datad
exampled = addons.exampled

@addons.using_psi4
def test_hcn_irc_psi4(localizer):
    """
    IRC test
    """
    shutil.copy2(os.path.join(datad, 'hcn_irc.psi4in'), os.path.join(os.getcwd(), 'hcn_irc.psi4in'))
    progress = geometric.optimize.run_optimizer(engine='psi4', input='hcn_irc.psi4in', converge=['set', 'GAU_LOOSE'],
                                                nt=4, reset=False, trust=0.05, irc=True, maxiter=50)
    e_ref1 = -92.35408411
    e_ref2 = -92.33971205
    max_e = np.max(progress.qm_energies)
    reac_e = progress.qm_energies[0]
    prod_e = progress.qm_energies[-1]

    # Check the mex_e is not from the end-points
    assert reac_e < max_e
    assert prod_e < max_e

    # Check the end-point energies
    assert np.isclose(min(reac_e, prod_e), e_ref1)
    assert np.isclose(max(reac_e, prod_e), e_ref2)

    # Check that the IRC converged in less than 100 iterations
    assert len(progress) < 100
