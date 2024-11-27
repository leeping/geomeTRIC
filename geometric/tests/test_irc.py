"""
A set of tests for IRC
"""

import os, shutil
from . import addons
import geometric
import numpy as np
import tempfile

localizer = addons.in_folder
datad = addons.datad


@addons.using_psi4
def test_psi4_hcn_irc(localizer, molecule_engine_hcn):
    """
    IRC test with Psi4
    """
    M, IC, engine, params = molecule_engine_hcn('psi4')
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr
    dirname = tempfile.mkdtemp()

    progress = geometric.optimize.Optimize(coords, M, IC, engine, dirname, params)
    e_ref1 = -92.35408411
    e_ref2 = -92.33971205
    max_e = np.max(progress.qm_energies)
    reac_e = progress.qm_energies[0]
    prod_e = progress.qm_energies[-1]

    # Check the max_e is not from the end-points
    assert reac_e < max_e
    assert prod_e < max_e

    # Check the end-point energies
    assert np.isclose(min(reac_e, prod_e), e_ref1)
    assert np.isclose(max(reac_e, prod_e), e_ref2)

    # Check that the IRC converged in less than 100 iterations
    assert len(progress) < 100


@addons.using_qchem
def test_qchem_hcn_irc(localizer, molecule_engine_hcn):
    """
    IRC test with QChem
    """
    M, IC, engine, params = molecule_engine_hcn('qchem')
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr
    dirname = tempfile.mkdtemp()

    progress = geometric.optimize.Optimize(coords, M, IC, engine, dirname, params)
    e_ref1 = -92.35408411
    e_ref2 = -92.33971205
    max_e = np.max(progress.qm_energies)
    reac_e = progress.qm_energies[0]
    prod_e = progress.qm_energies[-1]

    # Check the max_e is not from the end-points
    assert reac_e < max_e
    assert prod_e < max_e

    # Check the end-point energies
    assert np.isclose(min(reac_e, prod_e), e_ref1)
    assert np.isclose(max(reac_e, prod_e), e_ref2)

    # Check that the IRC converged in less than 100 iterations
    assert len(progress) < 100


@addons.using_gaussian
def test_gaussian_hcn_irc(localizer, molecule_engine_hcn):
    """
    IRC test with Gaussian
    """
    M, IC, engine, params = molecule_engine_hcn('gaussian')
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr
    dirname = tempfile.mkdtemp()

    progress = geometric.optimize.Optimize(coords, M, IC, engine, dirname, params)
    e_ref1 = -92.35408411
    e_ref2 = -92.33971205
    max_e = np.max(progress.qm_energies)
    reac_e = progress.qm_energies[0]
    prod_e = progress.qm_energies[-1]

    # Check the max_e is not from the end-points
    assert reac_e < max_e
    assert prod_e < max_e

    # Check the end-point energies
    assert np.isclose(min(reac_e, prod_e), e_ref1)
    assert np.isclose(max(reac_e, prod_e), e_ref2)

    # Check that the IRC converged in less than 100 iterations
    assert len(progress) < 100

@addons.using_terachem
def test_tera_hcn_irc(localizer, molecule_engine_hcn):
    """
    IRC test with TeraChem
    """
    shutil.copy2(os.path.join(datad, 'hcn_irc_input.xyz'), os.path.join(os.getcwd(), 'hcn_irc_input.xyz'))

    M, IC, engine, params = molecule_engine_hcn('tera')
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr
    dirname = tempfile.mkdtemp()

    progress = geometric.optimize.Optimize(coords, M, IC, engine, dirname, params)
    e_ref1 = -92.35408411
    e_ref2 = -92.33971205
    max_e = np.max(progress.qm_energies)
    reac_e = progress.qm_energies[0]
    prod_e = progress.qm_energies[-1]

    # Check the max_e is not from the end-points
    assert reac_e < max_e
    assert prod_e < max_e

    # Check the end-point energies
    assert np.isclose(min(reac_e, prod_e), e_ref1)
    assert np.isclose(max(reac_e, prod_e), e_ref2)

    # Check that the IRC converged in less than 100 iterations
    assert len(progress) < 100