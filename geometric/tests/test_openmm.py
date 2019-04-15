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

@addons.using_openmm
def test_dlc_openmm_water3(localizer):
    """
    Optimize the geometry of three water molecules using standard delocalized internal coordinates.
    The coordinate system will break down and have to be rebuilt.
    """
    progress = geometric.optimize.run_optimizer(openmm=True, pdb=os.path.join(datad,'water3.pdb'), coordsys='dlc', input='tip3p.xml')
    # The results here are in Angstrom
    #
    ref = np.array([[ 1.19172917, -1.71174316,  0.79961878],
                    [ 1.73335403, -2.33032763,  0.30714483],
                    [ 1.52818406, -0.83992919,  0.51498083],
                    [-0.31618326,  0.13417074,  2.15241103],
                    [ 0.07716192, -0.68377281,  1.79137674],
                    [-0.98942711, -0.18943265,  2.75288307],
                    [ 1.64949098,  0.96407596,  0.43451244],
                    [ 0.91641967,  0.91098247,  1.07801098],
                    [ 1.78727054,  1.90697627,  0.33206132]])
    e_ref = -0.0289248308
    xdiff = (progress.xyzs[-1] - ref).flatten()
    rmsd, maxd = geometric.optimize.calc_drms_dmax(progress.xyzs[-1], ref, align=True)
    print("RMS / Max displacement from reference:", rmsd, maxd)
    # This test is a bit stochastic and doesn't converge to the same minimized geometry every time.
    # Check that the energy is 0.01 a.u. above reference. Not really the qm_energy, this is a misnomer
    assert progress.qm_energies[-1] < (e_ref + 0.01)
    # Check that the optimization converged in less than 300 steps
    assert len(progress) < 300
    # Check that the geometry matches the reference to within 0.03 RMS 0.05 max displacement
    assert rmsd < 0.03
    assert maxd < 0.05

@addons.using_openmm
def test_tric_openmm_water6(localizer):
    """
    Optimize the geometry of six water molecules using translation-rotation internal coordinates.
    This optimization should be rather stable.
    """
    progress = geometric.optimize.run_optimizer(openmm=True, pdb=os.path.join(datad,'water6.pdb'), input='tip3p.xml')
    ref = np.array([[ 1.32539118, -1.69049000,  0.75057673],
                    [ 1.99955139, -2.21940859,  1.18382539],
                    [ 1.57912690, -0.77129578,  0.98807568],
                    [-1.38379009, -0.92093253,  0.85872908],
                    [-0.51149071, -1.29985588,  1.04548492],
                    [-1.20200783,  0.02764829,  0.79004674],
                    [ 1.73099256,  0.87457076,  1.40125992],
                    [ 0.91505292,  1.24056850,  1.00215384],
                    [ 2.16539319,  1.65091617,  1.75997418],
                    [-0.24547074,  1.01680646, -2.34970965],
                    [-0.12968321,  1.07011697, -3.30048872],
                    [-0.19545660,  0.05162253, -2.17758026],
                    [-0.17266482, -1.56348902, -1.60782169],
                    [ 0.53007928, -1.76265282, -0.96619742],
                    [-0.96767413, -1.61898974, -1.05756209],
                    [-0.55553348,  1.72499233,  0.21519466],
                    [-0.41554802,  1.52125570, -0.73793694],
                    [-1.23114402,  2.40806579,  0.18635652]])

    e_ref = -0.0777452561
    xdiff = (progress.xyzs[-1] - ref).flatten()
    rmsd, maxd = geometric.optimize.calc_drms_dmax(progress.xyzs[-1], ref, align=True)
    print("RMS / Max displacement from reference:", rmsd, maxd)
    assert progress.qm_energies[-1] < (e_ref + 0.01)
    # Check that the optimization converged in less than 100 steps
    assert len(progress) < 100
    # Check that the geometry matches the reference
    assert rmsd < 0.03
    assert maxd < 0.05

@addons.using_openmm
def test_openmm_ala_scan(localizer):
    # Requires amber99sb.xml which ships with OpenMM
    m = geometric.optimize.run_optimizer(openmm=True, enforce=0.1, pdb=os.path.join(datad, 'ala_a99sb_min.pdb'), input='amber99sb.xml',
                                         constraints=os.path.join(datad, 'ala_constraints.txt'))
    scan_final = geometric.molecule.Molecule('scan-final.xyz')
    scan_energies = np.array([float(c.split()[-1]) for c in scan_final.comms])
    ref_energies = np.array([-0.03368698, -0.03349261])
    # Check converged energies
    assert np.allclose(scan_energies, ref_energies, atol=1.e-3)
    # Check for performance regression (should be done in ~33 cycles)
    assert len(m) < 50

@addons.using_openmm
def test_openmm_ala_scan_conmethod(localizer):
    # Requires amber99sb.xml which ships with OpenMM
    m = geometric.optimize.run_optimizer(openmm=True, conmethod=1, pdb=os.path.join(datad, 'ala_a99sb_min.pdb'), input='amber99sb.xml',
                                         constraints=os.path.join(datad, 'ala_constraints.txt'))
    scan_final = geometric.molecule.Molecule('scan-final.xyz')
    scan_energies = np.array([float(c.split()[-1]) for c in scan_final.comms])
    ref_energies = np.array([-0.03368698, -0.03349261])
    # Check converged energies
    assert np.allclose(scan_energies, ref_energies, atol=1.e-3)
    # Check for performance regression (should be done in ~62 cycles)
    assert len(m) < 100

@addons.using_openmm
def test_openmm_h2o2_h2o_grad_hess(localizer):
    M, engine = geometric.optimize.get_molecule_engine(openmm=True, pdb=os.path.join(datad, 'h2o2_h2o.pdb'), input=os.path.join(datad, 'h2o2_h2o_system.xml'))
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr
    IC = geometric.internal.DelocalizedInternalCoordinates(M, build=True, connect=False, addcart=False)
    Gq_ana, Gq_num = geometric.optimize.CheckInternalGrad(coords, M, IC.Prims, engine, 'h2o2_h2o.tmp', False)
    Hq_ana, Hq_num = geometric.optimize.CheckInternalHess(coords, M, IC.Prims, engine, 'h2o2_h2o.tmp', False)
    assert np.allclose(Gq_ana, Gq_num, atol=1e-5)
    assert np.allclose(Hq_ana, Hq_num, atol=1e-5)
    #                                                    constraints=os.path.join(datad, 'ala_constraints.txt')
    # m = geometric.optimize.run_optimizer(openmm=True, enforce=0.1, pdb=os.path.join(datad, 'ala_a99sb_min.pdb'), input='amber99sb.xml',
    #                                      constraints=os.path.join(datad, 'ala_constraints.txt'))
    # scan_final = geometric.molecule.Molecule('scan-final.xyz')
    # scan_energies = np.array([float(c.split()[-1]) for c in scan_final.comms])
    # ref_energies = np.array([-0.03368698, -0.03349261])
    # # Check converged energies
    # assert np.allclose(scan_energies, ref_energies, atol=1.e-3)
    # # Check for performance regression (should be done in ~33 cycles)
    # assert len(m) < 50

@addons.using_openmm
def test_combination_detection(localizer):
    """Read in opls combination xml and make sure we find this to apply the combination rules"""
    M, engine = geometric.optimize.get_molecule_engine(openmm=True, pdb=os.path.join(datad, 'captan.pdb'), input=os.path.join(datad, 'captan.xml'))
    assert engine.combination == 'opls'

@addons.using_openmm
def test_opls_energy(localizer):
    """Test the opls energy evaluation of the molecule."""
    M, engine = geometric.optimize.get_molecule_engine(openmm=True, pdb=os.path.join(datad, 'captan.pdb'),input=os.path.join(datad, 'captan.xml'))
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr
    energy, grad = engine.calc_new(coords, None)
    opls_grad = np.array([-6.65563678e-03, -6.96108403e-03,  1.01219046e-02,
                           4.77186620e-03, -5.49923425e-03,  9.53944858e-03,
                           1.17318037e-02,  5.86619214e-03,  1.81590580e-03,
                           7.15324951e-03,  7.06591336e-03, -1.12101836e-02,
                           7.20033397e-03, -6.20852571e-03, -5.99141556e-04,
                          -7.09984632e-04,  6.16316713e-03, -6.55172883e-04,
                           3.58227761e-03,  5.81687632e-03,  8.24485640e-03,
                          -1.31141176e-03, -5.47527256e-03, -5.37463226e-05,
                          -4.62548834e-03,  1.16761467e-02, -1.22248256e-02,
                          -8.53129864e-03, -1.30094941e-03,  7.47111969e-04,
                           2.25825354e-02, -1.24891299e-02,  1.86977395e-02,
                          -1.40115259e-02, -6.63094157e-03, -2.94487154e-04,
                          -1.26591523e-03, -7.25662329e-04, -8.52036580e-03,
                          -7.91408908e-03,  1.17595326e-02, -1.00261913e-02,
                          -1.35293365e-02,  8.43635350e-03, -8.37737799e-03,
                           1.09957936e-03, -3.30495796e-03,  1.57696265e-03,
                           5.18478170e-03, -5.18710840e-03, -9.56596407e-05,
                           1.11783982e-03, -3.04045733e-03,  5.08006606e-03,
                          -6.20394358e-05,  2.60492060e-03, -9.76388473e-04,
                           2.88278170e-04, -2.18571329e-03, -1.84135131e-03,
                           8.43061547e-04,  5.81041130e-03, -4.49180435e-03,
                          -4.24613924e-03, -5.87608806e-03,  2.10134309e-03,
                          -9.18018280e-04, -2.62134372e-03,  4.31330813e-04,
                          -1.77472314e-03,  2.30695485e-03,  1.01002642e-03])
    assert energy == -0.004563862119216744
    assert grad.all() == opls_grad.all()
