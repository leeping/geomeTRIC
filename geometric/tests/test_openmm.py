"""
A set of tests for using the QCEngine project
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
