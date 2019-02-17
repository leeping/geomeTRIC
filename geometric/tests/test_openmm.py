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

pdb_water3 = """REMARK   1 CREATED WITH FORCEBALANCE 2019-02-17
HETATM    1  O   HOH A   1       0.856  -1.382   0.317  0.00  0.00           O  
HETATM    2  H1  HOH A   1       1.675  -1.848   0.486  0.00  0.00           H  
HETATM    3  H2  HOH A   1       1.118  -0.468   0.206  0.00  0.00           H  
TER
HETATM    4  O   HOH A   2      -1.099  -0.858   2.173  0.00  0.00           O  
HETATM    5  H1  HOH A   2      -0.403  -1.146   1.582  0.00  0.00           H  
HETATM    6  H2  HOH A   2      -1.085   0.097   2.113  0.00  0.00           H  
TER
HETATM    7  O   HOH A   3       1.864   1.220   1.333  0.00  0.00           O  
HETATM    8  H1  HOH A   3       2.010   1.883   0.658  0.00  0.00           H  
HETATM    9  H2  HOH A   3       2.642   0.663   1.295  0.00  0.00           H  
TER
"""

pdb_water6 = """REMARK   1 CREATED WITH FORCEBALANCE 2019-02-17
HETATM    1  O   HOH A   1       0.856  -1.382   0.317  0.00  0.00           O  
HETATM    2  H1  HOH A   1       1.675  -1.848   0.486  0.00  0.00           H  
HETATM    3  H2  HOH A   1       1.118  -0.468   0.206  0.00  0.00           H  
TER
HETATM    4  O   HOH A   2      -1.099  -0.858   2.173  0.00  0.00           O  
HETATM    5  H1  HOH A   2      -0.403  -1.146   1.582  0.00  0.00           H  
HETATM    6  H2  HOH A   2      -1.085   0.097   2.113  0.00  0.00           H  
TER
HETATM    7  O   HOH A   3       1.864   1.220   1.333  0.00  0.00           O  
HETATM    8  H1  HOH A   3       2.010   1.883   0.658  0.00  0.00           H  
HETATM    9  H2  HOH A   3       2.642   0.663   1.295  0.00  0.00           H  
TER
HETATM   10  O   HOH A   4       0.436   0.859  -1.799  0.00  0.00           O  
HETATM   11  H1  HOH A   4       1.295   0.594  -1.471  0.00  0.00           H  
HETATM   12  H2  HOH A   4       0.136   0.111  -2.315  0.00  0.00           H  
TER
HETATM   13  O   HOH A   5      -0.801  -2.091  -1.573  0.00  0.00           O  
HETATM   14  H1  HOH A   5      -0.220  -1.830  -0.858  0.00  0.00           H  
HETATM   15  H2  HOH A   5      -0.328  -1.846  -2.368  0.00  0.00           H  
TER
HETATM   16  O   HOH A   6      -1.497   2.268  -0.330  0.00  0.00           O  
HETATM   17  H1  HOH A   6      -0.847   1.694  -0.735  0.00  0.00           H  
HETATM   18  H2  HOH A   6      -2.325   1.798  -0.427  0.00  0.00           H  
TER
"""

@addons.using_openmm
def test_dlc_openmm_water3(localizer):
    """
    Optimize the geometry of three water molecules using standard delocalized internal coordinates. 
    The coordinate system will break down and have to be rebuilt.
    """
    with open('water3.pdb', 'w') as f:
        f.write(pdb_water3)
    progress = geometric.optimize.run_optimizer(openmm=True, pdb='water3.pdb', coordsys='dlc', input='tip3p.xml')
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
    with open('water6.pdb', 'w') as f:
        f.write(pdb_water6)
    progress = geometric.optimize.run_optimizer(openmm=True, pdb='water6.pdb', input='tip3p.xml')
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
    assert progress.qm_energies[-1] < (e_ref + 0.001)
    # Check that the optimization converged in less than 100 steps
    assert len(progress) < 100
    # Check that the geometry matches the reference
    assert rmsd < 0.01
    assert maxd < 0.02


