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
def test_internal_geometry_functions(localizer):
    """
    Use finite difference to test the derivative functions at the top of internal.py
    """
    # Two pseudo-random vectors
    a = np.array([ 1.40150344, -0.18842162,  2.04874367])
    b = np.array([-0.26840721,  2.48198241,  1.31996791])
    np.testing.assert_almost_equal(np.linalg.norm(geometric.internal.unit_vector(a)), 1.0)
    d_unit_vector_f = np.zeros((3, 3), dtype=float)
    for i in range(3):
        a[i] += 1e-6
        dplus = geometric.internal.unit_vector(a)
        a[i] -= 2e-6
        dminus = geometric.internal.unit_vector(a)
        a[i] += 1e-6
        d_unit_vector_f[i, :] = (dplus-dminus)/2e-6
    np.testing.assert_almost_equal(geometric.internal.d_unit_vector(a), d_unit_vector_f)

    d_cross_f = np.zeros((3, 3), dtype=float)
    for i in range(3):
        a[i] += 1e-6
        dplus = np.cross(a, b)
        a[i] -= 2e-6
        dminus = np.cross(a, b)
        a[i] += 1e-6
        d_cross_f[i, :] = (dplus-dminus)/2e-6
    np.testing.assert_almost_equal(geometric.internal.d_cross(a, b), d_cross_f)

    d_ncross_f = np.zeros(3, dtype=float)
    for i in range(3):
        a[i] += 1e-6
        dplus = geometric.internal.ncross(a, b)
        a[i] -= 2e-6
        dminus = geometric.internal.ncross(a, b)
        a[i] += 1e-6
        d_ncross_f[i] = (dplus-dminus)/2e-6
    np.testing.assert_almost_equal(geometric.internal.d_ncross(a, b), d_ncross_f)

    d_nudot_f = np.zeros(3, dtype=float)
    for i in range(3):
        a[i] += 1e-6
        dplus = geometric.internal.nudot(a, b)
        a[i] -= 2e-6
        dminus = geometric.internal.nudot(a, b)
        a[i] += 1e-6
        d_nudot_f[i] = (dplus-dminus)/2e-6
    np.testing.assert_almost_equal(geometric.internal.d_nudot(a, b), d_nudot_f)

    d_ucross_f = np.zeros((3, 3), dtype=float)
    for i in range(3):
        a[i] += 1e-6
        dplus = geometric.internal.ucross(a, b)
        a[i] -= 2e-6
        dminus = geometric.internal.ucross(a, b)
        a[i] += 1e-6
        d_ucross_f[i, :] = (dplus-dminus)/2e-6
    np.testing.assert_almost_equal(geometric.internal.d_ucross(a, b), d_ucross_f)

    d_nucross_f = np.zeros(3, dtype=float)
    for i in range(3):
        a[i] += 1e-6
        dplus = geometric.internal.nucross(a, b)
        a[i] -= 2e-6
        dminus = geometric.internal.nucross(a, b)
        a[i] += 1e-6
        d_nucross_f[i] = (dplus-dminus)/2e-6
    np.testing.assert_almost_equal(geometric.internal.d_nucross(a, b), d_nucross_f)

def test_update_internals():
    M = geometric.molecule.Molecule(os.path.join(datad, 'water3.pdb'))
    IC_1 = geometric.internal.PrimitiveInternalCoordinates(M, build=True, connect=False, addcart=False)
    IC_2 = geometric.internal.PrimitiveInternalCoordinates(M, build=True, connect=False, addcart=True)
    assert IC_1.update(IC_1) == False
    assert IC_1.update(IC_2) == True
    assert IC_1.join(IC_2) == False
