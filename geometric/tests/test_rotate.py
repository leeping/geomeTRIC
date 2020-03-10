"""
Tests the geomeTRIC rotate class.
"""

import pytest
import geometric
import os
import numpy as np
from . import addons

datad = addons.datad
test_logger = addons.test_logger

def test_q_der(test_logger):
    
    M = geometric.molecule.Molecule(os.path.join(datad, 'water5.xyz'))
    x = M.xyzs[0]
    y = M.xyzs[-1]
    a1, a2 = geometric.rotate.get_q_der(x, y, second=True, fdcheck=False)
    l1, l2 = geometric.rotate.get_q_der(x, y, second=True, fdcheck=True, use_loops=True)
    n1, n2 = geometric.rotate.get_q_der(x, y, second=True, fdcheck=True)
    q = geometric.rotate.get_quat(x, y)
    q_ref = np.array([0.41665412, 0.44088624, -0.20980728, 0.7668113])
    assert np.allclose(a1, n1, atol=1.e-7)
    assert np.allclose(a1, l1, atol=1.e-7)
    assert np.allclose(a2, n2, atol=1.e-7)
    assert np.allclose(a2, l2, atol=1.e-7)
    assert np.allclose(q, q_ref, atol=1.e-7)
    
def test_expmap_der(test_logger):
    M = geometric.molecule.Molecule(os.path.join(datad, 'water5.xyz'))
    x = M.xyzs[0]
    y = M.xyzs[-1]
    a1, a2 = geometric.rotate.get_expmap_der(x, y, second=True, fdcheck=False)
    l1, l2 = geometric.rotate.get_expmap_der(x, y, second=True, fdcheck=True, use_loops=True)
    n1, n2 = geometric.rotate.get_expmap_der(x, y, second=True, fdcheck=True)
    v = geometric.rotate.get_expmap(x, y)
    v_ref = np.array([1.10677773, -0.5266892, 1.92496294])
    assert np.allclose(a1, n1, atol=1.e-7)
    assert np.allclose(a1, l1, atol=1.e-7)
    assert np.allclose(a2, n2, atol=1.e-7)
    assert np.allclose(a2, l2, atol=1.e-7)
    assert np.allclose(v, v_ref, atol=1.e-7)
    
