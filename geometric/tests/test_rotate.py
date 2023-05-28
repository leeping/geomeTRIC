"""
Tests the geomeTRIC rotate class.
"""

import pytest
import geometric
import os
import numpy as np
from . import addons

datad = addons.datad
exampled = addons.exampled
localizer = addons.in_folder

def test_q_der():
    
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
    
def test_expmap_der():
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
    
def test_rot_der():
    M = geometric.molecule.Molecule(os.path.join(datad, 'water5.xyz'))
    x = M.xyzs[0]
    y = M.xyzs[-1]
    a1, a2 = geometric.rotate.get_rot_der(x, y, second=True, fdcheck=False)
    n1, n2 = geometric.rotate.get_rot_der(x, y, second=True, fdcheck=True)
    assert np.allclose(a1, n1, atol=1.e-7)
    assert np.allclose(a2, n2, atol=1.e-7)
    
def test_F_R_der():
    M = geometric.molecule.Molecule(os.path.join(datad, 'water5.xyz'))
    x = M.xyzs[0]
    y = M.xyzs[-1]
    a1 = geometric.rotate.get_F_der(x, y, fdcheck=False)
    n1 = geometric.rotate.get_F_der(x, y, fdcheck=True)
    assert np.allclose(a1, n1, atol=1.e-7)
    a1 = geometric.rotate.get_R_der(x, y, fdcheck=False)
    n1 = geometric.rotate.get_R_der(x, y, fdcheck=True)
    assert np.allclose(a1, n1, atol=1.e-7)
    
def test_rmsd():
    M = geometric.molecule.Molecule(os.path.join(datad, 'water5.xyz'))
    x = M.xyzs[0]
    y = M.xyzs[-1]
    rmsd = geometric.rotate.calc_rmsd(x, y)
    np.testing.assert_almost_equal(rmsd, 2.222, decimal=3)

def test_write_displacements(localizer):
    # This test generates 7 rotated structures of a buckyball at (-180, -120, ..., +180) degrees.
    M = geometric.molecule.Molecule(os.path.join(exampled, '0-performance-tests', 'bucky_catcher', 'start.xyz'))
    IC = geometric.internal.PrimitiveInternalCoordinates(M, build=True, connect=False, addcart=False)
    ic_select=["Rotation-A 76-135"]
    # Set weight of IC to 1.0 so IC value matches actual rotation
    for ic in IC.Internals:
        if type(ic) in [geometric.internal.RotationA, geometric.internal.RotationB, geometric.internal.RotationC]:
            ic.w = 1.0
    coords = M.xyzs[0].flatten()*geometric.nifty.ang2bohr
    M_rot = geometric.ic_tools.write_displacements(coords, M, IC, '.', displace_range=(-np.pi, np.pi, 7), ic_select=ic_select)
    rot_angles = []
    # Use get_quat to get the actual rotation angles
    for i in range(len(M_rot)):
        refxyz = M.atom_select(list(range(75, 135))).xyzs[0]
        rotxyz = M_rot.atom_select(list(range(75, 135))).xyzs[i]
        q = geometric.rotate.get_quat(refxyz, rotxyz)
        rot_angles.append(np.arccos(q[0])*2)
    rot_angles = np.array(rot_angles)
    ref_angles = np.linspace(-np.pi, np.pi, 7)
    rot_diffs = rot_angles - ref_angles
    # Calculate expected vs. actual rotation differences modulo 2*pi
    # Sometimes the sign of the rotation angle can be flipped and that's okay
    rot_diffs = np.vstack((rot_diffs, rot_angles - ref_angles - 2*np.pi))
    rot_diffs = np.vstack((rot_diffs, rot_angles - ref_angles + 2*np.pi))
    rot_diffs = np.vstack((rot_diffs, rot_angles + ref_angles))
    rot_diffs = np.vstack((rot_diffs, rot_angles + ref_angles - 2*np.pi))
    rot_diffs = np.vstack((rot_diffs, rot_angles + ref_angles + 2*np.pi))
    rot_diffs = np.min(np.abs(rot_diffs), axis=0)
    assert rot_diffs.shape[0] == 7
    np.testing.assert_almost_equal(rot_diffs, np.zeros(7))

