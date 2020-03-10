"""
Tests second derivatives of internal coordinates w/r.t. Cartesians.
"""

import pytest
import geometric
import os
import numpy as np
from . import addons
from geometric.internal import *

datad = addons.datad
test_logger = addons.test_logger

def test_hessian_assort(test_logger):
    M = geometric.molecule.Molecule(os.path.join(datad, 'assort.xyz'))
    coords = M.xyzs[0].flatten() * ang2bohr
    # Build TRIC coordinate system
    IC_ref = ['Distance 1-2', 'Distance 3-4', 'Distance 3-5', 'Distance 5-6', 'Distance 5-7', 'Distance 8-9',
              'Distance 8-10', 'Distance 8-11', 'Distance 8-12', 'Distance 12-13', 'Distance 12-14', 'Angle 3-5-7',
              'Angle 6-5-7', 'Angle 9-8-10', 'Angle 9-8-11', 'Angle 9-8-12', 'Angle 10-8-11', 'Angle 10-8-12',
              'Angle 11-8-12', 'Angle 8-12-13', 'Angle 8-12-14', 'Angle 13-12-14', 'LinearAngleX 4-3-5',
              'LinearAngleY 4-3-5', 'Out-of-Plane 5-3-6-7', 'Dihedral 9-8-12-13', 'Dihedral 9-8-12-14',
              'Dihedral 10-8-12-13', 'Dihedral 10-8-12-14', 'Dihedral 11-8-12-13', 'Dihedral 11-8-12-14',
              'Translation-X 1-2', 'Translation-X 3-7', 'Translation-X 8-14', 'Translation-Y 1-2',
              'Translation-Y 3-7', 'Translation-Y 8-14', 'Translation-Z 1-2', 'Translation-Z 3-7',
              'Translation-Z 8-14', 'Rotation-A 1-2', 'Rotation-A 3-7', 'Rotation-A 8-14', 'Rotation-B 1-2',
              'Rotation-B 3-7', 'Rotation-B 8-14', 'Rotation-C 1-2', 'Rotation-C 3-7', 'Rotation-C 8-14']
    IC = DelocalizedInternalCoordinates(M, build=True, connect=False, addcart=False, constraints=None, cvals=None)
    assert set([str(p) for p in IC.Prims.Internals]) == set(IC_ref)
    fgrad_prim = IC.Prims.checkFiniteDifferenceGrad(coords)
    agrad_prim = IC.Prims.derivatives(coords)
    fhess_prim = IC.Prims.checkFiniteDifferenceHess(coords)
    ahess_prim = IC.Prims.second_derivatives(coords)
    fgrad_dlc = IC.checkFiniteDifferenceGrad(coords)
    agrad_dlc = IC.derivatives(coords)
    fhess_dlc = IC.checkFiniteDifferenceHess(coords)
    ahess_dlc = IC.second_derivatives(coords)
    assert np.allclose(fgrad_prim, agrad_prim, atol=1.e-6)
    assert np.allclose(fhess_prim, ahess_prim, atol=1.e-6)
    assert np.allclose(fgrad_dlc, agrad_dlc, atol=1.e-6)
    assert np.allclose(fhess_dlc, ahess_dlc, atol=1.e-6)

