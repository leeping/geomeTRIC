"""
Tests the geomeTRIC molecule class.
"""

import pytest
import geometric
import os
import numpy as np
from . import addons

datad = addons.datad
localizer = addons.in_folder

def test_blank_molecule():
    mol = geometric.molecule.Molecule()

    assert len(mol) == 0

class TestAlaGRO:
    @classmethod
    def setup_class(cls):
        try: cls.molecule = geometric.molecule.Molecule(os.path.join(datad, 'alaglu.gro'))
        except:
            assert 0, "Failed to load ACE-ALA-NME ACE-GLU-NME structure"

    def test_topology(self):
        """Check for the correct number of bonds in a simple molecule"""
        # print(len(self.molecule.bonds))
        # self.logger.debug("\nTrying to read alanine dipeptide conformation... ")
        assert len(self.molecule.bonds) == 47, "Incorrect number of bonds for ACE-ALA-NME ACE-GLU-NME structure"
        assert len(self.molecule.molecules) == 2, "Incorrect number of molecules for ACE-ALA-NME ACE-GLU-NME structure"

    def test_measure_distances(self):
        measure = self.molecule.measure_distances(41,43)
        ref = 1.337198
        np.testing.assert_almost_equal(measure, ref, 4)
        
    def test_measure_angles(self):
        measure = self.molecule.measure_angles(40,14,39)
        ref = 9.429428
        np.testing.assert_almost_equal(measure, ref, 4)

    def test_measure_dihedrals(self):
        measure = self.molecule.measure_dihedrals(35,32,30,28)
        ref = 56.5389
        np.testing.assert_almost_equal(measure, ref, 4)

    def test_lattice(self):
        bx = self.molecule.boxes[0]
        np.testing.assert_almost_equal(bx.a, 20.0)
        np.testing.assert_almost_equal(bx.b, 20.0)
        np.testing.assert_almost_equal(bx.c, 20.0)
        np.testing.assert_almost_equal(bx.alpha, 90.0)
        np.testing.assert_almost_equal(bx.beta, 90.0)
        np.testing.assert_almost_equal(bx.gamma, 90.0)
        # This is how to override "ask for user input"
        # when running unit tests.
        # Three numbers = rectilinear box
        del self.molecule.Data['boxes']
        geometric.molecule.input = lambda userinput : '11 13 15'
        self.molecule.require_boxes()
        bx = self.molecule.boxes[0]
        np.testing.assert_almost_equal(bx.a, 11.0)
        np.testing.assert_almost_equal(bx.b, 13.0)
        np.testing.assert_almost_equal(bx.c, 15.0)
        np.testing.assert_almost_equal(bx.alpha, 90.0)
        np.testing.assert_almost_equal(bx.beta, 90.0)
        np.testing.assert_almost_equal(bx.gamma, 90.0)
        # Six numbers = specify alpha, beta, gamma
        del self.molecule.Data['boxes']
        geometric.molecule.input = lambda userinput : '11 13 15 81 82 93.5'
        self.molecule.require_boxes()
        bx = self.molecule.boxes[0]
        np.testing.assert_almost_equal(bx.a, 11.0)
        np.testing.assert_almost_equal(bx.b, 13.0)
        np.testing.assert_almost_equal(bx.c, 15.0)
        np.testing.assert_almost_equal(bx.alpha, 81.0)
        np.testing.assert_almost_equal(bx.beta, 82.0)
        np.testing.assert_almost_equal(bx.gamma, 93.5)
        # Nine numbers = specify box vectors
        # In this case the box vectors of a truncated octahedral box are given.
        del self.molecule.Data['boxes']
        geometric.molecule.input = lambda userinput : '7.65918   7.22115   6.25370   0.00000   0.00000   2.55306   0.00000  -2.55306   3.61057'
        self.molecule.require_boxes()
        bx = self.molecule.boxes[0]
        np.testing.assert_almost_equal(bx.a, 7.659, decimal=3)
        np.testing.assert_almost_equal(bx.b, 7.659, decimal=3)
        np.testing.assert_almost_equal(bx.c, 7.659, decimal=3)
        np.testing.assert_almost_equal(bx.alpha, 70.53, decimal=2)
        np.testing.assert_almost_equal(bx.beta, 109.47, decimal=3)
        np.testing.assert_almost_equal(bx.gamma, 70.53, decimal=3)
        # A single number = cubic box
        del self.molecule.Data['boxes']
        geometric.molecule.input = lambda userinput : '20'
        self.molecule.require_boxes()
        bx = self.molecule.boxes[0]
        np.testing.assert_almost_equal(bx.a, 20.0)
        np.testing.assert_almost_equal(bx.b, 20.0)
        np.testing.assert_almost_equal(bx.c, 20.0)
        np.testing.assert_almost_equal(bx.alpha, 90.0)
        np.testing.assert_almost_equal(bx.beta, 90.0)
        np.testing.assert_almost_equal(bx.gamma, 90.0)

    def test_add(self):
        # Test adding of Molecule objects and ensure that copies are created when adding
        M = self.molecule + self.molecule # __add__
        M += self.molecule                # __iadd__
        assert len(M) == 3
        assert np.allclose(M.xyzs[0], M.xyzs[1])
        M.xyzs[0][0,0] += 1.0
        assert not np.allclose(M.xyzs[0], M.xyzs[1], atol=0.1)
        assert not np.allclose(M.xyzs[0], M.xyzs[2], atol=0.1)
        assert np.allclose(M.xyzs[1], M.xyzs[2])
        M.xyzs[1][0,0] += 1.0
        assert np.allclose(M.xyzs[0], M.xyzs[1])
        assert not np.allclose(M.xyzs[0], M.xyzs[2], atol=0.1)
        assert not np.allclose(M.xyzs[1], M.xyzs[2], atol=0.1)
        M.xyzs[2][0,0] += 1.0
        assert np.allclose(M.xyzs[0], M.xyzs[1])
        assert np.allclose(M.xyzs[1], M.xyzs[2])
        assert np.allclose(M.xyzs[0], M.xyzs[2])

    def test_write(self, localizer):
        # print(len(self.molecule))
        # print(self.molecule.xyzs)
        # self.molecule.write('out.xyz')
        # M_xyz = geometric.molecule.Molecule('out.xyz')
        # assert np.allclose(self.molecule.xyzs[0], M_xyz.xyzs[0])
        for fmt in ['xyz', 'inpcrd', 'pdb', 'qdata', 'arc']:
            print("Testing reading/writing of %s format for AlaGlu system" % fmt)
            outfnm = "out.%s" % fmt
            self.molecule.write(outfnm)
            M_test = geometric.molecule.Molecule(outfnm)
            assert np.allclose(self.molecule.xyzs[0], M_test.xyzs[0])

    def test_select_stack(self):
        M1 = self.molecule.atom_select(range(22))
        assert len(M1.bonds) == 21
        assert len(M1.molecules) == 1
        M2 = self.molecule.atom_select(range(22, self.molecule.na))
        assert len(M2.bonds) == 26
        assert len(M2.molecules) == 1
        M3 = M1.atom_stack(M2)
        assert np.allclose(self.molecule.xyzs[0], M3.xyzs[0])
        M1.xyzs[0][0,0] += 1.0
        assert np.allclose(self.molecule.xyzs[0], M3.xyzs[0])

    def test_find_angles_dihedrals(self):
        a = self.molecule.find_angles()
        assert len(a) == 81
        d = self.molecule.find_dihedrals()
        assert len(d) == 97

    def test_remove_tr(self):
        IC = geometric.internal.DelocalizedInternalCoordinates(self.molecule, build=True, connect=False, addcart=False)
        IC_TR = geometric.internal.DelocalizedInternalCoordinates(self.molecule, build=True, connect=False, addcart=False, remove_tr=True)
        assert len(IC.Internals) == self.molecule.na*3
        assert len(IC_TR.Internals) == (self.molecule.na*3 - 6)

    def teardown_method(self, method):
        # This method is being called after each test case, and it will revert input back to original function
        geometric.molecule.input = input
        
