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

    def test_convert_gro(self, localizer):
        # print(len(self.molecule))
        # print(self.molecule.xyzs)
        # self.molecule.write('out.xyz')
        # M_xyz = geometric.molecule.Molecule('out.xyz')
        # assert np.allclose(self.molecule.xyzs[0], M_xyz.xyzs[0])
        for fmt in ['xyz', 'inpcrd', 'pdb', 'qdata', 'gro', 'arc']:
            print("Testing reading/writing of %s format for AlaGlu system" % fmt)
            outfnm = "out.%s" % fmt
            self.molecule.write(outfnm)
            M_test = geometric.molecule.Molecule(outfnm)
            assert np.allclose(self.molecule.xyzs[0], M_test.xyzs[0])
            if fmt in ['xyz', 'pdb', 'gro', 'arc']:
                assert self.molecule.elem == M_test.elem
            if fmt in ['pdb', 'gro']:
                assert self.molecule.resid == M_test.resid
                assert self.molecule.resname == M_test.resname
                assert self.molecule.atomname == M_test.atomname

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

    def test_align(self):
        # Test that alignment works
        # Create an Euler rotation matrix using some arbitrary rotations
        E = geometric.molecule.EulerMatrix(0.8*np.pi, 1.0*np.pi, -1.2*np.pi)
        # The rotated structure
        xyz1 = np.dot(self.molecule.xyzs[0], E)
        M = self.molecule + self.molecule + self.molecule
        M.xyzs[1] = xyz1.copy()
        # Perturb the last geometry, so RMSD is large even after alignment
        M.xyzs[2] = xyz1.copy() + np.arange(M.na*3).reshape(-1,3)*0.1
        # Calculate the RMSD a few different ways
        ref_rmsd_align = M.ref_rmsd(0)
        path_rmsd = M.pathwise_rmsd()
        pairwise_rmsd = M.all_pairwise_rmsd()
        ref_rmsd_noalign = M.ref_rmsd(0, align=False)
        assert ref_rmsd_align[0] < 1e-10
        assert ref_rmsd_align[1] < 1e-10
        assert ref_rmsd_align[2] > 1.0
        assert ref_rmsd_noalign[0] < 1e-10
        assert ref_rmsd_noalign[1] > 1.0
        assert ref_rmsd_noalign[2] > 1.0
        assert (path_rmsd + 1e-10 >= ref_rmsd_align[1:]).all()
        assert np.allclose(pairwise_rmsd[0], ref_rmsd_align)

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
        
class TestWaterQCOut:
    @classmethod
    def setup_class(cls):
        try: cls.molecule = geometric.molecule.Molecule(os.path.join(datad, 'water6_step2.qcout'))
        except:
            assert 0, "Failed to load water hexamer structure"

    def test_topology(self):
        """Check for the correct number of bonds in a simple molecule"""
        # print(len(self.molecule.bonds))
        # self.logger.debug("\nTrying to read alanine dipeptide conformation... ")
        assert len(self.molecule.bonds) == 12, "Incorrect number of bonds for water hexamer structure"
        assert len(self.molecule.molecules) == 6, "Incorrect number of molecules for water hexamer structure"

    def test_convert_qcout(self, localizer):
        # Test a variety of output formats
        # Q-Chem input file
        fmt = 'qcin'
        print("Testing reading/writing of %s format for water hexamer system" % fmt)
        outfnm = "out.%s" % fmt
        self.molecule.write(outfnm)
        M_test = geometric.molecule.Molecule(outfnm)
        assert np.allclose(self.molecule.xyzs[0], M_test.xyzs[0])
        assert M_test.charge == self.molecule.charge
        assert M_test.mult == self.molecule.mult
        assert M_test.qcrems == self.molecule.qcrems
        # ForceBalance qdata file
        fmt = 'qdata'
        print("Testing reading/writing of %s format for water hexamer system" % fmt)
        outfnm = "out.%s" % fmt
        self.molecule.write(outfnm)
        M_test = geometric.molecule.Molecule(outfnm)
        assert np.allclose(self.molecule.xyzs[0], M_test.xyzs[0])
        assert np.allclose(self.molecule.qm_energies, M_test.qm_energies)
        assert np.allclose(self.molecule.qm_grads[0], M_test.qm_grads[0])

def test_rings(localizer):
    ring_size_data = {'tetrahedrane.xyz': [3, 3, 3, 3],
                      'cholesterol.xyz' : [6, 6, 6, 5],
                      'bicyclo222octane.xyz' : [6, 6, 6],
                      'adamantane.xyz' : [6, 6, 6, 6],
                      'cubane.xyz' : [4, 4, 4, 4, 4, 4],
                      'coronene.xyz' : [6, 6, 6, 6, 6, 6, 6],
                      'porphin.xyz' : [5, 16, 5, 5, 5], 
                      'fenestradiene.xyz' : [6, 4, 5, 4, 6, 5, 6, 6, 4, 5, 4, 6, 5, 6],
                      'vancomycin.pdb' : [16, 16, 6, 16, 16, 6, 12, 6, 6, 6, 6, 6],
                      'c60.xyz' : [5, 6, 6, 6, 5, 6, 5, 6, 6, 5, 6, 6, 5, 6, 6, 5, 5, 6, 6, 5, 6, 6, 6, 5, 5, 6, 5, 6, 6, 6, 6, 5]}
    for fnm in ring_size_data.keys():
        M = geometric.molecule.Molecule(os.path.join(datad, fnm))
        ring_sizes = [len(i) for i in M.find_rings(max_size=20)]
        # Check that the number of rings is correct
        assert len(ring_sizes) == len(ring_size_data[fnm])
        # Check that ring sizes are correct and in the expected order
        assert ring_sizes == ring_size_data[fnm]
    
def test_rotate_bond(localizer):
    M = geometric.molecule.Molecule(os.path.join(datad, 'neu5ac.pdb'))
    M1, success1 = M.rotate_check_clash(0, (14, 16, 18, 20), printLevel=1)
    M2, success2 = M.rotate_check_clash(0, (14, 16, 18, 20), thresh_hyd=0.8, thresh_hvy=1.2)
    assert success1 == False
    assert success2 == True


def test_gaussian_input_single():
    """
    Test reading a gaussian input.
    """
    molecule = geometric.molecule.Molecule(os.path.join(datad, "ethane.com"))
    assert molecule.Data["charge"] == 0
    assert molecule.Data["mult"] == 1
    assert molecule.Data["ftype"] == "com"
    assert len(molecule.molecules) == 1
    xyz = np.array([[-4.13498124, 0.70342204, 0.],
           [-3.53650966, -0.1651672, -0.17967886],
           [-4.07172084, 1.36057017, -0.84205371],
           [-5.15338585, 0.41046458, 0.148081],
           [-3.62163902, 1.42937832, 1.25740497],
           [-3.68306794, 2.48719615, 1.10858318],
           [-4.22133413, 1.15219316, 2.09909028],
           [-2.60384227, 1.1531436, 1.43819258]])
    assert molecule.xyzs[0].tolist() == xyz.tolist()
    assert molecule.Data["comms"] == ["ethane"]
    assert molecule.Data["elem"] == ['C', 'H', 'H', 'H', 'C', 'H', 'H', 'H']
    assert molecule.Data["bonds"] == [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7)]


def test_gaussian_input_multiple():
    """
    Test reading a gaussian input with multiple molecules.
    """
    molecule = geometric.molecule.Molecule(os.path.join(datad, "waters.com"))
    assert len(molecule.molecules) == 6
    assert molecule.Data["bonds"] == [(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 8), (9, 10), (9, 11), (12, 13), (12, 14), (15, 16), (15, 17)]
