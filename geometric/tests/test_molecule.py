"""
Tests the geomeTRIC molecule class.
"""

import pytest
import geometric
from copy import deepcopy
import os
import numpy as np
from . import addons

datad = addons.datad
exampled = addons.exampled
localizer = addons.in_folder

def test_blank_molecule():
    mol = geometric.molecule.Molecule()
    assert len(mol) == 0

def test_cubic_box():
    box = geometric.molecule.CubicLattice(1.5)
    np.testing.assert_almost_equal(box.A, [1.5, 0.0, 0.0])
    np.testing.assert_almost_equal(box.B, [0.0, 1.5, 0.0])
    np.testing.assert_almost_equal(box.C, [0.0, 0.0, 1.5])
    np.testing.assert_almost_equal(box.V, 1.5**3)

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
        M.align()
        assert np.allclose(M.xyzs[1], M.xyzs[0])
        assert not np.allclose(M.xyzs[2], M.xyzs[0])

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

    def test_reorder(self):
        # Manually generated indices for scrambling atoms
        newidx = [8, 25, 15, 30, 28, 14, 27, 46, 39, 16, 36, 10, 13, 41, 11, 
                  12, 1, 2, 23, 21, 6, 20, 9, 43, 3, 44, 40, 34, 48, 0, 33, 
                  29, 19, 31, 32, 37, 7, 42, 18, 47, 22, 45, 24, 38, 17, 4, 35, 5, 26]

        # Scrambled molecule
        newmol = self.molecule.atom_select(newidx)
        
        newidx_2 = self.molecule.reorder_indices(newmol)
        assert newidx == newidx_2
        newmol2 = deepcopy(self.molecule)
        newmol2.reorder_according_to(newmol)
        np.testing.assert_almost_equal(newmol2.xyzs[0], newmol.xyzs[0])
        
        # Now find the indices that would map the scrambled molecule back to the original
        invidx = [newidx.index(i) for i in range(len(newidx))]
        newmol3 = newmol.reorder_indices(self.molecule)
        assert invidx == newmol3

    def test_write_lammps(self):
        # Outside of running LAMMPS we can't really check if it's correct.
        # At least we confirm it doesn't crash.
        self.molecule.xyzs[0] += np.array([10,10,10])[np.newaxis, :]
        self.molecule.write_lammps_data()

    def test_write_gro(self, localizer):
        self.molecule.write('out.gro')
        M1 = geometric.molecule.Molecule('out.gro')
        assert M1.atomname == self.molecule.atomname
        assert M1.elem == self.molecule.elem
        assert M1.resid == self.molecule.resid
        np.testing.assert_almost_equal(M1.boxes[0].A, self.molecule.boxes[0].A)
        np.testing.assert_almost_equal(M1.boxes[0].B, self.molecule.boxes[0].B)
        np.testing.assert_almost_equal(M1.boxes[0].C, self.molecule.boxes[0].C)
        np.testing.assert_almost_equal(M1.xyzs[0], self.molecule.xyzs[0])

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
        # Some dummy values to test the code for writing qdata
        self.qm_interaction = [0.0]
        self.qm_espxyzs = [np.array([0.0, 0.0, 0.0])]
        self.qm_espvals = [1.0]
        self.mm_energies = [0.0]
        self.molecule.write(outfnm)
        M_test = geometric.molecule.Molecule(outfnm)
        assert np.allclose(self.molecule.xyzs[0], M_test.xyzs[0])
        assert np.allclose(self.molecule.qm_energies, M_test.qm_energies)
        assert np.allclose(self.molecule.qm_grads[0], M_test.qm_grads[0])

    def test_fill_atomname(self, localizer):
        self.molecule.resname = ['HOH' for i in range(self.molecule.na)]
        self.molecule.resid = [i//3 + 1 for i in range(self.molecule.na)]
        self.molecule.boxes = [geometric.molecule.CubicLattice(20)]
        self.molecule.write("out.pdb")
        M = geometric.molecule.Molecule("out.pdb")
        assert M.atomname == ['O1', 'H2', 'H3', 'O1', 'H2', 'H3', 'O1', 'H2', 'H3', 
                              'O1', 'H2', 'H3', 'O1', 'H2', 'H3', 'O1', 'H2', 'H3']
        self.molecule.write("out.gro")
        M = geometric.molecule.Molecule("out.gro")
        assert M.atomname == ['O1', 'H2', 'H3', 'O1', 'H2', 'H3', 'O1', 'H2', 'H3', 
                              'O1', 'H2', 'H3', 'O1', 'H2', 'H3', 'O1', 'H2', 'H3']

    def test_add_quantum(self):
        molecule2 = geometric.molecule.Molecule(os.path.join(datad, "water6.pdb"))
        molecule2.add_quantum(self.molecule)
        print(self.molecule.Data.keys())#assert 'qm_energies' in molecule2.Data
        print(molecule2.Data.keys())#assert 'qm_energies' in molecule2.Data

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

def test_delete_atoms(localizer):
    """
    Test that atom deletion behaves properly.
    """
    M = geometric.molecule.Molecule(os.path.join(datad, 'chromophore.xyz'), ftype='tinker')
    M1 = M.delete_atoms([10, 26])
    old_bonds = [(8, 12), (23, 27), (46, 50)]
    new_bonds = [(8, 11), (22, 25), (44, 48)]
    for i in range(len(old_bonds)):
        assert old_bonds[i] in M.bonds
        assert new_bonds[i] in M1.bonds
    M1.write('test.xyz', ftype='tinker')
    M2 = geometric.molecule.Molecule('test.xyz', ftype='tinker')
    assert M2.bonds == M1.bonds

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

def test_quick_input():
    """
    Test reading a quick input
    """
    molecule = geometric.molecule.Molecule(os.path.join(exampled, "1-simple-examples","water2_quick","Water2.qkin"))
    assert molecule.Data["charge"] == 0
    assert molecule.Data["mult"] == 1
    assert molecule.Data["ftype"] == "qkin"
    assert len(molecule.molecules) == 2
    xyz = np.array([[ 0.85591, -1.38236,  0.31746],
           [ 1.67524, -1.84774,  0.4858 ],
           [ 1.11761, -0.46843,  0.20575],
           [-1.09863, -0.85837,  2.17319],
           [-0.4031 , -1.14608,  1.58184],
           [-1.08511,  0.09683,  2.11282]])
    assert molecule.xyzs[0].tolist() == xyz.tolist()
    assert molecule.Data["comms"] == ['HF BASIS=STO-3G cutoff=1.0e-9 denserms=1.0e-6 GRADIENT'] 
    assert molecule.Data["elem"] == ['O', 'H', 'H', 'O', 'H', 'H'] 

def test_gaussian_input_multiple():
    """
    Test reading a gaussian input with multiple molecules.
    """
    molecule = geometric.molecule.Molecule(os.path.join(datad, "waters.com"))
    assert len(molecule.molecules) == 6
    assert molecule.Data["bonds"] == [(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 8), (9, 10), (9, 11), (12, 13), (12, 14), (15, 16), (15, 17)]

def test_charmm_io(localizer):
    molecule = geometric.molecule.Molecule(os.path.join(datad, "boat.crd"), ftype='charmm')
    molecule.write("test.xyz")
    molecule2 = geometric.molecule.Molecule("test.xyz")
    np.testing.assert_almost_equal(molecule2.xyzs[0], molecule.xyzs[0])

def test_is_gro_coord_box():
    assert geometric.molecule.is_gro_coord('    1SOL     H6    6   0.018772  -0.140563  -0.004262') == True
    assert geometric.molecule.is_gro_coord(' 2500MAZ    H1730100   0.576250553  11.044091225   0.476635426') == True
    assert geometric.molecule.is_gro_coord('random string') == False
    assert geometric.molecule.is_gro_box('  3.000000000   3.000000000   3.000000000') == True
    assert geometric.molecule.is_gro_box('   7.41008   6.98629   6.05030   0.00000   0.00000   2.47003   0.00000  -2.47003   3.49314') == True
    assert geometric.molecule.is_gro_box('   7.41008   6.98629   6.05030   0.00000   0.00000   2.47003   0.00000  -2.47003   3.49314x') == False

def test_even_list():
    assert geometric.molecule.even_list(27, 5) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16], [17, 18, 19, 20, 21], [22, 23, 24, 25, 26]]

class TestReactionXYZ:
    @classmethod
    def setup_class(cls):
        try: cls.molecule = geometric.molecule.Molecule(os.path.join(datad, 'reaction_000_deci.xyz'))
        except:
            assert 0, "Failed to load reaction_000_deci structure"

    def test_arc_equal_spacing(self):
        m_eq = geometric.molecule.EqualSpacing(self.molecule)
        assert len(m_eq) == len(self.molecule)
        arclength = sum(geometric.molecule.arc(self.molecule))
        arclength2 = sum(geometric.molecule.arc(self.molecule[::2]))
        assert arclength2 < arclength

    def test_read_comm_charge_mult(self):
        self.molecule.read_comm_charge_mult()
        assert self.molecule.charge == 2
        assert self.molecule.mult == 5
    
    def test_compare_topology(self):
        m1 = self.molecule[0]
        m1.build_topology()
        assert geometric.molecule.TopEqual(m1, m1)
        assert geometric.molecule.MolEqual(m1, m1)
    
        indices = [ 5, 22, 27, 26, 18, 20, 28,  7,  8, 21,  1, 12, 29,  2, 24, 19, 13, 11, 10,  4,  3, 23,  6,  9, 15, 16, 25, 17,  0, 14]
        m2 = m1.atom_select(indices)
        assert not geometric.molecule.TopEqual(m1, m2)
        assert geometric.molecule.MolEqual(m1, m2)
    
        m3 = self.molecule[-1]
        m3.build_topology()
        assert not geometric.molecule.TopEqual(m1, m3)
        assert not geometric.molecule.MolEqual(m1, m3)

    def test_radius_gyration(self):
        np.testing.assert_almost_equal(self.molecule.radius_of_gyration()[0], 3.577, decimal=3)

    def test_iterate(self):
        counter = 0
        for i in self.molecule:
            assert type(i) is geometric.molecule.Molecule
            counter += 1
        assert counter == 35

    def test_repair(self):
        m2 = deepcopy(self.molecule)
        m2.boxes = [geometric.molecule.CubicLattice(20)]
        m2.comms = ['test comment']
        assert len(m2) == 35

class TestWaterBox:

    @classmethod
    def setup_class(cls):
        try: cls.molecule = geometric.molecule.Molecule(os.path.join(datad, 'spc216.gro'), toppbc=True)
        except:
            assert 0, "Failed to load spc216.gro structure"

    def test_num_water_molecules(self):
        """Check for the correct number of molecules in a cubic cell with broken molecules"""
        assert len(self.molecule.molecules) == 216, "Incorrect number of molecules for water box structure"

class TestWaterCluster:

    @classmethod
    def setup_class(cls):
        try: cls.molecule = geometric.molecule.Molecule(os.path.join(datad, 'water12.mdcrd'), top=os.path.join(datad, 'water12.pdb'))
        except:
            assert 0, "Failed to load water12 structure"

    def test_rigid_water(self, localizer):
        self.molecule.rigid_water()
        for i in range(len(self.molecule)):
            for j in range(12):
                np.testing.assert_almost_equal(self.molecule[i].measure_distances(j*3, j*3+1)[0], 0.9572)
                np.testing.assert_almost_equal(self.molecule[i].measure_distances(j*3, j*3+2)[0], 0.9572)
                np.testing.assert_almost_equal(self.molecule[i].measure_angles(j*3+1, j*3, j*3+2)[0], 104.52)
        self.molecule.write('frame0-2.mdcrd', selection=[2, 3, 4])
        molecule2 = geometric.molecule.Molecule('frame0-2.mdcrd', top=os.path.join(datad, 'water12.pdb'))
        np.testing.assert_almost_equal(self.molecule.xyzs[4], molecule2.xyzs[2], decimal=3)

def test_tinker_atom_select_stack():
    molecule = geometric.molecule.Molecule(os.path.join(datad, 'naphthax.xyz'), ftype='tinker')
    with pytest.raises(KeyError) as excinfo:
        b = molecule.atom_select(list(range(18)))
    a = molecule.atom_select([0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    b = molecule.atom_select([9, 10, 11, 12, 13, 14, 15, 16, 17, 27, 28, 29, 30, 31, 32, 33, 34, 35])
    assert b.tinkersuf[0].split() == ['2', '2', '5', '6']
    s = a.atom_stack(b)
    assert s.tinkersuf[18].split() == ['2', '20', '23', '24']
    
def test_read_write_qcfsm_in(localizer):
    molecule = geometric.molecule.Molecule(os.path.join(datad, 'qcfsm.in'), ftype='qcin')
    assert molecule.na == 8
    assert molecule.qcrems[0]['jobtype'] == 'fsm'
    assert len(molecule) == 2
    # Add a fictitious external charge
    molecule.qm_extchgs = [np.array([[0.0, 0.0, 1.0, 0.0]]), np.array([[0.0, 1.0, 0.0, 0.0]])]
    molecule.write('qcfsm_out.in', ftype='qcin')

def test_read_qcfsm_out(localizer):
    import bz2
    with bz2.open(os.path.join(datad, "qcfsm.out.bz2")) as f:
        content = f.read()
    with open("qcfsm.out", "wb") as fo:
        fo.write(content)
    molecule = geometric.molecule.Molecule("qcfsm.out")
    np.testing.assert_almost_equal(molecule.qm_energies[-1], -170.998405245)
    np.testing.assert_almost_equal(molecule.xyzs[-1][-1, 2], -0.5693570713)

def test_read_qcesp():
    M = geometric.molecule.Molecule(os.path.join(datad, 'test_qchem.esp'))
    assert 'qm_espxyzs' in M.Data
    assert 'qm_espvals' in M.Data
    assert len(M.qm_espvals[0]) == 280

def test_read_qcin_4part(localizer):
    # Read in a Q-Chem input file with 4 parts and a z-matrix (the Z-matrix is not parsed or interpreted).
    M = geometric.molecule.Molecule(os.path.join(datad, 'qchem_4part.in'))
    assert len(M.qcrems) == 4
    assert M.qcrems[3]['scf_guess'] == 'READ'
    M.write('qchem_4part_out.in')

def test_read_qcfreq(localizer):
    import bz2
    with bz2.open(os.path.join(datad, "qchem_tsfreq.out.bz2")) as f:
        content = f.read()
    with open("qchem_tsfreq.out", "wb") as fo:
        fo.write(content)
    molecule = geometric.molecule.Molecule("qchem_tsfreq.out")
    assert len(molecule.freqs) == 63

def test_read_qcirc(localizer):
    import bz2
    with bz2.open(os.path.join(datad, "qchem_irc.out.bz2")) as f:
        content = f.read()
    with open("qchem_irc.out", "wb") as fo:
        fo.write(content)
    molecule = geometric.molecule.Molecule("qchem_irc.out")
    # Todo: Verify IRC data.
    assert hasattr(molecule, 'Irc')

def test_read_triclinic_gro():
    M = geometric.molecule.Molecule(os.path.join(datad, 'triclinic.gro'))
    np.testing.assert_almost_equal(M.boxes[-1].A / 10, [1.824070000, 0.000000000, 0.000000000])
    np.testing.assert_almost_equal(M.boxes[-1].B / 10, [-0.151660337, 1.817797398, 0.000000000])
    np.testing.assert_almost_equal(M.boxes[-1].C / 10, [0.148010793, -0.149309580, 1.807615769])

