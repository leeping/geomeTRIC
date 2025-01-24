"""
Tests second derivatives of internal coordinates w/r.t. Cartesians.
"""

import pytest
import geometric
import subprocess
import os, shutil
import numpy as np
import tempfile
from . import addons
from geometric.internal import *

datad = addons.datad
localizer = addons.in_folder


def test_hessian_assort():
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

@addons.using_psi4
@addons.using_bigchem
def test_psi4_bigchem_hessian(bigchem_frequency):
    freqs = bigchem_frequency("psi4", "hcn_minimized.psi4in")

    np.testing.assert_almost_equal(freqs, [989.5974, 989.5992, 2394.0352, 3690.5745], decimal=0)
    assert len(freqs) == 4


@addons.using_qchem
@addons.using_bigchem
def test_qchem_bigchem_hessian(bigchem_frequency):
    # Optimized TS structure of HCN -> CNH reaction. 
    freqs = bigchem_frequency("qchem", "hcn_minimized_ts.qcin")

    np.testing.assert_almost_equal(freqs, [-1215.8587, 2126.6551, 2451.8599], decimal=0)
    assert freqs[0] < 0
    assert len(freqs) == 3


@addons.using_terachem
@addons.using_bigchem
def test_terachem_bigchem_hessian(bigchem_frequency):
    shutil.copy2(os.path.join(datad,"hcn_minimized.xyz"), os.path.join(os.getcwd(), "hcn_minimized.xyz"))
    freqs = bigchem_frequency("tera", "hcn_minimized.tcin")
    os.remove("hcn_minimized.xyz")

    np.testing.assert_almost_equal(freqs, [989.5482, 989.7072, 2394.4201, 3694.5272], decimal=0)
    assert len(freqs) == 4


class TestPsi4WorkQueueHessian:

    """ Tests are put into class so that the fixture can terminate the worker process. """

    @pytest.fixture(autouse=True)
    def work_queue_cleanup(self):
        self.workers = None
        yield
        if self.workers is not None:
            for worker in self.workers:
                worker.terminate()

    @addons.using_psi4
    @addons.using_workqueue
    def test_psi4_work_queue_hessian(self, localizer):
        shutil.copy2(os.path.join(datad, 'hcn_minimized.psi4in'), os.getcwd())

        geometric.nifty.createWorkQueue(9191, debug=False)
        wq = geometric.nifty.getWorkQueue()

        molecule, engine = geometric.prepare.get_molecule_engine(engine="psi4", input="hcn_minimized.psi4in")
        coords = molecule.xyzs[0].flatten()*geometric.nifty.ang2bohr
        worker_program = geometric.nifty.which('work_queue_worker')
        # Assume 4 threads are available
        self.workers = [subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", str(wq.port)],
                                         stdout=subprocess.PIPE) for i in range(4)]
        hessian = geometric.normal_modes.calc_cartesian_hessian(coords, molecule, engine, os.getcwd())
        freqs, modes, G = geometric.normal_modes.frequency_analysis(coords, hessian, elem=molecule.elem, verbose=True)
        np.testing.assert_almost_equal(freqs, [989.5974, 989.5992, 2394.0352, 3690.5745], decimal=0)
        assert len(freqs) == 4
        geometric.nifty.destroyWorkQueue()


class TestASEWorkQueueHessian:

    """ Tests are put into class so that the fixture can terminate the worker process. """

    @pytest.fixture(autouse=True)
    def work_queue_cleanup(self):
        self.workers = None
        yield
        if self.workers is not None:
            for worker in self.workers:
                worker.terminate()

    @addons.using_ase
    @addons.using_workqueue
    def test_ase_work_queue_hessian(self, localizer):

        shutil.copy2(os.path.join(datad, 'water1_coords_gfn2-xtb.xyz'), os.path.join(os.getcwd(), "start.xyz"))

        geometric.nifty.createWorkQueue(9191, debug=False)
        wq = geometric.nifty.getWorkQueue()

        molecule, engine = geometric.prepare.get_molecule_engine(engine="ase", input="start.xyz", ase_class="xtb.ase.calculator.XTB", ase_kwargs='{"method":"GFN2-xTB", "accuracy":0.001}')
        coords = molecule.xyzs[0].flatten()*geometric.nifty.ang2bohr
        worker_program = geometric.nifty.which('work_queue_worker')
        # Assume 4 threads are available
        self.workers = [subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", str(wq.port)],
                                         stdout=subprocess.PIPE) for i in range(4)]
        hessian = geometric.normal_modes.calc_cartesian_hessian(coords, molecule, engine, os.getcwd())
        freqs, modes, G = geometric.normal_modes.frequency_analysis(coords, hessian, elem=molecule.elem, verbose=True)
        np.testing.assert_almost_equal(freqs, [1539.536, 3642.893, 3651.028], decimal=1)
        assert len(freqs) == 3
        geometric.nifty.destroyWorkQueue()


@addons.using_psi4
def test_hessian_conversion(localizer):
    shutil.copy2(os.path.join(datad, 'hcn_tsguess.psi4in'), os.getcwd())
    molecule, engine = geometric.prepare.get_molecule_engine(engine="psi4", input="hcn_tsguess.psi4in")
    IC = geometric.internal.DelocalizedInternalCoordinates(molecule, build=True, connect=False, addcart=True)
    coords = molecule.xyzs[0].flatten()*geometric.nifty.ang2bohr
    gradient = engine.calc_new(coords, '.')['gradient']
    hessian = np.loadtxt(os.path.join(datad, 'hcn_tsguess_hessian.txt'))
    gradient_internal = IC.calcGrad(coords, gradient)
    hessian_internal = IC.calcHess(coords, gradient, hessian)
    hessian_backconv = IC.calcHessCart(coords, gradient_internal, hessian_internal)
    np.testing.assert_almost_equal(hessian, hessian_backconv)
