"""
A set of tests for NEB calculations
"""

import os, json, copy, shutil
from . import addons
import pytest
import geometric
import tempfile
import numpy as np
import subprocess

localizer = addons.in_folder
datad = addons.datad
exampled = addons.exampled

def test_hcn_neb_input(localizer, molecule_engine_hcn):
    """
    Test lengths of input chains
    """
    chain_M = geometric.molecule.Molecule(os.path.join(datad, "hcn_neb_input.xyz"))
    nimg = 7

    M1, engine = molecule_engine_hcn('psi4', nimg)

    # The number of images can't exceed the maximum number of images in the input chain
    M2, engine = molecule_engine_hcn('psi4', 9999)

    assert nimg == len(M1)
    assert len(M2) == len(chain_M)


@addons.using_psi4
def test_psi4_hcn_neb_optimize_1(localizer, molecule_engine_hcn):
    """
    Optimize a HCN chain without alignment
    """
    M, engine = molecule_engine_hcn('psi4', 11)

    params = geometric.params.NEBParams(**{"optep": True, "align": False, "verbose": 1})
    chain = geometric.neb.ElasticBand(
        M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
    )

    assert chain.coordtype == "cart"

    final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

    assert optCycle <= 10
    assert final_chain.maxg < params.maxg
    assert final_chain.avgg < params.avgg


@addons.using_psi4
def test_psi4_hcn_neb_optimize_2(localizer, molecule_engine_hcn):
    """
    Optimize a HCN chain with alignment
    """
    M, engine = molecule_engine_hcn('psi4', 11)

    # maxg and avgg are increased here to make them converge faster after the alignment
    params = geometric.params.NEBParams(**{"verbose": 1, "maxg": 3.0, "avgg": 2.0})
    chain = geometric.neb.ElasticBand(
        M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
    )

    assert chain.coordtype == "cart"

    final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

    assert optCycle <= 10
    assert final_chain.maxg < params.maxg
    assert final_chain.avgg < params.avgg


@addons.using_terachem
def test_tera_hcn_neb_optimize(localizer, molecule_engine_hcn):
    """
    NEB with TeraChem
    """
    M, engine = molecule_engine_hcn('tera', 11)

    params = geometric.params.NEBParams(**{"optep": False, "align": False})
    chain = geometric.neb.ElasticBand(
        M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
    )

    assert chain.coordtype == "cart"

    final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

    assert optCycle <= 10
    assert final_chain.maxg < params.maxg
    assert final_chain.avgg < params.avgg


@addons.using_qchem
def test_qchem_hcn_neb_optimize(localizer, molecule_engine_hcn):
    """
    NEB with QChem
    """
    M, engine = molecule_engine_hcn('qchem', 11)

    params = geometric.params.NEBParams(**{"optep": False, "align": False})
    chain = geometric.neb.ElasticBand(
        M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
    )

    assert chain.coordtype == "cart"

    final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

    assert optCycle <= 10
    assert final_chain.maxg < params.maxg
    assert final_chain.avgg < params.avgg


@addons.using_gaussian
def test_qchem_hcn_neb_optimize(localizer, molecule_engine_hcn):
    """
    NEB with Gaussian
    """
    M, engine = molecule_engine_hcn('gaussian', 11)

    params = geometric.params.NEBParams(**{"optep": False, "align": False})
    chain = geometric.neb.ElasticBand(
        M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
    )

    assert chain.coordtype == "cart"

    final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

    assert optCycle <= 10
    assert final_chain.maxg < params.maxg
    assert final_chain.avgg < params.avgg


@addons.using_psi4
@addons.using_bigchem
def test_psi4_bigchem(localizer, molecule_engine_hcn):
    """
    Optimize a HCN chain without alignment with BigChem and Psi4
    """
    M, engine = molecule_engine_hcn('psi4', 11)

    params = geometric.params.NEBParams(**{"align": False, "verbose": 1, "bigchem": True})
    chain = geometric.neb.ElasticBand(
        M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
    )

    assert chain.coordtype == "cart"

    final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

    assert optCycle <= 10
    assert final_chain.maxg < params.maxg
    assert final_chain.avgg < params.avgg


@addons.using_terachem
@addons.using_bigchem
def test_terachem_bigchem(localizer, molecule_engine_hcn):
    """
    Optimize a HCN chain without alignment with BigChem and TeraChem
    """
    M, engine = molecule_engine_hcn('tera', 11)

    params = geometric.params.NEBParams(**{"align": False, "verbose": 1, "bigchem": True})
    chain = geometric.neb.ElasticBand(
        M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
    )

    assert chain.coordtype == "cart"

    final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

    assert optCycle <= 10
    assert final_chain.maxg < params.maxg
    assert final_chain.avgg < params.avgg


@addons.using_qchem
@addons.using_bigchem
def test_qchem_bigchem(localizer, molecule_engine_hcn):
    """
    Optimize a HCN chain without alignment with BigChem and QChem
    """
    M, engine = molecule_engine_hcn('qchem', 11)

    params = geometric.params.NEBParams(**{"align": False, "verbose": 1, "bigchem": True})
    chain = geometric.neb.ElasticBand(
        M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
    )

    assert chain.coordtype == "cart"

    final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

    assert optCycle <= 10
    assert final_chain.maxg < params.maxg
    assert final_chain.avgg < params.avgg


class TestPsi4WorkQueueNEB:

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
    def test_psi4_work_queue_neb(self, localizer, molecule_engine_hcn):
        M, engine = molecule_engine_hcn('psi4', 11)
        params = geometric.params.NEBParams(**{"optep": True, "align": False, "verbose": 1})
        chain = geometric.neb.ElasticBand(
            M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
        )

        geometric.nifty.createWorkQueue(9191, debug=False)
        wq = geometric.nifty.getWorkQueue()
        worker_program = geometric.nifty.which('work_queue_worker')
        # Assume 4 threads are available
        self.workers = [subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", str(wq.port)],
                                         stdout=subprocess.PIPE) for i in range(4)]

        final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

        assert optCycle < 10
        assert final_chain.maxg < params.maxg
        assert final_chain.avgg < params.avgg
        geometric.nifty.destroyWorkQueue()


class TestQChemWorkQueueNEB:

    """ Tests are put into class so that the fixture can terminate the worker process. """

    @pytest.fixture(autouse=True)
    def work_queue_cleanup(self):
        self.workers = None
        yield
        if self.workers is not None:
            for worker in self.workers:
                worker.terminate()

    @addons.using_qchem
    @addons.using_workqueue
    def test_qchem_work_queue_neb(self, localizer, molecule_engine_hcn):
        M, engine = molecule_engine_hcn('qchem', 11)
        params = geometric.params.NEBParams(**{"optep": False, "align": False, "verbose": 1})
        chain = geometric.neb.ElasticBand(
            M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
        )

        geometric.nifty.createWorkQueue(9191, debug=False)
        wq = geometric.nifty.getWorkQueue()
        worker_program = geometric.nifty.which('work_queue_worker')
        # Assume 4 threads are available
        self.workers = [
            subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", str(wq.port)],
                             stdout=subprocess.PIPE) for i in range(4)]

        final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

        assert optCycle < 10
        assert final_chain.maxg < params.maxg
        assert final_chain.avgg < params.avgg
        geometric.nifty.destroyWorkQueue()


class TestTerachemWorkQueueNEB:

    """ Tests are put into class so that the fixture can terminate the worker process. """

    @pytest.fixture(autouse=True)
    def work_queue_cleanup(self):
        self.workers = None
        yield
        if self.workers is not None:
            for worker in self.workers:
                worker.terminate()

    @addons.using_terachem
    @addons.using_workqueue
    def test_terachem_work_queue_neb(self, localizer, molecule_engine_hcn):
        M, engine = molecule_engine_hcn('tera', 11)
        params = geometric.params.NEBParams(**{"optep": False, "align": False, "verbose": 1})
        chain = geometric.neb.ElasticBand(
            M, engine=engine, tmpdir=tempfile.mkdtemp(), params=params, plain=0
        )

        geometric.nifty.createWorkQueue(9191, debug=False)
        wq = geometric.nifty.getWorkQueue()
        worker_program = geometric.nifty.which('work_queue_worker')
        # Assume 4 threads are available
        self.workers = [
            subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", str(wq.port)],
                             stdout=subprocess.PIPE) for i in range(4)]

        final_chain, optCycle = geometric.neb.OptimizeChain(chain, engine, params)

        assert optCycle < 10
        assert final_chain.maxg < params.maxg
        assert final_chain.avgg < params.avgg
        geometric.nifty.destroyWorkQueue()


@addons.using_qcelemental
def test_hcn_neb_service_arrange(localizer):
    """
    Testing neb.arrange() function for QCFractal NEB service
    """
    from qcelemental.models import Molecule as qcmol

    chain_M = geometric.molecule.Molecule(os.path.join(datad, "hcn_neb_service.xyz"))
    coords = [M.xyzs/geometric.nifty.bohr2ang for M in chain_M]
    qcel_mols = [
        qcmol(
            symbols=chain_M[0].elem,
            geometry=coord,
            molecular_charge=0,
            molecular_multiplicity=1,
        )
        for coord in coords
    ]

    new_qcel_mols = geometric.qcf_neb.arrange(qcel_mols, True)
    count = sum(
        [
            1 if not np.allclose(i.geometry, j.geometry) else 0
            for i, j in zip(qcel_mols, new_qcel_mols)
        ]
    )

    # 5 images should change as the result of the respacing.
    assert count == 5


def test_hcn_neb_service_normal(localizer):
    """
    Testing neb.prepare() and neb.nextchain() function for QCFractal NEB service
    """

    # 1) neb.prepare()
    with open(os.path.join(datad, "prepare_json_in.json")) as prepare_in:
        in_dict = json.load(prepare_in)

    input_dict = copy.deepcopy(in_dict)
    new_coords, out_dict = geometric.qcf_neb.prepare(input_dict)
    new_coords_ang = np.array(new_coords) * geometric.nifty.bohr2ang
    old_coords_ang = out_dict["coord_ang_prev"]

    # All 11 images should be identical.
    count = sum(
        [1 if np.allclose(i, j) else 0 for i, j in zip(old_coords_ang, new_coords_ang)]
    )
    assert 11 == count

    # After prepare(), there should be just 1 Ys, GWs, and GPs
    assert 1 == len(out_dict["Ys"])
    assert 1 == len(out_dict["GWs"])
    assert 1 == len(out_dict["GPs"])

    # Input gradients and previous result gradients should be identical
    input_grad = in_dict["gradients"]

    for i in range(len(input_grad)):
        assert np.allclose(input_grad[i], out_dict["result_prev"][i]["gradient"])

    # 2-1) neb.nextchain() usual case test
    with open(os.path.join(datad, "nextchain_json_in.json")) as prepare_in:
        in_dict = json.load(prepare_in)

    input_dict = copy.deepcopy(in_dict)

    # testing the Iteration = 2 case
    input_dict["params"]["iteration"] = 2

    new_coords, out_dict_1 = geometric.qcf_neb.nextchain(input_dict)

    new_coords_ang = np.array(new_coords) * geometric.nifty.bohr2ang
    old_coords_ang = out_dict_1["coord_ang_prev"]

    # Only two images should be the same (first and last).
    count = sum(
        [1 if np.allclose(i, j) else 0 for i, j in zip(old_coords_ang, new_coords_ang)]
    )
    assert 2 == count
    assert np.allclose(new_coords_ang[0], old_coords_ang[0])
    assert np.allclose(new_coords_ang[-1], old_coords_ang[-1])

    # Output dictionary should have still have just one Ys, GWs, and GPs
    assert 1 == len(out_dict["Ys"])
    assert 1 == len(out_dict["GWs"])
    assert 1 == len(out_dict["GPs"])

    # testing the Iteration = 3 case (normal iteration)
    input_dict = copy.deepcopy(in_dict)
    new_coords, out_dict = geometric.qcf_neb.nextchain(input_dict)

    new_coords_ang = np.array(new_coords) * geometric.nifty.bohr2ang
    old_coords_ang = out_dict["coord_ang_prev"]

    # Only two images should be the same (first and last).
    count = sum(
        [1 if np.allclose(i, j) else 0 for i, j in zip(old_coords_ang, new_coords_ang)]
    )
    assert 2 == count
    assert np.allclose(new_coords_ang[0], old_coords_ang[0])
    assert np.allclose(new_coords_ang[-1], old_coords_ang[-1])

    # Output dictionary should have one more Ys, GWs, and GPs than input
    assert len(in_dict["Ys"]) + 1 == len(out_dict["Ys"])
    assert len(in_dict["GWs"]) + 1 == len(out_dict["GWs"])
    assert len(in_dict["GPs"]) + 1 == len(out_dict["GPs"])

    # geometry needs to be emptied.
    assert len(out_dict["geometry"]) == 0

    # Input gradients and previous result gradients should be identical
    input_grad = in_dict["gradients"]

    for i in range(len(input_grad)):
        assert np.allclose(input_grad[i], out_dict["result_prev"][i]["gradient"])

    # Output should not have any numpy array
    for k, v in out_dict.items():
        if isinstance(v, list):
            for i in v:
                assert not isinstance(i, np.ndarray)

    # 2-2) neb.nextchain() respaced case test
    input_dict = copy.deepcopy(in_dict)
    input_dict["respaced"] = True
    new_coords, out_dict = geometric.qcf_neb.nextchain(input_dict)

    new_coords_ang = np.array(new_coords) * geometric.nifty.bohr2ang
    old_coords_ang = out_dict["coord_ang_prev"]

    # Only two images should be the same (first and last).
    count = sum(
        [1 if np.allclose(i, j) else 0 for i, j in zip(old_coords_ang, new_coords_ang)]
    )
    assert 2 == count
    assert np.allclose(new_coords_ang[0], old_coords_ang[0])
    assert np.allclose(new_coords_ang[-1], old_coords_ang[-1])

    # Output dictionary should have just one Ys, GWs, and GPs
    assert 1 == len(out_dict["Ys"])
    assert 1 == len(out_dict["GWs"])
    assert 1 == len(out_dict["GPs"])

    # geometry needs to be emptied.
    assert len(out_dict["geometry"]) == 0

    # Output should not have any numpy array
    for k, v in out_dict.items():
        if isinstance(v, list):
            for i in v:
                assert not isinstance(i, np.ndarray)


def test_hcn_neb_service_special(localizer):
    """
    Testing neb.nextchain() bad quality step
    """

    # 1) Bad quality step
    with open(os.path.join(datad, "nextchain_json_in.json")) as prepare_in:
        in_dict = json.load(prepare_in)

    input_dict = copy.deepcopy(in_dict)

    # Sabotaging the step
    input_dict["attrs_prev"]["TotBandEnergy"] = (
        float(input_dict["attrs_prev"]["TotBandEnergy"]) - 1.0
    )
    input_dict["gradients"] = np.array(input_dict["gradients"]) * 3

    new_coords, out_dict = geometric.qcf_neb.nextchain(input_dict)

    # Trust radius should be decreased
    assert in_dict["trust"] > out_dict["trust"]

    # Number of stores IC and gradients shouldn't change
    assert len(in_dict["Ys"]) == len(out_dict["Ys"])
    assert len(in_dict["GWs"]) == len(out_dict["GWs"])
    assert len(in_dict["GPs"]) == len(out_dict["GPs"])

    # 2) Triggering recover
    input_dict = copy.deepcopy(in_dict)

    # If the smallest eigenvalue is smaller than epsilon, it will try to recover.
    input_dict["params"]["epsilon"] = 100

    new_coords, out_dict = geometric.qcf_neb.nextchain(input_dict)

    new_coords_ang = np.array(new_coords) * geometric.nifty.bohr2ang
    old_coords_ang = in_dict["coord_ang_prev"]

    prev_results = input_dict["result_prev"]
    new_results = out_dict["result_prev"]

    # Gradients and energies should stay the same
    assert len(prev_results) == len(new_results)
    for i in range(len(prev_results)):
        assert np.allclose(prev_results[i]["gradient"], new_results[i]["gradient"])
        assert np.isclose(prev_results[i]["energy"], new_results[i]["energy"])

    # Only two images should be the same (first and last). It takes a step after the recover.
    count = sum(
        [1 if np.allclose(i, j) else 0 for i, j in zip(old_coords_ang, new_coords_ang)]
    )
    assert 2 == count
    assert np.allclose(new_coords_ang[0], old_coords_ang[0])
    assert np.allclose(new_coords_ang[-1], old_coords_ang[-1])

    # There should be only one IC and gradients after the recover.
    assert 1 == len(out_dict["Ys"])
    assert 1 == len(out_dict["GWs"])
    assert 1 == len(out_dict["GPs"])
