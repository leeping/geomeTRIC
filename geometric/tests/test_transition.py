"""
A set of tests for transition state optimization
"""

import numpy as np
import os, shutil
from . import addons
import geometric
import pytest
import time
import subprocess

localizer = addons.in_folder
datad = addons.datad
exampled = addons.exampled

@addons.using_psi4
def test_transition_hcn_psi4(localizer):
    """
    Optimize the transition state of the HCN <-> HNC isomerization.
    """
    shutil.copy2(os.path.join(datad, 'hcn_tsguess.psi4in'), os.path.join(os.getcwd(), 'hcn.psi4in'))
    progress = geometric.optimize.run_optimizer(engine='psi4', transition=True, input='hcn.psi4in',
                                                converge=['gmax', '1.0e-5'], trust=0.1, tmax=0.3, hessian='first+last')
    # The results here are in Angstrom
    #
    molecule = geometric.molecule.Molecule(os.path.join(datad, 'hcn_ts_optim.xyz'))
    ref = molecule.xyzs[0]
    e_ref = -92.2460196061
    rmsd, maxd = geometric.optimize.calc_drms_dmax(progress.xyzs[-1], ref, align=True)
    # Check that the energy is 0.0001 a.u. above reference.
    assert progress.qm_energies[-1] < (e_ref + 0.0001)
    # Check that the optimization converged in less than 30 steps
    # LPW 2022-10-04: Increased from 20 to 30 because occasionally see 21 steps
    assert len(progress) < 30
    # Check that the geometry matches the reference to within 0.01 RMS 0.02 max displacement
    assert rmsd < 0.001
    assert maxd < 0.002
    coords = progress.xyzs[-1].flatten()*geometric.nifty.ang2bohr
    t0 = time.time()
    hessian = geometric.normal_modes.calc_cartesian_hessian(coords, progress[-1], None, 'hcn.tmp')
    # The latest Hessian should have been read in from file.
    assert (time.time() - t0) < 1
    # Todo: Add tests for correctness of Wigner sampling
    freqs, modes, G = geometric.normal_modes.frequency_analysis(coords, hessian, elem=progress.elem, energy=progress.qm_energies[-1], wigner=(-10, 'hcn.wigner'))
    np.testing.assert_almost_equal(G, -92.25677301, decimal=5)
    np.testing.assert_almost_equal(freqs[0]/10, -121.5855, decimal=0)

@addons.using_terachem
def test_transition_hcn_terachem(localizer):
    """
    Optimize the transition state of the HCN <-> HNC isomerization.
    """
    shutil.copy2(os.path.join(exampled, '0-regression-tests', 'hcn-hnc-ts', 'start.xyz'), os.getcwd())
    # shutil.copy2(os.path.join(exampled, '0-regression-tests', 'hcn-hnc-ts', 'run.tcin'), os.getcwd())
    geometric.engine.edit_tcin(fin=os.path.join(exampled, '0-regression-tests', 'hcn-hnc-ts', 'run.tcin'), fout='run.tcin', options={'guess':'c0'})
    shutil.copy2(os.path.join(datad, 'hcn_tsguess.c0'), os.path.join(os.getcwd(), 'c0'))
    shutil.copy2(os.path.join(datad, 'hcn_tsguess_hessian.txt'), os.getcwd())
    progress = geometric.optimize.run_optimizer(engine='terachem', transition=True, input='run.tcin',
                                                converge=['gmax', '1.0e-5'], trust=0.1, tmax=0.3, hessian='file:hcn_tsguess_hessian.txt')
    # The results here are in Angstrom
    #
    molecule = geometric.molecule.Molecule(os.path.join(datad, 'hcn_ts_optim.xyz'))
    ref = molecule.xyzs[0]
    e_ref = -92.2460196061
    rmsd, maxd = geometric.optimize.calc_drms_dmax(progress.xyzs[-1], ref, align=True)
    # Check that the energy is 0.0001 a.u. above reference.
    assert progress.qm_energies[-1] < (e_ref + 0.0001)
    # Check that the optimization converged in less than 20 steps
    assert len(progress) < 20
    # Check that the geometry matches the reference to within 0.01 RMS 0.02 max displacement
    assert rmsd < 0.001
    assert maxd < 0.002

class TestTransitionQchemWorkQueue:

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
    def test_transition_qchem_workqueue(self, localizer):
        import work_queue

        shutil.copy2(os.path.join(datad, 'propynimine-tsguess.qcin'), os.path.join(os.getcwd(), 'run.qcin'))
        shutil.copy2(os.path.join(datad, 'propynimine-tsguess-hessian.txt'), os.path.join(os.getcwd(), 'hessian.txt'))

        worker_program = geometric.nifty.which('work_queue_worker')
        # Assume 4 threads are available
        self.workers = [subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", "9191"],
                                         stdout=subprocess.PIPE) for i in range(4)]

        progress = geometric.optimize.run_optimizer(engine='qchem', port=9191, transition=True, input='run.qcin',
                                                    converge=['gmax', '1.0e-5'], trust=0.1, tmax=0.3, hessian='file+last:hessian.txt')

        M_ref = geometric.molecule.Molecule(os.path.join(datad, 'propynimine-ts-optimized.xyz'))

        # Check that the optimization converged in less than 10 steps
        assert len(progress) < 10
        # Check that the geometry matches the reference to within 0.01 RMS 0.02 max displacement
        rmsd, maxd = geometric.optimize.calc_drms_dmax(progress.xyzs[-1], M_ref.xyzs[0], align=True)
        assert rmsd < 0.001
        assert maxd < 0.002
        # Check the optimized energy
        np.testing.assert_almost_equal(progress.qm_energies[-1], -170.6848416207, decimal=5)


@addons.using_ase
@addons.using_xtb
def test_transition_hcn_ase_xtb(localizer):
    """
    Optimize the transition state of the HCN <-> HNC isomerization.

    A new guess structure was written for XTB, which is closer to the
    transition state. Apparently the one given for the other tests
    leads to the linear molecule and fails on a singularity of the
    coordinates with XTB.
    """
    shutil.copy2(
        os.path.join(datad, "hcn_tsguess.xyz"),
        os.path.join(os.getcwd(), "hcn.xyz"),
    )
    progress = geometric.optimize.run_optimizer(
        engine="ase",
        ase_class="xtb.ase.calculator.XTB",
        ase_kwargs='{"method": "GFN2-xTB"}',
        transition=True,
        input="hcn.xyz",
        converge=["gmax", "1.0e-5"],
        trust=0.1,
        tmax=0.3,
        hessian="first+last",
    )

    # ref results from running the same from the cli
    e_ref = -5.38737353  # elec. energy
    g_ref = -5.39851798  # gibbs free energy
    freq0_ref = -1425.942  # first imag. frequency

    # The results here are in Angstrom
    #
    molecule = geometric.molecule.Molecule(os.path.join(datad, "hcn_ts_optim_xtb.xyz"))
    ref = molecule.xyzs[0]
    rmsd, maxd = geometric.optimize.calc_drms_dmax(progress.xyzs[-1], ref, align=True)
    # Check that the energy is 0.0001 a.u. above reference.
    assert progress.qm_energies[-1] < (e_ref + 0.0001)
    # Check that the optimization converged in less than 20 steps
    assert len(progress) < 10
    # Check that the geometry matches the reference to within 0.01 RMS 0.02 max displacement
    assert rmsd < 0.001
    assert maxd < 0.002
    coords = progress.xyzs[-1].flatten() * geometric.nifty.ang2bohr
    t0 = time.time()
    hessian = geometric.normal_modes.calc_cartesian_hessian(
        coords, progress[-1], None, "hcn.tmp"
    )
    # The latest Hessian should have been read in from file.
    assert (time.time() - t0) < 1
    freqs, modes, G = geometric.normal_modes.frequency_analysis(
        coords,
        hessian,
        elem=progress.elem,
        energy=progress.qm_energies[-1],
        wigner=(-10, "hcn.wigner"),
    )
    np.testing.assert_almost_equal(G, g_ref, decimal=5)
    np.testing.assert_almost_equal(freqs[0], freq0_ref, decimal=0)
