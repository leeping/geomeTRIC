"""
A set of tests for using the CFOUR engine
"""

import copy
import numpy as np
import json, os, shutil
from . import addons
import geometric
import subprocess
import pytest
import itertools

localizer = addons.in_folder
datad = addons.datad
exampled = addons.exampled

@addons.using_cfour
def test_water1_cfour(localizer):
    """
    Optimize a single water molecule.
    """
    for fnm in ['ZMAT']:
        shutil.copy(os.path.join(exampled, '1-simple-examples', 'water1_cfour', fnm), os.getcwd())
    progress = geometric.optimize.run_optimizer(engine='cfour', input='ZMAT', converge=['grms', '1e-10'])
    e_ref = -75.9613385111
    assert progress.qm_energies[-1] < (e_ref + 1e-6)


class TestCFOURWorkQueue:

    """ Tests are put into class so that the fixture can terminate the worker process. """

    @pytest.fixture(autouse=True)
    def work_queue_cleanup(self):
        self.workers = None
        yield
        if self.workers is not None:
            for worker in self.workers:
                worker.terminate()

    @addons.using_cfour
    @addons.using_workqueue
    def test_cfour_workqueue(self, localizer):
        """
        Compute the gradient using Work Queue.
        """
        for fnm in ['ZMAT']:
            shutil.copy(os.path.join(exampled, '1-simple-examples', 'water1_cfour', fnm), os.getcwd())
        molecule, engine = geometric.prepare.get_molecule_engine(engine='cfour', input='ZMAT')
        coords = molecule.xyzs[0].flatten()*geometric.nifty.ang2bohr
        # Start the WQ master
        geometric.nifty.createWorkQueue(9191, debug=False)
        # Start the WQ worker program
        worker_program = geometric.nifty.which('work_queue_worker')
        # Assume 1 thread is available
        self.workers = [subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", str(9191)],
                                         stdout=subprocess.PIPE) for i in range(1)]
        # Submit the calculation to the queue
        engine.calc_wq(coords, 'run.tmp')
        # Wait for the calc to finish
        geometric.nifty.wq_wait(geometric.nifty.getWorkQueue(), print_time=10)
        # Read the result
        result = engine.read_wq(coords, 'run.tmp')['gradient']
        # Destroy Work Queue object
        geometric.nifty.destroyWorkQueue()
        # Compare result
        refgrad = np.array([[-0.000000000000000, -0.000000000000001,  0.018277635721552],
                            [-0.000000000000000, -0.011301923095225, -0.009138817860774],
                            [ 0.000000000000000,  0.011301923095228, -0.009138817860776]]).flatten()

        np.testing.assert_almost_equal(result, refgrad, decimal=8)

