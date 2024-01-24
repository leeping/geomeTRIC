"""
A set of tests for using the OpenMM engine
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

@addons.using_terachem
def test_meci_qmmm_terachem(localizer):
    """
    Find the MECI of a retinal protonated Schiff base model with static QM/MM water molecules.
    """
    pytest.skip("Skipping because TeraChem QM/MM is being updated.")
    for fnm in ['x.prmtop', 'qmindices.txt', 'run.tcin', 'run1.tcin']:
        shutil.copy(os.path.join(exampled, '0-regression-tests', 'psb3-qmmm-meci', fnm), os.getcwd())
    shutil.copy(os.path.join(exampled, '0-regression-tests', 'psb3-qmmm-meci', 'converged.inpcrd'), 
                os.path.join(os.getcwd(), 'x.inpcrd'))
    progress = geometric.optimize.run_optimizer(engine='terachem', input='run.tcin', meci=['run1.tcin'], meci_alpha=0.03)
    e_ref = -251.9957148967
    assert progress.qm_energies[-1] < (e_ref + 0.001)

class TestTerachemWorkQueue:

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
    def test_terachem_workqueue(self, localizer):
        pytest.skip("Skipping because TeraChem QM/MM is being updated.")
        # Copy needed files
        for fnm in ['x.prmtop', 'qmindices.txt', 'run.tcin']:
            shutil.copy(os.path.join(exampled, '0-regression-tests', 'psb3-qmmm-meci', fnm), os.getcwd())
        shutil.copy(os.path.join(exampled, '0-regression-tests', 'psb3-qmmm-meci', 'converged.inpcrd'), 
                    os.path.join(os.getcwd(), 'x.inpcrd'))
        # Create molecule and engine object
        molecule, engine = geometric.prepare.get_molecule_engine(engine='terachem', input='run.tcin')
        coords = molecule.xyzs[0].flatten()*geometric.nifty.ang2bohr
        # Start the WQ master
        geometric.nifty.createWorkQueue(9191, debug=False)
        # Start the WQ worker program
        worker_program = geometric.nifty.which('work_queue_worker')
        self.workers = [subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", "9191"], stdout=subprocess.PIPE)]
        # Submit the calculation to the queue
        engine.calc_wq(coords, 'run.tmp')
        # Wait for the calc to finish
        geometric.nifty.wq_wait(geometric.nifty.getWorkQueue(), print_time=10)
        # Read the result
        result = engine.read_wq(coords, 'run.tmp')['gradient']
        # Destroy Work Queue object
        geometric.nifty.destroyWorkQueue()
        # Compare result
        refgrad = np.array([[ 0.0183380081, -0.0045038141,  0.0007505074],
                            [-0.0165195434,  0.0049289444,  0.0097964202],
                            [-0.0005158430, -0.0005565642, -0.0001583164],
                            [ 0.0000519175,  0.0012185754, -0.0012528714],
                            [ 0.0012842094, -0.0020616916, -0.0005261028],
                            [ 0.0087362456, -0.0026767008,  0.0001638661],
                            [ 0.0129550718,  0.0156600443, -0.0308778074],
                            [-0.0041354590,  0.0009791807, -0.0060947589],
                            [-0.0060263353,  0.0036069996,  0.0035625801],
                            [-0.0438156463, -0.0246476930,  0.0258073558],
                            [ 0.0004496649, -0.0015933102, -0.0011358591],
                            [-0.0042658209, -0.0023538035,  0.0071018519],
                            [-0.0052052918,  0.0010504165, -0.0097522852],
                            [ 0.0404225583,  0.0116268521, -0.0007552128]]).flatten()
        np.testing.assert_almost_equal(result, refgrad, decimal=5)

def test_edit_tcin(localizer):
    read_tcin_1 = geometric.engine.edit_tcin(fin=os.path.join(exampled, '0-regression-tests', 'ivermectin', 'run.tcin'), fout='test.tcin', 
                                             options={'basis':'3-21g', 'threspdp':'1e-5'}, defaults={'method':'rhf', 'threspdp':'1e-4', 'precision':'mixed'})
    with pytest.raises(RuntimeError):
        read_tcin_2 = geometric.engine.edit_tcin(fin='test.tcin')
    read_tcin_2 = geometric.engine.edit_tcin(fin='test.tcin', reqxyz=False)
    assert read_tcin_1['run'] == read_tcin_2['run']
    assert read_tcin_2['basis'] == '3-21g'
    assert read_tcin_2['threspdp'] == '1e-5'
    assert read_tcin_2['method'] == 'rb3lyp'
    assert read_tcin_2['precision'] == 'mixed'
