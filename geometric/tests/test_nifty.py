"""
A set of tests for parsing inputs
"""

import copy
import numpy as np
import json, os, shutil
from . import addons
import geometric
import pytest
import itertools
import subprocess

localizer = addons.in_folder
datad = addons.datad

def test_isint():
    """Check utility functions in geometric.nifty"""
    assert geometric.nifty.isint("1")
    assert not(geometric.nifty.isint("1."))
    assert geometric.nifty.isint("-4")
    assert not(geometric.nifty.isint("-3.14"))

def test_isfloat():
    assert geometric.nifty.isfloat("1.5")
    assert geometric.nifty.isfloat("1")
    assert not(geometric.nifty.isfloat("a"))

def test_isdecimal():
    assert geometric.nifty.isdecimal("1.0")
    assert not(geometric.nifty.isdecimal("1"))

def test_get_least_squares():
    for result in geometric.nifty.get_least_squares(([0]), [0]):
        assert not(result.any())

    ##least squares function tests
    #   trivial fully determined
    X=((1,3,-2),(3,5,6),(2,4,3))
    Y=(5,7,8)
    result = geometric.nifty.get_least_squares(X,Y)[0]
    np.testing.assert_almost_equal(result[0], -15)
    np.testing.assert_almost_equal(result[1], 8)
    np.testing.assert_almost_equal(result[2], 2)

    #   inconsistent system
    X=((1,),(1,))
    Y=(0,1)
    result = geometric.nifty.get_least_squares(X,Y)[0]
    np.testing.assert_almost_equal(result[0], .5)

    #   overdetermined system
    X=((2,0),(-1,1),(0,2))
    Y=(1,0,-1)
    result = geometric.nifty.get_least_squares(X,Y)[0]
    np.testing.assert_almost_equal(result[0], 1./3.)
    np.testing.assert_almost_equal(result[1], -1./3.)

def test_flat_row_col():
    X=((2,0),(-1,1),(0,2))
    ##matrix manipulations
    X=geometric.nifty.flat(X)
    assert X.shape == (6,)
    X=geometric.nifty.row(X)
    assert X.shape == (1,6)
    X=geometric.nifty.col(X)
    assert X.shape == (6,1)

def test_exec(localizer):
    ##_exec
    assert type(geometric.nifty._exec("")) is list
    assert geometric.nifty._exec("echo test")[0] == "test"
    geometric.nifty._exec("touch .test")
    assert os.path.isfile(".test")
    geometric.nifty._exec("rm .test")
    assert not(os.path.isfile(".test"))
    with pytest.raises(Exception) as excinfo:
        geometric.nifty._exec("exit 255")

@addons.using_workqueue
def test_work_queue_functions():
    """Check work_queue functions behave as expected"""
    
    import work_queue

    # Work Queue will no longer be initialized to None
    assert geometric.nifty.WORK_QUEUE is None, "Unexpected initialization of geometric.nifty.WORK_QUEUE to %s" % str(geometric.nifty.WORK_QUEUE)

    geometric.nifty.createWorkQueue(9191, debug=False)
    assert type(geometric.nifty.WORK_QUEUE) is work_queue.WorkQueue, "Expected geometric.nifty.WORK_QUEUE to " \
                                                                     "be a WorkQueue object, but got a %s " \
                                                                     "instead" % str(type(geometric.nifty.WORK_QUEUE))

    wq = geometric.nifty.getWorkQueue()
    assert type(wq) is work_queue.WorkQueue, "Expected getWorkQueue() to return a " \
                                             "WorkQueue object, but got %s instead" % str(type(wq))

    worker_program = geometric.nifty.which('work_queue_worker')
    if worker_program != '':
        geometric.nifty.queue_up(wq, "echo work queue test > test.job", [], ["test.job"], tgt=None, verbose=False)
        assert wq.stats.tasks_waiting == 1, "Expected queue to have a task waiting"
        
        worker = subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", str(wq.port)],
                                  stdout=subprocess.PIPE)

        geometric.nifty.wq_wait1(wq, wait_time=5)
        worker.terminate()
        assert wq.stats.total_tasks_complete == 1, "Expected queue to have a task completed"
    
    # Destroy the Work Queue object so it doesn't interfere with the rest of the tests.
    geometric.nifty.destroyWorkQueue()
