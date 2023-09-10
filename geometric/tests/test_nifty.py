"""
A set of tests for nifty utility functions
"""

import copy
import numpy as np
import json, os, shutil
from . import addons
from geometric import nifty
import pytest
import filecmp
import itertools
import subprocess

localizer = addons.in_folder
datad = addons.datad

def test_isint():
    assert nifty.isint("1")
    assert not(nifty.isint("1."))
    assert nifty.isint("-4")
    assert not(nifty.isint("-3.14"))

def test_isfloat():
    assert nifty.isfloat("1.5")
    assert nifty.isfloat("1")
    assert not(nifty.isfloat("a"))

def test_isdecimal():
    assert nifty.isdecimal("1.0")
    assert not(nifty.isdecimal("1"))

def test_grouper():
    assert nifty.grouper('ABCDeFG', 3) == [['A', 'B', 'C'], ['D', 'e', 'F'], ['G']]

def test_get_least_squares():
    for result in nifty.get_least_squares(([0]), [0]):
        assert not(result.any())

    ##least squares function tests
    #   trivial fully determined
    X=((1,3,-2),(3,5,6),(2,4,3))
    Y=(5,7,8)
    result = nifty.get_least_squares(X,Y,w=[1, 1, 1])[0]
    np.testing.assert_almost_equal(result[0], -15)
    np.testing.assert_almost_equal(result[1], 8)
    np.testing.assert_almost_equal(result[2], 2)

    #   inconsistent system
    X=((1,),(1,))
    Y=(0,1)
    result = nifty.get_least_squares(X,Y)[0]
    np.testing.assert_almost_equal(result[0], .5)

    #   overdetermined system
    X=((2,0),(-1,1),(0,2))
    Y=(1,0,-1)
    result = nifty.get_least_squares(X,Y)[0]
    np.testing.assert_almost_equal(result[0], 1./3.)
    np.testing.assert_almost_equal(result[1], -1./3.)

def test_flat_row_col():
    X=((2,0),(-1,1),(0,2))
    ##matrix manipulations
    X=nifty.flat(X)
    assert X.shape == (6,)
    X=nifty.row(X)
    assert X.shape == (1,6)
    X=nifty.col(X)
    assert X.shape == (6,1)

def test_exec(localizer):
    ##_exec
    assert type(nifty._exec("")) is list
    assert nifty._exec("echo test")[0] == "test"
    nifty._exec("touch .test")
    assert os.path.isfile(".test")
    nifty._exec("rm .test")
    assert not(os.path.isfile(".test"))
    with pytest.raises(Exception) as excinfo:
        nifty._exec("exit 255")
    nifty._exec("echo error 1>&2; echo output; echo error2 1>&2 ; echo output2", cwd='.', print_to_screen=True, outfnm='test.out', logfnm='test.log', copy_stdout=True, copy_stderr=True)

class TestWorkQueue:

    """ Tests are put into class so that the fixture can terminate the worker process. """

    @pytest.fixture(autouse=True)
    def work_queue_cleanup(self):
        self.worker = None
        yield
        if self.worker is not None:
            self.worker.terminate()

    @addons.using_workqueue
    def test_work_queue_functions(self, localizer):
        """Check work_queue functions behave as expected"""
        try:
            import ndcctools.work_queue as work_queue
        except ImportError:
            import work_queue

        # Work Queue will no longer be initialized to None
        assert nifty.WORK_QUEUE is None, "Unexpected initialization of nifty.WORK_QUEUE to %s" % str(nifty.WORK_QUEUE)
        nifty.createWorkQueue(9191, debug=False)
        assert type(nifty.WORK_QUEUE) is work_queue.WorkQueue, "Expected nifty.WORK_QUEUE to " \
                                                                         "be a WorkQueue object, but got a %s " \
                                                                         "instead" % str(type(nifty.WORK_QUEUE))
        wq = nifty.getWorkQueue()
        assert type(wq) is work_queue.WorkQueue, "Expected getWorkQueue() to return a " \
                                                 "WorkQueue object, but got %s instead" % str(type(wq))
        worker_program = nifty.which('work_queue_worker')
        nifty.queue_up(wq, "echo work queue test > test.job", [], ["test.job"], tgt=None, verbose=False)
        assert wq.stats.tasks_waiting == 1, "Expected queue to have a task waiting"
        self.worker = subprocess.Popen([os.path.join(worker_program, "work_queue_worker"), "localhost", str(wq.port)],
                                  stdout=subprocess.PIPE)
        nifty.wq_wait1(wq, wait_time=5)
        assert wq.stats.total_tasks_complete == 1, "Expected queue to have a task completed"
        # 2022-09-09: The following test causes a problem in Work Queue, disabling until fixed.
        nifty.queue_up(wq, "sleep 1; exit 1", [], ['no_exist'], tgt=None, verbose=False)
        nifty.wq_wait1(wq, wait_time=5)
        assert wq.stats.tasks_submitted == 1 + wq.stats.total_tasks_complete
            
        # Destroy the Work Queue object so it doesn't interfere with the rest of the tests.
        nifty.destroyWorkQueue()

def test_commadash():
    a = [8, 1, 12, 18, 13, 5, 3, 0, 9, 2, 16, 14, 6, 4, 7, 10]
    assert nifty.commadash(a) == '1-11,13-15,17,19'
    assert nifty.uncommadash('1-11,13-15,17,19') == sorted(a)

def test_natural_sort():
    testlist = ['fs','gs0','gs1','gs10','gs10_ts','gs11','gs11_ts','gs12','gs12_ts','gs13','gs13_ts','gs14','gs14_ts',
                'gs15','gs15_ts','gs16','gs16_ts','gs17','gs17_ts','gs18','gs18_ts','gs19','gs19_ts','gs1_ts','gs2',
                'gs20','gs20_ts','gs21','gs21_ts','gs22','gs22_ts','gs23','gs23_ts','gs24','gs2_ts','gs3','gs3_ts',
                'gs4','gs4_ts','gs5','gs5_ts','gs6','gs6_ts','gs7','gs7_ts','gs8','gs9','reaction_002_split0_dc.xyz','reaction_002_split0.xyz']
    randomized = ['gs15', 'gs8', 'fs', 'gs23', 'gs5_ts', 'gs14', 'gs2', 'gs16_ts', 'gs19', 'gs4', 'gs6_ts', 'gs21', 'gs23_ts', 'gs7_ts', 
                  'gs2_ts', 'gs22', 'gs3', 'gs19_ts', 'reaction_002_split0_dc.xyz', 'gs22_ts', 'gs13', 'gs4_ts', 'gs9', 'gs10', 'gs12', 'gs18_ts',
                  'gs7', 'gs11_ts', 'gs17', 'gs24', 'gs21_ts', 'gs13_ts', 'gs12_ts', 'gs1_ts', 'gs10_ts', 'reaction_002_split0.xyz', 'gs14_ts', 
                  'gs5', 'gs20', 'gs18', 'gs17_ts', 'gs15_ts', 'gs20_ts', 'gs1', 'gs16', 'gs3_ts', 'gs11', 'gs0', 'gs6']
    result = ['fs', 'gs0', 'gs1', 'gs1_ts', 'gs2', 'gs2_ts', 'gs3', 'gs3_ts', 'gs4', 'gs4_ts', 'gs5', 'gs5_ts', 'gs6', 'gs6_ts', 
              'gs7', 'gs7_ts', 'gs8', 'gs9', 'gs10', 'gs10_ts', 'gs11', 'gs11_ts', 'gs12', 'gs12_ts', 'gs13', 'gs13_ts', 'gs14', 
              'gs14_ts', 'gs15', 'gs15_ts', 'gs16', 'gs16_ts', 'gs17', 'gs17_ts', 'gs18', 'gs18_ts', 'gs19', 'gs19_ts', 'gs20', 
              'gs20_ts', 'gs21', 'gs21_ts', 'gs22', 'gs22_ts', 'gs23', 'gs23_ts', 'gs24', 'reaction_002_split0.xyz', 'reaction_002_split0_dc.xyz']
    assert nifty.natural_sort(testlist) == result
    assert nifty.natural_sort(randomized) == result

def test_splitall():
    assert nifty.splitall('.') == ['.']
    assert nifty.splitall('/home/leeping/geomeTRIC') == ['/', 'home', 'leeping', 'geomeTRIC']
    assert nifty.splitall('leeping/geomeTRIC') == ['leeping', 'geomeTRIC']
    assert nifty.splitall('home/leeping/geomeTRIC') == ['home', 'leeping', 'geomeTRIC']
    assert nifty.splitall('./home/leeping/geomeTRIC') == ['.', 'home', 'leeping', 'geomeTRIC']

def test_printcool():
    assert nifty.printcool('geomeTRIC') == '----------------------------------------------------------\n'

def test_floatornan():
    assert nifty.floatornan("356") == 356.0
    assert nifty.floatornan("NaN") == 1e100
    assert nifty.floatornan("inf") == 1e100
    with pytest.raises(ValueError) as excinfo:
        nifty.floatornan("Mao")

def test_est124():
    assert nifty.est124(1.5) == 2.0
    assert nifty.est124(1.2) == 1.0
    assert nifty.est124(1.0) == 1.0
    assert nifty.est124(0.1) == 0.1
    assert nifty.est124(0.12) == 0.1
    assert nifty.est124(0.15) == 0.2
    assert nifty.est124(3) == 4.0
    assert nifty.est124(7) == 10.0
    assert nifty.est124(6) == 4.0

def test_monotonic_decreasing():
    energies = np.loadtxt(os.path.join(datad, 'bucky-energies.txt'))
    sorted_argmins = sorted(list(set([int(l.strip()) for l in open(os.path.join(datad, 'bucky-argmin-indices.txt')).readlines()])))
    assert list(nifty.monotonic_decreasing(energies)) == sorted_argmins
    assert list(nifty.monotonic_decreasing(energies[::-1], start=len(energies) - 1, end = 0)) == [len(energies)-i-1 for i in sorted_argmins]

def test_lp_dump_load(localizer):
    # This isn't really used in geomeTRIC but might as well test it here.
    test_dict = {'X': 0.18417850987695983,
                 'G': np.array([-0.1046131 , -0.09901273, -0.21102632,  0.07767315,  0.14657123,
                               -0.05605338, -0.00137182]),
                 'H': np.array([[ 2.06315884e+00, -2.57250804e-02, -6.29836909e-02,
                                  -8.82566192e-02, -6.08247392e-01, -5.84304473e-01,
                                  5.45802740e-04],
                                [-2.57250804e-02,  2.18584558e-01,  5.20280726e-01,
                                 -1.47406932e-01, -1.47031895e-01,  1.43416651e-01,
                                 8.99890458e-04],
                                [-6.29836909e-02,  5.20280726e-01,  1.30714108e+00,
                                 -3.98205308e-01, -3.90578323e-01,  3.85553976e-01,
                                 -1.19773349e-03],
                                [-8.82566192e-02, -1.47406932e-01, -3.98205308e-01,
                                 2.92590490e-01,  1.78075298e-01, -1.29469940e-01,
                                 5.99861100e-03],
                                [-6.08247392e-01, -1.47031895e-01, -3.90578323e-01,
                                 1.78075298e-01,  8.44482446e-01,  2.19087650e-01,
                                 1.30076858e-03],
                                [-5.84304473e-01,  1.43416651e-01,  3.85553976e-01,
                                 -1.29469940e-01,  2.19087650e-01,  7.65388001e-01,
                                 -1.71343364e-03],
                                [ 5.45802740e-04,  8.99890458e-04, -1.19773349e-03,
                                  5.99861100e-03,  1.30076858e-03, -1.71343364e-03,
                                  4.05421648e-04]])}
    nifty.lp_dump(test_dict, 'test_dict.p')
    test_dict1 = nifty.lp_load('test_dict.p')
    test_dict2 = nifty.lp_load(os.path.join(datad, 'test_dict.p'))
    for key in ['X', 'G', 'H']:
        np.testing.assert_almost_equal(test_dict1[key], test_dict2[key])

def test_statistical_inefficiency():
    a = np.loadtxt(os.path.join(datad, 'randn_1k.txt'))
    assert nifty.statisticalInefficiency(a) < 1.1
    assert nifty.statisticalInefficiency(np.repeat(a, 2)) > 2

def test_listfiles(localizer):
    # Confirm 'listfiles' raises error when a file doesn't exist in the list
    with pytest.raises(RuntimeError) as excinfo:
        nifty.listfiles(['ethanol.pdb', 'no_exist.file'], dnm=datad)
    # Also confirm if a string corresponding to a nonexistent file raises the error
    with pytest.raises(RuntimeError) as excinfo:
        nifty.listfiles('no_exist.file', dnm=datad)
    # Check file listing returns expected list of file names (in geometric/data)
    assert nifty.listfiles('ethanol.pdb', dnm=datad) == ['ethanol.pdb']
    assert nifty.listfiles(['ethanol.pdb', 'naphthax.xyz'], dnm=datad) == ['ethanol.pdb', 'naphthax.xyz']
    # Check that all listed files end with the expected extension
    for fnm in nifty.listfiles(ext='pdb', dnm=datad):
        assert fnm.endswith('.pdb')
    # Using absolute paths, check that file is copied over
    assert not os.path.exists('ethanol.pdb')
    nifty.listfiles(os.path.join(datad, 'ethanol.pdb'))
    assert os.path.exists('ethanol.pdb')
    # Nothing happens if identical file exists
    nifty.listfiles(os.path.join(datad, 'ethanol.pdb'))
    # Error is raised if files differ
    with open("ethanol.pdb", 'a') as f:
        f.write("extra-line\n")
    with pytest.raises(RuntimeError) as excinfo:
        nifty.listfiles(os.path.join(datad, 'ethanol.pdb'))
    # Also back up ethanol.pdb
    nifty.bak('ethanol.pdb', cwd='.')
    with open("ethanol.pdb", 'a') as f:
        f.write("extra-line\n")
    nifty.bak('ethanol.pdb', cwd='.')
        
def test_onefile(localizer):
    # Confirm error is raised if neither file name or path are provided
    with pytest.raises(RuntimeError) as excinfo:
        nifty.onefile(err=True)
    # Confirm that file is copied over
    assert not os.path.exists('ethanol.pdb')
    nifty.onefile(os.path.join(datad, "ethanol.pdb"))
    assert os.path.exists('ethanol.pdb')
    nifty.onefile(os.path.join(datad, "ethanol.pdb"))
    # Error is raised if file in abspath differs from local
    with open("ethanol.pdb", 'a') as f:
        f.write("extra-line\n")
    with pytest.raises(RuntimeError) as excinfo:
        nifty.onefile(os.path.join(datad, "ethanol.pdb"))
    # Expected behavior - return one existing file matching pattern
    assert nifty.onefile(ext='pdb') == "ethanol.pdb"
    # Return None if more than one file exists
    with open("other.pdb", 'w') as f:
        f.write("extra-line\n")
    assert nifty.onefile(ext='pdb') is None
    
def test_print_array():
    nifty.pvec1d(np.arange(5))
    nifty.pmat2d(np.arange(25).reshape(-1, 5))
    with pytest.raises(TypeError):
        nifty.pvec1d(np.arange(25).reshape(-1, 5))
    with pytest.raises(IndexError):
        nifty.pmat2d(np.arange(5))

def test_tar_extract(localizer):
    shutil.copy(os.path.join(datad, "test_compress.tar.gz"), os.getcwd())
    nifty.extract_tar("test_compress.tar.gz", ['ala_constraints.txt', 'parser_options.txt'])
    assert os.path.exists('ala_constraints.txt')
    assert os.path.exists('parser_options.txt')
    assert not os.path.exists('linalg_torsion_constraints.txt')

def test_gointo_leave_folder(localizer):
    os.makedirs("test_folder")
    nifty.GoInto("test_folder")
    assert os.getcwd().endswith("test_gointo_leave_folder/test_folder")
    nifty.Leave("test_folder")
    os.rmdir("test_folder")
    nifty.GoInto("test_folder")
    assert os.getcwd().endswith("test_gointo_leave_folder/test_folder")
    nifty.Leave("test_folder")
    os.rmdir("test_folder")

def test_missingfile_inspection():
    assert 'Make sure to install GROMACS' in nifty.MissingFileInspection('mdrun')

def test_wopen_link_copy(localizer):
    # Copy ethanol.pdb into current folder
    nifty.onefile(os.path.join(datad, "ethanol.pdb"))
    # Make a broken symlink
    os.symlink("no_exist.pdb", "ethanol_lnk.pdb")
    # The function should replace the broken symlink with a working one
    nifty.LinkFile("ethanol.pdb", "ethanol_lnk.pdb")
    # Raise an error if attempting to make a broken symlink
    with pytest.raises(RuntimeError):
        nifty.LinkFile("no_exist.pdb", "test_link.pdb")
    # Test that wopen can remove a symlink and replace it with a file
    with nifty.wopen("ethanol_lnk.pdb") as f:
        f.write("test content\n")
    assert "test content" in open("ethanol_lnk.pdb").readlines()[0]
    os.remove("ethanol_lnk.pdb")
    # Confirm that an error is raised if trying to copy to a symlink
    nifty.LinkFile("ethanol.pdb", "ethanol_lnk.pdb")
    with pytest.raises(RuntimeError):
        nifty.CopyFile("ethanol.pdb", "ethanol_lnk.pdb")
    os.remove("ethanol_lnk.pdb")
    # Confirm that CopyFile works as intended
    nifty.CopyFile("ethanol.pdb", "ethanol_copy.pdb")
    assert filecmp.cmp("ethanol.pdb", "ethanol_copy.pdb")
    # Remove the file if it exists
    nifty.remove_if_exists("ethanol_copy.pdb")
    assert not os.path.exists("ethanol_copy.pdb")
    nifty.remove_if_exists("ethanol.pdb")
    os.makedirs("a")
    nifty.GoInto("a")
    for fnm in ["ethanol.pdb", "boat.crd", "test_compress.tar.gz"]:
        nifty.onefile(os.path.join(datad, fnm))
    nifty.Leave("a")
    nifty.link_dir_contents("a", "b")
    # nifty.link_dir_contents(datad, os.getcwd())
    # for fnm in os.listdir(datad):
    #     assert os.path.exists(fnm)

def test_copy_tree_over(tmpdir):
    # tmpdir is a object
    tmpdir_str = str(tmpdir)

    src_dir = os.path.join(tmpdir_str, 'src')
    dest_dir = os.path.join(tmpdir_str, 'dest')

    # populate the source directory
    os.makedirs(os.path.join(src_dir, 'a', 'b'))
    with open(os.path.join(src_dir, 'a.txt'), 'w') as f:
        f.write('test file a')
    with open(os.path.join(src_dir, 'a', 'b', 'b.txt'), 'w') as f:
        f.write('test file b')

    nifty.copy_tree_over(src_dir, dest_dir)

    # Check the destination dir
    with open(os.path.join(dest_dir, 'a.txt'), 'r') as f:
        assert f.read() == 'test file a'
    with open(os.path.join(dest_dir, 'a', 'b', 'b.txt'), 'r') as f:
        assert f.read() == 'test file b'

    # Modify a file on the destination, then, and copy again
    with open(os.path.join(dest_dir, 'a', 'b', 'b.txt'), 'w') as f:
        assert f.write('test file b MODIFIED')

    nifty.copy_tree_over(src_dir, dest_dir)

    with open(os.path.join(dest_dir, 'a', 'b', 'b.txt'), 'r') as f:
        assert f.read() == 'test file b'
