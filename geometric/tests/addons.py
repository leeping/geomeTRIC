"""
Figures out currently available modules
"""
import shutil

import geometric
import pytest
import os
import logging.config
import pkg_resources


def _plugin_import(plug):
    """
    Tests to see if a module is available
    """
    import sys

    try:
        if sys.version_info >= (3, 4):
            from importlib import util

            plug_spec = util.find_spec(plug)
        else:
            import pkgutil

            plug_spec = pkgutil.find_loader(plug)
    except ModuleNotFoundError:
        return False

    if plug_spec is None:
        return False
    else:
        return True


def get_gaussian_version():
    """
    Try and work out the gaussian version if it can not be found return none.
    """
    if shutil.which("g16") is not None:
        return "g16"
    elif shutil.which("g09") is not None:
        return "g09"
    else:
        return None


# Modify paths for testing
os.environ["DQM_CONFIG_PATH"] = os.path.dirname(os.path.abspath(__file__))
os.environ["TMPDIR"] = "/tmp/"

# Add flags
using_psi4 = pytest.mark.skipif(
    _plugin_import("psi4") is False, reason="could not find psi4. please install the package to enable tests")
using_rdkit = pytest.mark.skipif(
    _plugin_import("rdkit") is False, reason="could not find rdkit. please install the package to enable tests")
using_qcengine = pytest.mark.skipif(
    _plugin_import("qcengine") is False, reason="could not find qcengine. please install the package to enable tests")
using_openmm = pytest.mark.skipif(
    _plugin_import("openmm") is False and _plugin_import("simtk.openmm") is False, reason="could not find openmm. please install the package to enable tests")
using_workqueue = pytest.mark.skipif(
    (_plugin_import("work_queue") is False) or (not geometric.nifty.which('work_queue_worker')), reason="could not find work_queue module or work_queue_worker executable. please install the package to enable tests")
using_terachem = pytest.mark.skipif(
    not geometric.nifty.which("terachem"), reason="could not find terachem. please make sure TeraChem is installed for these tests")
using_qchem = pytest.mark.skipif(
    not geometric.nifty.which("qchem"), reason="could not find qchem. please make sure Q-Chem is installed for these tests")
using_gaussian = pytest.mark.skipif(
    get_gaussian_version() is None,
    reason="could not find Gaussian. please make sure Gaussian is installed for these tests",
)
using_ase = pytest.mark.skipif(
    _plugin_import("ase") is False,
    reason="could not find ase. please install the package to enable tests",
)
using_xtb = pytest.mark.skipif(
    _plugin_import("xtb") is False,
    reason="could not find ase. please install the package to enable tests",
)


# Points to the folder where the data files are installed.
datad = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
exampled = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'examples')

# make tests run in their own folder
@pytest.fixture(scope="function")
def in_folder(request):

    # Build out a test folder
    cwd = os.path.abspath(os.getcwd())
    test_folder = os.path.join(cwd, 'test_generated_files', request.function.__name__)

    # Build and change to test folder
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    else:
        # If previous results exist in the test folder, archive them.
        prev_num = 0
        allfiles = os.listdir(test_folder)
        while os.path.exists(os.path.join(test_folder, 'previous.%03i' % prev_num)):
            allfiles.remove('previous.%03i' % prev_num)
            prev_num += 1
        if prev_num == 1000:
            raise IOError("There are too many previous result folders in %s" % test_folder)
        os.makedirs(os.path.join(test_folder, 'previous.%03i' % prev_num))
        for f in allfiles:
            src_path = os.path.join(test_folder, f)
            dst_path = os.path.join(test_folder, 'previous.%03i' % prev_num, f)
            os.rename(src_path, dst_path)

    os.chdir(test_folder)

    # Yield for testing
    yield test_folder

    # Change back to CWD
    os.chdir(cwd)
