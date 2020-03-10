"""
Figures out currently available modules
"""

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
    _plugin_import("simtk.openmm") is False, reason="could not find simtk.openmm. please install the package to enable tests")

# Points to the folder where the data files are installed.
datad = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')

# make tests run in their own folder
@pytest.fixture(scope="function")
def in_folder(request):

    # Build out a test folder
    cwd = os.path.abspath(os.getcwd())
    test_folder = os.path.join(cwd, 'test_generated_files', request.function.__name__)

    # Build and change to test folder
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    os.chdir(test_folder)

    # Yield for testing
    yield test_folder

    # Change back to CWD
    os.chdir(cwd)

# make tests run in their own folder
@pytest.fixture(scope="function")
def test_logger(request):

    # Adding these three lines here removes the extra newline that was printed 
    logIni = pkg_resources.resource_filename(geometric.optimize.__name__, 'config/logTest.ini')
    logging.config.fileConfig(logIni,disable_existing_loggers=False)
    
    # Yield for testing
    yield 
