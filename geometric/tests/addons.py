"""
Figures out currently available modules
"""

import pytest
import os


def _plugin_import(plug):
    """
    Tests to see if a module is available
    """
    import sys
    if sys.version_info >= (3, 4):
        from importlib import util
        plug_spec = util.find_spec(plug)
    else:
        import pkgutil
        plug_spec = pkgutil.find_loader(plug)
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

