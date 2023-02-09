"""
Test the gaussian related functions
"""

import pytest
from geometric.engine import Gaussian, GaussianEngineError
from geometric.molecule import Molecule
from geometric.prepare import get_molecule_engine
from geometric.nifty import bohr2ang
import numpy as np
import tempfile
import os
import platform
from . import addons
from .addons import get_gaussian_version

datad = addons.datad


def test_gaussian_version_wrong():
    """
    Test loading an engine with an incorrect version.
    """
    molecule = Molecule(os.path.join(datad, "ethane.com"))
    with pytest.raises(ValueError):
        _ = Gaussian(molecule=molecule, exe="gaussian09")


@pytest.mark.parametrize("version", [
    pytest.param("G09", id="G09"),
    pytest.param("G16", id="G16"),
])
def test_gaussian_correct_version(version):
    """
    Check that allowed versions do not raise errors.
    """
    molecule = Molecule(os.path.join(datad, "ethane.com"))
    engine = Gaussian(molecule=molecule, exe=version)

    assert engine.gaussian_exe == version.lower()


def test_find_gaussian_missing():
    """
    Make sure an error is raised if gaussian 09/16 is not available.
    """
    major, minor, _ = platform.python_version_tuple()
    if not int(major) >= 3 and int(minor) >= 5:
        pytest.skip("Python version below 3.5 shutil.which not available.")
    kwargs = {"engine": "gaussian", "input": os.path.join(datad, "ethane.com")}
    if get_gaussian_version() is None:
        with pytest.raises(ValueError):
            _ = get_molecule_engine(**kwargs)


def test_wrong_force_input():
    """
    Make sure an error is raised if force= is present in the input file but is not force=nostep.
    """
    molecule = Molecule(os.path.join(datad, "force_restart.com"))
    engine = Gaussian(molecule=molecule, exe="g09")
    with pytest.raises(GaussianEngineError):
        engine.load_gaussian_input(os.path.join(datad, "force_restart.com"))


def test_gaussian_template():
    """
    Make sure the template is formed properly when reading input files.
    """
    molecule = Molecule(os.path.join(datad, "ethane.com"))
    engine = Gaussian(molecule=molecule, exe="g09")
    engine.load_gaussian_input(os.path.join(datad, "ethane.com"))
    assert engine.gauss_temp == ['%Mem=6GB\n', '%NProcShared=2\n', '%Chk=ligand\n', '$!route@here', '\n',
                                 'ethane\n', '\n', '0 1\n', '$!geometry@here', '\n', '\n']


def test_setting_threads():
    """
    For an input file with threads make sure we can overwrite them to our desired value.
    """
    molecule = Molecule(os.path.join(datad, "ethane.com"))
    engine = Gaussian(molecule=molecule, exe="g09", threads=30)
    engine.load_gaussian_input(os.path.join(datad, "ethane.com"))
    assert "%NProcShared=30\n" in engine.gauss_temp


def test_adding_threads_value():
    """
    If we read an input file with no threads set but want them make sure they are added.
    """
    molecule = Molecule(os.path.join(datad, "ethane_no_data.com"))
    engine = Gaussian(molecule=molecule, exe="g09", threads=30)
    engine.load_gaussian_input(os.path.join(datad, "ethane_no_data.com"))
    assert "%NProcShared=30\n" in engine.gauss_temp


def test_adding_threads_none():
    """
    If we have an input with no threads and threads is None make sure we write 1.
    """
    molecule = Molecule(os.path.join(datad, "ethane_no_data.com"))
    engine = Gaussian(molecule=molecule, exe="g09", threads=None)
    engine.load_gaussian_input(os.path.join(datad, "ethane_no_data.com"))
    assert "%NProcShared=1\n" in engine.gauss_temp


def test_checkpoint_name():
    """
    If the user supplies a file with a different checkpoint name make sure we overwrite it.
    """
    molecule = Molecule(os.path.join(datad, "ethane_wrong_name.com"))
    engine = Gaussian(molecule=molecule, exe="g09", threads=None)
    engine.load_gaussian_input(os.path.join(datad, "ethane_wrong_name.com"))
    assert "%Chk=ligand\n" in engine.gauss_temp


def test_checkpoint_missing():
    """
    If the user passes a file with no checkpoint line make sure it is added.
    """
    molecule = Molecule(os.path.join(datad, "ethane_no_data.com"))
    engine = Gaussian(molecule=molecule, exe="g09", threads=None)
    engine.load_gaussian_input(os.path.join(datad, "ethane_no_data.com"))
    assert "%Chk=ligand\n" in engine.gauss_temp


def test_calc_new_gaussian():
    """
    Test calculating the force using gaussian.
    Note this is expected to fail due to gaussian not being installed.
    """
    major, minor, _ = platform.python_version_tuple()
    if not int(major) >= 3 and int(minor) >= 5:
        pytest.skip("Python version below 3.5 TemporaryDirectory not available.")
    molecule = Molecule(os.path.join(datad, "ethane.com"))
    engine = Gaussian(molecule=molecule, exe="g09")
    engine.load_gaussian_input(os.path.join(datad, "ethane.com"))
    home = os.getcwd()
    with tempfile.TemporaryDirectory() as temp:
        os.chdir(temp)
        # now we want to run calc new to make sure the file is written correctly
        g_version = get_gaussian_version()
        if g_version is None:
            with pytest.raises(GaussianEngineError):
                engine.calc(coords=molecule.xyzs[0] / bohr2ang, dirname="ethane.tmp")

        else:
            engine.gaussian_exe = g_version
            engine.calc(coords=molecule.xyzs[0] / bohr2ang, dirname="ethane.tmp")

        # now we want to read the file back in to make sure it is correct
        molecule_2 = Molecule(os.path.join("ethane.tmp", "gaussian.com"))
        # now check over the data
        assert molecule.Data["elem"] == molecule_2.Data["elem"]
        assert molecule.Data["bonds"] == molecule_2.Data["bonds"]
        assert np.allclose(molecule.Data["xyzs"][0], molecule_2.Data["xyzs"][0])

        os.chdir(home)

@addons.using_gaussian
def test_read_results_gaussian():
    """
    Test reading the results from a fchk and log gaussian file.
    """
    #molecule = Molecule(os.path.join(datad, "ethane.com"))
    #engine = Gaussian(molecule=molecule, exe="g09")
    molecule, engine = get_molecule_engine(input=os.path.join(datad, "ethane.com"), engine='gaussian')
    result = engine.read_result(dirname=datad)
    assert np.allclose(result["energy"], -79.2232076385)
    assert np.allclose(result["gradient"], np.array([-6.56042860e-03, -9.27746601e-03, -1.60692886e-02, -1.69810268e-03,  1.04014113e-02,
                                                     5.85661970e-03,  4.44716904e-04, -1.64409044e-03,  1.19361094e-02,  1.03726209e-02,
                                                     4.13886770e-03,  4.54429744e-03,  6.56039611e-03,  9.27750990e-03,  1.60693707e-02,
                                                     -4.58383008e-04, -1.11612438e-02, -4.53755817e-03,  1.70936522e-03,  1.18224187e-04,
                                                     -1.19346786e-02, -1.03701848e-02, -1.85321276e-03, -5.86487186e-03]))
    # make sure the gradient array is the same shape ass the coords
    assert molecule.xyzs[0].shape == result["gradient"].reshape(-1, 3).shape
