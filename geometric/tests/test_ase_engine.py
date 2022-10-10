"""Tests for the ASE Calculator Engine

Copyright 2022 Tamas K. Stenczel

Test the following:
- using ASE internal calculator class
- results are consistent with the ASE calc results
- import of class from string & import error
- setting of positions
- geometry optimisation w/ ASE internal calc
"""
import numpy as np
from ase.calculators.lj import LennardJones
from pytest import fixture, raises

from geometric.ase_engine import EngineASE
from geometric.errors import EngineError
from geometric.molecule import Molecule
from .addons import using_ase


@fixture
def molecule_h2o() -> Molecule:
    molecule = Molecule()
    molecule.elem = ["O", "H", "H"]
    molecule.xyzs = [
        np.array(
            (
                (0.0, 0.3, 0),
                (0.9, 0.8, 0),
                (-0.9, 0.5, 0),
            )
        )  # In Angstrom
    ]
    return molecule


@using_ase
def test_construction(molecule_h2o):
    lj_calc = LennardJones()
    engine = EngineASE(molecule_h2o, lj_calc)
    assert engine.calculator == lj_calc


@using_ase
def test_from_args(molecule_h2o):
    lj_calc = LennardJones(sigma=1.4, epsilon=3.0)

    # create equivalent engines in two ways
    engine_init = EngineASE(molecule_h2o, lj_calc)
    engine_cls = EngineASE.from_calculator_constructor(
        molecule_h2o, LennardJones, sigma=1.4, epsilon=3.0
    )

    # calculator parameters should be the same
    assert engine_init.calculator.parameters == engine_cls.calculator.parameters

    # calculated results should be the same as well
    xyz = np.arange(9)
    tmp_dir = "/tmp/"

    result_init = engine_init.calc(xyz, tmp_dir)
    result_cls = engine_cls.calc(xyz, tmp_dir)

    for key, val in result_init.items():
        assert np.all(val == result_cls[key])


@using_ase
def test_from_string(molecule_h2o):
    engine = EngineASE.from_calculator_string(
        molecule_h2o, calculator_import="ase.calculators.lj.LennardJones"
    )
    assert isinstance(engine.calculator, LennardJones)


@using_ase
def test_from_string_errors(molecule_h2o):
    with raises(EngineError):
        # does not exist
        EngineASE.from_calculator_string(
            molecule_h2o, calculator_import="module_that.does.not.exist"
        )
    with raises(EngineError):
        # does not exist
        EngineASE.from_calculator_string(
            molecule_h2o, calculator_import="ase.NonExistent"
        )
    with raises(EngineError):
        # exists, but not a calculator class
        EngineASE.from_calculator_string(
            molecule_h2o, calculator_import="ase.optimize.optimize.Optimizer"
        )


def test_importability_no_ase():
    # this should be able to import the ase_engine
    # without ase being installed
    import geometric.prepare
    import geometric.ase_engine
