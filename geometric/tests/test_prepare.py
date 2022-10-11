"""
A set of tests for functions in prepare.py not covered elsewhere (e.g. parsing constraints)
"""

import copy
import numpy as np
import json, os, shutil
from . import addons
import geometric
import pytest
import itertools

datad = addons.datad
localizer = addons.in_folder
exampled = addons.exampled

def test_parse_constraints(localizer):
    molecule = geometric.molecule.Molecule(os.path.join(exampled, '2-challenges', 'gal-glc_constraints', 'start.xyz'))
    # Test constraint freezing and setting
    constraint_str = """$freeze
distance 6 8
angle 8 6 22
dihedral 18 22 6 8
$set
distance  6 22 1.554
angle  8  9 16 110.139
dihedral  6  8  9 16 60.050
$end"""
    ics, vals = geometric.prepare.parse_constraints(molecule, constraint_str)
    assert ics[0] == geometric.internal.Distance(5, 7)
    assert ics[1] == geometric.internal.Angle(7, 5, 21)
    assert ics[2] == geometric.internal.Dihedral(7, 5, 21, 17)
    assert ics[3] == geometric.internal.Distance(5, 21)
    assert ics[4] == geometric.internal.Angle(7, 8, 15)
    assert ics[5] == geometric.internal.Dihedral(5, 7, 8, 15)
    assert vals[0][:3] == [None, None, None]
    np.testing.assert_almost_equal(np.array(vals[0][3:]), np.array([1.554/0.529177210903, 110.139*np.pi/180, 60.050*np.pi/180]))

    # Test multidimensional constraint scanning
    constraint_str = """$scan
distance 6 8 1.2 1.4 3
angle 8 6 22 110 119 4
dihedral 6 8 9 16 60 65 2
$end"""
    ics, vals = geometric.prepare.parse_constraints(molecule, constraint_str)
    assert ics[0] == geometric.internal.Distance(5, 7)
    assert ics[1] == geometric.internal.Angle(7, 5, 21)
    assert ics[2] == geometric.internal.Dihedral(5, 7, 8, 15)
    dvals, avals, tvals = np.meshgrid(np.linspace(1.2, 1.4, 3)/0.529177210903, np.linspace(110, 119, 4)*np.pi/180,
                                      np.linspace(60, 65, 2)*np.pi/180, indexing='ij')
    dvals = dvals.flatten()
    avals = avals.flatten()
    tvals = tvals.flatten()
    np.testing.assert_almost_equal(dvals, np.array([vals[i][0] for i in range(len(vals))]))
    np.testing.assert_almost_equal(avals, np.array([vals[i][1] for i in range(len(vals))]))
    np.testing.assert_almost_equal(tvals, np.array([vals[i][2] for i in range(len(vals))]))

    # Test freeze/set/scan for rotation constraints
    constraint_str = """$freeze
rotation 1-6
$set
rotation 7-12 1.0 0.0 0.0 30.0
$scan
rotation 13-18 0.0 0.0 1.0 0.0 20.0 3
$end"""
    ics, vals = geometric.prepare.parse_constraints(molecule, constraint_str)
    assert len(ics) == 9
    assert ics[0].a == (0,1,2,3,4,5)
    assert vals[0][:3] == [None, None, None]
    rg2 = 5.978207642
    rg3 = 7.668148025
    np.testing.assert_almost_equal(vals[0][3:6], [rg2*np.pi/6, 0.0, 0.0])
    np.testing.assert_almost_equal(vals[2][3:6], [rg2*np.pi/6, 0.0, 0.0])
    np.testing.assert_almost_equal(vals[0][6:9], [0.0, 0.0, 0.0])
    np.testing.assert_almost_equal(vals[1][6:9], [0.0, 0.0, rg3*np.pi/18])

    # Test freeze/set/scan for position constraints
    constraint_str = """$freeze
x 1
yz 5,6,8
xyz 11-14
$set
y 2 2.0
trans-xy 15,16 1.0 3.5
$scan
z 3 1.0 5.0 5
trans-xyz 7,9 0.0 1.0 0.0 1.0 0.0 0.0 3
$end"""
    ics, vals = geometric.prepare.parse_constraints(molecule, constraint_str)
    assert len(ics) == 1 + 2*3 + 3*4 + 1 + 2 + 1 + 3
    assert vals[0][:3] == [None, None, None]
    np.testing.assert_almost_equal(vals[10][19], 2.0/0.529177210903)
    np.testing.assert_almost_equal(vals[3][22], 2.0/0.529177210903)
    np.testing.assert_almost_equal(vals[13][24], 0.5/0.529177210903)
    np.testing.assert_almost_equal(vals[14][25], 0.0)


@addons.using_terachem
def test_get_molecule_engine_pdb__terachem():
    get_molecule_engine_pdb("terachem", "water12.tcin")


def test_get_molecule_engine_pdb__psi4():
    # no need for Psi4 to be installed
    get_molecule_engine_pdb("psi4", "water12.psi4in")


def test_get_molecule_engine_pdb__qchem():
    # no need for QChem
    get_molecule_engine_pdb("qchem", "water12.qcin")


@addons.using_gaussian
def test_get_molecule_engine_pdb__gaussian():
    get_molecule_engine_pdb("gaussian", "water12.gjf")


def test_get_molecule_engine_pdb__molpro():
    # no need for molpro
    get_molecule_engine_pdb("molpro", "water12.mol")


def get_molecule_engine_pdb(engine, inputf):
    # worker for the above tests which all require different engines
    molecule, engine = geometric.prepare.get_molecule_engine(
        input=os.path.join(datad, inputf),
        engine=engine,
        pdb=os.path.join(datad, "water12.pdb"),
        coords=os.path.join(datad, "water12.mdcrd"),
    )
    assert molecule.resid[-1] == 12
    np.testing.assert_almost_equal(molecule.xyzs[0][-1, 2], 1.288)
    assert engine.detect_dft() is False
