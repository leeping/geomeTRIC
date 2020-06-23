"""
A set of tests for using the QCEngine project
"""

import copy
import numpy as np
import json, os, sys, shutil
from . import addons
import geometric
import pytest

localizer = addons.in_folder

def _compare_constraint_strings(str1, str2, atol=1e-6):
    str1lines = str1.split('\n')
    str2lines = str2.split('\n')
    assert len(str1lines) == len(str2lines)
    for line1, line2 in zip(str1lines, str2lines):
        words1 = line1.split()
        words2 = line2.split()
        assert len(words1) == len(words2)
        for word1, word2 in zip(words1, words2):
            if geometric.nifty.isint(word1):
                assert geometric.nifty.isint(word2)
                assert int(word1) == int(word2)
            elif geometric.nifty.isfloat(word1):
                assert geometric.nifty.isfloat(word2)
                assert abs(float(word1) - float(word2)) < atol
            else:
                assert word1 == word2

def _build_input(molecule, program="rdkit", method="UFF", basis=None):
    qc_schema_input = {
        "schema_name": "qcschema_input",
        "schema_version": 1,
        "driver": "gradient",
        "model": {
            "method": method,
            "basis": basis
        },
        "keywords": {},
    } # yapf: disable
    in_json_dict = {
        "schema_name": "qcschema_optimization_input",
        "schema_version": 1,
        "keywords": {
            "coordsys": "tric",
            "maxiter": 100,
            "program": program
        },
        "initial_molecule": molecule,
        "input_specification": qc_schema_input,
    } # yapf: disable
    return in_json_dict


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher (ordered dict)")
def test_convert_constraint_dict_full():
    constraint_dict = {
        "freeze": [{
            "type": "xyz",
            "indices": [0, 1, 2, 3, 4]
        }],
        "set": [{
            "type": "distance",
            "indices": [1, 0],
            "value": 1.1
        }, {
            "type": "angle",
            "indices": [1, 0, 4],
            "value": 110.0
        }],
        "scan": [{
            "type": "distance",
            "indices": [1, 0],
            "start": 1.0,
            "stop": 2,
            "steps": 3
        }, {
            "type": "dihedral",
            "indices": [0, 4, 5, 6],
            "start": 110.0,
            "stop": 150.0,
            "steps": 3
        }]
    }
    constraint_string = geometric.run_json.make_constraints_string(constraint_dict)
    ref_constraint_string = """$freeze
xyz 1-5
$set
distance 2 1 0.582094931
angle 2 1 5 110.0
$scan
distance 2 1 0.52917721 1.05835442 3
dihedral 1 5 6 7 110.0 150.0 3"""
    _compare_constraint_strings(constraint_string, ref_constraint_string)

def test_convert_constraint_dict_failure():
    failing_constraint_dict = {'not_recognized_keyword': [('xyz', [0, 1])]}
    with pytest.raises(KeyError):
        geometric.run_json.make_constraints_string(failing_constraint_dict)


@addons.using_qcengine
@addons.using_rdkit
def test_run_json_scan_rejection(localizer):

    molecule = {
        "geometry": [
            0.0,  0.0, -0.5,
            0.0,  0.0,  0.5,
        ],
        "symbols": ["H", "H"],
        "connectivity": [[0, 1, 1]]
    } # yapf: disable

    in_json_dict = _build_input(molecule, program="psi4", method="HF", basis="sto-3g")
    in_json_dict["keywords"]["constraints"] = {
        'scan': [{
            "type": ('bond', [0, 1]),
            "start": 0.8,
            "stop": 1.2,
            "steps": 4
        }]
    }

    with open('in.json', 'w') as handle:
        json.dump(in_json_dict, handle, indent=2)

    with pytest.raises(KeyError) as excinfo:
        out_json = geometric.run_json.geometric_run_json(in_json_dict)
        print(out_json['stdout'])

    assert "'scan' keyword" in str(excinfo.value)


@addons.using_qcengine
@addons.using_rdkit
def test_run_json_rdkit_water(localizer):
    molecule = {
        "geometry": [
            0.0,  0.0,              -0.1294769411935893,
            0.0, -1.494187339479985, 1.0274465079245698,
            0.0,  1.494187339479985, 1.0274465079245698
        ],
        "symbols": ["O", "H", "H"],
        "connectivity": [[0, 1, 1], [0, 2, 1]]
    } # yapf: disable

    in_json_dict = _build_input(molecule)

    with open('in.json', 'w') as handle:
        json.dump(in_json_dict, handle, indent=2)
    out_json = geometric.run_json.geometric_run_json(in_json_dict)
    print(out_json['stdout'])

    with open('out.json', 'w') as handle:
        json.dump(out_json, handle, indent=2)

    result_geo = out_json['final_molecule']['geometry']

    # The results here are in Bohr
    ref = np.array([0., 0., -0.1218737, 0., -1.47972457, 1.0236449059, 0., 1.47972457, 1.023644906])
    assert pytest.approx(out_json["energies"][-1], 1.e-4) == 0.0
    assert np.allclose(ref, result_geo, atol=1.e-4)


@addons.using_qcengine
@addons.using_rdkit
def test_run_json_rdkit_hooh_constraint(localizer):
    molecule = {
        "geometry": [
             1.76498,  1.3431,  0.7946,
             1.24509, -0.0343, -0.3637,
            -1.24263,  0.0351, -0.3580,
            -1.75745, -1.3508,  0.7925
        ],
        "symbols": ["H", "O", "O", "H"],
        "connectivity": [[0, 1, 1], [1, 2, 1], [2, 3, 1]]
    } # yapf: disable


    in_json_dict = _build_input(molecule)
    in_json_dict["keywords"]["constraints"] = {"set": [{"type": "dihedral", "indices": [0, 1, 2, 3], "value": -180}]}

    with open('in.json', 'w') as handle:
        json.dump(in_json_dict, handle, indent=2)
    out_json = geometric.run_json.geometric_run_json(in_json_dict)
    print(out_json['stdout'])

    with open('out.json', 'w') as handle:
        json.dump(out_json, handle, indent=2)

    result_geo = out_json['final_molecule']['geometry']

    # The results here are in Bohr
    assert pytest.approx(out_json["energies"][-1], 1.e-4) == 0.0007534925


@addons.using_qcengine
@addons.using_rdkit
def test_run_json_distance_constraint(localizer):
    molecule = {
        "geometry": [
             1.76498,  1.3431,  0.7946,
             1.24509, -0.0343, -0.3637,
            -1.24263,  0.0351, -0.3580,
            -1.75745, -1.3508,  0.7925
        ],
        "symbols": ["H", "O", "O", "H"],
        "connectivity": [[0, 1, 1], [1, 2, 1], [2, 3, 1]]
    } # yapf: disable

    result_geo = np.array(molecule["geometry"]).reshape(-1, 3)

    in_json_dict = _build_input(molecule)
    in_json_dict["keywords"]["constraints"] = {"set": [{"type": "distance", "indices": [1, 2], "value": 2.4}]}

    with open('in.json', 'w') as handle:
        json.dump(in_json_dict, handle, indent=2)

    out_json = geometric.run_json.geometric_run_json(in_json_dict)
    print(out_json['stdout'])

    with open('out.json', 'w') as handle:
        json.dump(out_json, handle, indent=2)

    result_geo = np.array(out_json['final_molecule']['geometry']).reshape(-1, 3)
    assert pytest.approx(2.4) == np.linalg.norm(result_geo[1] - result_geo[2])
    assert "Converged" in out_json["stdout"]


@addons.using_qcengine
@addons.using_psi4
def test_run_json_psi4_hydrogen(localizer):

    molecule = {
        "geometry": [
            0.0,  0.0, -0.5,
            0.0,  0.0,  0.5,
        ],
        "symbols": ["H", "H"],
        "connectivity": [[0, 1, 1]]
    } # yapf: disable

    in_json_dict = _build_input(molecule, program="psi4", method="HF", basis="sto-3g")

    with open('in.json', 'w') as handle:
        json.dump(in_json_dict, handle, indent=2)

    out_json = geometric.run_json.geometric_run_json(in_json_dict)
    print(out_json['stdout'])

    with open('out.json', 'w') as handle:
        json.dump(out_json, handle, indent=2)

    assert out_json["success"], out_json["error"]

    result_geo = out_json['final_molecule']['geometry']

    # The results here are in Bohr
    ref = np.array([0., 0., -0.672954004258, 0., 0., 0.672954004258])
    #assert pytest.apprx
    assert pytest.approx(out_json["energies"][-1], 1.e-4) == -1.1175301889636524
    assert np.allclose(ref, result_geo, atol=1.e-5)
    assert out_json["schema_name"] == "qc_schema_optimization_output"
    assert "Converged" in out_json["stdout"]


@addons.using_qcengine
@addons.using_rdkit
def test_rdkit_run_error(localizer):
    molecule = {
        "geometry": [
            0.0,  0.0, -0.5,
            0.0,  0.0,  0.5,
        ],
        "symbols": ["H", "H"],
        "connectivity": [[0, 1, 1]]
    } # yapf: disable
    in_json_dict = _build_input(molecule, method="cookiemonster")
    # an error should be caught in the ret
    ret = geometric.run_json.geometric_run_json(in_json_dict)
    print(ret['stdout'])
    assert ret["success"] == False
    assert "UFF" in ret["error"]["error_message"]
