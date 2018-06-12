"""
A set of tests for using the QCEngine project
"""

import numpy as np
import json, os, shutil
from . import addons
import geometric
import pytest

localizer = addons.in_folder

@addons.using_qcengine
@addons.using_rdkit
def test_run_json_rdkit_water(localizer):
    # create a test input json file
    qc_schema_input = {
        "schema_name": "qc_schema_input",
        "schema_version": 1,
        "molecule": {
            "geometry": [
                0.0,  0.0,              -0.1294769411935893,
                0.0, -1.494187339479985, 1.0274465079245698,
                0.0,  1.494187339479985, 1.0274465079245698
            ],
            "symbols": ["O", "H", "H"],
            "connectivity": [[0, 1, 1], [0, 2, 1]]
        },
        "driver": "gradient",
        "model": {
            "method": "UFF",
            "basis": None
        },
        "keywords": {},
    }
    in_json_dict = {
        "schema_name": "qc_schema_optimization_input",
        "schema_version": 1,
        "keywords": {
            "coordsys": "tric",
            "maxiter": 100,
            "program": "rdkit"
        },
        "input_specification": qc_schema_input
    }

    with open('in.json', 'w') as handle:
        json.dump(in_json_dict, handle, indent=2)
    out_json = geometric.run_json.geometric_run_json(in_json_dict)

    with open('out.json', 'w') as handle:
        json.dump(out_json, handle, indent=2)

    result_geo = out_json['final_molecule']['molecule']['geometry']

    # The results here are in Bohr
    ref = np.array([0., 0., -0.1218737, 0., -1.47972457, 1.0236449059, 0., 1.47972457, 1.023644906])
    assert pytest.approx(out_json["energies"][-1], 1.e-4) == 0.0
    assert np.allclose(ref, result_geo, atol=1.e-4)

@addons.using_qcengine
@addons.using_psi4
def test_run_json_psi4_hydrogen(localizer):

    qc_schema_input = {
        "schema_name": "qc_schema_input",
        "schema_version": 1,
        "molecule": {
            "geometry": [
                0.0,  0.0, -0.5,
                0.0,  0.0,  0.5,
            ],
            "symbols": ["H", "H"],
            "connectivity": [[0, 1, 1]]
        },
        "driver": "gradient",
        "model": {
            "method": "HF",
            "basis": "sto-3g"
        },
        "keywords": {},
    }
    in_json_dict = {
        "schema_name": "qc_schema_optimization_input",
        "schema_version": 1,
        "keywords": {
            "coordsys": "tric",
            "maxiter": 100,
            "program": "psi4"
        },
        "input_specification": qc_schema_input
    }

    with open('in.json', 'w') as handle:
        json.dump(in_json_dict, handle, indent=2)

    out_json = geometric.run_json.geometric_run_json(in_json_dict)

    with open('out.json', 'w') as handle:
        json.dump(out_json, handle, indent=2)

    result_geo = out_json['final_molecule']['molecule']['geometry']

    # The results here are in Bohr
    ref = np.array([0., 0., -0.672954004258, 0., 0., 0.672954004258])
    #assert pytest.apprx
    assert pytest.approx(out_json["energies"][-1], 1.e-4) == -1.1175301889636524
    assert np.allclose(ref, result_geo, atol=1.e-5)
    assert out_json["schema_name"] == "qc_schema_optimization_output"
