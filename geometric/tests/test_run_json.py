"""
A set of tests for using the QCEngine project
"""

import numpy as np
import json, os, shutil
from . import addons
import geometric

@addons.in_folder
@addons.using_qcengine
@addons.using_rdkit
def test_run_json():
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
        "program": "rdkit"
    }
    in_json_dict = {
        "schema_name": "qc_schema_optimization_input",
        "schema_version": 1,
        "keywords": {
            "coordsys": "tric",
            "maxiter": 100
        },
        "input_specification": qc_schema_input
    }

    json.dump(in_json_dict, open('in.json','w'), indent=2)
    out_json_dict = geometric.run_json.geometric_run_json(in_json_dict)
    json.dump(out_json_dict, open('out.json','w'), indent=2)

    result_geo = out_json_dict['final_molecule']['molecule']['geometry']

    # The results here are in Bohr
    ref = np.array([0., 0., -0.1218737, 0., -1.47972457, 1.0236449059, 0., 1.47972457, 1.023644906])
    assert np.allclose(ref, result_geo, atol=1.e-5)
