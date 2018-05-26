"""
A set of tests for using the QCEngine project
"""

import numpy as np
import json, os, shutil
from . import addons
import geometric

@addons.using_qcengine
@addons.using_rdkit
def test_run_json():
    # create a test input json file
    qc_schema_input = {
        "schema_name": "UFF gradient",
        "schema_version": "v0.1",
        "molecule": {
            "geometry": [
                0.0, 0.0, -0.1294769411935893, 0.0,
               -1.494187339479985, 1.0274465079245698,
                0.0, 1.494187339479985, 1.0274465079245698
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
        "geometric_options": {
            "coordsys": "tric",
            "conv": 1.e-7
        },
        "input_specification": qc_schema_input
    }
    os.mkdir('run_json.tmp')
    os.chdir('run_json.tmp')
    json.dump(in_json_dict, open('in.json','w'), indent=2)
    out_json_dict = geometric.run_json.geometric_run_json(in_json_dict)
    json.dump(out_json_dict, open('out.json','w'), indent=2)
    os.chdir('..')
    #shutil.rmtree('run_json.tmp')
