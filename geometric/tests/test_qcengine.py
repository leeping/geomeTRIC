"""
A set of tests for using the QCEngine project
"""

import numpy as np

from . import addons
import geometric

@addons.using_qcengine
@addons.using_rdkit
def test_rdkit_simple():
    schema = {
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


    opts = {"qcengine": True, "qcschema": schema, "input": "tmp_data"}

    ret = geometric.optimize.run_optimizer(**opts)

    # Currently in angstrom
    ref = np.array([0., 0., -0.06851625, 0., -0.79068989, 0.54370128, 0., 0.79068989, 0.54370128])
    assert np.allclose(ref, ret.xyzs[0].ravel(), atol=1.e-5)
