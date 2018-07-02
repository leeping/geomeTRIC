"""
A set of tests for using the QCEngine project
"""

import numpy as np
from . import addons
import geometric

localizer = addons.in_folder

@addons.using_qcengine
@addons.using_rdkit
def test_rdkit_simple(localizer):
    schema = {
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

    opts = {"qcengine": True, "qcschema": schema, "input": "tmp_data", "qce_program": "rdkit"}

    ret = geometric.optimize.run_optimizer(**opts)

    # Currently in angstrom
    ref = np.array([0., 0., -0.0644928042, 0., -0.7830365196, 0.5416895554, 0., 0.7830365196, 0.5416895554])
    assert np.allclose(ref, ret.xyzs[-1].ravel(), atol=1.e-5)
