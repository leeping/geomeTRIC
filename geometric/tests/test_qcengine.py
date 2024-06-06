"""
A set of tests for using the QCEngine project
"""

import copy
import numpy as np
import pytest
from . import addons
import geometric

localizer = addons.in_folder

_base_schema = {
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
    } # yapf: disable

@addons.using_qcengine
@addons.using_rdkit
def test_rdkit_simple(localizer):

    schema = copy.deepcopy(_base_schema)
    opts = {"engine": "qcengine", "qcschema": schema, "input": "tmp_data", "qce_program": "rdkit"}

    ret = geometric.optimize.run_optimizer(**opts)

    # Currently in angstrom
    ref = np.array([0., 0., -0.0644928042, 0., -0.7830365196, 0.5416895554, 0., 0.7830365196, 0.5416895554])
    assert np.allclose(ref, ret.xyzs[-1].ravel(), atol=1.e-5)


@addons.using_qcengine_genopt
@pytest.mark.parametrize("bsse", [
    pytest.param("nocp"),
    pytest.param("cp"),
])
@pytest.mark.parametrize("qcprog", [
    # different programs just for flexibility of testing
    pytest.param("psi4", marks=addons.using_psi4),
    pytest.param("cfour", marks=addons.using_cfour),
    pytest.param("nwchem", marks=addons.using_nwchem),
])
def test_lif_bsse(localizer, bsse, qcprog):
    lif = {
        "symbols": ["Li", "F"],
        "geometry": [0, 0, 0, 0, 0, 3],
        "fragments": [[0], [1]],
        "fragment_charges": [+1, -1],
    }

    mbe_spec = {
        "schema_name": "qcschema_manybodyspecification",
        "specification": {
            "model": {
                "method": "hf",
                "basis": "6-31G",
            },
            "driver": "energy",
            "program": qcprog,
            "keywords": {},
            "protocols": {
                "stdout": False,
            },
        },
        "keywords": {
            "bsse_type": bsse,
            "supersystem_ie_only": True,
        },
        "protocols": {
            "component_results": "all",
        },
        "driver": "energy",
    }

    opt_data = {
        "initial_molecule": lif,
        "input_specification": mbe_spec,
        "keywords": {
            "program": "nonsense",
            "convergence_set": "interfrag_tight",
            "maxiter": 20,
        },
        "protocols": {
            "trajectory": "final",
        },
    }

    # LAB -- this could be made to work, but I'm not sure if qcengine is even advised to be used this route
    # opts = {"engine": "qcengine", "qcschema": opt_data, "input": "tmp_data", "qce_program": "qcmanybody"}
    # output_data = geometric.optimize.run_optimizer(**opts)

    output_data = geometric.run_json.geometric_run_json(opt_data)

    # printing will show up if job fails
    import pprint
    pprint.pprint(output_data, width=200)

    assert output_data["success"]

    assert len(output_data["trajectory"][0]["component_properties"]) == (5 if bsse == "cp" else 3)

    atres = list(output_data["trajectory"][0]["component_results"].values())[0]
    assert atres["provenance"]["creator"].lower() == qcprog

    Rlif = output_data["final_molecule"]["geometry"][5] - output_data["final_molecule"]["geometry"][2]  # Bohr
    Rref = 3.016 if bsse == "cp" else 2.969
    np.testing.assert_almost_equal(Rlif, Rref, decimal=3)
