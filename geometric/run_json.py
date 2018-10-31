#!/usr/bin/env python

import copy
import geometric
import json
import traceback


def parse_input_json_dict(in_json_dict):
    """
    Parse an input json dictionary into options, example:
    in_json_dict = {
        "schema_name": "qc_schema_optimization_input",
        "schema_version", 1,
        "keywords": {
            "coordsys": "tric",
            "conv": 1.e-7
        }
        "input_specification": qc_schema_input,
    }
    qc_schema_input = {
        "schema_version": 1,
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
    """

    in_json_dict = copy.deepcopy(in_json_dict)
    input_opts = in_json_dict['keywords']
    input_specification = in_json_dict['input_specification']

    # insert `fix_orientation` and `fix_com`
    input_specification['molecule'] = in_json_dict['initial_molecule']
    input_specification['molecule'].update({
        'fix_orientation': True,
        'fix_com': True,
    })

    # Here we force the use of qcengine because other engines don't support qc schema
    input_opts.update({'qcengine': True, 'qcschema': input_specification, 'qce_program': input_opts["program"]})
    return input_opts


def get_output_json_dict(in_json_dict, schema_traj):
    # copy the input json data
    out_json_dict = in_json_dict.copy()
    out_json_dict["schema_name"] = "qc_schema_optimization_output"

    energy_traj = []
    for x in schema_traj:
        try:
            energy_traj.append(x["properties"]["return_energy"])
        except KeyError:
            energy_traj.append(None)

    final_molecule = None
    if schema_traj:
        final_molecule = schema_traj[-1]["molecule"]

    out_json_dict.update({"trajectory": schema_traj, "energies": energy_traj, "final_molecule": final_molecule})
    return out_json_dict


def make_constraints_string(constraints_dict):
    """ Convert the new constraints dict format into the original string format """
    key_fields = {"freeze": ("type", ), "set": ("type", "value"), "scan": ("type", "start", "stop", "steps")}
    spec_numbers = {"xyz": None, "distance": 2, "angle": 3, "dihedral": 4}

    constraints_repr = []

    # Parse overall constraints
    for key, constraints_list in constraints_dict.items():

        if key not in key_fields:
            raise KeyError("Constraints key %s is not recognized" % key)
        key_args = key_fields[key]

        # Parse individual constraints within a key
        constraints_repr.append("$" + key)
        for constraint in constraints_list:

            # Check keys
            missing = set(key_args) - constraint.keys()
            if missing:
                raise KeyError("Constraint type '%s' requires fields '%s', found '%s'" % (key, key_args,
                                                                                          constraint.keys()))

            # Check types and length
            constraint_type = constraint["type"].lower()
            if constraint_type not in spec_numbers:
                raise KeyError("Constraint type '%s' not recognized." % constraint["type"][0])

            spec_length = spec_numbers[constraint_type]
            if (spec_length is not None) and (len(constraint["indices"]) != spec_length):
                raise ValueError("Expected constraint of type '%s' to have length '%d', found %s." %
                                 (constraint_type, spec_length, str(constraint["indices"])))

            # Get base values
            const_rep = [constraint_type]
            # Add one to make it consistent with normal input
            const_rep.extend([x + 1 for x in constraint["indices"]])
            for k in key_args[1:]:
                const_rep.append(constraint[k])

            rep = " ".join(map(str, const_rep))

            constraints_repr.append(rep)

    return "\n".join(constraints_repr)


def geometric_run_json(in_json_dict):
    """ Take a input dictionary loaded from json, and return an output dictionary for json """

    input_opts = parse_input_json_dict(in_json_dict)
    M, engine = geometric.optimize.get_molecule_engine(**input_opts)

    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr

    # Read in the constraints
    constraints_dict = input_opts.get('constraints', {})
    if "scan" in constraints_dict:
        raise KeyError("The constraint 'scan' keyword is not yet supported by the JSON interface")

    constraints_string = make_constraints_string(constraints_dict)
    Cons, CVals = None, None
    if constraints_string:
        Cons, CVals = geometric.optimize.ParseConstraints(M, constraints_string)

    # set up the internal coordinate system
    coordsys = input_opts.get('coordsys', 'tric')
    CoordSysDict = {
        'cart': (geometric.internal.CartesianCoordinates, False, False),
        'prim': (geometric.internal.PrimitiveInternalCoordinates, True, False),
        'dlc': (geometric.internal.DelocalizedInternalCoordinates, True, False),
        'hdlc': (geometric.internal.DelocalizedInternalCoordinates, False, True),
        'tric': (geometric.internal.DelocalizedInternalCoordinates, False, False)
    }

    CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]
    IC = CoordClass(
        M,
        build=True,
        connect=connect,
        addcart=addcart,
        constraints=Cons,
        cvals=CVals[0] if CVals is not None else None)

    # Print out information about the coordinate system
    if isinstance(IC, geometric.internal.CartesianCoordinates):
        print("%i Cartesian coordinates being used" % (3 * M.na))
    else:
        print("%i internal coordinates being used (instead of %i Cartesians)" % (len(IC.Internals), 3 * M.na))
    print(IC)

    params = geometric.optimize.OptParams(**input_opts)

    try:
        # Run the optimization
        if Cons is None:
            # Run a standard geometry optimization
            geometric.optimize.Optimize(coords, M, IC, engine, None, params)
        else:
            # Run a constrained geometry optimization
            if isinstance(IC, (geometric.internal.CartesianCoordinates,
                               geometric.internal.PrimitiveInternalCoordinates)):
                raise RuntimeError("Constraints only work with delocalized internal coordinates")
            for ic, CVal in enumerate(CVals):
                if len(CVals) > 1:
                    print("---=== Scan %i/%i : Constrained Optimization ===---" % (ic + 1, len(CVals)))
                IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVal)
                IC.printConstraints(coords, thre=-1)
                geometric.optimize.Optimize(coords, M, IC, engine, None, params)

        out_json_dict = get_output_json_dict(in_json_dict, engine.schema_traj)
        out_json_dict["success"] = True

    except Exception as e:
        out_json_dict = get_output_json_dict(in_json_dict, engine.schema_traj)
        out_json_dict["error_message"] = "geomeTRIC run_json error:\n" + traceback.format_exc()
        out_json_dict["success"] = False

    return out_json_dict


def main():
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_json', help='Input json file name')
    parser.add_argument('-o', '--out_json', default='out.json', help='Output Json file name')
    args = parser.parse_args()
    print(' '.join(sys.argv))

    in_json_dict = json.load(open(args.in_json))
    out_json_dict = geometric_run_json(in_json_dict)
    with open(args.out_json, 'w') as outfile:
        json.dump(out_json_dict, outfile, indent=2)


if __name__ == '__main__':
    main()
