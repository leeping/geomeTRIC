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


def _inc_commadash(arg):
    """
    '27-31,88-91,100,136-139' -> '28-32,89-92,101,137-140'
    """
    rep = []
    for a in (x.split("-") for x in arg.split(",")):
        if len(a) == 1:
            rep.append(str(int(a[0]) + 1))
        elif len(a) == 2:
            rep.append(str(int(a[0]) + 1) + "-" + str(int(a[1]) + 1))
        else:
            raise KeyError("Uncommadash exception, found incorrectly formated arg: %s" % a)
    return ",".join(rep)


def make_constraints_string(constraints_dict):
    """ Convert the new constraints dict format into the original string format """
    key_numbers = {"freeze": 0, "set": 1, "scan": 3}
    spec_numbers = {"xyz": 1, "distance": 2, "angle": 3, "dihedral": 4}

    constraints_string = []

    # Parse overall constraints
    for key, value_list in constraints_dict.items():

        if key not in key_numbers:
            raise KeyError("constraints key %s is not recognized" % key)
        key_args = key_numbers[key]

        # Parse individual constraints within a key
        constraints_string.append("$" + key)
        for spec_tuple in value_list:

            # If it is a string nothing to be done
            if isinstance(spec_tuple, str):
                constraints_string.append(spec_tuple)
                continue

            # Figure out total args
            if spec_tuple[0] not in spec_numbers:
                raise KeyError("constraint type %s is not yet supported % primary")

            rep = [spec_tuple[0]]
            spec_args = spec_numbers[spec_tuple[0]]
            total_args = 1 + key_args + spec_args
            if len(spec_tuple) != total_args:
                raise KeyError("Length of tuple should be %d not %d, %s" % (total_args, len(spec_tuple), spec_tuple))

            # We need to increment these by 1
            for arg in spec_tuple[1:(spec_args + 1)]:

                # Parse '27-31,88-91,100,136-139' and add one
                if isinstance(arg, str) and (("-" in arg) or ("," in arg)):
                    rep.append(_inc_commadash(arg))
                else:
                    rep.append(str(int(arg) + 1))

            # Append the set or scan values
            for arg in spec_tuple[spec_args + 1:]:
                rep.append(str(arg))

            constraints_string.append(" ".join(rep))

    return "\n".join(constraints_string)


def geometric_run_json(in_json_dict):
    """ Take a input dictionary loaded from json, and return an output dictionary for json """

    input_opts = parse_input_json_dict(in_json_dict)
    M, engine = geometric.optimize.get_molecule_engine(**input_opts)

    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr

    # Read in the constraints
    constraints_dict = input_opts.get('constraints', {})
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
