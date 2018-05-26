#!/usr/bin/env python

import json
import geometric

def get_qc_schema_traj(qc_schema_input, progress):
    qc_schema_traj = []

    return qc_schema_traj

def parse_json_input(in_json):
    """
    Parse an input json file into options, example:
    {
        "schema_name": "qc_schema_optimization_input",
        "schema_version", 1,
        "geometric_options": {
            "coordsys": "tric",
            "conv": 1.e-7
        }
        "input_specification": qc_schema_input,
    }

    qc_schema_input = {
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
    """
    d = json.load(open(in_json))
    input_opts = d['geometric_options']
    # insert `fix_orientation` and `fix_com`
    d['input_specification'].update({
        'fix_orientation': True,
        'fix_com': True
    })
    # Here we force the use of qcengine because other engines don't support qc schema
    input_opts.update({
        'qcengine': True,
        'qcschema': d['input_specification']
    })
    return input_opts

def write_json_output(in_json, schema_traj, out_json):
    # copy the input json data
    out_json_data = json.load(open(in_json))
    out_json_data.update({
        "trajectory": schema_traj,
        "energies": [s['properties']['return_energy'] for s in schema_traj],
        "final_molecule": schema_traj[-1]
    })
    json.dump(out_json_data, open(out_json, 'w'), indent=2)

def geometric_run_json(in_json, out_json):
    input_opts = parse_json_input(in_json)
    M, engine = geometric.optimize.get_molecule_engine(**input_opts)
    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr
    # Read in the constraints
    constraints = input_opts.get('constraints', None)
    if constraints is not None:
        Cons, CVals = geometric.optimize.ParseConstraints(M, constraints)
    else:
        Cons, CVals = None, None
    # set up the internal coordinate system
    coordsys = input_opts.get('coordsys', 'tric')
    CoordSysDict = {'cart':(geometric.internal.CartesianCoordinates, False, False),
                    'prim':(geometric.internal.PrimitiveInternalCoordinates, True, False),
                    'dlc':(geometric.internal.DelocalizedInternalCoordinates, True, False),
                    'hdlc':(geometric.internal.DelocalizedInternalCoordinates, False, True),
                    'tric':(geometric.internal.DelocalizedInternalCoordinates, False, False)}
    CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]
    IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVals[0] if CVals is not None else None)
    # Print out information about the coordinate system
    if isinstance(IC, geometric.internal.CartesianCoordinates):
        print("%i Cartesian coordinates being used" % (3*M.na))
    else:
        print("%i internal coordinates being used (instead of %i Cartesians)" % (len(IC.Internals), 3*M.na))
    print(IC)

    dirname = 'opt_tmp'
    params = geometric.optimize.OptParams(**input_opts)

    # Run the optimization
    if Cons is None:
        # Run a standard geometry optimization
        geometric.optimize.Optimize(coords, M, IC, engine, dirname, params)
    else:
        # Run a constrained geometry optimization
        if isinstance(IC, (geometric.internal.CartesianCoordinates, geometric.internal.PrimitiveInternalCoordinates)):
            raise RuntimeError("Constraints only work with delocalized internal coordinates")
        for ic, CVal in enumerate(CVals):
            if len(CVals) > 1:
                print("---=== Scan %i/%i : Constrained Optimization ===---" % (ic+1, len(CVals)))
            IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, cvals=CVal)
            IC.printConstraints(coords, thre=-1)
            geometric.optimize.Optimize(coords, M, IC, engine, dirname, params)

    write_json_output(in_json, engine.schema_traj, out_json)

def main():
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_json', help='Input json file name')
    parser.add_argument('-o', '--out_json', default='out.json', help='Output Json file name')
    args = parser.parse_args()
    print(' '.join(sys.argv))
    geometric_run_json(args.in_json, args.out_json)

if __name__ == '__main__':
    main()
