"""
Unit and regression tests for geometric.errors module
"""

import pytest
import os, shutil

import geometric
from geometric.errors import EngineError, GeomOptNotConvergedError, LinearTorsionError
from geometric.errors import Psi4EngineError, QChemEngineError, TeraChemEngineError, ConicalIntersectionEngineError, \
    OpenMMEngineError, GromacsEngineError, MolproEngineError, QCEngineAPIEngineError

from . import addons
localizer = addons.in_folder
datad = addons.datad

def test_error_types():
    """ Test error types """
    with pytest.raises(Exception):
        raise EngineError
    with pytest.raises(Exception):
        raise GeomOptNotConvergedError
    with pytest.raises(EngineError):
        raise Psi4EngineError
    with pytest.raises(EngineError):
        raise QChemEngineError
    with pytest.raises(EngineError):
        raise TeraChemEngineError
    with pytest.raises(EngineError):
        raise ConicalIntersectionEngineError
    with pytest.raises(EngineError):
        raise OpenMMEngineError
    with pytest.raises(EngineError):
        raise GromacsEngineError
    with pytest.raises(EngineError):
        raise MolproEngineError
    with pytest.raises(EngineError):
        raise QCEngineAPIEngineError

@addons.using_psi4
def test_reference_no_error(localizer):
    """ Test simple optimization with Psi4 """
    # setup an simple geometric optimization
    with open('water.in', 'w') as f:
        f.write('''
molecule {
0 1
O  -0.022933   0.13249    0.00000
H  -1.229594  -1.29574    0.00000
H   1.593571  -0.80708    0.00000
}
set {
basis sto-3g
}
gradient('hf')
'''
        )
    input_opts = {
        'coordsys': 'tric',
        'conv': 1.e-7,
        'engine' : 'psi4',
        'input': 'water.in'
    }
    M, engine = geometric.optimize.get_molecule_engine(**input_opts)
    IC = geometric.internal.DelocalizedInternalCoordinates(M, build=True)
    params = geometric.optimize.OptParams(**input_opts)
    # run the test optimization
    geometric.optimize.Optimize(M.xyzs[0].flatten(), M, IC, engine, 'tmp', params)

@addons.using_psi4
def test_engine_error(localizer):
    """ Test catching engine error with failed Psi4 """
    # modify the input file to make it fail
    with open('water.in', 'w') as f:
        f.write('''
molecule {
0 1
O  -0.022933   0.13249    0.00000
H  -1.229594  -1.29574    0.00000
H   1.593571  -0.80708    0.00000
}
set {
basis sto-3g
maxiter 2 # this will fail
}
gradient('hf')
'''
        )
    input_opts = {
        'coordsys': 'tric',
        'conv': 1.e-7,
        'engine': 'psi4',
        'input': 'water.in'
    }
    M, engine = geometric.optimize.get_molecule_engine(**input_opts)
    IC = geometric.internal.DelocalizedInternalCoordinates(M, build=True)
    params = geometric.optimize.OptParams(**input_opts)
    # EngineError
    with pytest.raises(EngineError):
        # run the test optimization
        geometric.optimize.Optimize(M.xyzs[0].flatten(), M, IC, engine, 'tmp', params)

@addons.using_psi4
def test_optimizer_not_converge_error(localizer):
    """ Test catching GeomOptNotConvergedError """
    # setup an simple geometric optimization
    with open('water.in', 'w') as f:
        f.write('''
molecule {
0 1
O  -0.02   0.13    0.00000
H  -1.22  -1.29    0.00000
H   1.59  -0.80    0.00000
}
set {
basis sto-3g
}
gradient('hf')
'''
        )
    input_opts = {
        'coordsys': 'tric',
        'conv': 1.e-7,
        'engine': 'psi4',
        'input': 'water.in',
        'maxiter': 1, # this will cause GeomOptNotConvergedError
    }
    M, engine = geometric.optimize.get_molecule_engine(**input_opts)
    IC = geometric.internal.DelocalizedInternalCoordinates(M, build=True)
    params = geometric.optimize.OptParams(**input_opts)
    # GeomOptNotConvergedError
    with pytest.raises(GeomOptNotConvergedError):
        # run the test optimization
        geometric.optimize.Optimize(M.xyzs[0].flatten(), M, IC, engine, 'tmp', params)

@addons.using_psi4
def test_linear_torsion_error(localizer):
    shutil.copy2(os.path.join(datad, 'linang.psi4in'), os.getcwd())
    shutil.copy2(os.path.join(datad, 'linalg_torsion_constraints.txt'), os.getcwd())
    with pytest.raises(LinearTorsionError):
        geometric.optimize.run_optimizer(engine='psi4', input='linang.psi4in', constraints='linalg_torsion_constraints.txt')
