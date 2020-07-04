#!/usr/bin/env python

import tempfile
import numpy as np
import geometric
import geometric.molecule
from . import addons

localizer = addons.in_folder

Bohr = 0.52917721

def model(coords):
    '''model Hamiltonian = sum_{AB} w_{AB} * (|r_A - r_B| - b_{AB})^2.
    All quantiles are in atomic unit
    '''
    dr = coords[:,None,:] - coords
    dist = np.linalg.norm(dr, axis=2)
    b = np.array([[0. , 1.8, 1.8,],
                  [1.8, 0. , 2.8,],
                  [1.8, 2.8, 0. ,]])
    w = np.array([[0. , 1.0, 1.0,],
                  [1.0, 0. , 0.5,],
                  [1.0, 0.5, 0. ,]])
    e = (w * (dist - b)**2).sum()

    grad = np.einsum('ij,ijx->ix', 2*w*(dist-b)/(dist+1e-60), dr)
    grad-= np.einsum('ij,ijx->jx', 2*w*(dist-b)/(dist+1e-60), dr)
    return e, grad

class CustomEngine(geometric.engine.Engine):
    def __init__(self, molecule):
        super(CustomEngine, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        energy, gradient = model(coords.reshape(-1,3))
        return {'energy': energy, 'gradient': gradient.ravel()}


def test_customengine(localizer):
    molecule = geometric.molecule.Molecule()
    molecule.elem = ['O', 'H', 'H']
    molecule.xyzs = [np.array((( 0. , 0.3, 0),
                               ( 0.9, 0.8, 0),
                               (-0.9, 0.5, 0),
                              ))  # In Angstrom
                    ]
    customengine = CustomEngine(molecule)

    tmpf = tempfile.mktemp()
    m = geometric.optimize.run_optimizer(customengine=customengine, check=1, input=tmpf)

    coords = m.xyzs[-1] / Bohr
    e = model(coords)[0]
    assert e < 1e-8

if __name__ == '__main__':
    test_customengine()
