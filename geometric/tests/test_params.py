"""
A set of tests for parsing inputs
"""

import json, os, shutil
from . import addons
import geometric

localizer = addons.in_folder
datad = addons.datad

def test_parse_command_line_opt_args(localizer):
    shutil.copy2(os.path.join(datad, 'hcn_tsguess.psi4in'), os.getcwd())
    shutil.copy2(os.path.join(datad, 'parser_options.txt'), os.getcwd())
    args = geometric.params.parse_optimizer_args('--engine psi4 --reset true --transition no hcn_tsguess.psi4in parser_options.txt'.split())
    assert args['engine'] == 'psi4'
    assert args['input'] == 'hcn_tsguess.psi4in'
    assert args['epsilon'] == 1e-6
    assert args['constraints'] == 'parser_options.txt'
    assert args['radii'] == ['O', '0.5']
    assert args['transition'] == False
    assert args['reset'] == True

def test_parse_command_line_neb_args(localizer):
    shutil.copy2(os.path.join(datad, 'hcn_neb_input.psi4in'), os.getcwd())
    shutil.copy2(os.path.join(datad, 'hcn_neb_input.xyz'), os.getcwd())
    args = geometric.params.parse_neb_args('--engine psi4 --nebk 10.3 --plain 3 --images 7 --optep yes hcn_neb_input.psi4in hcn_neb_input.xyz'.split())
    assert args['engine'] == 'psi4'
    assert args['input'] == 'hcn_neb_input.psi4in'
    assert args['chain_coords'] == 'hcn_neb_input.xyz'
    assert args['optep'] == True
    assert args['images'] == 7
    assert args['nebk'] == 10.3