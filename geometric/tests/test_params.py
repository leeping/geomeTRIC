"""
A set of tests for parsing inputs
"""

import copy
import numpy as np
import json, os, shutil
from . import addons
import geometric
import pytest
import itertools

localizer = addons.in_folder
datad = addons.datad

def test_parse_command_line_args(localizer):
    shutil.copy2(os.path.join(datad, 'hcn.psi4in'), os.getcwd())
    shutil.copy2(os.path.join(datad, 'parser_options.txt'), os.getcwd())
    args = geometric.params.parse_optimizer_args('--engine psi4 --reset true --transition no hcn.psi4in parser_options.txt'.split())
    assert args['engine'] == 'psi4'
    assert args['input'] == 'hcn.psi4in'
    assert args['epsilon'] == 1e-6
    assert args['constraints'] == 'parser_options.txt'
    assert args['radii'] == ['O', '0.5']
    assert args['transition'] == False
    assert args['reset'] == True
