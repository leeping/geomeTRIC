"""
Tests the geomeTRIC molecule class.
"""

import pytest
import geometric


def test_blank_molecule():
    mol = geometric.molecule.Molecule()

    assert len(mol) == 0
