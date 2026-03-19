"""
Molpro engine test
"""

from . import addons
import geometric
import os

datad = addons.datad

def test_molpro_parse_mp2_energy():
    """
    Testing to see if the engine parses MP2 energy correctly.
    """
    Molpro_engine = geometric.engine.Molpro()
    energy = Molpro_engine.read_result(os.path.join(datad, "molpro_mp2"))['energy']
    assert energy == -481.656215543708


