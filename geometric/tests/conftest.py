import logging
import _pytest.logging
import pytest
import geometric
import os, tempfile
from . import addons 

# 2022-09-22: This modification allows live and captured logs to be 
# formatted in the expected way when running tests using pytest. 
# This is because _pytest.LiveLoggingStreamHandler is a subclass of 
# logging.StreamHandler and is used instead of geomeTRIC's own loggers
# during test runs.
# It took a long time to figure out and I'm glad it only took one line to fix it!

logging.StreamHandler.terminator = ""

datad = addons.datad

@pytest.fixture
def molecule_engine_hcn_neb():
    """Return the Molecule and Engine for an NEB Calculation."""
    input_ext = {'psi4': 'psi4in', 'qchem': 'qcin', 'tera': 'tcin'}
    def get_molecule_engine(engine: str, images: int):
        
        return geometric.prepare.get_molecule_engine(
            input=os.path.join(datad, "hcn_neb_input.%s" %input_ext.get(engine)),
            chain_coords=os.path.join(datad, "hcn_neb_input.xyz"),
            images=images,
            neb=True,
            engine=engine,
    )

    return get_molecule_engine

@pytest.fixture
def bigchem_frequency():
    """Return frequency calculation results carried by BigChem"""
    def get_freqs(engine: str, input: str):
        molecule, engine = geometric.prepare.get_molecule_engine(engine=engine, input=os.path.join(datad, input))
        coords = molecule.xyzs[0].flatten()*geometric.nifty.ang2bohr
        hessian = geometric.normal_modes.calc_cartesian_hessian(coords, molecule, engine, tempfile.mkdtemp(), bigchem=True)
        freqs, modes, G = geometric.normal_modes.frequency_analysis(coords, hessian, elem=molecule.elem, verbose=True)
        return freqs

    return get_freqs
