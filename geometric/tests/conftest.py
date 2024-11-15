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
def molecule_engine_hcn():
    """Return the Molecule and Engine for NEB and IRC tests."""
    from typing import Optional
    input_ext = {'psi4': 'psi4in', 'qchem': 'qcin', 'tera': 'tcin', 'gaussian':'gjf'}
    def get_molecule_engine(engine: str, images: Optional[int]=None):
        if images:
            return geometric.prepare.get_molecule_engine(
                input=os.path.join(datad, "hcn_neb_input.%s" %input_ext.get(engine)),
                chain_coords=os.path.join(datad, "hcn_neb_input.xyz"),
                images=images,
                neb=True,
                engine=engine,
                )
        else:
            param_kwargs = {'engine':engine,
                            'input':os.path.join(datad, "hcn_irc_input.%s" %input_ext.get(engine)),
                            'converge':['set', 'GAU_LOOSE'],
                            'reset':False,
                            'trust':0.05, 'irc':True, 'xyzout':'test_irc.xyz'}
            params = geometric.params.OptParams(**param_kwargs)
            M, engine = geometric.prepare.get_molecule_engine(**param_kwargs)

            # Assuming the coordsys is tric
            IC = geometric.internal.DelocalizedInternalCoordinates(M, build=True, connect=False, addcart=False, constraints=None,
                            cvals=None,
                            conmethod=0, rigid=False)

            return M, IC, engine, params

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
