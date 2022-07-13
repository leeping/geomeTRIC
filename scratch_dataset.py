from geometric.molecule import Molecule as geoM
import numpy as np
from qcelemental.models import Molecule as qcel
from qcfractal import FractalSnowflake
from geometric.nifty import bohr2ang

M = geoM('ts.xyz')
s = FractalSnowflake()
client = s.client()

mol1 = qcel(symbols=["H", "H"], geometry=[0, 0, 0, 0, 0, 2]) # Unit is in Bohr
mol2 = qcel(symbols=["O", "O"], geometry=[0, 0, 0, 0, 0, 2])

sp_spec = {
    'program':"psi4",
    'driver':"energy",
    'method':"hf",
    'basis':"6-31g",
    'keywords':{"e_convergence": 1.0e-10, "d_convergence": 1.0e-10}
}

opt_spec = {
    'program':'geometric',
    'qc_specification':sp_spec,
    'keywords': {'transition':True},
}

#meta, ids = client.add_singlepoints([mol1, mol2], **sp_spec)
#s.await_results()
#
#
#result = client.get_singlepoints(ids)
#
#print(result) # This is a list with two SinglepointRecords
#print(result[0]) # This is an entire SinglepointRecord of H2
#print("\n Energy of H2",result[0].return_result) # Returning energy
#print("\n Energy of O2", result[1].return_result)

elmol = qcel(symbols=M.elem, geometry=np.array(M.xyzs) / bohr2ang)  # , molecular_charge = -1, molecular_multiplicity = 1)


_, ids = client.add_optimizations([elmol], **opt_spec)

s.await_results()
result = client.get_optimizations(ids)
print(result[0].stdout)
print(result[0].error)