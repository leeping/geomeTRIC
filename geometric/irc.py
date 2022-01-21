import os, sys, copy
import numpy as np
from .step import get_delta_prime
from .params import OptParams, parse_optimizer_args
from .internal import CartesianCoordinates, PrimitiveInternalCoordinates, DelocalizedInternalCoordinates
from .optimize import Optimizer
from .normal_modes import calc_cartesian_hessian
from .prepare import get_molecule_engine
from .nifty import ang2bohr
from .molecule import PeriodicTable

def main():
    args = parse_optimizer_args(sys.argv[1:])
    inputf = args.get('input')
    arg_prefix = args.get('prefix', None)
    prefix = arg_prefix if arg_prefix is not None else os.path.splitext(inputf)[0]
    dirname = prefix+".tmp"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    args['dirname'] = dirname


    M, engine = get_molecule_engine(**args)
    mass = np.array([PeriodicTable[i] for i in M.elem])
    invsqrtm3 = 1.0/np.sqrt(np.repeat(mass, 3))
    params = OptParams(**args)
    coords = M.xyzs[0].flatten()*ang2bohr
    mwcoords = coords/invsqrtm3[np.newaxis,:] 
    constraints = args.get('constraints', None)

    CoordSysDict = {'cart':(CartesianCoordinates, False, False),
                     'prim':(PrimitiveInternalCoordinates, True, False),
                     'dlc':(DelocalizedInternalCoordinates, True, False),
                     'hdlc':(DelocalizedInternalCoordinates, False, True),
                     'tric-p':(PrimitiveInternalCoordinates, False, False),
                     'tric':(DelocalizedInternalCoordinates, False, False)}

    coordsys = args.get('coordsys','cart')
    CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]

    params.xyzout = prefix+"_optim.xyz"
    params.trust = 0.005

    
        
    IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=None, cvals=None, conmethod=params.conmethod)
    Hx = calc_cartesian_hessian(coords, M, engine, dirname,read_data=True, verbose=params.verbose)
    opt = Optimizer(coords, M, IC, engine, dirname, params)
    opt.calcEnergyForce()
    print('gradx', opt.gradx)

    M = opt.progress 

    freq, normal, G = opt.frequency_analysis(Hx,None,False) 
    mw = normal/invsqrtm3[np.newaxis,:] # mass weighted cartesian displacement 
    mwnormal = mw/np.linalg.norm(mw, axis=1)[:, np.newaxis] # normalized mw cartesian displacement
    print('\nafter normalization again',mwnormal[0])
    disp = np.linalg.norm(mwnormal[0])
    if disp > 0.02:
        mwnormal[0] *= 0.5
    new_mwcoord = mwcoords + mwnormal[0]
    new_coords = (new_mwcoord*invsqrtm3).reshape(-1,3)
    new_M = copy.deepcopy(M)
    print('new coords',new_coords)
    new_M.xyzs = [np.array(new_coords/ang2bohr).reshape(-1,3)]
    coords = new_M.xyzs[0].flatten()*ang2bohr 
    print('new_M.qm_grads', new_M.qm_grads)
    print('new_M.xyzs', new_M.xyzs)
    IC = CoordClass(new_M, build=True, connect=connect, addcart=addcart, constraints=None, cvals=None, conmethod=params.conmethod)
    opt = Optimizer(coords, new_M, IC ,engine, dirname, params) 
    opt.optimizeGeometry()
    
    M_fwd = M + opt.progress 
    #M.xyzs *= invsqrtm3[np.newaxis,:]

    new_coords2 = ((mwcoords - mwnormal[0])*invsqrtm3).reshape(-1,3)
    new_M = copy.deepcopy(M)
    new_M.xyzs = [np.array(new_coords2/ang2bohr)]
    coords = new_M.xyzs[0].flatten()*ang2bohr 

    IC = CoordClass(new_M, build=True, connect=connect, addcart=addcart, constraints=None, cvals=None, conmethod=params.conmethod)
    opt = Optimizer(coords, new_M, IC ,engine, dirname, params) 
    opt.optimizeGeometry()
    M_bkw = opt.progress[::-1]

    M_final = M_bkw + M + M_fwd
    M_final.write('test.xyz')
    np.savetxt('Energies.txt',M_final.qm_energies)
    

   # spcalc = engine.calc()
   # E=spcalc['energy']
   # G=spcalc['gradient']
   # dy = get_delta_prime

if __name__ == '__main__':
    main()

