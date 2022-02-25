import os, sys, copy
import numpy as np
from .params import OptParams, parse_optimizer_args
from .internal import unit_vector, CartesianCoordinates, PrimitiveInternalCoordinates, DelocalizedInternalCoordinates
from .optimize import Optimizer
from .normal_modes import calc_cartesian_hessian
from .prepare import get_molecule_engine
from .nifty import ang2bohr
from .molecule import Molecule, PeriodicTable



def irc(M, engine, coords, IC, dirname, params, initial_disp=[], direction=1):
    if direction not in [-1, 1]:
        raise RuntimeError('direction for IRC should be either 1 or -1')
    
    initial_disp = np.array(initial_disp) 
    mass = np.repeat(np.array([PeriodicTable[i] for i in M.elem]), 3)
    sqrtm3 = np.sqrt(mass)
    iteration = 0
    acc_norm = 1
    E_diff = 1
    abs_E_diff = np.abs(E_diff)
    displacement = 1
    progress = Molecule()
    while abs_E_diff > 0.0001 or displacement > params.trust/2 or acc_norm > 0.005:
        print('\n---------------Itreation %i--------------' %iteration)
        if iteration > 100:
            print('IRC iteration went over 100. Completing the IRC calculation.')
            break
        mwcoords = coords*sqrtm3
        opt = Optimizer(coords, M, IC, engine, dirname, params) #Dirname
        opt.calcEnergyForce()   
        E1 = opt.E
        if iteration ==0 :
            progress = opt.progress[-1]
            print('Trust radius', params.trust)
            print('Maximum trust radius', params.tmax)
            if initial_disp.size==0:
                Hx = calc_cartesian_hessian(coords, M, engine, dirname, read_data=True, verbose=params.verbose)#Dirname
                freq, initial_disp, G=opt.frequency_analysis(Hx, 'iter%03i' % opt.Iteration, False)
                print('Imaginary Wavenumber:',freq[0])
            max_disp =  np.max(np.abs(initial_disp[0]))
            max_ind = np.where(np.abs(initial_disp[0])==max_disp)[0]
            
            mwcart = initial_disp[0]*sqrtm3            

            disp_vec = unit_vector(mwcart)
            max_vec = np.zeros_like(initial_disp[0]) 
            max_vec[max_ind] += initial_disp[0][max_ind]
            max_vec = unit_vector(max_vec)*direction
            init_uvec = unit_vector(disp_vec)*direction
            init_uvec_3 = unit_vector(np.mean(init_uvec.reshape(-1,3), axis=0))
            #print('init_uvec',init_uvec)
            max_vec_3 = np.mean(max_vec.reshape(-1,3), axis=0)
            max_uvec_3 = unit_vector(max_vec_3)
            #print('max_uvec_3', max_uvec_3)
            #print('init_uvec_3', init_uvec_3)
            init_max_vec = max_vec
            #print('init_max_vec', init_max_vec)
        
        acc = opt.gradx
        acc_norm = np.linalg.norm(acc)
        print('Gradients norm:',acc_norm)
        acc_uvec = acc/acc_norm
        #print('\nacc_uvec', acc_uvec)
        acc_vec_3 = np.mean(acc_uvec.reshape(-1,3), axis=0)
        #print('acc_vec_3', acc_vec_3)
        acc_uvec_3 = unit_vector(acc_vec_3)
        #print('acc_uvec_3', acc_uvec_3)
        dot_prod = np.dot(init_uvec_3, acc_uvec_3)
        print('Dot product:',dot_prod)
        #print('dot product', dot_prod)

        if dot_prod < 0 and iteration < 10:
            uvec = init_max_vec*2
            print('Initial 1D vec chosen instead of gradients')
        elif iteration == 0:
            uvec = init_uvec*2
        else:
            uvec = acc_uvec
            #print('Gradients chosen')
            
        shift = 0.5*params.trust*uvec

        pivot_coord = mwcoords - shift 
        M.xyzs[0] = (pivot_coord/sqrtm3).reshape(-1,3)/ang2bohr
        opt.X = pivot_coord /sqrtm3
        opt.coords = pivot_coord / sqrtm3
        opt.molecule = M
        opt.calcEnergyForce() # get E and grad
        opt.prepareFirstStep() # get Hessian
        opt.step()
        opt.calcEnergyForce()#self.profress saved
        opt_result = opt.progress[-1]

        E2 = opt.E #opt_result.qm_energies[-1]
        E_diff = E1-E2
        abs_E_diff = np.abs(E_diff)
        print('Energy:', E2)
        print('Energy Change:', E_diff)
        if E_diff < 0 and abs_E_diff > 0.0001:
            print('WARNING, Energy is increasing! iteration: %i' %iteration)
            if iteration >20:
                print('Displacement is heading towards a wrong direction. Ending IRC.')
                break
        new_coords = opt_result.xyzs[0].flatten()*ang2bohr
        new_M = opt_result
        progress += new_M
        displacement = np.linalg.norm(new_coords-coords)
        print('Cartesian Displacement:', displacement)
        M = new_M
        coords = new_coords 
        
        iteration += 1
    return progress, initial_disp
    

def main():
    args = parse_optimizer_args(sys.argv[1:])
    args['irc'] = True

    if args.get('trust') >= 0.3:
        args['tmax'] = args.get('trust')*1.01
        
    inputf = args.get('input')
    arg_prefix = args.get('prefix', None)
    prefix = arg_prefix if arg_prefix is not None else os.path.splitext(inputf)[0]
    dirname = prefix+".tmp"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    args['dirname'] = dirname

    
    M, engine = get_molecule_engine(**args)
    params = OptParams(**args)

    params.tmax = params.trust*1.01
    coords = M.xyzs[0].flatten()*ang2bohr
    constraints = args.get('constraints', None)

    CoordSysDict = {'cart':(CartesianCoordinates, False, False),
                     'prim':(PrimitiveInternalCoordinates, True, False),
                     'dlc':(DelocalizedInternalCoordinates, True, False),
                     'hdlc':(DelocalizedInternalCoordinates, False, True),
                     'tric-p':(PrimitiveInternalCoordinates, False, False),
                     'tric':(DelocalizedInternalCoordinates, False, False)}

    coordsys = args.get('coordsys','cart')
    CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]

    params.xyzout = prefix+"_IRC.xyz"

    IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=None, cvals=None, conmethod=params.conmethod)



    fwd, disp= irc(M, engine, coords, IC, dirname, params, direction = -1)
    print('Forward IRC is done')
    fwd.write('forward.xyz')
    bwd, disp= irc(M, engine, coords, IC, dirname, params, initial_disp=disp, direction = 1)
    print('\nBackward IRC is done')
    bwd.write('backward.xyz')
    final = bwd[::-1] + fwd[1:]
    final.write('IRC_%.2f.xyz' %params.trust)
    print('\n IRC calculations are done. \'IRC_%.2f.xyz\' was generated.' %params.trust)
    return final
    
if __name__ == '__main__':
    main()

