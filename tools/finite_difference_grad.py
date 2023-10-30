#!/usr/bin/env python

import os, sys, shutil, argparse
import numpy as np
from logging import *
from geometric.prepare import get_molecule_engine
from geometric.nifty import createWorkQueue, getWorkQueue, ang2bohr, bohr2ang, wq_wait, RawStreamHandler

logger = getLogger("FDLogger")
logger.setLevel(INFO)
handler = RawStreamHandler()
logger.addHandler(handler)

def displace_calc(coords, i, h, multiple, engine, dirname, wq=False, read=False):
    displace = h*multiple
    coords[i] += displace
    dirname_d = os.path.join(dirname, "displace/%%0%ii%%+i" % len(str(len(coords))) % (i, multiple))
    result = 0.0
    if wq:
        if read:
            result = engine.read_wq(coords, dirname_d)['energy']
        else:
            engine.calc_wq(coords, dirname_d)
    else:
        if read:
            raise RuntimeError("read can't be True if wq is False")
        result = engine.calc(coords, dirname_d)['energy']
    coords[i] -= displace
    return result

def stencil_calc(coords, i, h, stencil, engine, dirname, ezero = 0.0, wq=False, read=False):
    results = {0 : ezero}
    if stencil >= 2:
        results[1] = displace_calc(coords, i, h, 1, engine, dirname, wq=wq, read=read)
    if stencil >= 3:
        results[-1] = displace_calc(coords, i, h, -1, engine, dirname, wq=wq, read=read)
    if stencil >= 5:
        results[2] = displace_calc(coords, i, h, 2, engine, dirname, wq=wq, read=read)
        results[-2] = displace_calc(coords, i, h, -2, engine, dirname, wq=wq, read=read)
    if stencil >= 7:
        results[3] = displace_calc(coords, i, h, 3, engine, dirname, wq=wq, read=read)
        results[-3] = displace_calc(coords, i, h, -3, engine, dirname, wq=wq, read=read)
    if stencil == 2:
        G_fd_i = (results[1] - results[0])/h
    elif stencil == 3:
        G_fd_i = (results[1] - results[-1])/(2*h)
    elif stencil == 5:
        G_fd_i = np.dot(np.array([1, -8, 8, -1]), np.array([results[i] for i in [-2, -1, 1, 2]]))/(12*h)
    elif stencil == 7:
        G_fd_i = np.dot(np.array([-1, 9, -45, 45, -9, 1]), np.array([results[i] for i in [-3, -2, -1, 1, 2, 3]]))/(60*h)
    else:
        raise RuntimeError("Only valid values of stencil are 2, 3, 5, 7")
    return G_fd_i

def finite_difference_gradient(coords, molecule, engine, dirname, h=1e-3, stencil=3, verbose=0):
    """ 
    Calculate the Cartesian gradient using finite difference and compare with the
    analytic gradient.  This isn't the most efficient way because geomeTRIC only
    knows how to request gradients from the engine, and could be improved in the future.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of Cartesian coordinates in atomic units
    molecule : Molecule
        Molecule object
    engine : Engine
        Object containing methods for calculating energy and gradient
    dirname : str
        Directory name for files to be written, i.e. <prefix>.tmp
        
    """
    nc = len(coords)

    molecule.xyzs[0] = coords.reshape(-1, 3)*bohr2ang

    if os.path.exists(os.path.join(dirname, "displace")):
        shutil.rmtree(os.path.join(dirname, "displace"))
        
    # Calculate gradient using finite difference
    # Finite difference step
    wq = getWorkQueue()
    G_fd = np.zeros(nc, dtype=float)
    logger.info("Calculating gradient using finite difference on energies (%i grads total)\n" % ((stencil-stencil%2)*nc))
    if wq:
        engine.calc_wq(coords, dirname)
        for i in range(nc):
            if verbose >= 2: logger.info(" Submitting calculation for coordinate %i/%i\n" % (i+1, nc))
            stencil_calc(coords, i, h, stencil, engine, dirname, wq=True, read=False)
        # Wait for WQ calculations to finish.
        wq_wait(wq, print_time=60, verbose=verbose)
        # Read WQ results.
        central = engine.read_wq(coords, dirname)
        G_ana = central['gradient']
        ezero = central['energy']
        for i in range(nc):
            if verbose >= 2: logger.info(" Reading results for coordinate %i/%i\n" % (i+1, nc))
            G_fd[i] = stencil_calc(coords, i, h, stencil, engine, dirname, ezero=ezero, wq=True, read=True)
    else:
        central = engine.calc(coords, dirname)
        G_ana = central['gradient']
        ezero = central['energy']
        for i in range(nc):
            if verbose >= 1: logger.info(" Running calculation for coordinate %i/%i\n" % (i+1, nc))
            elif i%5 == 0: logger.info("%i / %i calculations complete\n" % (i*2, nc*2))
            G_fd[i] = stencil_calc(coords, i, h, stencil, engine, dirname, ezero=ezero, wq=False, read=False)
            if i == (nc-1) : logger.info("%i / %i calculations complete\n" % (nc*2, nc*2))

    # Delete displacement calcs because they take up too much space
    keep_displace = True
    if not keep_displace:
        if os.path.exists(os.path.join(dirname, "displace")):
            shutil.rmtree(os.path.join(dirname, "displace"))
            
    logger.info("#Coord       FDiff         Ana         Diff\n")
    for i in range(nc):
        logger.info("%3i %2s % 11.8f % 11.8f % 11.5e\n" % (i//3, 'xyz'[i%3], G_fd[i], G_ana[i], G_fd[i] - G_ana[i]))

def parse_fd_args(*args):

    """
    Read user input from the command line interface.
    Designed to be called by finite_difference_grad.main() passing in sys.argv[1:]
    """

    parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter, fromfile_prefix_chars='@')

    grp_univ = parser.add_argument_group('universal', 'Relevant to every job')
    grp_univ.add_argument('input', type=str, help='REQUIRED positional argument: input file for the QC program.\n')
    grp_univ.add_argument('engine', type=str, help='REQUIRED positional argument: name of the QC program (terachem, psi4).\n')
    grp_univ.add_argument('--step', type=float, default=1e-3, help='Finite difference step size (in a.u.).\n')
    grp_univ.add_argument('--stencil', type=int, default=3, help='Number of steps in the finite difference stencil. Valid values: 3, 5\n')
    grp_univ.add_argument('--port', type=int, help='Work Queue port number (optional).\n')
    grp_univ.add_argument('--verbose', type=bool, default=False, help='Print extra information.\n')

    grp_help = parser.add_argument_group('help', 'Get help')
    grp_help.add_argument('-h', '--help', action='help', help='Show this help message and exit')

    args_dict = {}
    for k, v in vars(parser.parse_args(*args)).items():
        if v is not None:
            args_dict[k] = v

    return args_dict

def main():
    args = parse_fd_args(sys.argv[1:])
    molecule, engine = get_molecule_engine(engine=args['engine'], input=args['input'], dirname='fdgrad.tmp')
    if 'port' in args:
        createWorkQueue(args['port'], debug=False)
    coords = molecule.xyzs[0].flatten() * ang2bohr
    finite_difference_gradient(coords, molecule, engine, 'fdgrad.tmp', h=args['step'], stencil=args['stencil'], verbose=args['verbose'])

if __name__ == '__main__':
    main()
