import tempfile
import numpy as np
from copy import deepcopy
from .prepare import get_molecule_engine
from .molecule import Molecule
from .nifty import bohr2ang, logger
from .neb import ElasticBand, print_forces, BFGSUpdate, compare, takestep, recover, converged, qualitycheck
from .params import NEBParams
from .engine import Blank


class nullengine(object):
    """
    Fake engine for QCFractal
    """

    def __init__(self, M):
        self.M = M


def add_attr(chain, attrs):
    """
    Add chain attributes to a given chain
    """
    chain.TotBandEnergy = attrs.get("TotBandEnergy")
    if attrs.get("climbSet", False):
        chain.climbSet = True
        chain.climbers = attrs.get("climbers")
        chain.locks = attrs.get("locks")
    return chain


def check_attr(chain):
    """
    Check a chain's attributes and extract them
    """
    attrs = {}
    if chain.climbSet:
        attrs["climbSet"] = True
        attrs["climbers"] = [int(i) for i in chain.climbers]
        attrs["locks"] = chain.locks
    attrs["TotBandEnergy"] = chain.TotBandEnergy

    return attrs


def chaintocoords(chain, ang=False):
    """
    Extracts Cartesian coordinates from an ElasticBand object
    Parameters
    ----------
    chain: ElasticBand object (in Ang)
    ang: Bool
        True will return Cartesian coordinates in Ang (False in Bohr)

    Returns
    -------
    newcoords: list
        Cartesian coordinates in list. It is not numpy array because neb record socket in QCF can't process nparray.
    """
    newcoords = []
    for i in range(len(chain)):
        M_obj = chain.Structures[i].M
        if ang:
            coord = M_obj.xyzs[0]
        else:
            coord = M_obj.xyzs[0] / bohr2ang
        newcoords.append(coord.tolist())
    return newcoords


def arrange(qcel_mols, align):
    """
    This function will respace a chain if images are too close to each other.
    Parameters
    ----------
    qcel_mols: [QCElemental Molecule object]
        QCElemental Molecule objects in a list.

    align: bool
        "True" will align the images.

    Returns
    -------
    respaced_chain: [QCElemental Molecule object]
        list of molecule objects
    """
    from qcelemental.models import Molecule as qcmol

    # Getting molecular information
    sym = qcel_mols[0].symbols.tolist()
    chg = qcel_mols[0].molecular_charge
    mult = qcel_mols[0].molecular_multiplicity
    M = Molecule()
    M.elem = sym
    M.xyzs = [mol.geometry*bohr2ang for mol in qcel_mols]
    M.build_topology()

    if align:
        logger.info("Aligning images \n")
        M.align()

    # Getting parameters and the chain
    neb_param = NEBParams()
    chain = ElasticBand(
        M, engine=None, tmpdir="tmp", params=neb_param, plain=0
    )

    # Respacing the chain
    chain.respace(0.01)
    chain.delete_insert(1.0)
    newcoords = chaintocoords(chain)

    respaced_chain = [qcmol(symbols=sym, geometry=coords, molecular_charge=chg, molecular_multiplicity=mult)
                      for coords in newcoords]

    return respaced_chain


def get_basic_info(info_dict, previous=False):
    """
    Extracting parameters and other basic objects from info_dict.
    """
    coords_bohr = np.array(info_dict.get("geometry"))
    coords_ang = coords_bohr * bohr2ang
    args_dict = info_dict.get("params")
    energies = info_dict.get("energies")
    gradients = info_dict.get("gradients")

    params_dict = {"images": args_dict.get("images"), "maxg": args_dict.get("maximum_force"),
        "avgg": args_dict.get("average_force"), "nebk": args_dict.get("spring_constant"),
        "neb_maxcyc": args_dict.get("maximum_cycle"), "plain": args_dict.get("spring_type"),
        "epsilon": args_dict.get("epsilon")}
    iteration = args_dict.get("iteration")
    params = NEBParams(**params_dict)

    M = Molecule()
    M.elem = info_dict.get("elems")
    M.charge = info_dict.get("charge")
    M.mult = info_dict.get("mult")
    if previous:
        M.xyzs = [coords.reshape(-1, 3) for coords in np.array(info_dict.get("coord_ang_prev"))]
    else:
        M.xyzs = [coords.reshape(-1, 3) for coords in coords_ang]

    params.customengine = nullengine(M)
    M, engine = get_molecule_engine(**{"customengine": params.customengine})

    result = []
    for i in range(len(energies)):
        if previous:
            result = info_dict.get("result_prev")
        else:
            result.append({"energy": energies[i], "gradient": gradients[i]})

    return params, M, engine, result, iteration - 1


def prepare(info_dict):
    """
    This function is for QCFractal. It takes a dictionary and prepares for the NEB calculation loops.
    """

    logger.info("\n-=# Chain optimization cycle 0 #=- \n")
    params, M, engine, result, _ = get_basic_info(info_dict)

    logger.info("Spring Force: %.2f kcal/mol/Ang^2 \n" % params.nebk)

    tmpdir = tempfile.mkdtemp()

    # Getting the initial chain.
    chain = ElasticBand(M, engine=engine, tmpdir=tmpdir, params=params, plain=params.plain)

    trust = params.trust
    chain.ComputeChain(result=result)
    chain.ComputeGuessHessian(blank=isinstance(engine, Blank))
    chain.PrintStatus()

    avgg_print, maxg_print = print_forces(chain, params.avgg, params.maxg)
    logger.info("-= Chain Properties =- \n")
    logger.info(
        "@\n%13s %13s %13s %13s %11s %13s %13s \n"
        % ("GAvg(eV/Ang)", "GMax(eV/Ang)", "Length(Ang)", "DeltaE(kcal)", "RMSD(Ang)", "TrustRad(Ang)", "Step Quality")
    )
    logger.info(
        "@%13s %13s %13s \n"
        % (
            "   %s  " % avgg_print,
            "     %s  " % maxg_print,
            "% 8.4f  " % sum(chain.calc_spacings()),
        )
    )

    GW = chain.get_global_grad("total", "working")
    GP = chain.get_global_grad("total", "plain")
    oldcoords = chaintocoords(chain)
    attrs_new = check_attr(chain)
    attrs_prev = check_attr(chain)

    temp = {"Ys": [chain.get_internal_all().tolist()], "GWs": [GW.tolist()], "GPs": [GP.tolist()], "attrs_new": attrs_new,
        "attrs_prev": attrs_prev, "trust": trust, "expect": None, "expectG": None, "respaced": False,
        "trustprint": "=", "frocerebuild": False,"lastforce": 0, "coord_ang_prev": chaintocoords(chain, True),
        "result_prev": result, "geometry": []}
    info_dict.update(temp)
    return oldcoords, info_dict


def nextchain(info_dict):
    """
    Generate next chain's Cartesian coordinate for QCFractal.
    """

    # Extracting information from the given dictionary.
    params, M, engine, result, iteration = get_basic_info(info_dict)
    params_prev, M_prev, engine_prev, result_prev, _ = get_basic_info(
        info_dict, previous=True
    )

    ThreLQ = 0.0
    ThreHQ = 0.5
    ThreRJ = 0.001

    logger.info("\n-=# Chain optimization cycle %i #=-\n" % iteration)
    tmpdir = tempfile.mkdtemp()

    # Define two chain objects for the previous and current iteration.
    chain = ElasticBand(M, engine=engine, tmpdir=tmpdir, params=params, plain=params.plain)

    chain_prev = ElasticBand(M_prev, engine=engine, tmpdir=tmpdir,
                             params=params_prev, plain=params_prev.plain)

    # Getting other information up to the previous iteration.
    trust = info_dict.get("trust")
    trustprint = info_dict.get("trustprint", "=")
    respaced = info_dict.get("respaced")
    expect = info_dict.get("expect")
    expectG = info_dict.get("expectG")
    Quality = info_dict.get("quality")
    LastForce = info_dict.get("lastforce", 0)
    ForceBuild = info_dict.get("forcerebuild", False)
    chain = add_attr(chain, info_dict.get("attrs_new"))
    chain_prev = add_attr(chain_prev, info_dict.get("attrs_prev"))
    chain.ComputeGuessHessian(blank=isinstance(engine, Blank))
    chain_prev.ComputeGuessHessian(blank=isinstance(engine, Blank))
    GWs = info_dict.get("GWs")
    GPs = info_dict.get("GPs")
    Ys = info_dict.get("Ys")
    Y_prev = np.array(Ys[-1])
    GW_prev = np.array(GWs[-1])
    GP_prev = np.array(GPs[-1])

    # Calculating the chain gradients.
    chain.ComputeChain(result=result)

    # Initial guess of the Hessians.
    HW0 = chain.guess_hessian_working.copy()
    HP0 = chain.guess_hessian_plain.copy()

    # Current chain's gradients and coordinates.
    GW = chain.get_global_grad("total", "working")
    GP = chain.get_global_grad("total", "plain")
    Y = chain.get_internal_all()

    if iteration == 1:
        logger.info("Taking the first NEB step\n")
        dy, expect, expectG, ForceRebuild = chain.CalcInternalStep(trust, HW0, HP0)
        newchain = chain.TakeStep(dy)
        respaced = newchain.delete_insert(1.5)
        newcoords = chaintocoords(newchain)
        attrs_new = check_attr(newchain)
        attrs_prev = check_attr(chain)
        temp = {"Ys": [chain.get_internal_all().tolist()], "GWs": [GW.tolist()], "GPs": [GP.tolist()],
                "attrs_new": attrs_new, "attrs_prev": attrs_prev, "trust": trust, "expect": expect,
                "expectG": expectG.tolist(), "respaced": respaced, "trustprint": "=", "frocerebuild": False,
                "lastforce": 0, "coord_ang_prev": chaintocoords(chain, True), "result_prev": result, "geometry": []}
        info_dict.update(temp)
        return newcoords, info_dict


    # Building the Hessian up to the previous iteration.
    for i in range(len(Ys) - 1):
        BFGSUpdate(np.array(Ys[i + 1]), np.array(Ys[i]), np.array(GPs[i + 1]), np.array(GPs[i]), HP0, params)
        BFGSUpdate(np.array(Ys[i + 1]), np.array(Ys[i]), np.array(GWs[i + 1]), np.array(GWs[i]), HW0, params)

    # Saving the Hessians for special cases such as rejecting a step.
    HW_bak = deepcopy(HW0)
    HP_bak = deepcopy(HP0)

    # Updating the Hessian for the current iteration
    BFGSUpdate(Y, np.array(Ys[-1]), GP, np.array(GPs[-1]), HP0, params)
    BFGSUpdate(Y, np.array(Ys[-1]), GW, np.array(GWs[-1]), HW0, params)

    HW_prev = HW0
    HP_prev = HP0

    # 1) First, compare two chains and determine the quality of the next step.
    chain, Y, GW, GP, HW, HP, c_hist, Quality = compare(chain_prev, chain, ThreHQ, ThreLQ, GW_prev, HW_prev, HP_prev,
                                        respaced, iteration, expect, expectG, trust, trustprint, params.avgg, params.maxg, Quality)
    if respaced:
        # 1-1) If the chain was respaced, take a new step using the guessed Hessians.
        (chain_prev, chain, expect, expectG, ForceRebuild, LastForce, Y_prev, GW_prev, GP_prev, respaced, _) \
            = takestep([chain_prev], chain, iteration, LastForce, ForceBuild, trust, Y, GW, GP, HW, HP, result_prev)
        attrs_new = check_attr(chain)
        attrs_prev = check_attr(chain_prev)
        newcoords = chaintocoords(chain)
        temp = {"Ys": [Y_prev.tolist()], "GWs": [GW_prev.tolist()], "GPs": [GP_prev.tolist()], "attrs_new": attrs_new,
                "attrs_prev": attrs_prev, "expect": expect, "expectG": expectG.tolist(), "respaced": respaced,
                "forcerebuild": ForceRebuild, "lastforce": LastForce, "coord_ang_prev": chaintocoords(chain_prev, True),
                "quality": Quality, "result_prev": result, "geometry": []}
        info_dict.update(temp)
        return newcoords, info_dict

    # 2) If the chain is converged, return None to QCFractal. This will end an NEB service.
    if converged(
        chain.maxg, chain.avgg, params.avgg, params.maxg, iteration, params.maxcyc
    ):
        return None, {}

    # 2) Checking the quality and increase/decrease the stepsize
    chain, trust, trustprint, Y, GW, GP, good = qualitycheck(chain_prev, chain, trust, Quality, ThreLQ, ThreRJ, ThreHQ,
                                                            Y, GW, GP, Y_prev, GW_prev, GP_prev, params.tmax)

    if not good:
        # 2-1) If the quality is bad, reject the step and take a new step with a decreased stepsize.
        chain.ComputeChain(result=result_prev)
        (chain_prev, chain, expect, expectG, ForceRebuild, LastForce, Y_prev, GW_prev, GP_prev, respaced, _) \
        = takestep([chain_prev], chain, iteration, LastForce, ForceBuild, trust, Y, GW, GP, HW_bak, HP_bak, result_prev)
        attrs_new = check_attr(chain)
        attrs_prev = check_attr(chain_prev)
        newcoords = chaintocoords(chain)
        temp = {"Ys": Ys, "GWs": GWs, "GPs": GPs, "attrs_new": attrs_new, "attrs_prev": attrs_prev, "trust": trust,
            "trustprint": trustprint, "expect": expect, "expectG": expectG.tolist(), "quality": Quality, "respaced": respaced,
            "coord_ang_prev": chaintocoords(chain_prev, True), "lastforce": LastForce, "forcerebuild": ForceRebuild,
            "result_prev": result_prev, "geometry": []}
        info_dict.update(temp)
        return newcoords, info_dict

    # 3) Reset the Hessians in case the eigenvalues are too small.
    Eig = np.linalg.eigh(HW)[0]
    Eig.sort()
    if np.min(Eig) <= params.epsilon:
        if params.skip:
            logger.info(
                "Eigenvalues below %.4e (%.4e) - skipping Hessian update \n"
                % (params.epsilon, np.min(Eig))
            )
        else:
            logger.info(
                "Eigenvalues below %.4e (%.4e) - will reset the Hessian \n"
                % (params.epsilon, np.min(Eig))
            )

            chain, Y, GW, GP, HW, HP = recover([chain_prev], result_prev)
            Ys = []
            GWs = []
            GPs = []
            result = result_prev

    # 4) Take the step based on the current Hessians and gradients. Pass the result Cartesian coordinates to QCFractal.
    (chain_prev, chain, expect, expectG, ForceRebuild, LastForce, Y_prev, GW_prev, GP_prev, respaced, _) \
        = takestep([chain_prev], chain, iteration, LastForce, ForceBuild, trust, Y, GW, GP, HW, HP, result_prev)
    attrs_new = check_attr(chain)
    attrs_prev = check_attr(chain_prev)
    Ys.append(Y.tolist())
    GWs.append(GW.tolist())
    GPs.append(GP.tolist())

    temp = {"Ys": Ys, "GWs": GWs, "GPs": GPs, "attrs_new": attrs_new, "attrs_prev": attrs_prev, "trust": trust,
        "trustprint": trustprint, "expect": expect, "expectG": expectG.tolist(), "quality": Quality, "respaced": respaced,
        "coord_ang_prev": chaintocoords(chain_prev, True), "lastforce": LastForce, "forcerebuild": ForceRebuild,
        "result_prev": result, "geometry": []}
    newcoords = chaintocoords(chain)
    info_dict.update(temp)
    return newcoords, info_dict
