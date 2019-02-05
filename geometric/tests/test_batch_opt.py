"""
A set of tests for using the QCEngine project
"""

import copy
import numpy as np
import tempfile
import logging
import math
from geometric.molecule import bohr2ang

log = logging.getLogger(__name__)

from . import addons
import geometric.optimize as gt 
from geometric.internal import CartesianCoordinates,\
    PrimitiveInternalCoordinates, DelocalizedInternalCoordinates
from geometric.nifty import ang2bohr

localizer = addons.in_folder

_base_schema = {
        "schema_version": 1,
        "molecule": {
            "geometry": [
                0.0,  0.0,              -0.1294769411935893,
                0.0, -1.494187339479985, 1.0274465079245698,
                0.0,  1.494187339479985, 1.0274465079245698
            ],
            "symbols": ["O", "H", "H"],
            "connectivity": [[0, 1, 1], [0, 2, 1]]
        },
        "driver": "gradient",
        "model": {
            "method": "UFF",
            "basis": None
        },
        "keywords": {},
        "program": "rdkit"
    } # yapf: disable

_geo2 = [0.0139,  -0.4830,   0.2848,
         0.0628,  -0.2860,   0.7675,
         0.0953,  -1.0031,   0.4339]

@addons.using_qcengine
@addons.using_rdkit




class BatchOptimizer(object):
    """ Demo BatchOptmizer for runnig pytest test """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.params = gt.OptParams(**kwargs)
        self.gtOptimizer = gt.Optimizer(self.params, None, None)
        
        
    def _initOptObjects(self, schemas):
        """ initilize all OptObjects for the schmas passed.
        
        Arguements
        ----------
        schemas: list of schemas for qcengine
        
        return
        ------
        list of OptOject's for each schema
        """
        
        optObjs = []
        for schema in schemas:
            M, engine = gt.get_molecule_engine(qcschema=schema, **self.kwargs)
            coords = M.xyzs[0].flatten() * ang2bohr
    
            # Read in the constraints
            constraints = self.kwargs.get('constraints', None) #Constraint input file (optional)
            if constraints is not None:
                Cons, CVals = gt.ParseConstraints(M, open(constraints).read())
            else:
                Cons = None
                CVals = None
        
            #=========================================#
            #| Set up the internal coordinate system |#
            #=========================================#
            # First item in tuple: The class to be initialized
            # Second item in tuple: Whether to connect non-bonded fragments
            # Third item in tuple: Whether to throw in all Cartesian (no effect if second item is True)
            CoordSysDict = {'cart':(CartesianCoordinates, False, False),
                            'prim':(PrimitiveInternalCoordinates, True, False),
                            'dlc':(DelocalizedInternalCoordinates, True, False),
                            'hdlc':(DelocalizedInternalCoordinates, False, True),
                            'tric':(DelocalizedInternalCoordinates, False, False)}
            coordsys = self.kwargs.get('coordsys', 'tric')
            CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]
    
            IC = CoordClass(M, build=True, connect=connect, addcart=addcart, constraints=Cons, 
                            cvals=CVals[0] if CVals is not None else None)
            tmpDir = tempfile.mkdtemp(".tmp", "batchOpt")

            obj = gt.OptObject(coords, M, IC, engine, self.params.trust, tmpDir)
            log.debug("[AU]: e=%.5f bl=%.5f,%.5f g=%.4f" % (
                    obj.E, obj.X[0],obj.X[3], obj.gradx[0]))
            optObjs.append(obj)
            
        return optObjs
    

    def _batchComputeEnergyAndForces(self, optObjs):
        """ This just an mockup. if this was NNP this would work in one batch
            on the GPU.
        """
        for obj in optObjs:
            if obj.state == gt.OPT_STATE.NEEDS_EVALUATION:
                obj.calcEnergyForce()
                log.debug("[AU]: e=%.5f bl=%.5f,%.5f g=%.4f" % (
                    obj.E, obj.X[0],obj.X[3], obj.gradx[0]))
    
    def optimizeMols(self, schemas):
        """ Optmize all molecules as represented by the schemas.
        
            return
            ------
            list of optimized Molecule's
        """
        optObjs = self._initOptObjects(schemas)
        res = []
        
        # Optimization Loop, while not all have completed optimization
        while len(optObjs) > 0:
            nextOptObjs = []
            
            # take one step, energy and gradient must have been stored in optObj
            for optObj in optObjs:    
                self.gtOptimizer.step(optObj)

            self._batchComputeEnergyAndForces(optObjs)

            # evaluate step
            for optObj in optObjs:    
                if optObj.state == gt.OPT_STATE.NEEDS_EVALUATION:
                    
                    optStatus = self.gtOptimizer.evaluateStep(optObj) 
                    if optStatus is not gt.OPT_RESULT.NOT_CONVERGED:
                        log.info("Optmization convereged!")
                        res.append(optObj.progress)
                        continue
                nextOptObjs.append(optObj)
            if len(nextOptObjs) == 0: break  ######## All Done
             
            # step and evaluation completed, next step for remaining conformations
            optObjs = nextOptObjs
            
        return res
 

def test_rdkit_simple():

    schema1 = copy.deepcopy(_base_schema)
    schema2 = copy.deepcopy(_base_schema)
    schema2['molecule']['geometry']= [c  / bohr2ang for c in _geo2]
    
    opts = {"qcengine": True, "input": "tmp_data", "qce_program": "rdkit"}

    bOptimizer = BatchOptimizer(**opts)
    ret = bOptimizer.optimizeMols([schema1, schema2])

    # Currently in angstrom
    ref = np.array([0., 0., -0.0644928042, 0., -0.7830365196, 0.5416895554, 0., 0.7830365196, 0.5416895554])
    assert np.allclose(ref, ret[0].xyzs[-1].ravel(), atol=1.e-5)
    
    # check that distances in ref are same as in ret[1]
    refAt = ref.reshape(-1,3)
    retAt = ret[1].xyzs[-1]
    for atRef,atRet in zip(refAt,retAt):
        for atRef2,atRet2 in zip(refAt,retAt):
            d2Ref = np.power(atRef[0]-atRef2[0],2) + np.power(atRef[1]-atRef2[1],2) +np.power(atRef[2]-atRef2[2],2)
            d2Ret = np.power(atRet[0]-atRet2[0],2) + np.power(atRet[1]-atRet2[1],2) +np.power(atRet[2]-atRet2[2],2)
            
            assert math.isclose(d2Ref, d2Ret, abs_tol=1e-3)
    
    
    

_N2_schema = {
        "schema_version": 1,
        "molecule": {
            "geometry": [
                0.0, 0., 0.,
                1.9, 0., 0.
            ],
            "symbols": ["N", "N"],
            "connectivity": [[0, 1, 3]]
        },
        "driver": "gradient",
        "model": {
            "method": "UFF",
            "basis": None
        },
        "keywords": {},
        "program": "rdkit"
    } # yapf: disable

_N2_geo2 = [0.0, 0., 0.,
            0.6, 0., 0.,]



def test_rdkit_N2():

    schema1 = copy.deepcopy(_N2_schema)
    schema2 = copy.deepcopy(_N2_schema)
    schema2['molecule']['geometry']= [c / bohr2ang for c in _N2_geo2]
    
    opts = {"qcengine": True, "input": "tmp_data", "qce_program": "rdkit"}

    bOptimizer = BatchOptimizer(**opts)
    ret = bOptimizer.optimizeMols([schema1, schema2])

    # Currently in angstrom
    ref = np.array([-0.05729, 0., 0.,   1.06272, 0., 0.])
    assert np.allclose(ref, ret[0].xyzs[-1].ravel(), atol=1.e-3)
    
    # check that distances in ref are same as in ret[1]
    refAt = ref.reshape(-1,3)
    retAt = ret[1].xyzs[-1]
    for atRef,atRet in zip(refAt,retAt):
        for atRef2,atRet2 in zip(refAt,retAt):
            d2Ref = np.power(atRef[0]-atRef2[0],2) + np.power(atRef[1]-atRef2[1],2) +np.power(atRef[2]-atRef2[2],2)
            d2Ret = np.power(atRet[0]-atRet2[0],2) + np.power(atRet[1]-atRet2[1],2) +np.power(atRet[2]-atRet2[2],2)
            
            assert math.isclose(d2Ref, d2Ret, abs_tol=1e-3)
