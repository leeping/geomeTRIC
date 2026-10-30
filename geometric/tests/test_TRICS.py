"""
A set of tests for TRICS interpolation
"""

import os, shutil
from . import addons
import geometric
import numpy as np
import tempfile

localizer = addons.in_folder
datad = addons.datad

def _calc_RMSD_ratio(result_M):
    rmsds=[]
    for i in range(len(result_M)-1):
        rmsd, rmsdm = geometric.step.calc_drms_dmax(result_M.xyzs[i] * geometric.nifty.ang2bohr, result_M.xyzs[i+1] * geometric.nifty.ang2bohr)
    
        rmsds.append(rmsd)
    
    return max(rmsds)/np.median(rmsds)


def test_TRICS(localizer):
    """
    Testing TRICS with two examples
    """
    interpolated_1 = geometric.interpolate.run_interpolator(**{'input':os.path.join(datad, 'TRICS_input_1.xyz')})
    ratio_1 = _calc_RMSD_ratio(interpolated_1)

    interpolated_2 = geometric.interpolate.run_interpolator(**{'input':os.path.join(datad, 'TRICS_input_2.xyz')})
    ratio_2 = _calc_RMSD_ratio(interpolated_2)


    assert ratio_1 < 3
    assert ratio_2 < 3
    assert len(interpolated_1) == 50
    assert len(interpolated_2) == 50
