#!/usr/bin/env python

from __future__ import division

import os

# The directory that this file lives in
rootdir = os.path.dirname(os.path.abspath(__file__))

# Conversion factors
bohr2ang = 0.529177210
ang2bohr = 1.0 / bohr2ang
au2kcal = 627.5096080306
kcal2au = 1.0 / au2kcal
au2kj = 2625.5002
kj2au = 1.0 / au2kj
grad_au2gmx = 49614.75960959161
grad_gmx2au = 1.0 / grad_au2gmx
# Gradient units
au2evang = 51.42209166566339
evang2au = 1.0 / au2evang

