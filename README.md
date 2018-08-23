# GeomeTRIC
[![Build Status](https://travis-ci.org/leeping/geomeTRIC.svg?branch=master)](https://travis-ci.org/leeping/geomeTRIC)
[![codecov](https://codecov.io/gh/leeping/geometric/branch/master/graph/badge.svg)](https://codecov.io/gh/leeping/geometric)

This is an open-source geometry optimization code for quantum
chemistry.  The code works by calling external software for the energy
and gradient through wrapper functions.  Currently Q-Chem, TeraChem, 
Psi4, and Molpro are supported.

Authors: Lee-Ping Wang, Chenchen Song

Contributors: Yudong Qiu (Psi4 engine); Sebastian Lee (Molpro engine); Daniel G. A. Smith (Testing framework); Chaya Stern (Travis, Conda)

Contact Email: leeping@ucdavis.edu

If this code has benefited your research, please support us by citing:

Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with translation and rotation coordinates", J. Chem, Phys. 144, 214108.
http://dx.doi.org/10.1063/1.4952956

## Quick Help

Package dependencies are:
Python 2.7, 3.6+
NumPy, Scipy, NetworkX

To install the code, run "python setup.py install".
To execute the geometry optimizer, run "geometric-optimize".

You will need a .xyz file for the coordinates and one of the supported 
quantum chemistry software packages.  The supported packages are:
TeraChem, Q-Chem, and Psi4.  You can also call Gromacs for MM forces.

Please refer to the example calculations for how to run the code.  
The commands to execute the code are contained in "command.sh".

