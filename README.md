![geomeTRIC](https://user-images.githubusercontent.com/1441560/56555299-600d1980-6549-11e9-9f05-b1e8b899a292.png)

[![Build Status](https://github.com/leeping/geomeTRIC/actions/workflows/test-conda.yml/badge.svg)](https://github.com/leeping/geomeTRIC/actions/workflows/test-conda.yml)
[![codecov](https://codecov.io/gh/leeping/geometric/branch/master/graph/badge.svg)](https://codecov.io/gh/leeping/geometric)

This is a geometry optimization code for molecular structures.
The code works by calling external software for the energy and 
gradient through wrapper functions.  Q-Chem, TeraChem, Psi4, 
Molpro, and Gaussian 09/16 are supported quantum chemistry
codes through the command line interface.  The PySCF and
QCArchive packages also provide interfaces to geomeTRIC for
optimization. MM optimizations using OpenMM and Gromacs are
also supported through the command line interface.

Documentation is hosted at https://geometric.readthedocs.io/ .

Authors: Lee-Ping Wang, Chenchen Song

Contributors: Yudong Qiu (Psi4 engine, error handling); Daniel G. A. Smith (Testing framework, QCEngine JSON API); Sebastian Lee (Molpro engine); Chaya Stern (Travis, Conda); Qiming Sun (Custom engine); Alberto Gobbi (Batch energy/gradient, logging); Josh Horton (Convergence criteria, Gaussian engine)

Contact Email: leeping@ucdavis.edu

If this code has benefited your research, please support us by citing:

Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with translation and rotation coordinates", J. Chem, Phys. 144, 214108.
http://dx.doi.org/10.1063/1.4952956

## Quick Help

Package dependencies are:
Python 2.7, 3.5+
NumPy, Scipy, NetworkX

To install the code from source, run "python setup.py install".
To install the latest release from pip, run "pip install geometric".
To install the latest release from conda-forge, run "conda install -c conda-forge geometric".

To execute the geometry optimizer, run "geometric-optimize".
Use "-h" to see the list of command line options.

Generally, you will need a .xyz file for the coordinates and 
one of the supported quantum chemistry software packages installed
on your system.

Supported QM packages are: TeraChem, Q-Chem, Molpro, Psi4, and Gaussian 09/16.
Supported MM packages are: OpenMM, Gromacs.

Please refer to the example calculations for how to run the code.  
The commands to execute the code are contained in "command.sh".

## Citation

If this code has benefited your research, please support us by citing our publication:

Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple
with translation and rotation coordinates", J. Chem, Phys. 144, 214108.
DOI: 10.1063/1.4952956
