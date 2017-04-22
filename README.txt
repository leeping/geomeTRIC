#==================#
#|    GeomeTRIC   |#
#==================#

This is an open-source geometry optimization code for quantum
chemistry.  The code works by calling external software for the energy
and gradient through wrapper functions.  Currently Q-Chem, TeraChem
and Psi4 are supported.

Authors: Lee-Ping Wang, Chenchen Song

Contact Email: leeping@ucdavis.edu

If this code has benefited your research, please support us by citing:

Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with translation and rotation coordinates", J. Chem, Phys. 144, 214108.
http://dx.doi.org/10.1063/1.4952956

#==================#
#|   Quick Help   |#
#==================#

Package dependencies are:
Python 2.7
NumPy, Scipy, NetworkX, ForceBalance

ForceBalance provides the molecule file format converter (molecule.py)
and some other helpful functions.  You don't need to install any of the 
optional dependencies of Forcebalance, such as lxml and Work Queue.

There is no installer and the optimize.py script may be called directly.
The script needs to be in the same folder as internal.py and rotate.py.
You need a .xyz file for the coordinates and one of the supported 
quantum chemistry software packages.  

Please refer to the example calculations for how to run the code.  
The commands to execute the code are contained in "command.sh".
