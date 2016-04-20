#==================#
#|    GeomeTRIC   |#
#==================#

This is an open-source geometry optimization code for quantum
chemistry.  The code works by calling external software for the energy
and gradient through wrapper functions.  Currently Q-Chem, TeraChem
and Psi4 are supported.

Authors: Lee-Ping Wang, Chenchen Song

#==================#
#|   Quick Help   |#
#==================#

Package dependencies are:
NumPy, Scipy, NetworkX, ForceBalance

ForceBalance provides the molecule file format converter (molecule.py)
and some helpful convenience functions.  You don't need to install any
of the optional dependencies of Forcebalance, such as lxml and Work Queue.

There is no installer and the optimize.py script may be called directly.
The script needs to be in the same folder as internal.py and rotate.py.
You need an .xyz file and one of the supported quantum chemistry software
packages.  

Please refer to the example calculations for how to run the code.  
The commands to execute the code are contained in "command.sh".
