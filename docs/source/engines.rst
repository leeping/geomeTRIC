.. _engines:

Engines
=======

This section lists the codes called "engines" that are interfaced to geomeTRIC for energy and gradient computations.
The choice of engine is set on the command line using the ``--engine`` option.

TeraChem
--------

Selected using ``--engine tera``.  This is also the default if ``--engine`` is not provided.

This interface supports the following features:

To use it, make sure `TeraChem <https://www.petachem.com/>`_ is installed and 
the ``$TeraChem`` environment variable is properly set (it should be the folder that 
contains the ``terachem`` executable under ``bin/``).

* Reuse of initial guesses for SCF and CASSCF calculations
* Distribution of parallel jobs via Work Queue (e.g. for numerical Hessians)
* Conical intersection optimization either by running separate jobs for each state, or having TeraChem directly compute the CI objective function.
* Setting number of CPU threads and GPUs via ``--nt`` option

The input file should contain at a minimum the keywords: ``charge``, ``spinmult``, ``method``, ``basis`` and ``coords``.
The ``coords`` value should match the name of a ``.xyz`` file that contains the initial coordinates in the same folder.
Guess orbitals for SCF and CASSCF should be specified using the ``guess`` or ``casguess`` keywords, with the corresponding
files placed in the same folder.

Q-Chem
------

Selected using ``--engine qchem``.

To use it, make sure `Q-Chem <https://www.q-chem.com/>`_ is installed and 
environment variables are properly set.

This interface supports the following features:

* Reuse of scratch folder from previous single-point calculations during optimizations
* Conical intersection optimization by running separate jobs for each state
* Setting number of CPU threads via ``--nt`` option

Not supported at the moment:

* No distribution of parallel jobs via Work Queue (e.g. for numerical Hessians)

