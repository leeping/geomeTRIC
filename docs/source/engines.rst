.. _engines:

Engines
=======

This section lists the codes called "engines" that are interfaced to geomeTRIC for energy and gradient computations.
The choice of engine is set on the command line using the ``--engine`` option.

    Note: geomeTRIC disables the engines' symmetry features by default because the automatic re-orientation of the molecule into standard coordinates can confuse the optimizer.
    This limitation may be removed at some point in the future.

Different engines have different features available.  Here is a simple summary:

+-------------+--------------+-----------------+---------+---------------+------------+
| Engine name | Scratch dir. | Work Queue task | Setting | Conical       | Bond order |
|             | handling     | distribution    | threads | intersections |            |
+=============+==============+=================+=========+===============+============+
| TeraChem    | Yes          | Yes             | CPU/GPU | Yes           | Yes        |
+-------------+--------------+-----------------+---------+---------------+------------+
| Q-Chem      | Yes          | Limited         | CPU     | Yes           | Yes        |
+-------------+--------------+-----------------+---------+---------------+------------+
| Psi4        | No           | No              | CPU     | Untested      | No         |
+-------------+--------------+-----------------+---------+---------------+------------+
| Molpro      | No           | No              | CPU     | Untested      | No         |
+-------------+--------------+-----------------+---------+---------------+------------+
| Gaussian    | Yes          | No              | CPU     | Untested      | No         |
+-------------+--------------+-----------------+---------+---------------+------------+
| OpenMM      | No           | No              | No      | No            | No         |
+-------------+--------------+-----------------+---------+---------------+------------+
| Gromacs     | No           | No              | No      | No            | No         |
+-------------+--------------+-----------------+---------+---------------+------------+
| QCEngine    | No           | No              | No      | No            | No         |
+-------------+--------------+-----------------+---------+---------------+------------+
| Custom      | No           | No              | No      | No            | No         |
+-------------+--------------+-----------------+---------+---------------+------------+
| ASE         | No           | No              | No      | No            | No         |
+-------------+--------------+-----------------+---------+---------------+------------+

In what follows, all examples can be found in the ``[root]/examples/`` folder of the source distribution.

TeraChem
--------

Selected using ``--engine tera``.  This is also the default if ``--engine`` is not provided.

Make sure `TeraChem <https://www.petachem.com/>`_ is installed and
the ``$TeraChem`` environment variable is properly set (it should be the folder that
contains the ``terachem`` executable under ``bin/``).
Versions 1.9 and above are supported (possibly earlier, but not guaranteed).

The input file should contain at a minimum the keywords: ``charge``, ``spinmult``, ``method``, ``basis`` and ``coords``.
The ``coords`` value should match the name of a ``.xyz`` file that contains the initial coordinates in the same folder.
Guess orbitals for SCF and CASSCF should be specified using the ``guess`` or ``casguess`` keywords, with the corresponding
files placed in the same folder.

This interface supports the following features:

* Reuse of initial guesses for SCF and CASSCF calculations
* Distribution of parallel jobs via Work Queue (e.g. for numerical Hessians)
* Conical intersection optimization either by running separate jobs for each state, or having TeraChem directly compute the CI objective function.
* Setting number of CPU threads and GPUs via ``--nt`` option

An example is provided in the ``[root]/examples/2-challenges/trp-cage_terachem`` folder.  This calculation takes a rather long time to run due to the large system size.

    Note: TeraChem contains a C++ implementation of geomeTRIC that avoids the need to call the program repeatedly for single point gradients.
    It currently does not have some of the newer features such as transition state optimization and MECI optimization.
    The user interface is similar to what is described here; refer to the TeraChem manual for more details.

Q-Chem
------

Selected using ``--engine qchem``.

Make sure `Q-Chem <https://www.q-chem.com/>`_ is installed and
environment variables are properly set.
Versions 4.4 and above are supported (possibly earlier, but not guaranteed).
The input file should contain the molecular structure in Cartesian coordinates.

This interface supports the following features:

* Reuse of scratch folder from previous single-point calculations during optimizations
* Distribution of parallel jobs via Work Queue (e.g. for numerical Hessians). At this time there is no capability to send the qcdir folder to Work Queue workers.
* Conical intersection optimization by running separate jobs for each state (note: only tested for two *ground* states that differ in the initial guess.)
* Setting number of CPU threads via ``--nt`` option

An example is provided in the ``[root]/examples/0-regression-tests/water6_qchem`` folder.

Psi4
----

Selected using ``--engine psi4``.

Make sure `Psi4 <https://www.psicode.org/>`_ is installed and environment variables are properly set.
Versions 1.3 and above are supported (possibly earlier, but not guaranteed).
The input file should contain the molecular structure in Cartesian coordinates.

This interface supports the following features:

* Setting number of CPU threads via ``--nt`` option
* Conical intersection optimization should work, but not tested

Examples of energy minimization are provided in the ``[root]/examples/1-simple-examples/water6_psi4`` and ``[root]/examples/1-simple-examples/water6_psi4_mbe`` folders.
The latter example includes a basis set superposition error (BSSE) correction in the energy and gradient and shows how to use Psi4 fragment syntax.

A two-dimensional scan of an improper torsion and bond angle is provided in ``[root]/examples/2-challenges/improper-2D``.

Molpro
------

Selected using ``--engine molpro``.

Make sure `Molpro <https://www.molpro.net/>`_ is installed and
environment variables are properly set.
Versions 2015.1 and above are supported (possibly earlier, but not guaranteed).
The input file should contain the molecular structure in Cartesian coordinates.

This interface supports the following features:

* Setting number of CPU threads via ``--nt`` option
* Conical intersection optimization should work, but not tested

An example is provided in the ``[root]/examples/1-simple-examples/water6_molpro`` folder.

Gaussian
--------

Selected using ``--engine gaussian``.

Make sure `Gaussian <https://gaussian.com/>`_ is installed and
environment variables are properly set.
Gaussian versions 09 and 16 are supported.
The input file should contain the molecular structure in Cartesian coordinates.

This interface supports the following features:

* Setting number of CPU threads via ``--nt`` option
* Conical intersection optimization should work, but not tested

Examples are provided in the ``[root]/examples/1-simple-examples/ethane_pcm_gaussian`` and ``[root]/examples/1-simple-examples/water2_gaussian`` folders.

OpenMM
------

Selected using ``--engine openmm``.

Make sure `OpenMM <https://www.openmm.org>`_ is installed and environment variables are properly set.
Versions 7.1 and above are supported (possibly earlier, but not guaranteed).

You will need a ``.pdb`` file containing the structure and topology, and either a force field ``.xml`` or system ``.xml`` file (geomeTRIC will autodetect the type).
(If you provide the name of a force field ``.xml`` file that is not in the current folder but is in the search path of OpenMM, that also works.))

The engine contains an OpenMM Simulation object which is created using the topology information in the ``.pdb`` file and a parameterized system;
the latter is either created from the force field ``.xml`` file, or read in from the system ``.xml`` file.

Because this is an MM engine, optimizing conical intersections is not recommended.
There is also no way to set the number of threads, as the engine is hard-coded to use the Reference platform.

    Note: geomeTRIC's internal routines are currently not efficient for systems containing more than a few hundred atoms,
    so this is currently not recommended for optimizing systems that OpenMM is typically used to simulate (>10,000 atoms).

Gromacs
-------

Selected using ``--engine gromacs``.

Make sure `Gromacs <https://www.gromacs.org>`_ is installed and environment variables are properly set.
This engine also requires `ForceBalance <https://www.github.com/leeping/forcebalance>`_ to be installed.
Versions 4.6.7 and 5.1.4 are known to work; it has not been tested with older or newer versions.

The input file to the calculation is a GROMACS ``.gro`` coordinate file named *exactly* ``conf.gro``.
Also required is a GROMACS topology and run parameter file, named *exactly* ``topol.top`` and ``shot.mdp``.
The ForceBalance interface to GROMACS is used to compute single-point energies and gradients.

    Note: As the GROMACS engine is not extensively used, it is not guaranteed to work well with newer GROMACS versions
    so proceed with caution.

An example is provided in ``<root>/examples/1-simple-examples/trp-cage_gromacs``.

QCEngine
--------

This engine enables geomeTRIC to work with MolSSI's `QCArchive <https://qcarchive.molssi.org>`_ ecosystem.
It works a bit differently in that `QCEngine <https://github.com/MolSSI/QCEngine>`_ is another quantum chemistry program executor/wrapper that supports a number of packages.

This engine is typically used by running geomeTRIC using the JSON API instead of the command line.
Examples are provided in ``<root>/geometric/tests/test_run_json.py``.

CustomEngine
------------

This is yet another way for quantum chemistry programs to work with geomeTRIC, contributed by the developers of `PySCF <https://github.com/pyscf/pyscf>`_.
Basically any class that defines a method to calculate the energy and gradient given the coordinates (all in atomic units) can be used to optimize the geometry.

The custom engine cannot be used via geomeTRIC's command line, but an example for how to code one up is provided in ``<root>/geometric/tests/test_customengine.py``.

ASE Engine
----------

This is a wrapper engine for any `ASE <https://gitlab.com/ase/ase/>`_-compatible calculator to be used for geometry
optimisation. The calculator needs to be importable in your Python environment, as well as ASE installed.
Nb. this means that not only the calculators in the main ASE repo, but any calculator from other projects
that is subclassed from ASE can be used, eg. `XTB <https://github.com/grimme-lab/xtb-python>`_,
`GAP (with quippy) <https://github.com/libatoms/quip>`_.

Usage:

* Selected using ``--engine ase``
* set the class of your calculator with ``--ase-class``, eg. ``--ase-class=xtb.ase.calculator.XTB``, ``--ase-class=quippy.potential.Potential``
* set any initialisation keyword arguments for the calculator class with ``--ase-kwargs``, where the given argument is parsed as a JSON string. Note, this requires correct quoting, eg. ``--ase-kwargs='{"method":"GFN2-xTB"}'``.
