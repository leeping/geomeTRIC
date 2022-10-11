.. _options:

Option Index
============

This page provides a full list of options that are available via the command line interface.

The command line is called as::

    geometric-optimize [optional args with 1 value] input [constraints] [optional args with >=1 value]

The design of the command line interface is that there are no flags (optional arguments with no values) other than ``-h``.
For optional args that are boolean (i.e. designed to be toggled on/off), pass one of ``yes / on / true / t / y / 1`` to turn on,
and ``no / off / false / f / n / 0`` to turn off.

Passing the ``-h`` flag on the command line will produce a concise version of this documentation.

Universal options
-----------------

These options are relevant to every job.

....

``input``

Input file for the calculation. This is a **required** positional argument.
The file type to be used depends on the engine; see :ref:`Engines <engines>` for more details.

....

``constraints``

File containing constraint specifications and/or additional command line options.
This is an **optional** positional argument that comes after the input file.

    Note: Constraints have not been tested for transition state optimizations and may not work.

....

``--coordsys [tric]``

Internal coordinate (IC) system to be used. Valid values for this argument are:

- ``cart`` : Cartesian coordinate system, i.e. no internal coordinates are used.  Can be used for debugging.
- ``prim`` : Primitive (a.k.a. redundant) internal coordinates.  Distances, angles, and dihedral angles are used to span the whole system.  The number of ICs can exceed the number of degrees of freedom.
- ``dlc`` : Delocalized internal coordinates.  Same as ``prim`` but a set of non-redundant linear combinations of "delocalized" internal coordinates is formed by diagonalizing the G matrix and keeping only the eigenvectors with non-zero eigenvalues.  The default for native optimization codes in several quantum chemistry programs.
- ``hdlc`` : Hybrid delocalized internal coordinates.  Distances, angles, and dihedral angles are used to span individual molecules in the system, Cartesian coordinates for all atoms are added, then delocalized internal coordinates are formed.
- ``tric`` (**default**) : Translation-rotation internal coordinates.  Distances, angles, and dihedral angle¯s are used to span individual molecules in the system, three translation and three rotation coordinates are added for each molecule, then delocalized internal coordinates are formed.
- ``tric-p`` : A primitive (redundant) version of translation-rotation internal coordinates where the delocalization step is not applied.

....

``--engine [tera]``

The "engine" is the software to be used for computing energies and gradients. Also see :ref:`Engines <engines>`.

- ``tera`` (**default**) : Use TeraChem. Provide a TeraChem input file. A .xyz coordinate file must also be present (name should match the ``coords`` option in the input file).
- ``qchem`` : Use Q-Chem. Provide a Q-Chem input file with Cartesian coordinates.
- ``molpro`` : Use Molpro. Provide a Molpro input file with Cartesian coordinates.
- ``psi4`` : Use Psi4. Provide a Psi4 input file with Cartesian coordinates.
- ``gaussian`` : Use Gaussian (version 09 or 16). Provide a Gaussian job file with Cartesian coordinates.
- ``openmm`` : Use OpenMM. Provide an OpenMM force field or system ``.xml`` file and a PDB file using ``--pdb`` for the initial structure and topology.
- ``gmx`` : Use Gromacs (experimental). Provide a Gromacs ``.gro`` file. A topology file ``topol.top`` and run parameters ``shot.mdp`` are required, with those exact names.

``--nt [1]``

Number of threads for running parallel jobs on a single machine (i.e. no multi-node jobs at the moment).
The number of threads can be set in TeraChem (also sets the number of GPUs), Q-Chem, Psi4, Molpro, and Gaussian.

Job type options
----------------

These options are used if you're requesting something other than the default energy minimization.

....

``--transition [yes/no]``

Provide ``yes`` to optimize a transition state (first order saddle point).
This option changes the behavior of the optimization algorithm and convergence criteria.
Also, a numerical Hessian will be computed at the start of the optimization (not the end) unless otherwise specified.

    Note: Transition state optimizations are notorious for requiring an initial guess in order to arrive at the desired structure, so it is recommended to start with a high-quality initial guess from constraint scanning or other approaches.

....

``--meci [second_input.in]``

``--meci_sigma [3.5]``

``--meci_alpha [0.025]``

Provide a second input file to search for a minimum-energy conical intersection or crossing point between two potential energy surfaces.
The potential energy and gradient will be computed for each input file, then a penalty constrained objective function will be optimized.
The objective function is defined following `Levine et al. <https://pubs.acs.org/doi/10.1021/jp0761618>`_

Presently only TeraChem and Q-Chem have been tested, but this presumably works with other QC engines as well.
This option slightly changes the behavior of the optimization algorithm, in particular the lower bound on step size for rejecting a bad step is reduced from ``1e-2`` to ``1e-4``.

Additionally, ``--meci engine`` specifies that the engine itself computes the penalty constrained objective function, which means from geomeTRIC's perspective it is similar to an energy minimization,
except for the change in threshold mentioned above.

The parameters to the MECI penalty function are specified using ``--meci_sigma`` (a multiplicative scaling) and ``--meci_alpha`` (a width parameter).
Generally, decreasing ``--meci_alpha`` will result in a smaller gap between the states at convergence but will also require more iterations to convergence.
Setting ``--meci_alpha`` to ``1.0e-3`` often results in convergence of the energy gap to ``1.0e-4`` a.u. or tighter.

....

Additionally, a frequency analysis / harmonic free energy calculation may be specified without any optimization
by providing ``--hessian stop`` (see below).

Hessian options
---------------

These options control the calculation of Hessian (force constant) matrices and derived quantities.

....

``--hessian [never/first/last...]``

Specify whether and when to compute the Hessian matrix for optimization and/or frequency analysis.
The Hessian data will be written to a text file in NumPy format under ``[prefix].tmp/hessian/hessian.txt``.
The ``<prefix.tmp>/hessian`` folder contains a coordinate file corresponding to the Hessian matrix;
if the coordinates at run time matches the existing coordinate file, the Hessian will be read from file instead.

Currently, Hessian matrices are computed by geomeTRIC by numerical central difference of the gradient, requiring 1+6*N(atoms) total calculations.
The finite difference step size is ``1.0e-3`` a.u. which is the default in many QC programs.
Independent gradient jobs can be computed either locally in serial or in parallel using the Work Queue distributed computing library (see ``--port`` below).

Individual gradient calculations are stored in folders such as ``[prefix].tmp/hessian/displace/001[m/p]`` which stands for "coordinate 1 minus/plus displacement".
If the job is interrupted and restarted, existing completed gradient calculations will be read instead of recomputed.
At the conclusion of the Hessian calculation, the ``[prefix].tmp/hessian/displace`` folder is deleted to save space.

Several software packages contain native routines to compute the Hessian matrix using analytic or numerical second derivatives.
Interfaces for using native Hessian calculation routines will be added in the future.

Possible values to pass to this argument are:

- ``never`` (**default for minimization and MECI**) : Do not calculate the Hessian or read Hessian data.
- ``first`` (**default for transition state**) : Calculate the Hessian for the initial structure.
- ``last`` : Calculate the Hessian for the final (optimized) structure.
- ``first+last`` : Calculate the Hessian for both the initial and final structure.
- ``stop`` : Calculate the Hessian for the initial structure, and then stop (do not optimize).
- ``each`` : Calculate the Hessian for each step in the optimization (costly).
- ``file:folder/hessian.txt`` : Provide ``file:`` followed by a relative path to read initial Hessian data in NumPy format.

....

``--port [9876]``

Provide a port number for the Work Queue distributed computing server.
This is only used for distributing gradient calculations in numerical Hessian calculations.
This number can range up to 65535, and a number in the high 4-digit range is acceptable.
Do not use privileged port numbers (less than 1024).

The port number should not be used by other servers running on your machine, and should match
the port number provided to Work Queue worker processes whose job is to execute the gradient calculations.

....

``--frequency [yes]``

Perform a frequency and thermochemical analysis whenever a Hessian calculation is requested; default value is ``yes``.
This will compute harmonic frequencies and vibrational modes, as well as an ideal gas / rigid rotor / harmonic oscillator
approximation to the Gibbs free energy.

The information printed to the screen and log file can be controlled using the ``--verbose`` flag.
Additionally, the frequencies and Cartesian displacements of vibrational modes are written to ``<prefix>.vdata`` files,
which is a ForceBalance-readable vibrational data format.

    Note 1: Cartesian displacements are not orthogonal, because the orthogonal vectors are mass-weighted
    (i.e. Cartesian displacements multiplied by square root of mass).

    Note 2: Because the frequency analysis doesn't cost anything, there usually isn't a reason to disable it.

....

``--thermo [300.0] [1.0]``

Provide temperature (K) and pressure (bar) for thermochemical analysis and Wigner sampling (if applicable).
Default values are as above.

....

``--wigner [100]``

Generate a number of samples from the Wigner distribution, which maps the ground state wavepacket to a phase space distribution.
Samples are written to ``[prefix.tmp]/wigner/[000]/coords.xyz, vel.xyz, fms.dat`` where ``vel.xyz`` is in AMBER units and ``fms.dat`` is in `FMS <https://doi.org/10.1063/1.3103930>`_ readable format.

Provide a positive or negative number to keep or overwrite any existing samples in this folder respectively. (Useful if you generate some samples, then want to generate more while keeping the originals.)

Optimization parameters
-----------------------

This section controls various aspects of the optimization algorithm.

....

``--maxiter [300]``

This sets the maximum number of optimization steps.
Most calculations should converge well within 100 steps, so 300 is a safe upper limit for most jobs.
If convergence fails after 300 steps, then it might be worth taking a close look at the inputs, or if all else fails, contacting the developers.

....

``--converge [energy 1e-6 ...]``

This sets the values of convergence criteria. Units are in atomic units (Bohr and Hartree).
geomeTRIC uses five convergence criteria, using the same values as Gaussian:

- The change in energy from the previous step (default ``1.0e-6``)
- The RMS gradient (default ``3.0e-4``)
- The maximum gradient (default ``4.5e-4``)
- The RMS displacement from the previous step (default ``1.2e-3``)
- The maximum displacement from the previous step (default ``1.8e-3``)

geomeTRIC computes these quantities by taking the norm on each atom
then calculating the RMS/maximum values using the atomic values.
Convergence is reached when all five variables drop below the criteria.

To set one or more convergence criteria individually, provide one or more pairs of values such as
``--converge energy 1.0e-6 grms 3.0e-4 gmax 4.5e-4 drms 1.2e-3 dmax 1.8e-3``.

Hard-coded sets of convergence criteria can also be specified by providing ``--converge set SET_NAME``
where ``set`` must be entered exactly and ``SET_NAME`` is one of the entries in the following table:

+----------------------+----------------+--------------+--------------+--------------+--------------+
| Set name             | Energy         | Grad RMS     | Grad Max     | Disp RMS     | Disp Max     |
+======================+================+==============+==============+==============+==============+
| ``GAU_LOOSE``        | ``1.0e-6``     | ``1.7e-3``   | ``2.5e-3``   | ``6.7e-3``   | ``1.0e-2``   |
+----------------------+----------------+--------------+--------------+--------------+--------------+
| ``NWCHEM_LOOSE``     | ``1.0e-6``     | ``3.0e-3``   | ``4.5e-3``   | ``3.6e-3``   | ``5.4e-3``   |
+----------------------+----------------+--------------+--------------+--------------+--------------+
| ``TURBOMOLE``        | ``1.0e-6``     | ``5.0e-4``   | ``1.0e-3``   | ``5.0e-4``   | ``1.0e-3``   |
+----------------------+----------------+--------------+--------------+--------------+--------------+
| ``GAU`` (*default*)  | ``1.0e-6``     | ``3.0e-4``   | ``4.5e-4``   | ``1.2e-3``   | ``1.8e-3``   |
+----------------------+----------------+--------------+--------------+--------------+--------------+
| ``INTERFRAG_TIGHT``  | ``1.0e-6``     | ``1.0e-5``   | ``1.5e-5``   | ``4.0e-4``   | ``6.0e-4``   |
+----------------------+----------------+--------------+--------------+--------------+--------------+
| ``GAU_TIGHT``        | ``1.0e-6``     | ``1.0e-5``   | ``1.5e-5``   | ``4.5e-5``   | ``6.0e-5``   |
+----------------------+----------------+--------------+--------------+--------------+--------------+
| ``GAU_VERYTIGHT``    | ``1.0e-6``     | ``1.0e-6``   | ``2.0e-6``   | ``4.0e-6``   | ``6.0e-6``   |
+----------------------+----------------+--------------+--------------+--------------+--------------+

    Note 1: The user is responsible for setting the SCF / CASSCF / other convergence thresholds
    sufficiently tight in the engine, especially when tighter than default convergence criteria are used.
    Otherwise, the energy may jump around erratically instead of reaching convergence.

    Note 2: For the case of constrained optimizations, an additional condition is that constrained degrees of freedom
    must be within 0.01 Angstrom / degrees of their target values.

    Note 3: To simulate Q-Chem or Molpro-style convergence criteria, a separate option ``--qccnv`` or ``--molcnv``
    needs to be set.  This is because the logic for determining convergence is different (for example, Q-Chem
    converges when the gradient and *either the RMS displacement or energy change* falls below the threshold.



....

``--trust [0.1]``

``--tmax [0.3]``

These options control the starting and maximum values of the trust radius.
The trust radius is the maximum allowed Cartesian displacement of an optimization step measured in Angstrom.

Depending on the quality of individual optimization steps, the trust radius can be increased from its current value up to the ``--tmax`` value, or it can be decreased down to a minimum value.

The minimum trust radius cannot be user-set; its value is ``0.0`` for transition state and MECI jobs, and the smaller of the ``drms`` convergence criteria and ``1.2e-3`` for energy minimizations.
The purpose of the minimum trust radius is to prevent unpredictable behavior when the trust radius becomes extremely small (e.g. if the step is so small that the energy change is smaller than the SCF convergence criteria).

....

``--enforce [0.1]``

If provided, enforce exact constraint satisfaction when deviation of current values of constrained internal coordinates from target values falls below this threshold.

The default constrained optimizer in geomeTRIC can result in final structures that deviate very slightly from target values (e.g. 0.01 degrees in the dihedral angle).
Provide this option to activate an algorithm that ensures constraints are exactly satisfied the moment the deviations drop below the threshold value.
This can also speed up convergence, but the stability of the algorithm is not very widely tested.
If tested widely enough, setting a threshold of 0.1 may become the default behavior in the future.

....

``--conmethod [0]``

Provide a value of ``1`` to use an alternative way of building the delocalized internal coordinates that satisfies constraints more rapidly, but may be less stable.
Use only if the default method fails for constrained optimizations.

....

``--reset [yes/no]``

``--epsilon [1e-5]``

Specify ``--reset yes`` to reset the approximate Hessian matrix to the initial guess if any of the eigenvalues drop below the threshold specified by ``--epsilon``.
This is enabled by default in energy minimizations, and disabled in transition state / conical intersection optimizations.

....

``--check [10]``

If a number is provided, the internal coordinate system will be rebuilt at the specified interval as if the current structure were the input structure.
This is disabled by default because it tends to lower performance, but may be useful for debugging.

Structure options
-----------------

These options provide flexibility for modifying the initial molecular structure or connectivity.

....

``--radii [Na 0.0]``

Provide pairs of values to modify the covalent radii parameters (two atoms are considered to be bonded if their distance is below 1.2 times the sum of their covalent radii).
Default values are taken from `Cordero et al. <https://doi.org/10.1039/B801115J>`_ with the value for ``Na`` (sodium) set to ``0.0``.

Fine-tuning these values can lead to changes in the number of independent fragments used in TRIC optimizations; for example, if you want to treat a transition metal ion and its ligands as separate molecules, set the radius of the metal to ``0.0``.

....

``--pdb [molecule.pdb]``

Provide a PDB file name. This is important for OpenMM optimizations because the PDB file name contains topology information (i.e. atom names and residue names) needed to parameterize the system.
The residue numbers in the PDB file will also be used to make translation/rotation internal coordinates for individual residues.
If provided, the coordinates in the PDB file will override any coordinates in the input file (but will be overridden by any coordinates passed via ``--coords``).

....

``--coords [coords.xyz]``

Provide a coordinate file to use as the starting structure in the optimization.
If this file contains multiple structures, the **last** structure will be used.
This will override any coordinates present in the PDB file or input file.

....

``--frag [yes]``

Provide ``--frag yes`` to delete bonds between residues, producing separate fragments in the TRIC coordinate system.
This tends to slightly decrease optimization performance in terms of the total number of steps, but in the future could be used to speed up G-matrix inversion and other routines by making the matrices block-diagonal.

Output options
--------------

These options control the format and amount of output.

....

``--prefix [jobname]``

This specifies the base name of files and temporary folders generated by geomeTRIC, such as ``[prefix]_optim.xyz``, ``[prefix].tmp/`` and ``[prefix].log``.
The default value is the input file path with the extension removed.

    Note: This means geomeTRIC can in principle be run in a different folder from the input file, but this is not recommended.

....

``--verbose [0-3]``

This specifies the amount of information printed to the terminal and log files.

- ``0`` : Default, concise print level.
- ``1`` : Include basic information about the optimization step.
- ``2`` : Include detailed information including micro-iterations to determine the optimization step.
- ``3`` : Lots of printout from low-level functions.

....

``--qdata [yes]``

Activating this option will generate a ForceBalance-readable ``qdata.txt`` file containing coordinates, energies and gradients for each structure.

....

``--logINI [log.ini]``

Provide a custom ``log.ini`` file to customize the logger.
This is most useful when using geomeTRIC in ways other than the command line.
Examples are provided in the source distribution under ``<root>/geometric/config/[log.ini, logJson.ini]``.

....

``--write_cart_hess [output.txt]``

At convergence of the optimization, write the approximate Hessian to the specified file name.
This is an experimental feature and not often used, but could be interesting for analysis of the approximate BFGS Hessian.

Software-specific
-----------------

These options are either specific to particular software packages or intended to mimic the behavior of the native optimizer in a software package.

....

``--molproexe [/path/to/molpro.exe]``

Specify the absolute path of the Molpro executable.

....

``--molcnv [yes]``

Use Molpro-style convergence criteria; maximum gradient and displacement are computed differently, and convergence is reached if the maximum gradient and *either maximum displacement or energy change* falls below the threhsold.

....

``--qcdir [qchem.d]``

Provide a Q-Chem scratch folder containing temporary files (e.g. initial molecular orbitals) for the initial step of the geometry optimization.  After the first step in the optimization, temporary files generated by previous optimization steps will be used.

....

``--qccnv [yes]``

Use Q-Chem style convergence criteria; convergence is reached if the RMS gradient and *either RMS displacement or energy change* falls below the threhsold.

....

``--ase-class [string]``

Specify the calculator class to import and use for ASE engine. This needs to be in your python environment, and hence
importable. Under the hood, ``importlib`` is used to import it by name if it exists. eg. ``ase.calculators.lj.LennardJones``
This can be pointing to any class that is a subclass of ``ase.calculators.calculator.Calculator``.

....

``--ase-kwargs [JSON string]``

Specify the keyword arguments for the calculator's initialisation. This is interpreted as a JSON string,
becoming a dictionary that is passed in at construction of the calculator.

Be mindful of quoting, since JSON uses ``"`` for strings, so it it convenient to pack the command line option into
single quotes ``'``. For example: ``--ase-kwargs='{"param_filename":"path/to/file.xml"}'``.


Debugging options
-----------------

These infrequently-used options are mainly for development and debugging.

....

``--displace [yes]``

Write a series of coordinate files containing displacements of various sizes along individual internal coordinates, then exit (no optimizations or QC calculations performed).

....

``--fdcheck [yes]``

Perform finite difference tests for the correctness of internal coordinate first and second derivatives, then exit (no optimizations or QC calculations performed).

