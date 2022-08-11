.. _examples:

Examples
========

Example input and output files for geomeTRIC can be found in the ``examples/`` folder of the source tree.  They are organized in the following way:

- ``examples/0-regression-tests``:  These example calculations are also used to measure performance changes.  The purpose is to ensure that geomeTRIC's performance does not degrade over time.  Despite the folder name, they are complete and easily runnable example calculations from the user's perspective.
- ``examples/1-simple-examples``:  Example calculations designed to familiarize the user with running geomeTRIC using different engines and job types.  These examples are designed to run in a few minutes or less, and are an ideal starting point for new users.
- ``examples/2-challenges``:  A collection of calculations that represent "difficult" optimizations, some of which inspired new feature development.  They may require a large number of iterations to converge or passing non-default arguments.  

Organization of an example
--------------------------

The root folder of an example calculation contains the input file(s) and a script ``command.sh``, which contains a single line that shows the command line arguments to geomeTRIC.  The example may be run as follows\: ::

   sh command.sh

In simple examples, the command line arguments specify only the input file and engine name.  More complex examples may require additional command line arguments or auxiliary input files (such as a constraints.txt file).

A reference set of saved calculation outputs are provided in the ``saved/`` subfolder.  Each set of outputs has its own sub-subfolder such as ``saved/2022-07-10`` for the run date, or ``saved/v0.9.7.2`` corresponding to a released version.  These files are for ensuring your calculation is running as expected, and for developers to check that performance has not degraded from past versions.  

.. note::
    If you wish to run the examples, make sure you have the corresponding QC program installed.  Many of the examples use Q-Chem and TeraChem; both are commercial software packages.  geomeTRIC also supports Psi4, which is freely available, as well as the commercial packages Gaussian (G09 or G16) and Molpro.  For more information, refer to the :ref:`Engines <engines>` page.

Descriptions of individual examples
-----------------------------------

The following descriptions are given in rough order of simple to complex.

Optimization of 6 water molecules using Q-Chem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Location: ``examples/0-regression-tests/water6_qchem``
- Command line: ``geometric-optimize --engine qchem water6.qcin``
- Number of atoms: 18
- Optimization cycles (approx.): 65
- Run time (approx.): <5 minutes
- Description: Optimization of 6 water molecules using Q-Chem at the HF/3-21G level of theory. This is a simple example that shows how translation-rotation internal coordinates (TRIC) performs well for optimizing clusters of molecules where translation and rotation are important degrees of freedom.

.. image:: images/water6.png
   :width: 400

The above image shows the initial structure (in green) and final optimized structure.

Optimization of trp-cage miniprotein using OpenMM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Location: ``examples/0-regression-tests/trpcage_openmm``
- Command line: ``geometric-optimize --engine openmm --pdb trpcage.pdb amber99sb.xml --prefix run``
- Number of atoms: 304
- Optimization cycles (approx.): 163
- Run time (approx.): 20 minutes
- Description: Optimization of Trp-cage miniprotein (20 amino acids) using OpenMM in the gas phase using the AMBER ff99SB force field.  The TRIC coordinate system adds an explicit translation and rotation coordinate for each amino acid residue. The force field XML file ``amber99sb.xml`` is provided as the required input file, and the PDB file containing the initial coordinates is provided using the ``--pdb trpcage.pdb`` option. The output file prefix is changed to ``run``, otherwise it would have defaulted to ``amber99sb`` (taken from the input file name). Note that the energies that are output by MM calculations are much lower than QC calculations because they do not include the binding energies of electrons.

.. image:: images/trpcage.png
   :width: 400

The above image shows the initial structure (in green) and final optimized structure, as well as the secondary structure of the optimized structure (transparent).

Optimization of azithromycin using Gaussian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Location: ``examples/2-challenges/azithromycin_gaussian``
- Command line: ``geometric-optimize --engine gaussian start.gjf``
- Number of atoms: 124
- Optimization cycles (approx.): 47
- Run time (approx.): 9 hours
- Description: Optimization of azithromycin using Gaussian at the B3LYP/6-31G* level of theory.  The long runtime is due to the cost of the single point Gaussian calculations, which average around 12 minutes using 4 cores on the test machine (Intel i7-6850K CPU @ 3.60GHz).  For this particular example geomeTRIC converges more rapidly than Gaussian 16's native optimizer.

.. image:: images/azithromycin.png
   :width: 600

The above image shows the initial structure (in green) and final optimized structure, with the 2D structure on the right for reference.

