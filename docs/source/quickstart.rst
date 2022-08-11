.. _quickstart:

Quick Start
-----------

The easiest way to use geomeTRIC is the command line executable, ``geometric-optimize``.

To get started, you need an input file
for your chosen quantum chemistry (QC) program or "engine" for computing energies and gradients (see :ref:`Engines <engines>`.)
This file will be parsed for the initial structures
and run parameters for the QC program.  
You also need the QC program to be installed into your environment
and in your path.

.. note:: The QC program's native geometry optimization methods will not be used, so setting the job type to optimization in the QC input file will have no effect.

Here we will show how to run an energy minimization on a water dimer with `Psi4 <http://www.psicode.org/>`_, a freely available QC program.
Before running this example, you will need to have Psi4 installed (i.e. the ``psi4`` program needs to be in your ``$PATH`` environment variable.)
Here is a Psi4 input file for a calculation on two water molecules (found in the ``examples/1-simple-examples/water2_psi4`` folder)::

    molecule {
    0 1
    O        0.8559100000   -1.3823600000    0.3174600000
    H        1.6752400000   -1.8477400000    0.4858000000
    H        1.1176100000   -0.4684300000    0.2057500000
    O       -1.0986300000   -0.8583700000    2.1731900000
    H       -0.4031000000   -1.1460800000    1.5818400000
    H       -1.0851100000    0.0968300000    2.1128200000
    }

    set basis sto-3g

    gradient('hf')

With this file saved as ``water2.psi4in``, an energy minimization is carried out using::

    geometric-optimize --engine psi4 water2.psi4in

When running, geomeTRIC will write output to the terminal, and also to ``water2.log``, showing progress of the optimization.
The calculation will finish when all of the convergence criteria are reached (individually met criteria shown in green):

.. parsed-literal::
    Step   16 : Displace = 2.827e-03/4.521e-03 (rms/max) Trust = 3.000e-01 (=) Grad = :green-bold:`7.995e-05`/:green-bold:`1.114e-04` (rms/max) E (change) = -149.9414045323 (-1.387e-06) Quality = 1.449
    Hessian Eigenvalues: 1.25260e-03 7.20014e-03 2.64588e-02 ... 5.65827e-01 5.84730e-01 7.21463e-01
    Step   17 : Displace = 4.522e-03/6.044e-03 (rms/max) Trust = 3.000e-01 (=) Grad = :green-bold:`3.165e-05`/:green-bold:`4.256e-05` (rms/max) E (change) = -149.9414053051 (:green-bold:`-7.728e-07`) Quality = 1.136
    Hessian Eigenvalues: 1.23141e-03 7.53885e-03 2.40697e-02 ... 5.65379e-01 5.84679e-01 7.15469e-01
    Step   18 : Displace = :green-bold:`7.794e-04`/:green-bold:`1.128e-03` (rms/max) Trust = 3.000e-01 (=) Grad = :green-bold:`7.100e-06`/:green-bold:`8.910e-06` (rms/max) E (change) = -149.9414053470 (:green-bold:`-4.191e-08`) Quality = 1.137
    Hessian Eigenvalues: 1.23141e-03 7.53885e-03 2.40697e-02 ... 5.65379e-01 5.84679e-01 7.15469e-01
    Converged! =D

The structures of each optimization step are written to ``water2_optim.xyz``, therefore when the job is completed, the optimized structure is the final structure in this file.
The folder ``water2.tmp`` is the working folder of the engine and contains the QC input and output files, as well as any temporary files for the most recent optimization step.

.. note::
    You can control the number of Psi4 parallel threads using the command line option ``--nt``, for example, ``geometric-optimize --engine psi4 --nt 4 water2.psi4in`` to run Psi4 with 4 threads.  This does not affect geomeTRIC's internal behavior.

.. note::
    Psi4 has an `interface to geomeTRIC <http://www.psicode.org/psi4manual/master/optking.html#interface-to-geometric>`_ 
    that allows the user to use geomeTRIC for geometry optimization by running Psi4 directly. 
    Although the geomeTRIC optimization routines are being used in both cases, the user interface differs from what's being described here.
