.. _install:

Installation
============

You can install `geomeTRIC` with ``conda``, with ``pip``, or by installing from source.

Conda
-----

You can install or update geomeTRIC using the ``conda`` package manager.

.. parsed-literal::

    conda install geometric -c conda-forge

This installs geomeTRIC and its dependencies.

The geomeTRIC package is maintained on the
`conda-forge channel <https://conda-forge.github.io/>`_.

Pip
---

To install geomeTRIC with ``pip`` ::

    pip install geometric

Install from Source
-------------------

To install geomeTRIC from source, clone the repository from `github
<https://github.com/leeping/geometric>`_::

    git clone https://github.com/lpwgroup/geomeTRIC.git
    cd geomeTRIC
    python setup.py install

Dependencies
------------

The required packages for geomeTRIC are as follows. Older versions of packages may work, no guarantees.

* Python : Versions 2.7, 3.5, 3.6, 3.7 are supported. Presumably it will work with version 3.8.
* NumPy: Version 1.15 or above
* NetworkX : Version 2.2 or above

You will also need at least one quantum chemistry (QM) or molecular mechanics (MM) package to evaluate energies and gradients. This requirement is already satisfied if you are using software like QCFractal or PySCF which uses geomeTRIC as a library.

Supported QM packages are as follows. Older versions of packages may work, no guarantees.

* Q-Chem : Version 4.2 or above
* TeraChem : Version 1.5 or above
* Psi4 : Version 1.2 or above
* Molpro : Version 2015 and 2019
* Gaussian : Version 09 or 16
  
Supported MM packages are:
* OpenMM : Version 6.3 or above
* Gromacs : Version 4.6.7 or 5.1.4. Newer versions are not tested.

Additionally, a number of QM, MM, semiempirical and machine learning potentials are supported through the `QCEngine <https://github.com/MolSSI/QCEngine>`_ package. Version 0.15.0 or above required.

A testing environment can be set up using ``conda`` which includes OpenMM, Psi4, QCEngine, and RDKit. This can be accomplished by::

    cd geomeTRIC
    python devtools/scripts/create_conda_env.py -n=geo_test -p=3.7 devtools/conda-envs/omm_psi4_rdkit.yaml

  
Testing
-------

Test geomeTRIC with ``pytest``::

    cd geomeTRIC
    pytest

More information is available in the developer documentation.

Installation of cctools
------------------------
The Work Queue library in the `CCTools <https://github.com/cooperative-computing-lab/cctools>`_ package is utilized to provide distributed computing features in geomeTRIC, primarily the computation of numerical Hessian matrices.

Installation of ``cctools`` is done separately. A convenient bash script has been made to simplify the process::

    $bash geomeTRIC/devtools/travis-ci/install-cctools.sh
