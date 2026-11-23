.. _mace:

MACE machine-learning potentials
================================

geomeTRIC can use `MACE <https://github.com/ACEsuit/mace>`_ machine-learning interatomic potentials (MLIPs)
through the generic **ASE engine**. There is no dedicated ``--engine mace``; pass the MACE calculator via
``--ase-class`` and ``--ase-kwargs``.

Supported workflows include:

* Geometry minimization with MACE
* Constrained optimization and constraint scans with MACE
* Nudged elastic band (NEB) calculations driven by MACE

Installation
------------

Install ASE and MACE in the same Python environment as geomeTRIC, for example::

    pip install ase mace-torch

Download a pretrained checkpoint (first use may pull weights into ``~/.cache/mace/``).
Common molecular / materials models:

+------------------+------------------------------------------+------------------------------------------+
| Family           | Typical use                              | Example checkpoint name                  |
+==================+==========================================+==========================================+
| MACE-OMOL        | Molecules, ions, open-shell systems      | ``MACE-omol-0-extra-large-1024.model``   |
+------------------+------------------------------------------+------------------------------------------+
| MACE-OFF         | Neutral organic molecules                | ``MACE-OFF23_small.model``               |
+------------------+------------------------------------------+------------------------------------------+
| MACE-MP          | Materials / crystals (PBE-like)          | ``20231210mace128L0_energy_epoch249model``|
+------------------+------------------------------------------+------------------------------------------+

For molecular work with net charge or spin multiplicity, **MACE-OMOL** is recommended.

Command-line interface
----------------------

Select the ASE engine and the MACE calculator class::

    --engine ase \
      --ase-class=mace.calculators.mace.MACECalculator \
      --ase-kwargs='{"model_paths":"/path/to/checkpoint.model","mace_device":"cpu","default_dtype":"float64"}'

Notes on ``--ase-kwargs``:

* ``model_paths`` : Path to a MACE ``.model`` file (required).
* ``mace_device`` : ``"cpu"`` or ``"cuda"`` (mapped to MACE's ``device`` argument).
* ``default_dtype`` : Prefer ``"float64"`` for optimization work.
* Multi-head OMOL checkpoints need ``"mace_head":"omol"`` (mapped to MACE's ``head``).
* Prefer ``mace_device`` / ``mace_head`` over bare ``device`` / ``head`` in ``ase-kwargs``
  to avoid clashing with other meanings of those names.
* ``charge`` and ``mult``: see :ref:`mace_charge_mult` below.

Direct minimization with MACE
-----------------------------

Provide a coordinate file (e.g. ``.xyz``) as the input::

    geometric-optimize --engine ase \
      --ase-class=mace.calculators.mace.MACECalculator \
      --ase-kwargs='{"model_paths":"'"$HOME"'/.cache/mace/MACE-omol-0-extra-large-1024.model","mace_device":"cpu","default_dtype":"float64","mace_head":"omol"}' \
      start.xyz

Constrained optimization and scans
----------------------------------

Constraints work the same as with QC engines. Pass a constraints file as the optional
positional argument, and keep the ASE/MACE settings above.

Example (fixed O–H distance)::

    geometric-optimize --engine ase \
      --ase-class=mace.calculators.mace.MACECalculator \
      --ase-kwargs='{"model_paths":"'"$HOME"'/.cache/mace/MACE-omol-0-extra-large-1024.model","mace_device":"cpu","default_dtype":"float64","mace_head":"omol"}' \
      start.xyz constraints.txt

For ``$scan`` blocks, each scan point is optimized with the same MACE calculator.

NEB with MACE
-------------

NEB always requires two positional arguments: a structure template ``input`` and a
multi-frame ``chain_coords`` XYZ. With ASE/MACE both can be XYZ files (they may be the
same multi-frame file if the topology matches)::

    geometric-neb --engine ase \
      --ase-class=mace.calculators.mace.MACECalculator \
      --ase-kwargs='{"model_paths":"'"$HOME"'/.cache/mace/MACE-omol-0-extra-large-1024.model","mace_device":"cpu","default_dtype":"float64","mace_head":"omol"}' \
      --images 11 --align no --prefix hcn_mace \
      HCN.xyz HCN.xyz

Useful NEB options (unchanged):

* ``--images`` : number of images along the band
* ``--align no`` : disable rigid alignment of images
* ``--optep yes`` : optimize chain endpoints with the same ASE engine before NEB

**Example:** ``examples/1-simple-examples/hcn_hnc_neb_MACE/``

NEB for the HCN ↔ HNC isomerization driven by MACE through ASE (multi-frame chain XYZ).

.. note::
   MACE NEB evaluates images sequentially on the selected device unless Work Queue or
   other parallel backends are configured for the ASE engine.

.. _mace_charge_mult:

Charge and multiplicity
-----------------------

For MACE-OMOL, total charge and spin multiplicity matter. There is **no** separate
``--charge`` or ``--mult`` command-line flag for the ASE engine. Electronic state is
resolved entirely when the ASE engine is built, as follows.

**How values are chosen (precedence)**

1. If ``"charge"`` and/or ``"mult"`` appear in ``--ase-kwargs``, those values are used.
2. Else, if the geomeTRIC ``Molecule`` object already has ``.charge`` / ``.mult``
   attributes (for example after reading certain QC-style inputs), those are used as a
   fallback.
3. Else, defaults are charge ``0`` and multiplicity ``1``.

A plain ``.xyz`` file does **not** set ``Molecule.charge`` / ``Molecule.mult``, so for
typical ASE/MACE XYZ workflows you must put ``charge`` and ``mult`` in ``--ase-kwargs``
whenever they are not the defaults.

**If kwargs and Molecule disagree**

``--ase-kwargs`` always wins for the energy/gradient evaluation. The ``Molecule``
attributes are **not** updated to match kwargs, and no warning is printed. Only the
ASE side used by the calculator is set from the resolved values.

**How values are applied (not as MACE constructor args)**

``charge`` and ``mult`` are stripped out of the calculator constructor kwargs. They are
**not** passed as ``MACECalculator(..., charge=..., mult=...)``. Instead, geomeTRIC
applies them to the ASE ``Atoms`` object:

* ``atoms.info["charge"] = charge`` (total charge; used by MACE-OMOL-style models)
* ``atoms.info["spin"] = mult`` (spin multiplicity, not :math:`S_z` or :math:`2S`)
* Atom-0 initial charge and magnetic moment (``mult - 1``) for calculators that read
  those ASE fields (e.g. some XTB setups)

Example::

    --ase-kwargs='{"model_paths":"...","mace_device":"cpu","default_dtype":"float64","mace_head":"omol","charge":-1,"mult":1}'

Limitations and caveats
-----------------------

* Use ``--engine ase`` with a MACE calculator; there is no separate ``mace`` engine.
* Absolute energies from materials-trained models (MACE-MP) are not comparable to hybrid
  DFT molecular energies; treat barriers and relative energies with appropriate care.
* Large models can be memory-intensive on CPU; use ``--ase-kwargs`` with ``"mace_device":"cuda"``
  when a GPU is available.
