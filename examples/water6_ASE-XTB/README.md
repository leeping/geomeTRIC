# 6 water molecules with XTB through ASE

The initial structure was copied from the `water6_psi4` example and converted to simple xyz format.

This example illustrates how to use any ASE [1] calculator that you can import and evaluate energy
and forces with for optimisations.

The calculator class is specified with the `--ase-class` key, which does not need to be from the
main ASE project, just like here where we use XTB imported as `xtb.ase.calculator.XTB` [2].

The parameters of the calculator, as keyword args in a JSON string are supplied with the
`--ase-kwargs='{"method":"GFN2-xTB"}'` key here. Notice that JSON uses double quotes (`"`) and on
needs to put the string into quotes (single quotes then `'`, or use `\"` appropriately)

References
----------

- [1] Atomic Simulation Environment (ASE) https://wiki.fysik.dtu.dk/ase/about.html
- [2] Extended Tight Binding (XTB) https://github.com/grimme-lab/xtb-python
