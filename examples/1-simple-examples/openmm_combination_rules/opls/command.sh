cp ../captan.pdb .
geometric-optimize --pdb captan.pdb --engine openmm state.xml --converge set GAU_TIGHT energy 2e-2
