geometric-neb --engine ase \
  --ase-class=mace.calculators.mace.MACECalculator \
  --ase-kwargs='{"model_paths":"'"$HOME"'/.cache/mace/MACE-omol-0-extra-large-1024.model","mace_device":"cpu","default_dtype":"float64","mace_head":"omol"}' \
  --images 11 --align no --prefix hcn_mace \
  HCN.xyz HCN.xyz
