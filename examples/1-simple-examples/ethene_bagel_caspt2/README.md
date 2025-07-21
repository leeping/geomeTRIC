An example optimisation of the $\pi-\pi^\*$ state of ethene using XMS-CASPT2 
as implemented in Bagel. 
Here, Bagel is executed in parallel using MPI managed by slurm
with 4 MPI processes and 2 Bagel threads per process.

This optimisation will not converge using default parameters, since 
the minimum is located at a conical intersection where the 
adiabatic state energies are not differentiable. Therefore, the gradients 
evaluated by Bagel are not reliable and will not vanish at the minimum.
