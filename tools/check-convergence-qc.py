#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)

from forcebalance.molecule import *
import os, sys

# This code checks to see whether a Q-Chem optimization has converged
# to within the same criteria as GeomeTRIC.

M = Molecule(sys.argv[1], build_topology=False)
M.align(smooth=True)

Convergence_energy = 1e-6
Convergence_grms = 3e-4
Convergence_gmax = 4.5e-4
Convergence_drms = 1.2e-3
Convergence_dmax = 1.8e-3

for frame in range(1, len(M)):
    # Compute energy change, gradient mean norm / max norm, 
    Eprev = M.qm_energies[frame-1]
    E = M.qm_energies[frame]
    G = M.qm_grads[frame]
    displacement = M.xyzs[frame]-M.xyzs[frame-1]
    atomgrad = np.sqrt(np.sum((G.reshape(-1,3))**2, axis=1))
    rms_displacement = np.sqrt(np.mean(displacement**2))
    rms_gradient = np.sqrt(np.mean(atomgrad**2))
    max_displacement = np.max(displacement)
    max_gradient = np.max(atomgrad)
    # Convergence criteria
    Converged_energy = np.abs(E-Eprev) < Convergence_energy
    Converged_grms = rms_gradient < Convergence_grms
    Converged_gmax = max_gradient < Convergence_gmax
    Converged_drms = rms_displacement < Convergence_drms
    Converged_dmax = max_displacement < Convergence_dmax
   # Print status
    logger.info("Step %4i :" % frame),
    logger.info("Displace = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("\x1b[92m" if Converged_drms else "\x1b[0m", rms_displacement, "\x1b[92m" if Converged_dmax else "\x1b[0m", max_displacement)),
    logger.info("Grad = %s%.3e\x1b[0m/%s%.3e\x1b[0m (rms/max)" % ("\x1b[92m" if Converged_grms else "\x1b[0m", rms_gradient, "\x1b[92m" if Converged_gmax else "\x1b[0m", max_gradient)),
    logger.info("E (change) = % .10f (%s%+.3e\x1b[0m)" % (E, "\x1b[92m" if Converged_energy else "\x1b[0m", E-Eprev))
    if Converged_energy and Converged_grms and Converged_drms and Converged_gmax and Converged_dmax:
        logger.info("Converged! =D (Q-Chem takes %i)" % (len(M)))
        sys.exit()

logger.warning("Warning: Not Converged!")
