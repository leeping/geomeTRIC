.. _meci:

Conical intersections
=====================

Theory
------

A conical intersection (CI) is a sub-manifold or "seam" of the configuration space where the energy difference between two states becomes degenerate and there exists a derivative discontinuity in the ground and excited state potential energy surfaces along a two-dimensional space called the branching plane.
There also exist other types of potential energy surface crossings, for example those induced by molecular symmetry, and surface crossings between states of different spins.
A minimum energy conical intersection (MECI) or crossing point (MECP) is defined as the structure that (locally) minimizes the energy subject to the constraint that the energy gap is zero between the two states.

In geomeTRIC, the optimization of MECI / MECP geometries is possible using the penalty-constrained algorithm of `Levine et al. <https://pubs.acs.org/doi/10.1021/jp0761618>`_
This method does not require the use of nonadiabatic couplings as many other methods do.
Conceptually, this method minimizes an objective function that is the sum of the average energy of two states, plus a penalty function that depends on the energy gap between the states as:

.. math::
    L = \frac{1}{2}(E_I + E_J) + \sigma \frac{\Delta E_{IJ}^2}{\Delta E_{IJ} + \alpha}

The two parameters of the penalty function control, :math:`\sigma` (``meci_sigma``) and :math:`\alpha` (``meci_alpha``), correspond to scaling and width parameters that control the shape of the penalty function. 

As :math:`\alpha` goes to zero, the penalty function tends toward a limiting value of :math:`\sigma |E_I - E_J|`, producing a "V-shaped" derivative discontinuity of the objective function perpendicular to the seam that is also a minimum if a large enough value of :math:`\sigma` is used.
Although the energy gap is exactly zero at the minimum, the derivative discontinuity makes the application of unconstrained optimization methods impossible; that is essentially the reason why other methods require the nonadiabatic coupling vectors in order to project out the directions along the branching plane containing the discontinuity.

Here, using a finite value of :math:`\alpha` produces a smooth objective function but also a finite energy gap, and the larger the value, the smoother the function but the larger the gap at the optimized structure.
The default values of parameters are :math:`\sigma = 3.5, \alpha = 0.025`, and can produce a final optimized structure with an energy gap of :math:`\Delta E_{IJ} < 5 \times 10^{-4}` a.u. (0.01 eV or 0.3 kcal/mol) for the the green fluorescent protein (GFP) chromophore at the SA-CASSCF(2,2)/cc-pVDZ level of theory.

Usage
-----

To use MECI / MECP optimization in geomeTRIC, you may use any engine, although when this was written only TeraChem has been tested for MECI calculations using SA-CASSCF and Q-Chem has been tested for finding the MECP between two SCF solutions.

The input file for the calculation should calculate the gradient for one of the states desired. Provide an input file for the other state using ``--meci <second_input_file>``.
Internally, geomeTRIC will create a "MECI engine" that contains two Engine objects for computing :math:`E_I` and :math:`E_J`, then these values are used to compute the MECI objective function and its gradient.
The two input files should use the same level of theory and other settings, differing only in the energy and gradient of the target state.
Alternatively, if the quantum chemistry code is able to compute the MECI objective function directly, use ``--meci engine`` instead of passing a second input file. 

    Note: Because this is advanced usage, you may need to modify the output file parsers in the Engine in order to correctly obtain the energy and gradient of each state. In that case, please consider making a code contribution back to the main repository.

For a smaller energy gap, it is recommended to start from the optimized structure using default parameters and reducing ``meci_alpha``.
Values of ``meci_alpha`` going down to ``1e-3`` have been tested, resulting in energy gaps of :math:`\Delta E_{IJ} \approx 1 \times 10^{-4}` a.u.