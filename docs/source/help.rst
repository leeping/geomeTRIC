.. _help:

Help & Troubleshooting
======================

This page includes tips on how to resolve issues that may be encountered when using geomeTRIC.

Convergence failure
-------------------

There are many potential causes of convergence failure, including:

Energy and gradient are not consistent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If there is any inconsistency between the gradient and the energy, then the optimization may perform poorly.
Because geomeTRIC uses the gradient to predict the first-order energy change from the optimization step,
inconsistencies in the gradient can result in large differences between the predicted and actual energy change.
This will be seen as "poor step quality" by the optimizer, resulting in a reduction of the trust radius, however,
poor step quality resulting from an inaccurate gradient is independent of the step size.  
As a result, the optimizer may take a large number of steps with a very small trust radius, eventually reaching the step size limit.
An example of this behavior is below:

.. parsed-literal::
    Step  291 : Displace = :green-bold:`1.197e-03`/2.616e-03 (rms/max) Trust = 1.200e-03 (:red-bold:`-`) Grad_T = 8.775e-04/3.583e-03 (rms/max) E (change) = -723.3158195891 (:red-bold:`+4.858e-06`) Quality = :red-bold:`-0.096`
    Hessian Eigenvalues: 3.87882e-04 2.26181e-02 2.30000e-02 ... 8.22727e-01 8.48434e-01 1.91366e+00
    Step  292 : Displace = :green-bold:`1.182e-03`/2.861e-03 (rms/max) Trust = 1.200e-03 (:red-bold:`-`) Grad_T = 1.176e-03/4.190e-03 (rms/max) E (change) = -723.3158066655 (:red-bold:`+1.292e-05`) Quality = :red-bold:`-0.180`
    Eigenvalues below 0.0000e+00 (-6.9363e-03) - returning guess
    Hessian Eigenvalues: 2.30000e-02 2.30000e-02 2.30000e-02 ... 8.16921e-01 8.39493e-01 8.90193e-01
    Step  293 : Displace = 1.266e-03/2.572e-03 (rms/max) Trust = 1.200e-03 (:red-bold:`-`) Grad_T = 9.509e-04/3.856e-03 (rms/max) E (change) = -723.3158095680 (-2.903e-06) Quality = 0.076
    Hessian Eigenvalues: 2.90187e-03 2.30000e-02 2.30000e-02 ... 8.24402e-01 8.47009e-01 1.35974e+00
    Step  294 : Displace = :green-bold:`1.191e-03`/3.339e-03 (rms/max) Trust = 1.200e-03 (:red-bold:`-`) Grad_T = 9.151e-04/3.595e-03 (rms/max) E (change) = -723.3157993647 (:red-bold:`+1.020e-05`) Quality = :red-bold:`-0.139`
    Eigenvalues below 0.0000e+00 (-6.4253e-01) - returning guess
    Hessian Eigenvalues: 2.30000e-02 2.30000e-02 2.30000e-02 ... 8.16921e-01 8.39493e-01 8.90193e-01
    Step  295 : Displace = 1.297e-03/2.610e-03 (rms/max) Trust = 1.200e-03 (:red-bold:`-`) Grad_T = 1.030e-03/4.287e-03 (rms/max) E (change) = -723.3157880464 (:red-bold:`+1.132e-05`) Quality = :red-bold:`-0.161`


To test for whether this is causing the convergence failure, calculate a high-quality numerical gradient 
using the final structure from the failed or failing optimization. (A recommended setting is to use a 5-point 
finite difference stencil and a step size of :math:`10^{-3}` Bohr). If there is a maximum difference of :math:`10^{-4}` Hartree/Bohr
or greater between the analytic and numerical gradient, then there may be a problem.  A gradient difference 
of :math:`10^{-3}` Hartree/Bohr or greater indicates a major problem.  A script for computing the numerical gradient is provided
in ``tools/finite_difference_grad.py``, but using the engine's native finite difference code may be more efficient.

This problem is common for DFT calculations that use the default quadrature grid.
In multiple packages, an inconsistency between the analytic and numerical gradient on the order of :math:`10^{-4}` Hartree/Bohr
will exist for medium to large-sized systems (30-100 atoms) when using a wide range of common DFT functionals (such as B3LYP, PBE, and many others).
This is a well-known issue that also causes problems when performing frequency analysis in systems containing
low-frequency vibrational modes.
To resolve the issue, increase the DFT quadrature grid size to 99 radial points / 590 angular points or greater
(this is larger than the default in most software packages).
This will increase the computational cost, but it also highlights the need for accurate gradients if a
high-quality optimized structure is desired.
If errors still persist in the analytic gradient, then tightening other numerical thresholds (e.g. integral thresholds or GPU precision) is recommended.

GeomeTRIC has some features in the optimization algorithm to mitigate this problem.
Sometimes, the erroneous forces have a translational and/or rotational component which manifest in the output structures
as an overall drifting or tumbling motion.
When a low-quality step dominated by overall translation/rotation is detected, geomeTRIC will switch on a
projection step that removes the overall translational/rotational component of the force.
This is the default behavior as of version 1.0, but it can be removed (or turned on from the beginning) using the ``--subfrctor`` option.

In the development version as of Nov. 2023 (and to be released in version 1.1), the optimization behavior is further modified
by allowing the trust radius to decrease below the RMS displacement convergence threshold.
This will cause the optimization to converge instead of taking a very large number of small, low-quality steps when the gradient
and energy change criteria have already been met.
Because this behavior is arguably causing convergence "artificially", a note is printed out when this occurs::

    *_* Trust radius is lower than RMS displacement convergence criterion; please check gradient accuracy. *_*

Users who need a higher-quality optimized structure are advised to run a second geometry optimization using the converged structure from the preceding one,
with the gradient accuracy increased by tightening numerical thresholds as discussed above.

Depending on the electronic structure method and software being used, there may be other sources of error in the analytic gradient.
Finite difference testing of the gradient is always recommended as a "first line" diagnosis of poor convergence behavior.

Energy / gradient calculation failed to converge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common problem in electronic structure calculations is SCF convergence failure, which could occur for many reasons including ground state degeneracy, 
the presence of transition metals in the structure, "physically unreasonable" structures, insufficient integral / quadrature precision, and others.
If the convergence failure occurs for the initial energy/gradient calculation, the structure should be visually inspected for close contacts, unit conversion errors, and/or other "physically unreasonable" features.
If the initial structure is reasonable, the user is advised to adjust SCF convergence settings and other thresholds in the input file such that a single-point gradient calculation converges successfully, and in some engines (e.g. TeraChem / Q-Chem), files containing initial orbitals from a separate single-point calculation may be provided.
If the convergence failure occurs for a structure other than the initial one and the structure is physically unreasonable, then another error may be the root cause, such as inconsistency between the energy and gradient, unsatisfiable constraint, or a problem with geomeTRIC's internal coordinate system and/or optimization algorithm.

Potential energy surface is discontinuous or not smooth
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The potential energy surface (PES) can sometimes be discontinuous, for example, 
when the system has multiple possible solutions for the electronic ground state depending on the structure or initial wavefunction guess.
The optimization could still continue normally if the optimization step crosses over a discontinuity, the step is not rejected, and further steps do not "re-cross" the surface of discontinuity; however, such fortuitous behavior is by no means guaranteed.
Unfortunately, it is not currently possible in geomeTRIC (or other known optimization software) to optimize a structure reliably when there are discontinuities in the potential energy surface.
There is no single recommended solution, and a good initial step would be to characterize these discontinuities by finding the crossing point of the potential surfaces using the ``--meci`` feature.

Constraints cannot be satisfied
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GeomeTRIC can handle optimizations with multiple constraints, however, it is easy to specify constraints that are chemically unreasonable or cannot be satisfied.

By default the atom indices used in the geomeTRIC constraint file are 1-based, i.e. there is no atom numbered zero.
This could cause errors in the constraint specification, for example, if the chosen atoms for the constraint atoms are different from the intended ones.
The optimization log file prints out constraint values that are far from being satisfied, and these should be double-checked to ensure constraints are specified correctly.

In some cases, specifying multiple constraints can cause the optimization problem to become overdetermined (for example, if six angle constraints are specified to freeze the bond angles around a tetrahedrally bonded atom; there are only five degrees of freedom). 
The user is advised to gradually reduce the number of constraints until the optimization behaves normally, and check the output for correctness.

In other cases, a reasonable constraint specification can still fail to converge, either because the constraint cannot be satisfied, and/or the optimization does not take good-quality steps.  
This happens more commonly if the constraint causes large forces to appear.
The behavior of the optimization can be modified by setting ``--enforce 0.1`` (to enforce exact constraint satisfaction when the structure comes "close" to satisfying the constraint), or ``--conmethod 1`` (changing the constraint algorithm to satisfy constraints more rapidly).
The performance of constrained optimization for a certain setting varies depending on the system; for example, ``--conmethod 1`` is more reliable for rigid-molecule optimizations and intermolecular distance constraints, but is less reliable for dihedral angle constraints.

Choice of internal coordinate system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Incorrect choice of the internal coordinate system could be a cause of geometry optimization failure.  For example, ``--coordsys dlc`` is not recommended for optimizing structures that contain multiple molecules; the default ``--coordsys tric`` is recommended instead (geomeTRIC was originally developed to implement the TRIC coordinate system).  If incorrect drifting or tumbling of the entire system is observed during optimization, then ``dlc`` may be used instead of ``tric``, however, this should not be necessary given the safeguards already implemented to prevent this behavior.  It is generally not recommended to use ``hdlc``, ``cart``, ``prim`` or ``tric-p`` for production calculations, as they are provided only for testing and comparison.  

Internal coordinate system, optimization algorithm, or some other part of geomeTRIC is incorrect or inefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is always possible for a bug or inefficiency in geomeTRIC to be the root cause of a failing optimization.  In general, an optimization is more difficult if the potential energy surface is slowly varying along some collective coordinates, or if the system is very large (such as a protein).
If the optimization is "on the way" to convergence but it has reached the maximum number of iterations, you may restart the calculation from the latest structure and/or increase the maximum number of iterations with the ``--maxiter`` option.  However, most systems should be converged within 300 cycles so we are interested in cases that fail to converge within this limit.

We are always interested in finding examples that help to improve the code.  If you have a failing calculation that you believe is due to a bug or inefficiency in geomeTRIC (and not the other potential causes listed above), please send us a message via GitHub or email.  
