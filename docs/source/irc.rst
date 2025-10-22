.. _irc:

Intrinsic reaction coordinate
=============================

Basics
------

The Intrinsic Reaction Coordinate (IRC) method aims to trace a minimum energy pathway on a potential energy surface (PES) starting with an optimized transition state (TS) structure.
This optimized TS structure sits at a first-order saddle point on the PES where the structure has only one imaginary vibrational frequency mode.
To begin, geomeTRIC calculates the Hessian and vibrational frequencies to confirm this imaginary mode. The calculation of Hessian can be carried in parallel using :ref:`BigChem <installbigchem>` or :ref:`Work Queue <installcctools>`.

Once the Hessian calculation is completed, the first step is taken in the positive direction of the corresponding eigenvector of the imaginary mode using mass-weighted internal coordinates.
The Hessian is updated using the :ref:`BFGS method <bfgs-update>` and the succeeding steps are guided by the instantaneous acceleration vectors.
The IRC method continues taking the steps until it meets the same convergence criteria as :ref:`geometry optimization <convergence_criteria>`.

geomeTRIC repeats the same procedure for the negative direction after the positive direction IRC converges. Once both directions have converged, one of the paths is reversed to concatenated with the other, using the TS structure as the connecting point.

Usage
-----

The IRC method can be used by passing ``--irc yes`` on the command line with ``geometric-optimize``. The input geometry should be an optimize TS structure.
geomeTRIC adjusts the IRC step size based on the step quality, and the maximum step size will be set equal to the initial trust radius ``--trust [0.3]``. The direction of IRC can be specified using ``--irc_direction [both]``.
The ``forward`` direction follows the imaginary mode from the frequency analysis as is, while the ``backward`` direction follows its opposite.
Once convergence is achieved, geomeTRIC will generate an output xyz file containing the IRC pathway.

Example
-------

See ``examples/1-simple-examples/hcn_hnc_irc`` directory.


Theory
------

The IRC method uses a large portion of the same code as the optimization algorithm. The two main differences are in obtaining steps and adjusting the trust radius.

Obtaining the IRC step
""""""""""""""""""""""

geomeTRIC implemented Gonzalez and Schlegel's `mass-weighted internal coordinates IRC method <https://doi.org/10.1021/j100377a021>`_.
The mass-weighted Wilson B-matrix (:math:`\mathbf{B}`) has elements of :math:`dq_i / (dx_j \sqrt{m_j})`, where :math:`m_j` is the atomic mass, and the G-matrix is calculated as :math:`\mathbf{G} = \mathbf{B}\mathbf{B}^T` (See :ref:`internal coordinate setup <internal_coordinate>`).

At the beginning of the first iteration, the mass-weighted step size (:math:`\mathbf{s}`) is calculated from the trust radius (:math:`R_{\mathrm{trust}}`) as follows:

.. math::
    \begin{aligned}
    & A = \sqrt{\sum_{i=1}^{N_{\mathrm{atoms}}} (\Delta x_i^2 + \Delta y_i^2 + \Delta z_i^2) \times m_i} \\
    & \mathrm{s} = R_{\mathrm{trust}} \times A
    \end{aligned}

where :math:`\Delta x`, :math:`\Delta y`, and :math:`\Delta z` are the normalized Cartesian displacements along the imaginary mode on the saddle-point.
The conversion factor :math:`A` is used in every iteration to convert the trust radius.

Each IRC step starts with taking a half-step towards a pivot point following the instantaneous accelerations (:math:`-\mathbf{G} \cdot \mathbf{g}`).
The pivot point (:math:`\mathbf{\mathrm{q}}^*`) is obtained by:

.. math::
    \mathbf{q}^* = \mathbf{q} - \frac{\mathrm{s}}{2} \cdot \frac{\mathbf{G} \cdot \mathbf{g}}{(\mathbf{g}^T \cdot \mathbf{G} \cdot \mathbf{g})^{1/2}}

where :math:`\mathbf{g}` is the gradients of internal coordinates :math:`\mathbf{q}`.
To reach the next point on the reaction path, another half-step needs to be taken from the pivot point along a vector that is 1) parallel to the acceleration vector of the next point and 2) has a scalar value equal to half of the mass-weighted step size (:math:`\mathrm{s}/2`).

First, a guessed point (:math:`\mathbf{q}_1^\prime`) is obtained by taking the same step as the initial half-step starting from the pivot point.
The guessed point is guided to the next guessed point (:math:`\mathbf{q}_2^\prime`) until the vector pointing from the pivot point to the guessed point satisfies the two conditions.

If we define the following two vectors:

.. math::
    \begin{aligned}
    & \mathbf{p} = \mathbf{q}_n^\prime - \mathbf{q}^* \\
    & \Delta\mathbf{q} = \mathbf{q}_{n+1}^\prime - \mathbf{q}_n^\prime
    \end{aligned}
    :label: vectors

:math:`\Delta\mathbf{q}` can be updated while keeping the scaler value of :math:`\mathbf{p}` equal to :math:`\mathrm{s}/2` until :math:`\mathbf{q}_n^\prime` reaches the point where :math:`\mathbf{p} + \Delta\mathbf{q}` is parallel to the acceleration vectors at point :math:`\mathbf{q}_{n+1}^\prime`.
The vectors, Hessian, and gradients in mass-weighted internal coordinates can be expressed as

.. math::
    \begin{aligned}
    & \Delta\mathbf{q}_\mathrm{M} = \mathbf{G}^{-1/2} \Delta\mathbf{q}\\
    & \mathbf{p}_\mathrm{M} = \mathbf{G}^{-1/2} \mathbf{p}\\
    & \mathbf{g}_\mathrm{M} = \mathbf{G}^{1/2} \mathbf{g}^{\prime}\\
    & \mathbf{H}_\mathrm{M} = \mathbf{G}^{1/2} \mathbf{H} \mathbf{G}^{1/2}\\
    \end{aligned}
    :label: mwic

where :math:`\mathbf{g}^{\prime}` represents the estimated gradients at the point :math:`\mathbf{q}_n^\prime`, using a quadratic expansion.
:math:`\mathbf{G}` is calculated at :math:`\mathbf{q}_n^\prime` as well.

The step size constraint can be expressed as:

.. math::
    (\mathbf{p}_\mathrm{M} + \Delta\mathbf{q}_\mathrm{M})^{T}(\mathbf{p}_\mathrm{M} + \Delta\mathbf{q}_\mathrm{M}) = (\frac{\mathrm{s}}{2})^2
    :label: const1

The other condition is satisfied at the convergence point (the next point), when the following equation holds true:

.. math::
    (\mathbf{g}_\mathrm{M} - \lambda \mathbf{p}_\mathrm{M}) + (\mathbf{H}_\mathrm{M} - \lambda \mathbf{I})\Delta\mathbf{q}_\mathrm{M} = 0
    :label: const2

where :math:`\lambda` is the Lagrangian multiplier and :math:`\mathbf{I}` is the identity matrix.

Eq. :eq:`const2` can be rearranged as follows:

.. math::
    \Delta\mathbf{q}_\mathrm{M} = -(\mathbf{H}_\mathrm{M} - \lambda \mathbf{I})^{-1}(\mathbf{g}_\mathrm{M} - \lambda \mathbf{p}_\mathrm{M})
    :label: delqm

:math:`\lambda` is calculated iteratively after introducing Eq. :eq:`delqm` to :eq:`const1`.
:math:`\Delta\mathbf{q}_\mathrm{M}` is then used to move :math:`\mathbf{q}_n^\prime` to :math:`\mathbf{q}_{n+1}^\prime` and new Eq. :eq:`vectors` and Eq. :eq:`mwic` are defined to calculate the next :math:`\Delta\mathbf{q}_\mathrm{M}`.
This process repeats until the norm of :math:`\Delta\mathbf{q}` falls below 1e-6. It then takes the rest of the half-step along :math:`\mathbf{p} + \Delta\mathbf{q}` from the pivot point, which completes an iteration.


Trust radius adjustment
"""""""""""""""""""""""

The step quality (:math:`Q`) is calculated in the same way as the :ref:`energy minimization step quality <step_quality>`.
The trust radius is adjusted as follows:

* :math:`Q \geq 0.75` : "Good" step, trust radius is increased by a factor of :math:`\sqrt{2}`, but not greater than the maximum.
* :math:`0.75 > Q \geq 0.50` : "Okay" step, trust radius is unchanged.
* :math:`Q < 0.50` : Step is rejected, trust radius is decreased by setting it to :math:`0.5 \times \mathrm{min}(R_{\mathrm{trust}}, \mathrm{RMSD})`, but not lower than the minimum