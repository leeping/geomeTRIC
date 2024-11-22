.. _neb:

Nudged elastic band
===================

Basics
------
The Nudged Elastic Band (NEB) method is a chain-of-states approach used to identify the minimum energy pathway (MEP) by optimizing a series of molecular structures (images) that connect two energy basins on a potential energy surface (PES).
The resulting optimized sequence of images approximates the MEP, with the highest-energy image serving as a starting structure for :ref:`transition state optimization <transition>` to precisely locate the first-order saddle point.

There are two main components of forces that guide each image closer to MEP. These forces, which each image experiences, can be expressed as:

.. math::
    \mathbf{F}_i = \mathbf{F}_{\mathrm{PES}}^{\perp}(\mathbf{r}_i) + \mathbf{F}_{\mathrm{spring}}^{\parallel}(\Delta \mathbf{r}_{i+1,i}, \Delta \mathbf{r}_{i-1, i})
    :label: neb_force

where :math:`\mathbf{r}_i` is the geometry of the :math:`i`-th image and :math:`\perp, \parallel` indicate that each force contribution is decomposed into perpendicular and parallel components to the path, and only the indicated component of each force contribution is included for NEB.
If all the components of both forces (:math:`\mathbf{F}_{\mathrm{PES}}` + :math:`\mathbf{F}_{\mathrm{spring}}`) are applied to the images, the chain is called a plain band. A hybrid band omits the parallel force from the PES (:math:`\mathbf{F}_{\mathrm{PES}}^{\perp}` + :math:`\mathbf{F}_{\mathrm{spring}}`).
geomeTRIC is capable of optimizing all three types of bands by minimizing the force components iteratively.

The root-mean-squared (RMS) Cartesian gradients of each images (:math:`g_\mathrm{RMS}`) are calculated as:


.. math::
    g_\mathrm{RMS} = \sqrt{\frac{1}{N_{\mathrm{atoms}}} \displaystyle\sum_{i=1}^{N_{\mathrm{atoms}}} ({g_{x_i}}^2 + {g_{y_i}}^2 + {g_{z_i}}^2)}
    :label: grms

where :math:`g_{x_i}`, :math:`g_{y_i}`, and :math:`g_{z_i}` represent the gradients along the x, y, and z Cartesian axes, respectively.
The NEB calculation will continue iterating until the average and maximum :math:`g_\mathrm{RMS}` of all the images fall below convergence criteria.

Usage
-----

To run the NEB calculation with geomeTRIC, you need two input files:a QC input file and an xyz file containing the input chain coordinates.
The input xyz file must contain at least the numer of images specified by the ``--images [11]`` argument.
These two input files can be provided by running ``geometric-neb qc.input chain.xyz`` on the command line.
The chain coordinates from the input xyz file will override the molecular geometry from the QC input file.

By default, geomeTRIC will align all the images with the first image. This could be turned off by passing ``--align no``.
If ``--optep yes`` is passed, the two endpoints of the input chain will be optimized before alignment.

During optimization, geomeTRIC writes an xyz file of an image that climbs up towards the first-order saddle point(``qc.tsClimb.xyz``) in the working directory.
Once the NEB calculation converges, the climbing image can be used as a high-quality initial guess for the :ref:`transition state optimization <transition>`.

Since the single point energy/gradient calculations for each images in :math:`\mathbf{F}_{\mathrm{PES}}` are independent, they can be carried out as separate parallel tasks using tools such as :ref:`Work Queue <installcctools>`, :ref:`BigChem <installbigchem>`, or `QCFractal <http://docs.qcarchive.molssi.org/projects/QCFractal/en/stable/>`_.

.. note::
    geomeTRIC only supports Cartesian coordinate system for the NEB method.
    If ``--coordsys [tric]`` is specified and ``--optep yes`` is passed, the coordinate system will be used to optimize the two endpoints of the input chain.

Example
-------

See ``examples/1-simple-examples/hcn_hnc_neb`` directory.


Theory
------

The NEB method follows the similar procedures for :ref:`optimization steps <optimization_step>` and :ref:`step size control <optimization_stepsize>` as geometry optimizations.
The details of how each force component is applied will be explained here, along with how the step quality is calculated.

Force components
""""""""""""""""

The gradients are calculated based on the force components described in Equation (:eq:`neb_force`).
geomeTRIC implemented Henkelman and Jónsson's `improved tangent <https://doi.org/10.1063/1.1323224>`_ and `climibing image <https://doi.org/10.1063/1.1329672>`_ method.
The perpendicular and parallel forces are obtained as following:

.. math::
    \begin{aligned}
    & \mathbf{F}_{\mathrm{PES}}^{\perp}(\mathbf{r}_i) = \mathbf{F}_{\mathrm{PES}}(\mathbf{r}_i) - \mathbf{F}_{\mathrm{PES}}(\mathbf{r}_i) \cdot \hat{\mathbf{\tau}}_i\\
    & \mathbf{F}_{\mathrm{spring}}^{\parallel}(\Delta \mathbf{r}_{i+1,i}, \Delta \mathbf{r}_{i-1, i}) = k([(\mathbf{r}_{i+1} - \mathbf{r}_i) - (\mathbf{r}_i - \mathbf{r}_{i-1})] \cdot \hat{\mathbf{\tau}}_i) \hat{\mathbf{\tau}}_i
    \end{aligned}

The tangent vector (:math:`\hat{\mathbf{\tau}}_i`) is defined as:

.. math::
    \hat{\mathbf{\tau}}_i=
    \begin{cases}
        \hat{\mathbf{\tau}}_i^+ = \mathbf{r}_{i+1} - \mathbf{r}_i& \textrm{if}\qquad E_{i+1} > E_{i} > E_{i-1}\\
        \hat{\mathbf{\tau}}_i^- = \mathbf{r}_i - \mathbf{r}_{i-1}& \textrm{if}\qquad E_{i+1} < E_{i} < E_{i-1}
    \end{cases}

where :math:`E_{i}` is the energy of :math:`i`-th image.

For the images located at extrema, the following tangent is applied:

.. math::
    \hat{\mathbf{\tau}}_i=
    \begin{cases}
        \hat{\mathbf{\tau}}_i^+ \Delta E_i^{\mathrm{max}} + \mathbf{\hat{\tau}}_i^- \Delta E_i^{\mathrm{min}}  & \textrm{if}\qquad E_{i+1} > E_{i-1}\\
        \hat{\mathbf{\tau}}_i^+ \Delta E_i^{\mathrm{min}} + \mathbf{\hat{\tau}}_i^- \Delta E_i^{\mathrm{max}}  & \textrm{if}\qquad E_{i+1} < E_{i-1}
    \end{cases}

where

.. math::
    \Delta E_i^{\mathrm{max}} = max(|E_{i+1} - E_i|, |E_{i-1} - E_i|) \\
    \Delta E_i^{\mathrm{min}} = min(|E_{i+1} - E_i|, |E_{i-1} - E_i|)

The tangent vector is normalized and applied to project the force components accordingly.
During the optimization, when the maximum RMS gradient of the chain falls below a threshold (default set at 0.5 ev/Ang using ``--climb [0.5]``), the highest energy image (:math:`i_{\mathrm{max}}`) is switched to climbing mode.
The climbing image receives a newly defined force, which is:

.. math::
    \mathbf{F}_i = -\nabla E(\mathbf{r}_{i_{\mathrm{max}}}) + 2 (\nabla E(\mathbf{r}_{i_{\mathrm{max}}}) \cdot \hat{\mathbf{\tau}}_{i_{\mathrm{max}}})\hat{\mathbf{\tau}}_{i_{\mathrm{max}}}

Once both the average and maximum gradient of :math:`i`-th image satisfy the convergence criteria, which are 0.025 and 0.05 eV/Ang, respectively by default, the image is locked.
The locked images won't be moved until their gradients exceed the convergence criteria and they are unlocked.
The NEB calculation will converge when the average and maximum of RMS-gradient of all the images fall below the criteria.

Trust radius adjustment
"""""""""""""""""""""""

The NEB method assesses step quality through changes in band energy and gradients. The step quality based on energy (:math:`Q_E`) is expressed as:

.. math::
    Q_E =
    & \begin{cases}
    & \frac{2\Delta E_{\mathrm{pred}} - \Delta E_{\mathrm{actual}}}{\Delta E_{\mathrm{pred}}} & \textrm{if }\Delta E_{\mathrm{actual}} > \Delta E_{\mathrm{pred}} > 0\\
    & \frac{\Delta E_{\mathrm{actual}}}{\Delta E_{\mathrm{pred}}} & \textrm{else }
    & \end{cases} \\

where :math:`\Delta E_{\mathrm{actual}}` represents the energy difference between the previously iterated and current chain.
The :math:`\Delta E_{\mathrm{pred}}` is calculated using the same equation as the :ref:`predicted energy <step_quality>` of the geometry optimization step.

The step quality based on gradients (:math:`Q_g`) is calculated as:

.. math::
    Q_g = 2 - \frac{g_{\mathrm{curr}}}{max(g_{\mathrm{pred}}, \frac{g_{\mathrm{prev}}}{2}, \frac{g_{\mathrm{conv}}}{2})}

where :math:`g_{\mathrm{curr}}` and :math:`g_{\mathrm{prev}}` are the average RMS Cartesian gradients of current and previously iterated chain, respectively.
:math:`g_{\mathrm{conv}}` is the average gradient convergence criterion (``--avgg [0.025]``).
Predicted Cartesian gradients of each image (:math:`g_{\mathrm{cart}}`) is calculated as

.. math::
    g_{\mathrm{cart}} = \mathbf{H} \cdot \boldsymbol \delta +  \mathbf{g}

where :math:`\boldsymbol \delta, \mathbf{g}, \mathbf{H}` are the displacement, gradient, and Hessian in Cartesian coordinate.
:math:`g_{\mathrm{pred}}` is the average RMS of :math:`g_{\mathrm{cart}}` calculated using Eq. :eq:`grms`.

The larger value is chosen as the overall step quality (:math:`Q`) between the two values.
The overall step quality is then used to adjust the trust radius (:math:`R_{\mathrm{trust}}`) as following:

* :math:`Q \geq 0.50` : "Good" step, trust radius is increased by a factor of :math:`\sqrt{2}`, but not greater than the maximum.
* :math:`0.50 > Q \geq 0.00` : "Okay" step, trust radius is unchanged.
* :math:`0.00 > Q \geq -1.0` : "Poor" step, trust radius is decreased by setting it to :math:`0.5 \times \mathrm{min}(R_{\mathrm{trust}}, \mathrm{RMSD})`, but not lower than the minimum.
* :math:`Q < -1.0` : Step is rejected in addition to decreasing the trust radius as above.
