*****************
Force Coefficient
*****************

The **Force Coefficient**, :math:`C_f`, is a dimensionless parameter that provides a generalized representation of the **resultant forces** experienced by an object within a fluid flow.
It offers a means to evaluate the **cumulative effect** of pressure coefficients, :math:`c_p` across different regions of an object's surface and how these pressures translate into aerodynamic forces.

:math:`C_f` is a fundamental tool for assessing lift, drag, and other forces crucial for the design and analysis of aerodynamic components.

Definition
==========

This coefficient is defined as a net resulting force coefficient of a body.
A body is composed by a **set of surfaces**.
For example, consider a building's canopy, where the lower surface is marked on red, and the upper surface is marked on green:

.. image:: /_static/pressure/marquee.png
    :width: 90 %
    :align: center

The net resulting force coefficient is defined as:

.. math::
   C_{f} = \frac{\sum F_{res}}{q A_{rep}} = \frac{F_{green} + F_{red}}{q A_{rep}} = \frac{\sum{c_{pi} A_{i}} + \sum{c_{pj} A_{j}}}{A_{rep}}

.. important:: Note that the net force coefficient has a direction attached to its definition. Its direction is the same as the resulting force direction.

It can also be defined for each axis direction:

.. math::
   C_{fx} = \frac{\sum Fx_{res}}{q A_{x}} = \frac{Fx_{green} + Fx_{red}}{q A_{x}} = \frac{\sum{c_{pi} A_{ix}} + \sum{c_{pj} A_{jx}}}{A_{x}}

.. math::
   C_{fy} = \frac{\sum Fy_{res}}{q A_{y}} = \frac{Fy_{green} + Fy_{red}}{q A_{y}} = \frac{\sum{c_{pi} A_{iy}} + \sum{c_{pj} A_{jy}}}{A_{y}}

.. math::
   C_{fz} = \frac{\sum Fz_{res}}{q A_{z}} = \frac{Fz_{green} + Fz_{red}}{q A_{z}} = \frac{\sum{c_{pi} A_{iz}} + \sum{c_{pj} A_{jz}}}{A_{z}}


The representative area is defined as a **projection of the surface area** for the body composed by the selected surfaces, for each direction.

To define the representative areas for each direction of the resulting force, we can sum the **projected area of each of one of the surfaces triangles**:

.. math::
   A_x = \sum Ax_{i}

   A_y = \sum Ay_{i}

   A_z = \sum Az_{i}

One can also define the representative area as a vector:

.. math::
   A_{rep} = [A_x, A_y, A_z]

.. important::
  This method is valid for bodies composed by **parallel surfaces that have the same surface area**. For more complex geometries, the method has to account for the different representative areas for each surface, **not implemented yet**.

Use Case
========

A common application of the net force coefficient requires sectioning the body in different **sub-bodies**.
To do so, a similar logic applied to the shape coefficient is used to **determine the respective sub-body** of each of the body's triangles.
If its center lies inside the sub-body volume, then it belongs to it.

The result is a sectionated body in different **sub-bodies for each interval**.
When sectioning the body, the respective representative area should be the same as the sub-body representative area.

.. note:: Check out the `definitions <./definitions.rst>`_ section for more information about **surface, body and sub-body** definitions.

Like the other coefficients, we can apply statistical analysis to the net force coefficient.

By definition, the net force coefficient is a **property of a body**.

It is used for **primary and secondary structures design**, such as **canopies** and **roof vents**.
It can also be used for evaluating the resultant wind action over a **building** or the **building paviments**.
It can be seen as the resulting effect of the wind induced force over a body.

Artifacts
=========

In order to use the force coefficient module, the user has to provide a **set of artifacts**:

#. **A lnas file**: It contains the information about the mesh.
#. **Parameters file**: It contains which surface inside the mesh is going to be used for evaluating net force coefficient as well as other configs parameters.
#. **HDF time series**: It contains the pressure coefficient signals indexed by each of the mesh triangles.

Which outputs the following data:

#. **Dimensionless time series**: force coefficient time series for each body.
#. **Statistical results**: maximum, minimum, RMS and average values for the force coefficient time series, for each body.
#. **VTK File**: contains the statistical values inside the original mesh (VTK).

An illustration of the force coefficient module pipeline can be seen below:

.. image:: /_static/pressure/Cf_pipeline.png
    :width: 90 %
    :align: center

Usage
=====

An example of the parameters file required for calculating the net force coefficient can be seen below:

.. literalinclude:: /_static/pressure/Cf_params.yaml
    :language: yaml

To invoke and run the calculation, the following command can be used:

.. code-block:: Bash

   poetry run python -m cfdmod.use_cases.pressure.Cf \
      --output {OUTPUT_PATH} \
      --cp     {CP_SERIES_PATH} \
      --mesh   {LNAS_PATH} \
      --config {CONFIG_PATH}

Or it can be generated together with the pressure data conversion:

.. code-block:: Bash

   poetry run python -m cfdmod.use_cases.pressure \
      --output {OUTPUT_PATH} \
      --cp     {CP_SERIES_PATH} \
      --mesh   {LNAS_PATH} \
      --config {CONFIG_PATH} \
      --Cf

Another way to run the force coefficient calculation, is through the `notebook <calculate_Cf.ipynb>`_

Data format
===========

.. note:: The rule for determining the region_idx is based on the **region index and the body name**.
        Input mesh can have multiple bodies, and each of them can be applied a specific zoning/region rule.
        Because of that, region_idx has to be composed by the **zoning region index joined by "-" and the body name**.
        This also guarantee that even if different bodies lie on the same region, the interpreted region for each of them will be different

.. list-table:: :math:`C_{fx}(t)`
   :widths: 15 15 15 15 15
   :header-rows: 1

   * - time_step_idx/region_idx
     - time_step
     - 0-Body1
     - 1-Body1
     - 0-Body2
   * - 0
     - 10000
     - 1.25
     - 1.15
     - -1.1
   * - 1
     - 11000
     - 1.5
     - 0.9
     - -1.15

.. list-table:: :math:`C_{fy}(t)`
   :widths: 15 15 15 15 15
   :header-rows: 1

   * - time_step_idx/region_idx
     - time_step
     - 0-Body1
     - 1-Body1
     - 0-Body2
   * - 0
     - 10000
     - 1.25
     - 1.15
     - -1.1
   * - 1
     - 11000
     - 1.5
     - 0.9
     - -1.15

.. list-table:: :math:`C_{fz}(t)`
   :widths: 15 15 15 15 15
   :header-rows: 1

   * - time_step_idx/region_idx
     - time_step
     - 0-Body1
     - 1-Body1
     - 0-Body2
   * - 0
     - 10000
     - 1.25
     - 1.15
     - -1.1
   * - 1
     - 11000
     - 1.5
     - 0.9
     - -1.15

.. list-table:: :math:`C_{fx} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - max
     - min
     - mean
     - std
     - skewness
     - kurtosis
   * - 0-Body1
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1-Body1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_{fy} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - max
     - min
     - mean
     - std
     - skewness
     - kurtosis
   * - 0-Body1
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1-Body1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_{fz} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - max
     - min
     - mean
     - std
     - skewness
     - kurtosis
   * - 0-Body1
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1-Body1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`Regions(indexing)`
   :widths: 50 50
   :header-rows: 1

   * - region_idx
     - point_idx
   * - 0-Body1
     - 0
   * - 1-Body1
     - 1

.. list-table:: :math:`Regions(definition)`
   :widths: 10 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - x_min
     - x_max
     - y_min
     - y_max
     - z_min
     - z_max
   * - 0-Body1
     - 0
     - 100
     - 0
     - 50
     - 0
     - 20
   * - 1-Body1
     - 100
     - 200
     - 0
     - 50
     - 0
     - 20

.. toctree::
   :maxdepth: -1
   :hidden:

   Calculate Cf <calculate_Cf.ipynb>
