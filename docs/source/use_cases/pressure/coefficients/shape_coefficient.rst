*****************
Shape Coefficient
*****************

Shape coefficient describes the behavior of the **resulting pressure applied to a surface**.
It can be interepreted as a **resulting pressure coefficient** inside a given region.

It allows us to assess how the local pressure at different points inside a region combine themselves.

Definition
==========

The shape coefficient is a **dimensionless form of the resulting pressure signal**.

Consider a rectangular surface composed by triangles.
At each of its triangles' center, pressure signals are obtained.
However, we can calculate the resulting force by summing the pressure load of each triangle.
To do so, pressure is transformed into a force by multiplying by the respective **triangle area**.

.. image:: /_static/pressure/surface.png
    :width: 50 %
    :align: center

We can define a resulting force for each triangle as:

.. math::
   f_{i} = p_{i} A_{i} = c_{pi} q A_{i}

Even so, we can define a resulting force for the surface, by summing its triangles forces:

.. math::
   F_{res} = \sum{f_{i}} = p_{1} A_{1} + p_{2} A_{2} + p_{3} A_{3} + p_{4} A_{4} = q (c_{p1} A_{1} + c_{p2} A_{2} + c_{p3} A_{3} + c_{p4} A_{4})

.. important::
   Some peaks (minimum or maximum) of the pressure coefficient signal can be cancelled when calculating the shape coefficient.

The shape coefficient is based on the definition of an area of influence.
This area is delimited by x, y and z intervals.
Each combination of the intervals results in a **region**.

.. note:: Check out the `definitions <./definitions.rst>`_ section for more information about **region and surface** definitions.

To calculate the shape coefficient of a given region, the **surface triangles** which center lies inside this region must be filtered.
Then, the resulting force is evaluated for the filtered triangles data. 
After that, the shape coefficient can be calculated.
For the previous example, considering a case where only one region exists:

.. math::
   C_{e} = \frac{F_{res}}{q A_{region}} = \frac{(c_{p1} A_{1} + c_{p2} A_{2} + c_{p3} A_{3} + c_{p4} A_{4})}{A_{region}}

And we can obatin its **maximum, minimum, RMS and average** values.

Use Case
========

By definition, the shape coefficient is a property of a surface or an area.
It is used to evaluate **wind loads on primary and secondary structures**, such as beans, coating and sealing components.

Structural engineers might use the shape coefficient for wind load evaluation on superficial and long elements.

For smaller components, it is essential to define small intervals for the **region definition**, which size should be comparable to the component of interest size.

For example, to evaluate the effect of wind pressure on windows, the intervals used for defining the shape coefficient regions should be about the window size, in a way that all triangles that form the window lie inside of this region.

It can also be used to evaluate the resulting wind effect over **coating elements**, mounted on roofs or walls, such as panels.
Or even to evaluate the resulting effect over **doors**, and calculate the stress over its hinges.

Artifacts
=========

In order to use the shape coefficient module, the user has to provide a **set of artifacts**:

#. **A lnas file**: It contains the information about the mesh.
#. **HDF time series**: It contains the pressure coefficient signals indexed by each of the mesh triangles.
#. **Parameters file**: It contains the zoning information for defining the bounding area, as well as other configs parameters.

Which outputs the following data:

#. **Dimensionless time series**: shape coefficient time series for each region.
#. **Regions**: definition of each region generated with its bounds (x_min, x_max), (y_min, y_max), (z_min, z_max), and the region index.
#. **Regions mesh**: new mesh generated using the region information and the original mesh.
#. **Statistical results**: maximum, minimum, RMS and average values for the shape coefficient time series, for each region.
#. **VTK File**: contains the statistical values inside the region mesh (VTK).

An illustration of the shape coefficient module pipeline can be seen below:

.. image:: /_static/pressure/Ce_pipeline.png
    :width: 90 %
    :align: center

Usage
=====

An example of the parameters file required for calculating the shape coefficient can be seen below:

.. literalinclude:: /_static/pressure/Ce_params.yaml
    :language: yaml

.. literalinclude:: /_static/pressure/zoning_params.yaml
    :language: yaml
    :caption: zoning_params.yaml

To invoke and run the calculation, the following command can be used:

.. code-block:: Bash

   poetry run python -m cfdmod.use_cases.pressure.Ce \
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
      --Ce

# TODO: reference the notebooks

Data format
===========

.. important:: All tables for shape coefficient listed below are defined for **each of the body's surfaces**, unlike the other coefficients. The idea is to keep the processing for a single surface and not account for unrelated data. 

.. list-table:: :math:`C_e(t)`
   :widths: 33 33 33
   :header-rows: 1

   * - region_idx
     - timestep
     - C_e
   * - 0
     - 10000
     - 1.25
   * - 1
     - 10000
     - 1.15

.. list-table:: :math:`C_e (stats)`
   :widths: 20 10 10 10 10 20 20
   :header-rows: 1

   * - region_idx
     - max
     - min
     - avg
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_e(regions)`
   :widths: 10 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - x_min
     - x_max
     - y_min
     - y_max
     - z_min
     - z_max
   * - 0
     - 0
     - 100
     - 0
     - 50
     - 0
     - 20
   * - 1
     - 100
     - 200
     - 0
     - 50
     - 0
     - 20