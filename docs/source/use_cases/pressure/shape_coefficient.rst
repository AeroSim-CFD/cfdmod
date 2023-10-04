*****************
Shape Coefficient
*****************

Shape coefficient describe the behavior of the **resulting force applied to a surface**.

For example, consider a rectangular surface composed by triangles.
At each of its triangles' center, pressure signals are obtained.
However, we can calculate the resulting force by summing the pressure load of each triangle.
To do so, pressure is transformed into a force by multiplying by the respecting **triangle area**.

.. image:: /_static/pressure/surface.png
    :width: 50 %
    :align: center

In that way, we can define a resulting force for each triangle as:

.. math::
   f_{i} = p_{i} A_{i} = c_{pi} q A_{i}

Even so, we can define a resulting force for the surface, by summing its triangles forces:

.. math::
   F_{res} = \sum{f_{i}} = p_{1} A_{1} + p_{2} A_{2} + p_{3} A_{3} + p_{4} A_{4} = q (c_{p1} A_{1} + c_{p2} A_{2} + c_{p3} A_{3} + c_{p4} A_{4})

.. important::  Note that each triangle force has a **direction** attached to it. Its direction is the same as the normal for the given triangle. The direction of each triangle force must be accounted for when summing to compose the **surface resulting force**. If all triangles are **coplanar**, then it can be perfomed a scalar sum and the resulting direction is the same as to the triangles.

The shape coefficient is based on the definition of an area of influence.
This area is delimited by x, y and z intervals.
Each combination of the intervals results in a **region**.

.. note:: Check out the `definitions <./definitions.rst>`_ section for more information about **region and surface** definitions.

To calculate the shape coefficient of a given region, the **surface triangles** which center lies inside this region must be filtered.
Then, the resulting force is evaluated for the filtered triangles data. 
After that, the shape coefficient can be calculated.
For the previous example, considering that it only exists one region:

.. math::
   C_{e} = \frac{F_{res}}{q A_{surf}} = \frac{(c_{p1} A_{1} + c_{p2} A_{2} + c_{p3} A_{3} + c_{p4} A_{4})}{A_{surf}}

And we can obatin its **maximum, minimum, RMS and average** values.

By definition, the shape coefficient is a property of a surface or an area.
It is used to evaluate **wind loads on primary and secondary structures**, such as beans, coating and sealing components.

Structural engineers might use the shape coefficient for wind load evaluation on superficial and long elements.

For smaller components, it is essential to define small intervals for the **region definition**, which size should be comparable to the component of interest size.

For example, to evaluate the effect of wind pressure on windows, the intervals used for defining the shape coefficient regions should be about the window size, in a way that all triangles that form the window lie inside of this region.

An example of the parameters file required for calculating the shape coefficient can be seen below:

.. literalinclude:: /_static/pressure/Ce_params.yaml
    :language: yaml

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


Use cases:
==========

* **Walls**
* **Roofs**
* **Doors**
* **Windows**

Artifacts:
==========

#. **A lnas file**: It contains the information about the mesh.
#. **HDF time series**: It contains the pressure coefficient signals indexed by each of the mesh triangles.
#. **Parameters file**: It contains the zoning information for defining the bounding area, as well as other configs parameters.

Outputs:
========

#. **Dimensionless time series**: shape coefficient time series for each region.
#. **Regions mesh**: new mesh generated using the region information and the original mesh.
#. **Statistical results**: maximum, minimum, RMS and average values for the shape coefficient time series, for each region.
#. **VTK File**: contains the statistical values inside the region mesh (VTK).