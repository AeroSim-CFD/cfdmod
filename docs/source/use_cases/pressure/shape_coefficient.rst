*****************
Shape Coefficient
*****************

Shape coefficient describe the behavior of the resulting force applied to a surface.

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

.. important:: Note that each triangle force has a **direction** attached to it. Its direction is the same as the normal for the given triangle. 

Even so, we can define a resulting force for the surface, by summing its triangles forces:

.. math::
   F_{res} = \sum{f_{i}} = p_{1} A_{1} + p_{2} A_{2} + p_{3} A_{3} + p_{4} A_{4} = q (c_{p1} A_{1} + c_{p2} A_{2} + c_{p3} A_{3} + c_{p4} A_{4})

.. important:: Note that the resulting force has a **direction** attached to it. The direction of each triangle force must be accounted for when summing to compose the **surface resulting force**. If all triangles are **coplanar**, then it can be perfomed a scalar sum and the resulting direction is the same as to the triangles.


The shape coefficient is based on the definition of an area of influence.
This area is delimited by x, y and z intervals.
Each combination of the intervals results in a **region**.

To calculate the shape coefficient of a given region, the triangles which center lies inside this region must be filtered.
Then, the resulting force is evaluated. After that, the shape coefficient can be calculated as:

.. math::
   C_{e} = \frac{F_{res}}{q A_{surf}} = \frac{(c_{p1} A_{1} + c_{p2} A_{2} + c_{p3} A_{3} + c_{p4} A_{4})}{A_{surf}}

And we can obatin its **maximum, minimum, RMS and average** values.

By definition, the shape coefficient is a property of a surface or an area.
It is used to evaluate **wind loads on primary and secondary structures**, such as beans, coating and sealing components.

Structural engineers might use the shape coefficient for wind load evaluation on superficial and long elements.

Artifacts:
==========

#. A lnas file: It contains the information about the mesh.
#. HDF time series: It contains the pressure coefficient signals indexed by each of the mesh triangles.
#. Zoning information: Necessary for defining the bounding area for calculating shape and liquid force coefficients.

Outputs:
========

#. **Adimensionalized time series**: shape coefficient time series for each region.
#. **Regions mesh**: new mesh generated using the region information and the original mesh.
#. **Statistical results**: maximum, minimum, RMS and average values for the shape coefficient time series, for each region.
#. **VTK File**: contains the statistical values inside the region mesh (VTK).