*****************
Shape Coefficient
*****************

Shape coefficient describe the behavior of the resulting force applied to a surface.

For example, consider a triangular surface.
At each of its vertices, pressure signals are obtained.
However, we can calculate the resulting force by summing the pressure load of each vertex.
To do so, pressure is transformed into a force by multiplying by an **area of influence**.

For triangles, we can consider that a third of its area is influenced by each vertex.

.. image:: /_static/pressure/triangle_area.png
    :width: 50 %
    :align: center

In that way, we can define a resulting force for each vertex as:

.. math::
   f_{i} = p_{i} A_{i} = \frac{p_{i} A_{T}}{3}

Even so, we can define a resulting force for the triangle, by summing its vertex forces:

.. math::
   F_{res} = \sum{f_{i}} = p_{1} A_{1} + p_{2} A_{2} + p_{3} A_{3} = (p_{1} + p_{2} + p_{3}) \frac{A_{T}}{3}

The shape coefficient is based on the definition of an area of influence, that can be a set of triangles.
Or it can also be defined for a whole surface, or even a set of surfaces.
If it contains more than one triangle, the calculation must be recursive for each of the triangles, then each of the surface.
To get the shape coefficient, the **resulting pressure** for a set of triangles, that define a surface or a set of surfaces, must be calculated:

.. math::
   p_{res} = \sum{\frac{F_{res}}{A_{res}}} = \frac{p_{1} A_{1} + p_{2} A_{2} + p_{3} A_{3}}{A_{1} + A_{2} + A_{3}} = \frac{p_{1} + p_{2} + p_{3}}{3}

Then the shape coefficient for the defined area is:

.. math::
   C_{e} = \frac{p_{res}(t) - p_{\infty}(t)}{q}

And we can obatin its **maximum, minimum and average** values.

By definition, the shape coefficient is a property of a surface or an area.
It is used to evaluate **wind loads on primary and secondary structures**, such as beans, coating and sealing components.

Structural engineers might use the shape coefficient for wind load evaluation on superficial and long elements.

Artifacts:
==========

#. A lnas file: It contains the information about the mesh.
#. HDF time series: It contains the pressure signals indexed by each of the mesh vertices.
#. Domain static pressure time series: It contains the pressure signals for probes far away from the building.
#. Zoning information: Necessary for defining the bounding area for calculating shape and liquid force coefficients. It is not necessary for pressure coefficient use case only. 

Outputs:
========

#. **Adimensionalized time series**: shape coefficient time series for each region.
#. **Regions mesh**: new mesh generated using the region information and the original mesh.
#. **Statistical results**: maximum, minimum, RMS and average values for the shape coefficient time series, for each region.
#. **VTK File**: contains the statistical values inside the region mesh (VTK).