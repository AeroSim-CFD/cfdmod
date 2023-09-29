********
Pressure
********

The **Pressure** module handles the analysis and post processing of pressure time series data over a body.
Data comes from CFD transient simulations, and by definition is attached to a mesh.

Mesh describes a body geometry, and contains a set of **discrete vertices** and a **set of triangles**.
Triangles represent the link between 3 different vertices. 
Illustration of a mesh and a mesh triangle can be seen below:

.. image:: /_static/pressure/mesh.png
    :width: 65 %
.. image:: /_static/pressure/triangle.png
    :width: 30 %

Pressure data is extracted at each of the mesh vertices.
The frequency for exporting pressure is set during the simulation setup.
The resulting data has the form of a signal, or a time series.
An example of a pressure signal is shown below:

.. image:: /_static/pressure/pressure_signal.png
    :width: 65 %
    :align: center

The analysis of this signal is based on statistical operations, such as finding the **maximum, minimum or average** values for each vertex.
But first, it needs to be adimensionalized, dividing it by a **dynamic pressure**.
When applied this transformation to the pressure signal, **pressure coefficients** are obtained.
The pressure coefficients can be defined as follows:

.. math::
   c_{p} = \frac{p_{v}}{\frac{1}{2} \rho V ^ 2}

For engineering purposes, other coefficients come in handy, such as the **shape coefficient**.
Shape coefficient describe the behavior of the resulting force applied to a surface.

For example, consider a triangular surface.
At each of its vertices, pressure coefficients are computed based on the signal obtained.
However, we can calculate the resulting force by summing the pressure load of each vertex.
To do so, pressure is transformed into a force by multiplying by an **area of influence**.

For triangles, we can consider that a third of its area is influenced by each vertex.

.. image:: /_static/pressure/triangle_area.png
    :width: 50 %
    :align: center

In that way, we can define a resulting force for each vertex as:

.. math::
   f_{v} = p_{v} * A_{v}

Or we can define a resulting force for the triangle, by summing its vertex forces:

.. math::
   F_{res} = \sum{f_{v}} = p_{v1} * A_{v1} + p_{v2} * A_{v2} + p_{v3} * A_{v3}

The shape coefficient is based on the definition of an area of influence, that can be a set of triangles.
Or it can also be defined for a whole surface, or even a set of surfaces.
To get the shape coefficient, the **resulting pressure** for a set of triangles, that define a surface or a set of surfaces, must be calculated:

.. math::
   p_{res} = \sum{\frac{F_{res}}{A_{res}}} = \frac{p_{v1} * A_{v1} + p_{v2} * A_{v2} + p_{v3} * A_{v3}}{A_{v1} + A_{v2} + A_{v3}}

Then the shape coefficient for the defined area is:

.. math::
   C_{e} = \frac{p_{res}}{\frac{1}{2} \rho V ^ 2}

And we can obatin its **maximum, minimum and average** values.

Another important coefficient is the **liquid force coefficient**.
This is defined as a liquid resulting pressure between two surfaces.

.. math::
   C_{f} = \frac{p_{res1} * A_{res1} - p_{res2} * A_{res2}}{\frac{1}{2} \rho V ^ 2 A_{rep}}

Like the other coefficients, we can apply statistical analysis to the liquid force coefficient.