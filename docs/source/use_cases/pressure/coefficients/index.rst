*********************
Pressure Coefficients
*********************

As stated before, the pressure data is relevant for structural engineers to understand the **wind induced stress**.

However, it comes in handy to adimensionalize the pressure data into **coefficients**.
Therefore, the analysis is **independent of the scale or unit system**.
It is a way to generalize the pressure data.

Available Coefficients
======================

There are several different coefficients, that are commonly used in the wind industry, such as:

* `Pressure Coefficient <./pressure_coefficient.rst>`_: It is a fundamental adimensionalization of the pressure data. It is obtained by dividing a pressure difference by the **dynamic pressure**.
* `Shape Coefficient <./shape_coefficient.rst>`_: It is equivalent to a resulting pressure coefficient over an **area of interest**. It is used to combine the pressure effects over the area, in a way to **sum the exerted force in each triangle** inside this area. Some peaks in each triangle **may cancel each other** in when calculating the shape coefficient.
* `Force Coefficient <./force_coefficient.rst>`_: It is a general adimensionalization of the resulting **wind induced force** over a body. It is calculated by summing the resulting force of each triangle, and dividing it by a **representative area**.
* `Momentum Coefficient <./momentum_coefficient.rst>`_: It is a general adimensionalization of the resulting **wind induced momentum** over a body. It is calculated by summing the resulting momentum of each triangle, and dividing it by a **representative volume**. The anchor point to define the momentum lever is an **arbitrary point for the whole body**.

Geometry Artifact
=================

Each coefficient has its own artifacts dependencies.
However, a common artifact shared between all of them, is a **description of the structure geometry**.

This module uses the **LNAS geometry** format to represent the geometry of interest.
The LNAS format contains the **vertices and triangles** necessary build the structure.
It also contains some functions to evaluate geometric properties such as triangle normals, and to filter the data for a given surface.

To get more information about the LNAS format, please see the documentation inside the `LNAS repository <https://github.com/AeroSim-CFD/stl2lnas>`_

.. toctree::
   :maxdepth: 1
   :caption: Pressure use Cases
   :hidden:

   Pressure Coefficient <./pressure_coefficient.rst>
   Shape Coefficient <./shape_coefficient.rst>
   Force Coefficient <./force_coefficient.rst>
   Momentum Coefficient <./momentum_coefficient.rst>