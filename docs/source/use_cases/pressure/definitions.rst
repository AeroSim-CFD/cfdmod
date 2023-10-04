***********
Definitions
***********

Some definitions are needed to abstract some use cases rules.

Surface
=======

Surface is a collection of triangles that share points with its neighbors.
For example, consider the left side of a building's roof.
Its geometry is described as a STL file, which looks like this:

.. image:: /_static/pressure/surface_mesh.png
    :width: 85 %
    :align: center

.. important:: All surfaces of a structure **must be defined in the pre-processing** (before running the simulation).


Body
====

Body is a collection of surfaces that together define a volume.
For example, a generic building is composed by a left + right side roofs and walls, and a front + back side walls
Each surface is colored with different colors in the image below:

.. image:: /_static/pressure/body.png
    :width: 85 %
    :align: center

.. important:: All of the body's surfaces **must be separated before converting to LNAS** (before running the simulation).

Regions
=======

Regions are defined by x, y and z intervals.
Each combination of the three intervals result in different regions.
For example, consider the following intervals definiton:

.. code-block:: python

    x_intervals = [0,100,200,300,400]
    y_intervals = [0,50,100]
    z_intervals = [0,15]

There will be 8 different regions as a result:

.. code-block:: python

    R1: 0   <= x <  100, 0  <= y <  50,  0 <= z <= 15
    R2: 0   <= x <  100, 50 <= y <= 100, 0 <= z <= 15
    R3: 100 <= x <  200, 0  <= y <  50,  0 <= z <= 15
    R4: 100 <= x <  200, 50 <= y <= 100, 0 <= z <= 15
    R5: 200 <= x <  300, 0  <= y <  50,  0 <= z <= 15
    R6: 200 <= x <  300, 50 <= y <= 100, 0 <= z <= 15
    R7: 300 <= x <= 400, 0  <= y <  50,  0 <= z <= 15
    R8: 300 <= x <= 400, 50 <= y <= 100, 0 <= z <= 15

.. image:: /_static/pressure/regions.png
    :width: 85 %
    :align: center

The regions are used to filter triangles for a surface or for a body, grouping them together for processing.

.. important:: The rule to apply and guarantee that **every triangle belongs to a region and one only**, is to include the upper limit only if it is the last one. Otherwise, the upper limit is not included, only the lower limit is.