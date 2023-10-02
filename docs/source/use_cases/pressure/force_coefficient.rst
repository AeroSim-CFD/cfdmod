*****************
Force Coefficient
*****************

This coefficient is defined as a net resulting pressure coefficient between two surfaces.
For example, consider a building's canopy, where the lower surface is marked on red, and the upper surface is marked on green:

.. image:: /_static/pressure/marquee.png
    :width: 90 %
    :align: center

The net resulting pressure coefficient is defined as:

.. math::
   C_{f} = \frac{F_{res1} - F_{res2}}{q A_{rep}} = \frac{\sum{c_{pi} A_{i}} - \sum{c_{pj} A_{j}}}{A_{rep}}

.. important:: Note that the net force coefficient has a direction attached to its definition. Its direction is the same as the resulting force direction.

It can also be defined for a given direction, for example in the x axis direction:

.. math::
   C_{fx} = \frac{Fx_{res1} - Fx_{res2}}{q A_{rep}} = \frac{\sum{c_{pi} A_{ix}} - \sum{c_{pj} A_{jx}}}{A_{rep}}


But it can also be defined as a net resulting momentum between two surfaces:

.. math::
   C_{fx} = \frac{Mx_{res1} - Mx_{res2}}{q A_{rep} Lx_{rep}}


Like the other coefficients, we can apply statistical analysis to the net force coefficient.

By definition, the net force coefficient is a **property of a body**.

It is used for primary and secondary structures design.
It can be seen as the resulting effect of the wind induced stress over a body.

Artifacts:
==========

#. A lnas file: It contains the information about the mesh.
#. Surface selection: It contains which surface inside the mesh is going to be used for evaluating net force coefficient.
#. HDF time series: It contains the pressure coefficient signals indexed by each of the mesh triangles.

Outputs:
========

#. **Adimensionalized time series**: force coefficient time series for each body.
#. **Statistical results**: maximum, minimum, RMS and average values for the force coefficient time series, for each body.
#. **VTK File**: contains the statistical values inside the original mesh (VTK).