*****************
Force Coefficient
*****************

This coefficient is defined as a liquid resulting pressure coefficient between two surfaces.
For example, consider a shed's marquee, where the lower surface is marked on red, and the upper surface is marked on green:

.. image:: /_static/pressure/marquee.png
    :width: 90 %
    :align: center

The liquid resulting pressure coefficient is defined as:

.. math::
   C_{f} = \frac{F_{res1} - F_{res2}}{q A_{rep}}

But it can also be defined as a liquid resulting momentum between two surfaces:

.. math::
   C_{f} = \frac{M_{res1} - M_{res2}}{q A_{rep} L_{rep}}

Like the other coefficients, we can apply statistical analysis to the liquid force coefficient.

By definition, the liquid force coefficient is a **property of a body**.

It is used for primary and secondary structures design.
It can be seen as the resulting effect of the wind induced stress over a body.

Artifacts:
==========

#. A lnas file: It contains the information about the mesh.
#. HDF time series: It contains the pressure signals indexed by each of the mesh vertices.
#. Domain static pressure time series: It contains the pressure signals for probes far away from the building.

Outputs:
========

#. **Adimensionalized time series**: force coefficient time series for each body.
#. **Statistical results**: maximum, minimum, RMS and average values for the force coefficient time series, for each body.
#. **VTK File**: contains the statistical values inside the original mesh (VTK).