********************
Pressure Coefficient
********************

The pressure coefficient is an **adimensionalized form of the pressure signal**.
It is obtained by the following expression:

.. math::
   c_{p}(t) = \frac{p(t) - p_{\infty}(t)}{q}

By definition, the pressure coefficient is a local property for each point of the mesh.
It is used primarily for analysis and interpretation of the measured data.

It should always be generated, since it is the first analysis step. 
It is a fundamental property of the pressure normalization.
However, it is not the final result to be delivered to clients.

Artifacts:
==========

#. A lnas file: It contains the information about the mesh.
#. HDF time series: It contains the pressure signals indexed by each of the mesh vertices.
#. Domain static pressure time series: It contains the pressure signals for probes far away from the building.

Outputs:
========

#. **Adimensionalized time series**: pressure coefficient time series for each vertex.
#. **Statistical results**: maximum, minimum, RMS and average values for the pressure coefficient time series, for each vertex.
#. **VTK File**: contains the statistical values inside a mesh representation (VTK).
