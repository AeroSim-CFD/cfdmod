********************
Pressure Coefficient
********************

The pressure coefficient is an **dimensionless form of the pressure signal**.
It is obtained by the following expression:

.. math::
   c_{p}(t) = \frac{p(t) - p_{\infty}(t)}{q}

By definition, the pressure coefficient is a local property for each triangle of the mesh.
It is used primarily for analysis and interpretation of the measured data.

It should always be generated, since it is the first analysis step. 
It is a fundamental property of the pressure normalization, and **it is used to calculate the other coefficients**.
However, it is not the final result to be delivered to clients.

The parameter file for converting the pressure data into pressure coefficient looks as follows:

.. literalinclude:: /_static/pressure/cp_params.yaml
    :language: yaml

To invoke and run the conversion, the following command can be used:

.. code-block:: Bash

   poetry run python -m cfdmod.use_cases.pressure \
      --output {OUTPUT_PATH} \
      --p      {PRESS_SERIES_PATH} \
      --s      {STATIC_PRESS_PATH} \
      --mesh   {LNAS_PATH} \
      --config {CONFIG_PATH}

# TODO: reference the notebooks

Artifacts:
==========

#. **A lnas file**: It contains the information about the mesh.
#. **HDF time series**: It contains the pressure signals indexed by each of the mesh triangles.
#. **Parameters file**: It contains the values for adimensionalization as well as other configs parameters.
#. **Static pressure time series**: It contains the pressure signals for probes far away from the building.

Outputs:
========

#. **Dimensionless time series**: pressure coefficient time series for each triangle.
#. **Statistical results**: maximum, minimum, RMS and average values for the pressure coefficient time series, for each triangle.
#. **VTK File**: contains the statistical values inside a mesh representation (VTK).
