********************
Pressure Coefficient
********************

The **Pressure Coefficient**, :math:`c_p`, is a dimensionless quantity that provides a **generalized representation** of the pressure distribution on a surface, or body, exposed to a fluid flow.
It allows us to assess how the local pressure at a specific point differs from the surrounding free-stream pressure, **accounting for the dynamic pressure** of the fluid flow.

Definition
==========

The pressure coefficient is a **dimensionless form of the pressure signal**.
It is obtained by the following expression:

.. math::
   c_{p}(t) = \frac{p(t) - p_{\infty}(t)}{q}

By definition, the pressure coefficient is a local property for each triangle of the mesh.

Use Case
========

It is used primarily for **analysis and interpretation** of the measured data.

It should always be generated, since it is the first analysis step. 
It is a fundamental property of the pressure normalization, and **it is used to calculate the other coefficients**.
However, it is not the final result to be delivered to clients.

Artifacts
=========

In order to use the pressure normalization module, the user has to provide a **set of artifacts**:

#. **A lnas file**: It contains the information about the mesh.
#. **HDF time series**: It contains the pressure signals indexed by each of the mesh triangles.
#. **Parameters file**: It contains the values for adimensionalization as well as other configs parameters.
#. **Static reference pressure time series**: It contains the pressure signals for probes far away from the building.

Which outputs the following data:

#. **Dimensionless time series**: pressure coefficient time series for each triangle.
#. **Statistical results**: statistical values for the pressure coefficient time series, for each triangle.
#. **VTK File**: contains the statistical values inside a mesh representation (VTK).

An illustration of the pressure coefficient module pipeline can be seen below:

.. image:: /_static/pressure/cp_pipeline.png
    :width: 90 %
    :align: center

Usage
=====

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

Another way to run the pressure coefficient conversion, is through the `notebook <calculate_cp.ipynb>`_

Data format
===========

.. list-table:: :math:`c_p(t)`
   :widths: 33 33 33
   :header-rows: 1

   * - point_idx
     - timestep
     - cp
   * - 0
     - 10000
     - 1.25
   * - 1
     - 10000
     - 1.15

.. list-table:: :math:`c_p (stats)`
   :widths: 20 10 10 10 10 20 20
   :header-rows: 1

   * - point_idx
     - max
     - min
     - mean
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. toctree::
   :maxdepth: -1
   :hidden:

   Transform cp <calculate_cp.ipynb>
