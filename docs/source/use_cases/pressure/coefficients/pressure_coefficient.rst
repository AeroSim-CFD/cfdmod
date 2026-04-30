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

The user provides:

#. **Body pressure XDMF+H5**: per-timestep pressure on every mesh triangle.
#. **Reference probe XDMF+H5** (optional): atmospheric reference pressure
   probe signal. If omitted, the reference pressure is taken as 0.
#. **Parameters** (``CpCaseConfig``): adimensionalisation values, statistic
   list, time-step range. The config can be a YAML file or built in code.
#. **Mesh** (optional): ``.lnas`` / ``.stl`` / ``.h5`` / ``.xdmf``. When
   omitted, the geometry is read from the body H5's embedded
   ``/Triangles + /Geometry``.

Outputs (under the ``output`` directory; layout is flat, no subfolders):

#. ``cp.{label}.time_series.{h5,xdmf}`` -- per-timestep Cp on the full
   mesh, ParaView-readable.
#. ``stats.{h5,xdmf}`` -- combined statistics for every coefficient in
   the run, with one ``<Grid>`` per leaf group on the matching mesh.
   Cp lands under ``/cp/{label}/``.

Each output H5 carries the post-processing config under
``/processing_metadata/`` for downstream debugging; read it back with
:func:`cfdmod.read_processing_metadata`.

Usage
=====

A reference parameters file:

.. literalinclude:: /_static/pressure/cp_params.yaml
    :language: yaml

Driving the pipeline from Python:

.. code-block:: python

   from cfdmod import run_cp, CpCaseConfig
   run_cp(
       body_h5="body.h5",
       probe_h5="probe.h5",                    # or None for zero reference
       cfg_path=CpCaseConfig.from_file("cp.yaml"),
       output="output",
       # mesh_path optional; omitting it reads from body.h5
   )

Or via the CLI:

.. code-block:: Bash

   python -m cfdmod pressure cp \
      --body   {BODY_H5} \
      --probe  {PROBE_H5} \
      --config {CONFIG_PATH} \
      --output {OUTPUT_PATH}

The same flow is also exercised in the `calculate_cp.ipynb <calculate_cp.ipynb>`_
notebook, with a fuller end-to-end version (including container
partition, Cf, and Cm) available at ``notebooks/process_container_pack.ipynb``
in the repository root.

Data format
===========

.. note::
    For more information about the normalized time scale (:math:`t^*`), check the `Normalization section <./normalization.rst>`_ 

.. list-table:: :math:`c_p(t)`
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - time_idx/point_idx
     - Normalized time (:math:`t^*`)
     - 0
     - 1
     - 2
   * - 0
     - 0.0
     - 1.25
     - 1.15
     - 1.32
   * - 0
     - 1.0
     - 1.1
     - 1.5
     - 1.13

.. list-table:: :math:`c_p (stats)`
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - scalar
     - 0
     - 1
     - 2
     - 3
   * - min
     - -1.25
     - -0.9
     - -1.1
     - -0.2
   * - max
     - 1.15
     - 0.95
     - 1.13
     - 0.19
   * - mean
     - 0.83
     - 0.9
     - 0.5
     - 0.13
   * - rms
     - 0.26
     - 0.25
     - 0.13
     - 0.19
   * - skewness
     - 1.15
     - -0.95
     - 1.13
     - 0.19

.. toctree::
   :maxdepth: -1
   :hidden:

   Transform cp <calculate_cp.ipynb>
