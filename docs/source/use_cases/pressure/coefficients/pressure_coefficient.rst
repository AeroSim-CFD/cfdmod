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

The Cp template declares:

#. **Body pressure** (``kind: surface``): per-timestep pressure on every
   mesh triangle.
#. **Reference probe** (``kind: points``, optional): atmospheric reference
   pressure signal. If omitted, the reference pressure is taken as 0.

and a pipeline that subtracts the reference (``sub``), divides by the
dynamic pressure (``scale`` by ``1 / q``), and reduces to per-element
``statistics``. The outputs are a ``cp`` time series and a ``cp`` statistics
data source, each writable to an XDMF+H5 pair (ParaView-readable).

Usage
=====

Run the shipped template from the command line:

.. code-block:: bash

   cfdmod run fixtures/tests/pressure/templates/cp.yaml

or drive it from Python over any storage backend:

.. code-block:: python

   from cfdmod import load_template, run_template, XdmfH5Storage

   template = load_template("cp.yaml")
   bindings = run_template(template, storage=XdmfH5Storage(root="."))
   cp_t = bindings["cp_t"]          # SurfaceDataSource, one row per triangle

The `calculate_cp.ipynb <calculate_cp.ipynb>`_ notebook walks through this
template step by step; a fuller end-to-end version (Cp / Cf / Cm / Ce over
a container pack) lives at ``examples/container_pack/process_container_pack.ipynb`` in
the repository.

Data format
===========

.. note::
    For more information about the normalized time scale (:math:`t^*`), check the `Time Normalization section <./time_normalization.rst>`_

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
