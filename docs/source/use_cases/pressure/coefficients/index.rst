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
* `Moment Coefficient <./moment_coefficient.rst>`_: It is a general adimensionalization of the resulting **wind induced momentum** over a body. It is calculated by summing the resulting momentum of each triangle, and dividing it by a **representative volume**. The anchor point to define the momentum lever is an **arbitrary point for the whole body**.

Geometry Artifact
=================

Each coefficient has its own artifacts dependencies.
However, a common artifact shared between all of them, is a **description of the structure geometry**.

The pipeline accepts geometry in any of the following formats and dispatches by suffix via :func:`cfdmod.io.load_mesh`:

* ``.lnas`` -- the AeroSim native LNAS file with authored surfaces.
* ``.stl`` -- a triangle mesh; treated as a single ``"all"`` surface.
* ``.h5`` -- an XDMF+H5 with embedded ``/Triangles + /Geometry``; single ``"all"`` surface.
* ``.xdmf`` -- redirects to its sibling ``.h5``.

The mesh is attached to a Cp time series by the ``mesh_attach`` op, which pulls per-triangle areas, normals and centroids from the ``.lnas`` (or ``.h5``) geometry. The Cf / Cm / Ce templates all start with a ``mesh_attach`` step; the Cp template itself does not need a mesh.

When the same building is simulated at several wind directions, each solver run produces a body H5 in its own wind-aligned ("spun") coordinate frame. Point every template's ``mesh_attach`` at a single fixed-frame mesh and all coefficients are expressed in that shared frame. Triangle ordering must match the body H5 -- only vertex coordinates may differ.

.. note::
   For LNAS-specific details, see the documentation inside the `LNAS repository <https://github.com/AeroSim-CFD/stl2lnas>`_.

Filtering between coefficients
==============================

A coefficient time series is carried by a :class:`cfdmod.DataSource`, so
signal-processing steps are just more ops in the pipeline. To smooth a
series, insert a ``moving_average`` step; the window is expressed in the
input time units and the result is another field on the same data source,
ready to feed the next step.

.. code-block:: yaml

   - id: cp_smoothed
     kind: moving_average
     source: cp_t
     field: cp
     window: 3.0
     out: cp

Placing smoothing in the pipeline (rather than inside a statistics block)
keeps the lineage explicit: every step is recorded in the template that
produced the output.

Worked example
==============

An end-to-end example lives at ``examples/container_pack/process_container_pack.ipynb``
in the repo. It builds a Cp / Cf / Cm / Ce pipeline as YAML templates,
runs them with ``run_template`` over the on-disk XDMF+H5 storage, and
streams the results back onto a coarse mesh. The per-coefficient
tutorials below run the same shipped templates
(``fixtures/tests/pressure/templates/``) step by step.

.. toctree::
   :maxdepth: 1
   :caption: Pressure use Cases
   :hidden:

   Pressure Coefficient <./pressure_coefficient.rst>
   Shape Coefficient <./shape_coefficient.rst>
   Force Coefficient <./force_coefficient.rst>
   Moment Coefficient <./moment_coefficient.rst>
   Time Normalization <./time_normalization.rst>