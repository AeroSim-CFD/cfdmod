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

The pipeline accepts geometry in any of the following formats and dispatches by suffix via :func:`cfdmod.io.load_mesh`:

* ``.lnas`` -- the AeroSim native LNAS file with authored surfaces.
* ``.stl`` -- a triangle mesh; treated as a single ``"all"`` surface.
* ``.h5`` -- an XDMF+H5 with embedded ``/Triangles + /Geometry``; single ``"all"`` surface.
* ``.xdmf`` -- redirects to its sibling ``.h5``.

The ``mesh_path`` argument is optional on every ``run_*`` entry point. When omitted, the geometry is read from the source H5 itself (the body H5 for ``run_cp``, the cp timeseries H5 for ``run_cf``/``run_cm``/``run_ce``) -- so a single-body pipeline does not need a separate mesh file.

.. note::
   For LNAS-specific details, see the documentation inside the `LNAS repository <https://github.com/AeroSim-CFD/stl2lnas>`_.

v2 Quickstart
=============

A worked end-to-end example lives at ``notebooks/process_container_pack.ipynb`` in the repo. It

* loads geometry directly from the body H5 (no LNAS authoring),
* auto-detects container partition via a triangle-centroid gap sweep,
* runs Cp / Cf / Cm and writes everything flat under ``./output/``,
* uses ``lever_strategy="region_bbox_corners_xy"`` to produce four overturning-moment scans per container.

Outputs land flat in ``output/`` (no nested per-coefficient subfolders); per-coefficient stats are merged into a single ``stats.{h5,xdmf}`` and every output H5 carries the post-processing config under ``/processing_metadata/`` for downstream debugging.

.. toctree::
   :maxdepth: 1
   :caption: Pressure use Cases
   :hidden:

   Pressure Coefficient <./pressure_coefficient.rst>
   Shape Coefficient <./shape_coefficient.rst>
   Force Coefficient <./force_coefficient.rst>
   Moment Coefficient <./moment_coefficient.rst>
   Time Normalization <./time_normalization.rst>