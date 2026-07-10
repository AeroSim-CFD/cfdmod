***************
Getting Started
***************

This page takes you from a fresh install to a first pressure-coefficient
(``Cp``) result you can open in ParaView or load into pandas. It assumes
you have a body pressure H5 and a static-pressure probe H5 produced by the
AeroSim CFD solver; if you are working from a repository checkout, a
complete runnable fixture ships with the code (see `Run your first Cp`_).

Install
=======

Base install (Cp / Cf / Cm / Ce pipeline, IO helpers, and the ``cfdmod``
CLI):

.. code-block:: bash

   pip install aerosim-cfdmod

Optional extras, added only when you need them:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - When you need it
   * - ``[vtk]``
     - ParaView snapshot automation, VTK polydata writers, S1 probe-on-line.
   * - ``[geometry]``
     - Altimetry section and loft helpers (trimesh).
   * - ``[notebook]``
     - jupyter / ipykernel for the worked-example notebooks.
   * - ``[docs]``
     - sphinx + theme + nbsphinx to build this documentation.
   * - ``[legacy]``
     - pandas-HDFStore compatibility readers (inflow, HFPI static).

Install several at once:

.. code-block:: bash

   pip install "aerosim-cfdmod[vtk,geometry,notebook]"

.. note::
   ``pymeshlab`` is intentionally not an extra: its GPL license would
   force GPL on downstream code. The few code paths that need it expect
   you to install it explicitly, at your own license risk.

What you need on disk
=====================

A pressure post-processing run consumes two data sources, each an
XDMF+H5 pair (a ``.h5`` payload with a sibling ``.xdmf`` sidecar):

* **A body surface** -- per-triangle pressure for every timestep. Declared
  in the template as ``kind: surface``.
* **A static-pressure probe** -- the reference pressure timeseries.
  Declared as ``kind: points``.

Optionally, a **mesh** (``.lnas``, ``.stl``, or an XDMF+H5 with embedded
geometry) when you go on to Cf / Cm / Ce, which attach per-triangle areas,
normals, and centroids.

.. note::
   **Filename convention.** The XDMF+H5 storage infers a source's kind
   from its filename: a probe must be named ``points.*`` to load as a
   points source; everything else loads as a surface. The ``kind:`` you
   declare in the template is checked against the loaded kind, so a
   mismatch fails fast.

Run your first Cp
=================

Post-processing is expressed as a **pipeline template**: a YAML document
with ``inputs`` (data sources on disk), a ``pipeline`` of ops, and
``outputs``. A minimal Cp template subtracts the static reference,
divides by the dynamic pressure ``q``, and reduces the time axis to
per-triangle statistics:

.. code-block:: yaml

   name: cp
   inputs:
     body:                          # surface pressure per triangle per timestep
       kind: surface
       path: bodies.my_case         # -> bodies.my_case.h5 (+ .xdmf)
     p_ref:                         # static reference probe; must be named points.*
       kind: points
       path: points.static_pressure
   pipeline:
     - id: cp_unscaled              # p - p_ref  (column-wise broadcast)
       kind: sub
       source: body
       rhs: p_ref
       field: pressure
       out: cp
     - id: cp_t                     # / dynamic pressure q  (factor = 1/q)
       kind: scale
       source: cp_unscaled
       field: cp
       factor: 800.0
     - id: cp_stats                 # collapse the time axis
       kind: statistics
       source: cp_t
       field: cp
       kinds: [mean, rms, min, max]
   outputs:
     cp_timeseries: {source: cp_t, path: out/cp.time_series}
     cp_stats:      {source: cp_stats, path: out/cp.stats}

Run it with the CLI:

.. code-block:: bash

   cfdmod run cp.yaml

Or drive it from Python. The same recipe runs against an in-memory store
(handy for notebooks and tests, no files touched) or the on-disk XDMF+H5
store:

.. code-block:: python

   from cfdmod import load_template, run_template, XdmfH5Storage

   template = load_template("cp.yaml")
   bindings = run_template(template, storage=XdmfH5Storage(root="."))
   cp_t = bindings["cp_t"]          # a SurfaceDataSource
   cp_series = cp_t.fields.read("cp")

``load_template`` validates the whole template up front -- unknown op
kinds, dangling ``source`` / ``rhs`` refs, duplicate ids, typo'd fields --
before any file is read.

.. note::
   **Working from a repository checkout?** Complete, runnable templates
   and a bundled ``galpao`` wind-tunnel fixture ship under
   ``fixtures/tests/pressure/``. Copy ``templates/`` and ``data/`` into a
   writable directory and run ``cfdmod run templates/cp.yaml`` there; the
   Cf / Cm / Ce templates in the same folder chain off the Cp output.

What you get
============

Each declared output is written as an XDMF+H5 pair under the template's
directory:

* ``out/cp.time_series.{h5,xdmf}`` -- the time-resolved Cp, one value per
  triangle per timestep (group ``/cp`` with one dataset ``t{T}`` per
  timestep, plus ``/Triangles`` and ``/Geometry``).
* ``out/cp.stats.{h5,xdmf}`` -- the per-triangle statistics (group
  ``/stats`` with ``mean`` / ``rms`` / ``min`` / ``max``).

The ``.xdmf`` sidecar is what you open in ParaView; the ``.h5`` holds the
arrays. See :doc:`reading_outputs` to pull these back into pandas, export
CSV, quick-plot, or open them in ParaView.

Next steps
==========

* :doc:`reading_outputs` -- ParaView, pandas, CSV, and reproducibility
  metadata.
* :doc:`../use_cases/pressure/index` -- the full Cp / Cf / Cm / Ce recipes,
  one page each.
* :doc:`../architecture/data_sources` -- the data-source + pipeline
  paradigm end to end.
* :doc:`../architecture/v3_migration` -- moving off the pre-v3 entry
  points.

.. toctree::
   :maxdepth: 1
   :hidden:

   Reading outputs <reading_outputs.rst>
