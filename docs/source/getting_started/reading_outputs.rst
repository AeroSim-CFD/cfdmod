***************
Reading Outputs
***************

A pipeline run writes each output as an XDMF+H5 pair. This page shows the
four things you typically do with them: open them in ParaView, load a
timeseries into pandas, export to CSV, quick-plot, and round-trip the
embedded reproducibility metadata. The examples assume the Cp output from
:doc:`index` (``out/cp.time_series.h5`` and ``out/cp.stats.h5``).

Open in ParaView
================

Open the ``.xdmf`` sidecar, not the ``.h5``:

.. code-block:: bash

   paraview out/cp.stats.xdmf

ParaView reads the geometry and the per-triangle fields (``mean`` /
``rms`` / ``min`` / ``max`` for a stats file; the per-timestep Cp for a
time-series file) directly through the XDMF description; the ``.h5`` next
to it supplies the arrays.

Into a pandas DataFrame
=======================

:func:`cfdmod.read_timeseries_df` flattens a coefficient timeseries into a
wide-form DataFrame indexed by normalized time, one column per triangle:

.. code-block:: python

   from cfdmod import read_timeseries_df

   df = read_timeseries_df("out/cp.time_series.h5", "cp", triangles=[0, 1, 2])
   #   index   -> time_normalized
   #   columns -> triangle indices 0, 1, 2
   #   values  -> Cp(t, triangle)

The second argument is the coefficient group inside the file: ``"cp"`` in
a Cp file, ``"cf_x"`` / ``"cf_y"`` / ``"cf_z"`` in a Cf file, ``"cm_x"`` /
``"cm_y"`` / ``"cm_z"`` in a Cm file.

Because a per-triangle Cp file can be tens of thousands of columns wide,
``read_timeseries_df`` refuses to return more than ``max_columns`` (200 by
default) unless you narrow it:

* ``triangles=[...]`` -- keep an explicit subset of triangle indices.
* ``regions=True`` -- for **Cf / Cm** files, where every triangle in a
  region carries the same value, deduplicate to one representative column
  per region. Do not use this on per-triangle Cp.
* ``timestep_range=(t_min, t_max)`` -- restrict to a raw-time window before
  building the frame.

.. code-block:: python

   # Cf: one column per body/region, windowed in time
   cf = read_timeseries_df(
       "out/cf_x.time_series.h5", "cf_x",
       regions=True, timestep_range=(0.0, 5.0),
   )

Export to CSV
=============

:func:`cfdmod.to_csv` writes the wide-form frame (first column
``time_normalized``, one column per retained triangle/region) -- it drops
straight into a spreadsheet. Extra keyword arguments pass through to
``pandas.DataFrame.to_csv``:

.. code-block:: python

   from cfdmod import to_csv

   to_csv(df, "out/cp_selected.csv")

Quick plot
==========

:func:`cfdmod.plot_timeseries` is a one-line matplotlib plot of the frame,
returning the Axes so you can keep styling it:

.. code-block:: python

   from cfdmod import plot_timeseries

   ax = plot_timeseries(df, title="Cp on selected triangles", ylabel="Cp")
   ax.figure.savefig("out/cp_selected.png", dpi=150)

Reproducibility metadata
=========================

You can embed the parameters that produced a result as HDF5 attributes
plus a YAML string dataset under ``/{group}/processing_metadata``, so the
file records how it was made. :func:`cfdmod.write_processing_metadata`
attaches the record to an existing output group;
:func:`cfdmod.read_processing_metadata` reads it back:

.. code-block:: python

   from cfdmod import write_processing_metadata, read_processing_metadata

   write_processing_metadata(
       "out/cp.stats.h5", "stats",
       {"note": "galpao Cp demo", "dynamic_pressure_factor": 800.0},
   )

   meta = read_processing_metadata("out/cp.stats.h5", "stats")
   meta["config"]          # -> {'note': 'galpao Cp demo', 'dynamic_pressure_factor': 800.0}
   meta["cfdmod_version"]  # package version that wrote the record
   meta["produced_at"]     # ISO-8601 UTC timestamp

The ``config`` dict is free-form -- record whatever parameters you want to
reproduce. Both helpers live in :mod:`cfdmod.io.xdmf`.
