*************
API Reference
*************

This page documents the public Python surface of ``cfdmod``. Every symbol
listed in ``cfdmod.__all__`` is reachable via ``from cfdmod import X``;
deeper module paths are also stable.

Top-level entry points
======================

The pipeline is driven through four functions, one per coefficient. They
accept either a YAML config path or an in-memory case-config instance,
and the geometry is read from the source H5 by default.

.. autofunction:: cfdmod.run_cp

.. autofunction:: cfdmod.run_cf

.. autofunction:: cfdmod.run_cm

.. autofunction:: cfdmod.run_ce

Configuration models
====================

All configuration types are plain Pydantic v2 ``BaseModel`` subclasses
(``BasePressureConfig`` for the pressure coefficient configs, which itself
extends ``BaseModel``). Each ``*CaseConfig`` exposes a ``from_file(path)``
classmethod for loading from YAML; ``model_dump()`` / ``model_dump_json()``
give back dict / JSON. There is no project-specific base class.

Pressure coefficient (Cp)
-------------------------

.. autoclass:: cfdmod.CpConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.CpCaseConfig
   :members:
   :show-inheritance:

Force coefficient (Cf)
----------------------

.. autoclass:: cfdmod.CfConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.CfCaseConfig
   :members:
   :show-inheritance:

Moment coefficient (Cm)
-----------------------

.. autoclass:: cfdmod.CmConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.CmCaseConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.MomentBodyConfig
   :members:
   :show-inheritance:

The Cm body config is the place to configure the moment center per region.
Three strategies are supported:

- ``lever_strategy="fixed"`` -- single ``lever_origin`` for the body
  (default).
- ``lever_strategy="region_base"`` -- each region uses
  ``(mean_x, mean_y, min_z)`` of its triangle vertices, i.e. the
  footprint centroid at the lowest z. Useful for overturning moments
  about the base of each container.
- ``lever_strategy="region_bbox_corners_xy"`` -- expand the body into
  four independent runs (``xmin_ymin``, ``xmin_ymax``, ``xmax_ymin``,
  ``xmax_ymax``); each run produces its own timeseries file and stats
  group. Useful for a worst-case overturning-moment scan around the
  footprint.

Per-region overrides via ``region_lever_origins`` (single run) or
``lever_origin_cases`` (multi-run scan) take precedence over the
strategy.

Shape coefficient (Ce)
----------------------

.. autoclass:: cfdmod.CeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.CeCaseConfig
   :members:
   :show-inheritance:

Geometry / zoning
-----------------

.. autoclass:: cfdmod.BodyDefinition
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.BodyConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.ZoningModel
   :members:
   :show-inheritance:

Triangle-grouping pipeline
==========================

A grouping pipeline partitions or selects triangles of a parent
:class:`lnas.LnasFormat` mesh into named groups. It is the geometry
counterpart to :mod:`cfdmod.pressure.filters` for time-series
processing: specs are Pydantic models in a discriminated union,
composed left-to-right with :func:`cfdmod.apply_groupings`, and a
triangle may belong to **zero, one, or many** groups.

The pressure pipeline (Cf, Cm, Ce) is built on top of this abstraction.
The legacy ``BodyConfig.sub_bodies`` / ``CeConfig.zoning`` YAML form
keeps working unchanged; the canonical
``[BySurfaceGrouping, ByZoningGrouping]`` chain is synthesized
internally. New configurations may instead set
``BodyConfig.groupings`` to an explicit chain to express compositions
the legacy fields cannot (for instance, splitting a body by
shared-edge connectivity).

Driver
------

.. autofunction:: cfdmod.apply_groupings

.. autoclass:: cfdmod.GroupingResult
   :members:

Built-in grouping kinds
-----------------------

Each kind is dispatched on its ``kind`` discriminator in the
:data:`cfdmod.GroupingSpec` union. A new kind is added by defining a
new Pydantic model under ``cfdmod/geometry/grouping/kinds/`` with a
unique ``kind`` literal, registering it in the union, and adding a
dispatch branch in ``cfdmod.geometry.grouping.base._dispatch``.

.. autoclass:: cfdmod.BySurfaceGrouping
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.ByZoningGrouping
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.ByConnectivityGrouping
   :members:
   :show-inheritance:

Persistence
-----------

A grouping chain can be recorded alongside a coefficient's
``processing_metadata`` block via the convention
``{"groupings": cfdmod.dump_groupings(chain)}`` (sibling to
``"filters"``); :func:`cfdmod.load_groupings` rehydrates the typed
spec instances on read.

.. autofunction:: cfdmod.dump_groupings

.. autofunction:: cfdmod.load_groupings

I/O helpers
===========

Mesh resolver
-------------

.. autofunction:: cfdmod.load_mesh

.. autofunction:: cfdmod.mesh_from_h5

Embedded post-processing metadata
---------------------------------

Every pipeline output H5 carries a ``processing_metadata`` group with the
config used to produce it. The two helpers below let external pipelines
write or read that block without depending on the layout details.

.. autofunction:: cfdmod.write_processing_metadata

.. autofunction:: cfdmod.read_processing_metadata

Timeseries access
-----------------

Pull a coefficient timeseries out of any output H5 into a wide-form
``pandas.DataFrame``, save it as CSV for spreadsheet ingest, or plot
it with one matplotlib call.

.. autofunction:: cfdmod.read_timeseries_df

.. autofunction:: cfdmod.to_csv

.. autofunction:: cfdmod.plot_timeseries

Geometry I/O (STL)
------------------

.. autofunction:: cfdmod.read_stl

.. autofunction:: cfdmod.export_stl

Migration (legacy formats)
==========================

The migration helpers convert legacy pandas-HDFStore body / probe files
into the v2 XDMF+H5 layout.

.. autofunction:: cfdmod.pressure.migrate.migrate_body_h5

.. autofunction:: cfdmod.pressure.migrate.migrate_probe_h5

Notebook utilities
==================

.. autofunction:: cfdmod.mesh_summary

.. autofunction:: cfdmod.show_config

.. autofunction:: cfdmod.load_lnas
