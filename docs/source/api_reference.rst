*************
API Reference
*************

This page documents the public Python surface of ``cfdmod``. Every symbol
listed in ``cfdmod.__all__`` is reachable via ``from cfdmod import X``;
deeper module paths are also stable.

Top-level entry points
======================

Post-processing is expressed as a **pipeline template**: a YAML document
declaring inputs, a sequence of composable ops, and outputs. Templates run
from the command line or in Python; the same recipe code runs whether the
backend is an in-memory store (notebooks, tests) or the on-disk XDMF+H5
store (production). See :doc:`architecture/data_sources` for the paradigm
and :doc:`architecture/v3_migration` for the mapping from the legacy
per-coefficient functions.

.. code-block:: bash

   cfdmod run path/to/template.yaml

.. autofunction:: cfdmod.load_template

.. autofunction:: cfdmod.run_template

.. autoclass:: cfdmod.PipelineTemplate
   :members:
   :show-inheritance:

The op registry backing the template loader is extensible; register a new
op with :func:`cfdmod.register_op` and inspect the built-ins in
:data:`cfdmod.OP_REGISTRY`.

.. autofunction:: cfdmod.register_op

Data sources
============

Every result -- pressures on a surface, cell values in a volume, probe
timeseries, group aggregates, modal coordinates -- is carried by a frozen
:class:`cfdmod.DataSource`: elements on one axis, timesteps on the other,
one or more named fields sharing that shape, plus element / time / field
metadata. Ops consume and produce data sources; a
:class:`cfdmod.Pipeline` (built with :func:`cfdmod.compose`) chains them.

.. autoclass:: cfdmod.DataSource
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.SurfaceDataSource
   :show-inheritance:

.. autoclass:: cfdmod.VolumeDataSource
   :show-inheritance:

.. autoclass:: cfdmod.PointsDataSource
   :show-inheritance:

.. autoclass:: cfdmod.GroupsDataSource
   :show-inheritance:

.. autoclass:: cfdmod.ModesDataSource
   :show-inheritance:

Axes and topology
-----------------

.. autoclass:: cfdmod.TimeAxis
   :members:

.. autoclass:: cfdmod.Topology
   :members:

.. autoclass:: cfdmod.FieldMeta
   :members:

.. autoclass:: cfdmod.Grouping
   :members:

Containers
----------

A :class:`cfdmod.Container` aggregates many data sources under complex
keys (case, direction, ...), the way a parametric study fans out over
inflow angles.

.. autoclass:: cfdmod.Container
   :members:

Pipelines and storage
---------------------

.. autoclass:: cfdmod.Pipeline
   :members:

.. autofunction:: cfdmod.compose

.. autoclass:: cfdmod.MemoryStorage
   :members:

.. autoclass:: cfdmod.XdmfH5Storage
   :members:

Recipe configs
==============

The pressure and wind recipes ship as small-data Pydantic configs under
:mod:`cfdmod.recipes`; each mirrors one legacy ``*CaseConfig`` and
builds the equivalent pipeline. Example templates live under
``fixtures/tests/pressure/templates/``.

.. autoclass:: cfdmod.recipes.CpRecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.recipes.CfRecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.recipes.CmRecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.recipes.CeRecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.recipes.S1RecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.recipes.DynamicAnalysisConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.recipes.PedestrianComfortConfig
   :members:
   :show-inheritance:

Triangle-grouping pipeline
==========================

A grouping pipeline partitions or selects triangles of a parent
:class:`lnas.LnasFormat` mesh into named groups: specs are Pydantic
models in a discriminated union, composed left-to-right with
:func:`cfdmod.apply_groupings`, and a triangle may belong to **zero,
one, or many** groups.

The force / moment / shape recipes (Cf, Cm, Ce) are built on top of this
abstraction: the ``body_grouping``, ``zoning_grouping`` and
``connectivity_grouping`` ops attach a grouping to a
:class:`cfdmod.SurfaceDataSource`, and per-group field series are then
aggregated onto a :class:`cfdmod.GroupsDataSource`. ``body_grouping``
splits by surface name, ``zoning_grouping`` by a rectangular
centroid-binned grid, and ``connectivity_grouping`` by shared-edge
connected component (one physical body per component, no axis
projection). More elaborate partitions are expressed by composing the
built-in grouping kinds below.

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

.. autoclass:: cfdmod.ByDivisionsGrouping
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.BySizeGrouping
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.ByConnectivityGrouping
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.ByNormalGrouping
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.ByPlaneGrouping
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.ByPercentileGrouping
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.ByCylindricalGrouping
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.CustomGrouping
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

Remesh (geometry coarsening)
============================

``cfdmod.remesh`` is a small API-only module for coarsening grouped
``LnasFormat`` meshes -- typically the output of the ``regroup`` pipeline,
where each named surface holds many triangles that came out of the
``aggregation="sliced"`` 90-degree cuts. The default path is **exact
coplanar-fan collapse** (lossless, numpy-only); a flat NxN-subdivided
square inside one surface comes back as 2 triangles, a curved patch
unchanged.

The module follows a small convention split. The two array-level
functions take raw ``(vertices, triangles)`` and operate on a single
sub-mesh:

.. autofunction:: cfdmod.merge_coplanar

.. autofunction:: cfdmod.decimate_qem

``decimate_qem`` requires the optional ``fast-simplification`` dep
(``pip install 'aerosim-cfdmod[remesh]'``); calling it without that extra
raises ``ImportError`` with a clear install hint.

The top-level entry point dispatches both array-level operations over
every named surface in an ``LnasFormat`` and restitches the per-surface
outputs back into a fresh ``LnasFormat`` (same surface names,
``vertex`` order recompacted, optional tolerance-based seam dedup):

.. autofunction:: cfdmod.remesh_per_group

Regroup (disk regroup pipeline)
===============================

``cfdmod.regroup`` takes a geometry plus a per-triangle HDF5 timeseries,
applies a chain of triangle-grouping specs, and writes two aligned outputs: a
new ``LnasFormat`` mesh with one named surface per group, and a new HDF5
timeseries whose columns line up with the new triangle order (or are
area-weighted aggregates per group). It reuses the grouping kinds above and
adds one regroup-local spec that fans out per-component target-size
subdivisions. Runnable as ``python -m cfdmod.regroup``.

.. autoclass:: cfdmod.RegroupConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.BySizeRoundedPerComponent
   :members:
   :show-inheritance:

.. autofunction:: cfdmod.build_regroup_mapping

.. autofunction:: cfdmod.build_regrouped_mesh

.. autofunction:: cfdmod.apply_regroup_to_timeseries

.. autofunction:: cfdmod.expand_regroup_chain

.. autofunction:: cfdmod.run_regroup

Building wind-load post-processing
==================================

``cfdmod.building`` turns a pressure timeseries on a building surface into
the engineering deliverables of a wind study: per-floor force and moment
coefficients, the modal dynamic response, occupant-comfort accelerations,
design load cases, and a multi-direction / multi-body fan-out driver. It
composes the v3 recipes and ops; nothing here is high-rise-specific -- the
same helpers serve low-rise studies. Import from the sub-package::

    from cfdmod.building import BuildingCase, cf_per_floor, solve_building_response

Case aggregation
----------------

.. autoclass:: cfdmod.building.BuildingCase
   :members:
   :show-inheritance:

.. autofunction:: cfdmod.building.cp_from_pressure

Per-floor coefficients
----------------------

.. autofunction:: cfdmod.building.cf_per_floor

.. autofunction:: cfdmod.building.cm_per_floor

Peak statistics
---------------

The peak of a fluctuating series is taken by one of three selectable
methods (raw maximum, gust peak-factor, or a Gumbel fit).

.. autofunction:: cfdmod.building.gust_peak_factor

.. autofunction:: cfdmod.building.peak_value

Dynamic response
----------------

.. autofunction:: cfdmod.building.solve_building_response

.. autofunction:: cfdmod.building.floor_accelerations

.. autofunction:: cfdmod.building.peak_response_table

Occupant comfort
----------------

Peak top-floor accelerations are checked against the comfort limits of
three standards -- NBR 6123, Melbourne (1992) and the NBCC.
``comfort_limit`` dispatches on the selected standard and occupancy;
the per-standard helpers are also exposed directly.

.. autofunction:: cfdmod.building.comfort_limit

.. autofunction:: cfdmod.building.nbr6123_acceleration_limit

.. autofunction:: cfdmod.building.melbourne1992_acceleration_limit

.. autofunction:: cfdmod.building.nbcc_acceleration_limit

Design load cases
-----------------

.. autofunction:: cfdmod.building.generate_load_cases

.. autofunction:: cfdmod.building.save_load_case_tables

Multi-direction fan-out
-----------------------

A single driver runs the whole per-floor / dynamic / comfort chain over
every (direction, body, config) combination of a parametric study.

.. autoclass:: cfdmod.building.FanoutPlan
   :members:
   :show-inheritance:

.. autofunction:: cfdmod.building.run_fanout

Structural model import (dynamics)
==================================

The building dynamic-response recipe needs a modal model of the structure
-- per-floor mass, polar inertia, centre of mass, natural periods and the
per-floor mode shapes. ``cfdmod.dynamics`` reads that model out of the
structural engineer's design software (TQS, Eberick) and converts it to the
internal :class:`~cfdmod.dynamics.structural.BuildingStructuralData`. See
:doc:`use_cases/dynamics/index` for the supported file formats and the
conversion in detail. ::

    from cfdmod.dynamics import read_tqs_portels, read_tqs_portico, read_eberick

Structural model
----------------

.. autoclass:: cfdmod.dynamics.structural.BuildingStructuralData
   :members:
   :show-inheritance:

Importers
---------

.. autofunction:: cfdmod.dynamics.read_tqs_portels

.. autofunction:: cfdmod.dynamics.read_tqs_portico

.. autofunction:: cfdmod.dynamics.read_eberick

Conversion
----------

.. autofunction:: cfdmod.dynamics.imports.aggregate_to_building

.. autoclass:: cfdmod.dynamics.imports.EberickUnits
   :members:
   :show-inheritance:

Analytical wind profiles
========================

Code-based mean-velocity profiles :math:`U(z)` for reference and inflow
target curves.

.. autoclass:: cfdmod.WindProfile_NBR
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.WindProfile_EU
   :members:
   :show-inheritance:

Notebook utilities
==================

.. autofunction:: cfdmod.mesh_summary

.. autofunction:: cfdmod.show_config

.. autofunction:: cfdmod.load_lnas
