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
:mod:`cfdmod.core.recipes`; each mirrors one legacy ``*CaseConfig`` and
builds the equivalent pipeline. Example templates live under
``fixtures/tests/pressure/templates/``.

.. autoclass:: cfdmod.core.recipes.CpRecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.core.recipes.CfRecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.core.recipes.CmRecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.core.recipes.CeRecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.core.recipes.S1RecipeConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.core.recipes.DynamicAnalysisConfig
   :members:
   :show-inheritance:

.. autoclass:: cfdmod.core.recipes.PedestrianComfortConfig
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
abstraction: the ``body_grouping`` and ``zoning_grouping`` ops attach a
grouping to a :class:`cfdmod.SurfaceDataSource`, and per-group field
series are then aggregated onto a :class:`cfdmod.GroupsDataSource`. An
explicit chain (for instance, splitting a body by shared-edge
connectivity) is expressed by composing the built-in grouping kinds
below.

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

Notebook utilities
==================

.. autofunction:: cfdmod.mesh_summary

.. autofunction:: cfdmod.show_config

.. autofunction:: cfdmod.load_lnas
