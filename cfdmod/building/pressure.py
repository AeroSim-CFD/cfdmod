"""v3 recipe/op wiring for the high-rise pressure stages.

Thin composition of library ops -- no new maths lives here. The high-rise
choices baked in:

    Cp  = (p - p_ref) / q,  q = 0.5 * rho * U_H^2   (from BuildingCase)
    Cf  per floor: force_contribution (explicit reference area) summed per floor.
    Cm  per floor: moment_contribution about the case lever origin, summed per
        floor (normalised by the reference volume).

Per-floor partitioning has two methods, chosen by ``method``:

- ``"face_cut"`` (default) -- geometrically slice each triangle at the floor
    z-edges so a triangle straddling a boundary contributes its *real partial
    area* to each floor. Exact force/moment by floor.
- ``"centroid"`` -- the fast/approximate ``zoning_grouping``: assign each whole
    triangle to one floor by its centroid. Cheaper, but a triangle spanning a
    boundary lands entirely on one side.

Both attach a ``"floor"`` grouping (raster region id == floor index, since x/y
are open) and sum the per-triangle contributions with
``field_series_for_groups(agg="sum")``. The reference-area normalisation (vs the
legacy per-region bounding-box area) is the convention chosen for 3.2.
"""

from __future__ import annotations

from typing import Literal

from cfdmod.core.data_source import DataSource, GroupsDataSource
from cfdmod.core.ops.data_source_create.face_cut import FaceCutParams, face_cut
from cfdmod.core.ops.data_source_create.field_series_for_groups import (
    FieldSeriesForGroupsParams,
    field_series_for_groups,
)
from cfdmod.core.ops.field.force_contribution import ForceContributionParams, force_contribution
from cfdmod.core.ops.field.moment_contribution import (
    MomentContributionParams,
    moment_contribution,
)
from cfdmod.core.ops.geometric.mesh_attach import MeshAttachParams, mesh_attach
from cfdmod.core.ops.geometric.zoning_grouping import ZoningGroupingParams, zoning_grouping
from cfdmod.core.recipes import CpRecipeConfig, build_cp

from .case import BuildingCase

_FLOOR = "floor"

FloorMethod = Literal["face_cut", "centroid"]


def cp_from_pressure(
    body: DataSource,
    p_ref,
    case: BuildingCase,
    *,
    statistics: list[str] | None = None,
    time_rescale_factor: float | None = None,
) -> DataSource:
    """Cp time series (or stats) non-dimensionalised by the case dynamic pressure.

    ``p_ref`` is a scalar reference pressure or a points/surface DataSource
    (broadcast per timestep). Pass ``statistics`` to collapse the time axis to
    mean/rms/peak fields instead of the full series.
    """
    cfg = CpRecipeConfig(
        dynamic_pressure=case.dynamic_pressure,
        statistics=statistics or [],
        time_rescale_factor=time_rescale_factor,
    )
    return build_cp(body, p_ref=p_ref, cfg=cfg)


def _partition_floors(
    ds: DataSource, mesh_path: str, case: BuildingCase, method: FloorMethod
) -> DataSource:
    """Attach a per-floor partition; ``face_cut`` slices, ``centroid`` groups whole triangles.

    ``face_cut`` returns a new (fragmented) surface whose fields are inherited
    from the parent, so force/moment must be computed *after* it to pick up the
    partial fragment areas. ``centroid`` attaches a grouping to the same surface.
    """
    if method == "face_cut":
        return face_cut(ds, FaceCutParams(z_intervals=list(case.floor_heights), name=_FLOOR))
    if method == "centroid":
        return zoning_grouping(
            ds,
            ZoningGroupingParams(
                mesh=mesh_path, z_intervals=list(case.floor_heights), name=_FLOOR
            ),
        )
    raise ValueError(f"unknown floor method {method!r}; expected 'face_cut' or 'centroid'")


def _sum_per_floor(ds: DataSource, fields: list[str]) -> GroupsDataSource:
    """field_series_for_groups (agg=sum) for each field, merged onto one groups source."""
    result: GroupsDataSource | None = None
    for field in fields:
        partial = field_series_for_groups(
            ds,
            FieldSeriesForGroupsParams(grouping=_FLOOR, field=field, agg="sum", out=field),
        )
        if result is None:
            result = partial
        else:
            result = result.with_field(
                field, partial.fields.read(field), meta=partial.field_meta[field]
            )
    assert result is not None
    return result


def _attach_and_partition(
    cp_ds: DataSource, mesh_path: str, case: BuildingCase, method: FloorMethod
) -> DataSource:
    """mesh_attach + per-floor partition, shared by the Cf / Cm recipes.

    ``face_cut`` returns a fragmented surface with fragment areas/normals;
    ``centroid`` attaches a grouping to the mesh-attached surface. Either way the
    result is ready for :func:`force_contribution`.
    """
    ds = mesh_attach(cp_ds, MeshAttachParams(mesh=mesh_path))
    return _partition_floors(ds, mesh_path, case, method)


def _force(ds: DataSource, case: BuildingCase, directions: list[str]) -> DataSource:
    return force_contribution(
        ds, ForceContributionParams(nominal_area=case.nominal_area, directions=directions)
    )


def _moment(ds: DataSource, case: BuildingCase, directions: list[str]) -> DataSource:
    return moment_contribution(
        ds,
        MomentContributionParams(
            lever_origin=tuple(case.lever_origin),
            nominal_area=case.nominal_area,
            nominal_volume=case.nominal_volume,
            directions=directions,
        ),
    )


def cf_per_floor(
    cp_ds: DataSource,
    mesh_path: str,
    case: BuildingCase,
    *,
    directions: tuple[str, ...] = ("x", "y"),
    method: FloorMethod = "centroid",
) -> GroupsDataSource:
    """Per-floor force coefficients cf_<dir>, one row per floor slice.

    ``method="centroid"`` (default) assigns each whole triangle to a floor by its
    centroid -- fast, bounded memory, and matching the v2 sub-body grouping.
    ``method="face_cut"`` slices triangles at the floor edges for an exact
    partial-area split, but fragments the mesh (much heavier); prefer centroid at
    production sizes. See :func:`cf_cm_per_floor` when you need both Cf and Cm.
    """
    ds = _attach_and_partition(cp_ds, mesh_path, case, method)
    ds = _force(ds, case, list(directions))
    return _sum_per_floor(ds, [f"cf_{d}" for d in directions])


def cm_per_floor(
    cp_ds: DataSource,
    mesh_path: str,
    case: BuildingCase,
    *,
    directions: tuple[str, ...] = ("z",),
    method: FloorMethod = "centroid",
) -> GroupsDataSource:
    """Per-floor moment coefficients cm_<dir> about the case lever origin.

    See :func:`cf_per_floor` for the ``method`` trade-off.
    """
    ds = _attach_and_partition(cp_ds, mesh_path, case, method)
    # moment_contribution reads all three force components, so produce them all.
    ds = _force(ds, case, ["x", "y", "z"])
    ds = _moment(ds, case, list(directions))
    return _sum_per_floor(ds, [f"cm_{d}" for d in directions])


def cf_cm_per_floor(
    cp_ds: DataSource,
    mesh_path: str,
    case: BuildingCase,
    *,
    cf_directions: tuple[str, ...] = ("x", "y"),
    cm_directions: tuple[str, ...] = ("z",),
    method: FloorMethod = "centroid",
) -> tuple[GroupsDataSource, GroupsDataSource]:
    """Per-floor Cf and Cm from a **single** mesh-attach + partition + force pass.

    Computing Cf and Cm separately (``cf_per_floor`` + ``cm_per_floor``) runs the
    heavy ``mesh_attach`` + partition + ``force_contribution`` twice. When both
    are needed -- the normal high-rise case -- this fuses them: the three force
    components are computed once and reused for the Cf sums and the moment.
    Returns ``(cf, cm)``.
    """
    ds = _attach_and_partition(cp_ds, mesh_path, case, method)
    ds = _force(ds, case, ["x", "y", "z"])
    cf = _sum_per_floor(ds, [f"cf_{d}" for d in cf_directions])
    ds = _moment(ds, case, list(cm_directions))
    cm = _sum_per_floor(ds, [f"cm_{d}" for d in cm_directions])
    return cf, cm


def per_floor_loads(
    body: DataSource,
    p_ref,
    mesh_path: str,
    case: BuildingCase,
    *,
    cf_directions: tuple[str, ...] = ("x", "y"),
    cm_directions: tuple[str, ...] = ("z",),
    method: FloorMethod = "centroid",
    chunk_size: int | None = None,
) -> tuple[GroupsDataSource, GroupsDataSource]:
    """Memory-bounded per-floor Cf / Cm straight from body + reference pressure.

    Fuses ``Cp -> force -> moment -> per-floor sum`` and streams it over time
    windows of ``chunk_size`` timesteps, so the full ``(n_triangles,
    n_timesteps)`` Cp / force arrays never materialise at once: peak memory is
    ``O(n_triangles * chunk_size)`` while the returned per-floor series is small.
    With ``chunk_size=None`` (default) it runs whole-series (identical result);
    pass a chunk (e.g. a few thousand) for production-size cases. Returns
    ``(cf, cm)`` with the full time axis, ready for
    :func:`cfdmod.building.floor_load_source`.
    """
    from cfdmod.core.chunked import concat_time, slice_time, time_windows

    def _window(b: DataSource, r) -> tuple[GroupsDataSource, GroupsDataSource]:
        cp = cp_from_pressure(b, r, case)
        return cf_cm_per_floor(
            cp,
            mesh_path,
            case,
            cf_directions=cf_directions,
            cm_directions=cm_directions,
            method=method,
        )

    n_t = body.time.n_timesteps
    if chunk_size is None or n_t == 0 or chunk_size >= n_t:
        return _window(body, p_ref)

    cf_parts: list[GroupsDataSource] = []
    cm_parts: list[GroupsDataSource] = []
    for sl in time_windows(n_t, chunk_size):
        cf_w, cm_w = _window(slice_time(body, sl), slice_time(p_ref, sl))
        cf_parts.append(cf_w)
        cm_parts.append(cm_w)
    return concat_time(cf_parts), concat_time(cm_parts)
