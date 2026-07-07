"""v3 recipe/op wiring for the high-rise pressure stages.

Thin composition of library ops -- no new maths lives here. The high-rise
choices baked in:

    Cp  = (p - p_ref) / q,  q = 0.5 * rho * U_H^2   (from HighRiseCase)
    Cf  per floor: force_contribution (explicit reference area) summed over the
        triangles whose centroid falls in each floor's z-slice.
    Cm  per floor: moment_contribution about the case lever origin, summed per
        floor (normalised by the reference volume).

Per-floor slicing uses ``zoning_grouping`` with the floor z-edges as z_intervals
and open x/y bins, so each floor is one group. The reference-area normalisation
(vs the legacy per-region bounding-box area) is the convention chosen for 3.2.
"""

from __future__ import annotations

from cfdmod.core.data_source import DataSource, GroupsDataSource
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
from pp.case import HighRiseCase

_FLOOR = "floor"


def cp_from_pressure(
    body: DataSource,
    p_ref,
    case: HighRiseCase,
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


def _floor_grouping(ds: DataSource, mesh_path: str, case: HighRiseCase) -> DataSource:
    return zoning_grouping(
        ds,
        ZoningGroupingParams(mesh=mesh_path, z_intervals=list(case.floor_heights), name=_FLOOR),
    )


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


def cf_per_floor(
    cp_ds: DataSource,
    mesh_path: str,
    case: HighRiseCase,
    *,
    directions: tuple[str, ...] = ("x", "y"),
) -> GroupsDataSource:
    """Per-floor force coefficients cf_<dir>, one row per floor slice."""
    ds = mesh_attach(cp_ds, MeshAttachParams(mesh=mesh_path))
    ds = force_contribution(
        ds, ForceContributionParams(nominal_area=case.nominal_area, directions=list(directions))
    )
    ds = _floor_grouping(ds, mesh_path, case)
    return _sum_per_floor(ds, [f"cf_{d}" for d in directions])


def cm_per_floor(
    cp_ds: DataSource,
    mesh_path: str,
    case: HighRiseCase,
    *,
    directions: tuple[str, ...] = ("z",),
) -> GroupsDataSource:
    """Per-floor moment coefficients cm_<dir> about the case lever origin."""
    ds = mesh_attach(cp_ds, MeshAttachParams(mesh=mesh_path))
    # moment_contribution reads all three force components, so produce them all.
    ds = force_contribution(
        ds, ForceContributionParams(nominal_area=case.nominal_area, directions=["x", "y", "z"])
    )
    ds = moment_contribution(
        ds,
        MomentContributionParams(
            lever_origin=tuple(case.lever_origin),
            nominal_area=case.nominal_area,
            nominal_volume=case.nominal_volume,
            directions=list(directions),
        ),
    )
    ds = _floor_grouping(ds, mesh_path, case)
    return _sum_per_floor(ds, [f"cm_{d}" for d in directions])
