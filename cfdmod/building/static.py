"""Static-equivalent floor loads from per-floor Cf / Cm.

The step between the coefficient stage (:mod:`cfdmod.building.pressure`, which
returns dimensionless per-floor ``cf_x`` / ``cf_y`` / ``cm_z``) and every
downstream deliverable (floor-load profiles, peak tables, base envelopes).

    F_i(t) = cf_i(t) * q_load * A_ref
    M_i(t) = cm_i(t) * q_load * V_ref
    q_load = 0.5 * rho * U_load^2

``q_load`` is **not** the case dynamic pressure. ``BuildingCase.dynamic_pressure``
is built from the *simulation* inlet speed, which is the speed Cp was
non-dimensionalised by; a structural deliverable must be referenced at the
*design* speed for the wind direction being processed (e.g. the 50-year NBR
``U_H``). The two differ by ``(U_design / U_simul)^2`` -- for a real tower that
is routinely a factor of 0.8 to 1.5, and it varies direction by direction, so
using the simulation speed distorts the directional envelope as well as the
magnitude. :func:`static_floor_loads` therefore takes ``reference_velocity`` as
a **required** keyword: there is no default that could silently be the wrong
speed.

The returned :class:`~cfdmod.core.data_source.PointsDataSource` carries the
``feq_x`` / ``feq_y`` / ``meq_z`` fields consumed by
:mod:`cfdmod.dynamics.cases` / :mod:`cfdmod.dynamics.plotting`, positioned at
the per-floor lever heights so the base overturning moments
(:func:`cfdmod.dynamics.cases.global_load_history`) have a lever arm to work
with.
"""

from __future__ import annotations

__all__ = [
    "FloorLever",
    "floor_band_indices",
    "floor_lever_heights",
    "scatter_to_floor_bands",
    "static_floor_loads",
]

from typing import Literal

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, Topology
from cfdmod.core.data_source import DataSource
from cfdmod.core.field_meta import FieldMeta

from .case import BuildingCase

FloorLever = Literal["top", "mid", "bottom"]


def floor_lever_heights(floor_edges, n_floors: int, *, lever: FloorLever = "top") -> np.ndarray:
    """Per-floor lever-arm heights from the ``n_floors + 1`` band edges.

    ``lever="top"`` (default) puts the load at the slab level that bounds the
    band above -- the convention the structural handoff uses, since the floor
    loads are applied at storey levels. ``"mid"`` uses the band centroid (the
    physical line of action of a uniformly-loaded band) and ``"bottom"`` the
    lower edge.

    Raises when the edge count does not match ``n_floors``: a silent fallback
    to an index ladder would put the lever arms in the wrong units and quietly
    scale every base moment.
    """
    edges = np.asarray(floor_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size != n_floors + 1:
        raise ValueError(
            f"floor_edges must hold {n_floors + 1} ascending z-edges for {n_floors} "
            f"floors; got {edges.size}"
        )
    if lever == "top":
        return edges[1:]
    if lever == "bottom":
        return edges[:-1]
    if lever == "mid":
        return 0.5 * (edges[:-1] + edges[1:])
    raise ValueError(f"unknown lever {lever!r}; expected 'top', 'mid' or 'bottom'")


def floor_band_indices(source: DataSource, *, grouping: str = "floor") -> np.ndarray | None:
    """Which floor band each row of a per-floor source belongs to.

    ``zoning_grouping`` only emits rows for bands that actually contain
    triangles, so a tower whose lowest bands sit below the modelled geometry
    comes back with fewer rows than the storey table has floors. The row -> band
    mapping is recoverable from the grouping labels the aggregation carried
    forward (the raster region id, which equals the z-band index when the x / y
    axes are open).

    Returns ``None`` when the mapping cannot be recovered (a different grouping,
    or non-numeric labels), so the caller can fall back to requiring an exact
    row count rather than guessing an alignment.
    """
    group = source.groupings.get(grouping) if source.groupings else None
    if group is None or group.id_to_label is None:
        return None
    try:
        bands = [int(group.id_to_label[i]) for i in range(len(group.id_to_label))]
    except (KeyError, TypeError, ValueError):
        return None
    return np.asarray(bands, dtype=np.int64)


def scatter_to_floor_bands(rows: np.ndarray, band_indices: np.ndarray, n_bands: int) -> np.ndarray:
    """Place ``rows`` (one per populated band) onto the full ``n_bands`` ladder.

    Unpopulated bands are zero-filled. Keeping the profile aligned to the full
    storey table is what lets the per-floor deliverable be read row-by-row
    against ``alturas.csv``.
    """
    rows = np.asarray(rows, dtype=np.float64)
    bands = np.asarray(band_indices, dtype=np.int64)
    if bands.shape[0] != rows.shape[0]:
        raise ValueError(f"got {bands.shape[0]} band indices for {rows.shape[0]} rows")
    if bands.size and (bands.min() < 0 or bands.max() >= n_bands):
        raise ValueError(f"band indices {bands.min()}..{bands.max()} outside 0..{n_bands - 1}")
    full = np.zeros((n_bands, rows.shape[1]), dtype=np.float64)
    full[bands] = rows
    return full


def static_floor_loads(
    cf: DataSource,
    cm: DataSource,
    case: BuildingCase,
    *,
    reference_velocity: float,
    lever: FloorLever = "top",
    floor_edges=None,
) -> PointsDataSource:
    """Per-floor Cf / Cm -> static-equivalent floor loads at ``reference_velocity``.

    Args:
        cf: Per-floor groups source carrying ``cf_x`` / ``cf_y``
            ``(n_floors, n_t)`` (from :func:`cfdmod.building.per_floor_loads`).
        cm: Per-floor groups source carrying ``cm_z``.
        case: Supplies ``fluid_density``, ``nominal_area``, ``nominal_volume``
            and (unless ``floor_edges`` is given) the floor z-edges.
        reference_velocity: The speed the loads are referenced at, in m/s --
            normally the design ``U_H`` for this wind direction, **not** the
            simulation inlet speed. Required; see the module docstring.
        lever: Which point of each floor band carries the load, for the lever
            arms written into ``elements.position``. See
            :func:`floor_lever_heights`.
        floor_edges: Override for the ``n_floors + 1`` band edges. Defaults to
            ``case.floor_heights``.

    Returns:
        A points source with ``feq_x`` / ``feq_y`` [N] and ``meq_z`` [N.m] on
        floor points whose Z is the lever height. Rows always span the *full*
        band ladder: when the coefficient source only covers the populated
        bands (``zoning_grouping`` drops empty ones), the rows are scattered
        back onto the ladder by their recovered band index and the rest are
        zero -- so the profile lines up with the storey table row for row.

    Raises:
        ValueError: If the three coefficient arrays disagree on the floor
            count, if the rows cannot be aligned to the band ladder, or if
            ``reference_velocity`` is not positive.
    """
    if not np.isfinite(reference_velocity) or reference_velocity <= 0:
        raise ValueError(
            f"reference_velocity must be a positive speed; got {reference_velocity!r}"
        )

    cf_x = np.asarray(cf.fields.read("cf_x"), dtype=np.float64)
    cf_y = np.asarray(cf.fields.read("cf_y"), dtype=np.float64)
    cm_z = np.asarray(cm.fields.read("cm_z"), dtype=np.float64)
    n_floors = cf_x.shape[0]
    if not (cf_y.shape[0] == cm_z.shape[0] == n_floors):
        raise ValueError(
            f"Cf/Cm floor counts disagree: cf_x={cf_x.shape[0]}, cf_y={cf_y.shape[0]}, "
            f"cm_z={cm_z.shape[0]}"
        )

    edges = np.asarray(
        case.floor_heights if floor_edges is None else floor_edges, dtype=np.float64
    )
    n_bands = edges.size - 1

    if n_floors != n_bands:
        bands = floor_band_indices(cf)
        if bands is None or bands.shape[0] != n_floors:
            raise ValueError(
                f"per-floor rows ({n_floors}) do not match the {n_bands} floor bands and the "
                "row -> band mapping could not be recovered from the grouping; pass "
                "floor_edges matching the rows you have"
            )
        cf_x = scatter_to_floor_bands(cf_x, bands, n_bands)
        cf_y = scatter_to_floor_bands(cf_y, bands, n_bands)
        cm_z = scatter_to_floor_bands(cm_z, bands, n_bands)

    z = floor_lever_heights(edges, n_bands, lever=lever)

    q = 0.5 * float(case.fluid_density) * float(reference_velocity) ** 2
    fields = {
        "feq_x": cf_x * q * case.nominal_area,
        "feq_y": cf_y * q * case.nominal_area,
        "meq_z": cm_z * q * case.nominal_volume,
    }

    pts = np.zeros((n_bands, 3), dtype=np.float64)
    pts[:, 2] = z
    return PointsDataSource(
        time=cf.time,
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(fields),
        field_meta={
            "feq_x": FieldMeta(name="feq_x", unit="N"),
            "feq_y": FieldMeta(name="feq_y", unit="N"),
            "meq_z": FieldMeta(name="meq_z", unit="N.m"),
        },
    )
