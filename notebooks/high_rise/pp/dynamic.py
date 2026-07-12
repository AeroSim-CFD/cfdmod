"""v3 recipe wiring for the high-rise dynamic-response stage.

Thin composition of the library dynamic recipe -- no new structural maths
lives here. It bridges the per-floor Cf/Cm produced by the pressure stage
(:mod:`pp.pressure`) into the building dynamic-response recipe
(:func:`cfdmod.core.recipes.dynamic.build_building_dynamic_response`) and the
comfort acceleration recipe (:func:`build_point_accelerations`).

The pipeline the stage assembles:

    per-floor Cf/Cm timeseries (GroupsDataSource, from pp.pressure)
        -> floor-load PointsDataSource (cf_x / cf_y / cm_z, dimensionalised)
        -> generalized modal loads -> SDOF RK45 -> floor response
           (disp_x / disp_y / rot_z + static-equivalent feq_x / feq_y / meq_z)
        -> off-centre horizontal accelerations (acc_x / acc_y / acc_mag)

Structural inputs (mode shapes, floor masses, radii, natural frequencies)
come from a case's modes/floors/mode-shape CSVs via
:func:`structure_from_csvs`. For the headless demo (no CSVs on disk) a
self-contained :func:`example_building_structure` synthesises plausible
cantilever sway + torsion modes tuned to the case geometry, so the whole
chain runs on the in-repo fixtures.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, Topology
from cfdmod.core.data_source import DataSource
from cfdmod.core.recipes import (
    ComfortConfig,
    build_building_dynamic_response,
    build_point_accelerations,
)
from cfdmod.dynamics import BuildingStructuralData, mass_normalize_mode_shapes
from pp.case import HighRiseCase

# Ellis (1980) empirical fundamental frequency: f1 ~ 46 / H [Hz], H in metres.
_ELLIS_COEFF = 46.0


def floor_load_source(
    cf: DataSource,
    cm: DataSource,
    case: HighRiseCase,
    *,
    dimensionalize: bool = True,
) -> PointsDataSource:
    """Merge per-floor Cf/Cm groups into one floor-load ``PointsDataSource``.

    ``cf`` carries ``cf_x`` / ``cf_y`` and ``cm`` carries ``cm_z``, each a
    ``(n_floors, n_t)`` groups source from :func:`pp.pressure.cf_per_floor` /
    :func:`pp.pressure.cm_per_floor`. The result has the three fields the
    building recipe expects (``cf_x`` / ``cf_y`` / ``cm_z``) on floor points
    stacked along Z.

    With ``dimensionalize`` (default) the force coefficients are scaled back
    to physical loads with the case dynamic pressure and reference area /
    volume (``F = cf * q * A``, ``M = cm_z * q * V``) so the response comes
    out in metres / newtons. Set it ``False`` to feed the raw coefficients
    (the response is then linear in an arbitrary scale).
    """
    cf_x = np.asarray(cf.fields.read("cf_x"), dtype=np.float64)
    cf_y = np.asarray(cf.fields.read("cf_y"), dtype=np.float64)
    cm_z = np.asarray(cm.fields.read("cm_z"), dtype=np.float64)
    n_floors = cf_x.shape[0]
    if not (cf_y.shape[0] == cm_z.shape[0] == n_floors):
        raise ValueError(
            f"Cf/Cm floor counts disagree: cf_x={cf_x.shape[0]}, cf_y={cf_y.shape[0]}, "
            f"cm_z={cm_z.shape[0]}"
        )

    if dimensionalize:
        q = case.dynamic_pressure
        cf_x = cf_x * q * case.nominal_area
        cf_y = cf_y * q * case.nominal_area
        cm_z = cm_z * q * case.nominal_volume

    pts = np.zeros((n_floors, 3), dtype=np.float64)
    pts[:, 2] = _floor_mid_heights(case, n_floors)

    fields = {"cf_x": cf_x, "cf_y": cf_y, "cm_z": cm_z}
    return PointsDataSource(
        time=cf.time,
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(fields),
    )


def example_building_structure(
    case: HighRiseCase,
    n_floors: int,
    *,
    n_modes: int = 3,
    frequencies_hz: list[float] | None = None,
    floor_mass: float = 1.0,
) -> BuildingStructuralData:
    """A self-contained :class:`BuildingStructuralData` tuned to the case.

    Synthesises three canonical mode shapes -- sway-X, sway-Y and torsion,
    each growing linearly with height like a cantilever's first mode -- for a
    building of ``n_floors`` uniform floors. Natural frequencies default to
    the Ellis ``46/H`` fundamental with higher modes at 1.1x / 1.25x. For real
    work use :func:`structure_from_csvs`.

    ``n_modes`` is clamped to ``[1, 3]`` (only three canonical shapes are
    defined).
    """
    n_modes = int(np.clip(n_modes, 1, 3))
    z = _floor_mid_heights(case, n_floors)
    z_norm = z / max(float(z.max()), 1e-9)  # 0..1 up the height

    # Canonical shapes: [sway-X, sway-Y, torsion], each linear with height.
    shape_defs = [
        (z_norm, np.zeros(n_floors), np.zeros(n_floors)),  # sway X
        (np.zeros(n_floors), z_norm, np.zeros(n_floors)),  # sway Y
        (np.zeros(n_floors), np.zeros(n_floors), z_norm),  # torsion
    ]
    phi = np.stack(
        [np.column_stack(shape_defs[m]) for m in range(n_modes)], axis=1
    )  # (n_floors, n_modes, 3)

    floors_mass = np.full(n_floors, float(floor_mass), dtype=np.float64)
    # Radius of gyration ~ 0.4 * plan dimension (rectangular-plan rule of thumb).
    floors_radius = np.full(n_floors, 0.4 * max(case.characteristic_length, 1e-6))
    phi = mass_normalize_mode_shapes(phi, floors_mass, floors_radius)

    if frequencies_hz is None:
        f1 = _ELLIS_COEFF / max(case.reference_height, 1e-6)
        factors = [1.0, 1.1, 1.25][:n_modes]
        frequencies_hz = [f1 * k for k in factors]
    wp = 2.0 * np.pi * np.asarray(frequencies_hz[:n_modes], dtype=np.float64)

    floor_points = np.column_stack([np.zeros(n_floors), np.zeros(n_floors), z])
    cm_positions = np.zeros((n_floors, 2), dtype=np.float64)

    return BuildingStructuralData(
        mode_shapes=phi,
        natural_frequencies=wp,
        floor_points=floor_points,
        cm_positions=cm_positions,
        floors_mass=floors_mass,
        floors_radius=floors_radius,
    )


def structure_from_csvs(
    modes_csv: str | pathlib.Path,
    floors_csv: str | pathlib.Path,
    mode_shape_csvs: list[str | pathlib.Path],
    *,
    active_modes: list[int] | None = None,
) -> BuildingStructuralData:
    """Load real structural data from the modes / floors / mode-shape CSVs.

    Thin passthrough to :meth:`BuildingStructuralData.from_csvs`; see its
    docstring for the CSV column layouts. Mode shapes come back
    mass-normalised, ready for :func:`solve_building_response`.
    """
    return BuildingStructuralData.from_csvs(
        pathlib.Path(modes_csv),
        pathlib.Path(floors_csv),
        [pathlib.Path(p) for p in mode_shape_csvs],
        active_modes=active_modes,
    )


def solve_building_response(
    load_source: PointsDataSource,
    structure: BuildingStructuralData,
    *,
    damping_ratio: float = 0.02,
) -> PointsDataSource:
    """Floor loads + structure -> per-floor dynamic response.

    Returns a ``PointsDataSource`` over the floors with displacement fields
    ``disp_x`` / ``disp_y`` / ``rot_z`` and static-equivalent load fields
    ``feq_x`` / ``feq_y`` / ``meq_z`` (each ``(n_floors, n_t)``).
    """
    cfg = structure.to_config(damping_ratio=damping_ratio)
    return build_building_dynamic_response(load_source, cfg)


def floor_accelerations(
    response: PointsDataSource,
    structure: BuildingStructuralData,
    *,
    point: tuple[float, float] = (0.0, 0.0),
) -> PointsDataSource:
    """Per-floor horizontal accelerations at an off-centre occupant point.

    Augments ``response`` with ``acc_x`` / ``acc_y`` / ``acc_mag`` for the
    comfort assessment. ``point`` is in the same frame as the structure's CM
    offsets.
    """
    cfg = ComfortConfig(cm_positions=structure.cm_positions, point=point)
    return build_point_accelerations(response, cfg)


def peak_response_table(
    response: PointsDataSource,
    accelerations: PointsDataSource,
    case: HighRiseCase,
) -> pd.DataFrame:
    """Per-floor peak magnitudes for the engineer-facing deliverable table.

    One row per floor: mid-height Z, peak absolute displacement / rotation,
    peak static-equivalent loads, and peak acceleration magnitude. Peaks are
    the maximum absolute value over the time record.
    """

    def peak_abs(ds: PointsDataSource, field: str) -> np.ndarray:
        return np.nanmax(np.abs(np.asarray(ds.fields.read(field), dtype=np.float64)), axis=1)

    n_floors = np.asarray(response.fields.read("disp_x")).shape[0]
    return pd.DataFrame(
        {
            "floor": np.arange(n_floors),
            "z_mid": _floor_mid_heights(case, n_floors),
            "disp_x_peak": peak_abs(response, "disp_x"),
            "disp_y_peak": peak_abs(response, "disp_y"),
            "rot_z_peak": peak_abs(response, "rot_z"),
            "feq_x_peak": peak_abs(response, "feq_x"),
            "feq_y_peak": peak_abs(response, "feq_y"),
            "meq_z_peak": peak_abs(response, "meq_z"),
            "acc_mag_peak": peak_abs(accelerations, "acc_mag"),
        }
    )


def _floor_mid_heights(case: HighRiseCase, n_floors: int) -> np.ndarray:
    """Floor mid-heights from the case z-edges, or a unit ladder if they disagree.

    The pressure stage may return fewer floor rows than the case has z-edges
    (empty slices are dropped), so fall back to an integer ladder when the
    edge count does not yield exactly ``n_floors`` mid-points.
    """
    edges = np.asarray(case.floor_heights, dtype=np.float64)
    if edges.size == n_floors + 1:
        return 0.5 * (edges[:-1] + edges[1:])
    return np.arange(n_floors, dtype=np.float64)
