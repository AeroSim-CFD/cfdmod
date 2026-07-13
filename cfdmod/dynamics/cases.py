"""Directional / parametric case orchestration for building dynamics.

Ports the legacy ``cfdmod.hfpi.handler`` multi-case orchestration onto the
generic :class:`cfdmod.core.container.Container`. A case is identified by a
frozen :class:`BuildingCaseParameters` key (direction, damping, recurrence
period, and the mass / frequency / integral-scale multipliers); the results
form a ``Container[BuildingCaseParameters, PointsDataSource]`` that groups
(``join_by``) and filters (``filter_by``) with no bespoke machinery.

The wind-profile-driven reference-speed lookup and per-direction force
selection stay in the shell (the caller supplies a ``solve_fn`` that turns a
case into a response), keeping this layer free of I/O -- exactly the split
the v3 paradigm asks for.
"""

from __future__ import annotations

__all__ = [
    "BuildingCaseParameters",
    "build_cases",
    "solve_building_cases",
    "filter_by_recurrence_period",
    "filter_by_xi",
    "filter_by_kd",
    "join_by_direction",
    "join_by_recurrence_period",
    "get_max_acceleration",
    "get_max_acceleration_by_recurrence_period",
    "get_stats_forces_effective",
    "get_global_peaks_by_direction",
]

import itertools
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from cfdmod.core.container import Container
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.protocols import Pool

if TYPE_CHECKING:
    from cfdmod.building.peaks import PeakMethod

StatType = Literal["min", "max", "mean"]

_STAT_REDUCERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "min": lambda a: np.min(a, axis=1),
    "max": lambda a: np.max(a, axis=1),
    "mean": lambda a: np.mean(a, axis=1),
}


class BuildingCaseParameters(BaseModel, frozen=True):
    """Identifies one building dynamic-analysis case (a Container key).

    Frozen and hashable so it can key a
    :class:`~cfdmod.core.container.Container`.
    """

    direction: float
    xi: float
    recurrence_period: float
    use_kd: bool = False
    frequency_multiplier: float = 1.0
    integral_scale_multiplier: float = 1.0
    mass_multiplier: float = 1.0


def build_cases(
    *,
    directions: list[float],
    xis: list[float],
    recurrence_periods: list[float],
    use_kd: list[bool] = [False],
    frequency_multipliers: list[float] = [1.0],
    integral_scale_multipliers: list[float] = [1.0],
    mass_multipliers: list[float] = [1.0],
) -> list[BuildingCaseParameters]:
    """Cartesian product of case knobs -> list of case parameters."""
    return [
        BuildingCaseParameters(
            direction=direction,
            xi=xi,
            use_kd=kd,
            recurrence_period=period,
            frequency_multiplier=fm,
            integral_scale_multiplier=ism,
            mass_multiplier=mm,
        )
        for direction, xi, kd, period, fm, ism, mm in itertools.product(
            directions,
            xis,
            use_kd,
            recurrence_periods,
            frequency_multipliers,
            integral_scale_multipliers,
            mass_multipliers,
        )
    ]


def solve_building_cases(
    cases: list[BuildingCaseParameters],
    solve_fn: Callable[[BuildingCaseParameters], PointsDataSource],
    *,
    pool: Pool | None = None,
) -> Container[BuildingCaseParameters, PointsDataSource]:
    """Solve every case and collect the responses in a Container.

    ``solve_fn`` maps a case to its response data source (the caller wires
    the direction -> forces / dimensional scaling / structural multipliers).
    With ``pool`` the fanout runs through ``pool.map``; otherwise it is
    sequential. Group the result with ``container.join_by(lambda c:
    c.direction)`` and slice it with ``container.filter_by(...)``.
    """
    if pool is None:
        results = [solve_fn(c) for c in cases]
    else:
        results = pool.map(solve_fn, cases)
    return Container(items=dict(zip(cases, results)))


# --- Multi-direction result queries -----------------------------------------
#
# Thin, domain-named views over the generic ``Container`` primitives, plus the
# reducers the legacy ``cfdmod.hfpi.analysis`` result object exposed. A result
# set stays a plain ``Container[BuildingCaseParameters, PointsDataSource]`` (the
# key already carries direction / xi / use_kd / recurrence_period), so filtering
# is just ``Container.filter_by`` keyed on those fields and there is no bespoke
# result class to maintain.

ResultContainer = Container[BuildingCaseParameters, PointsDataSource]


def filter_by_recurrence_period(
    container: ResultContainer, recurrence_period: float
) -> ResultContainer:
    """Sub-container of cases at ``recurrence_period``.

    Convenience wrapper over :meth:`Container.filter_by`. Unlike the legacy
    ``filter_by_recurrence_period`` (which indexed ``join_by_*`` and raised
    ``KeyError`` on a missing value), an absent value yields an empty
    sub-container -- it composes with further filters without a guard.
    """
    return container.filter_by(lambda k: k.recurrence_period == recurrence_period)


def filter_by_xi(container: ResultContainer, xi: float) -> ResultContainer:
    """Sub-container of cases at damping ratio ``xi`` (empty if none match)."""
    return container.filter_by(lambda k: k.xi == xi)


def filter_by_kd(container: ResultContainer, use_kd: bool) -> ResultContainer:
    """Sub-container of cases with the given ``use_kd`` flag."""
    return container.filter_by(lambda k: k.use_kd == use_kd)


def join_by_direction(container: ResultContainer) -> dict[float, ResultContainer]:
    """Partition by wind direction (``{direction: sub-container}``)."""
    return container.join_by(lambda k: k.direction)


def join_by_recurrence_period(container: ResultContainer) -> dict[float, ResultContainer]:
    """Partition by recurrence period (``{recurrence_period: sub-container}``)."""
    return container.join_by(lambda k: k.recurrence_period)


def get_max_acceleration(
    container: ResultContainer,
    *,
    floor: int = -1,
    method: "PeakMethod" = "gumbel",
    field: str = "acc_mag",
    **peak_kwargs,
) -> float:
    """Largest design peak comfort acceleration across the container's cases.

    For every case, the ``field`` history at ``floor`` (default the top floor)
    is reduced to a single design peak by :func:`cfdmod.building.peaks.peak_value`
    (``method`` selects max / peak-factor / gumbel), and the maximum over cases
    is returned. The response must carry accelerations (see
    :func:`cfdmod.building.dynamic.floor_accelerations`). Returns ``nan`` for an
    empty container.
    """
    from cfdmod.building.peaks import peak_value

    peaks: list[float] = []
    for response in container.values():
        acc = np.asarray(response.fields.read(field), dtype=np.float64)  # (n_floors, n_t)
        peaks.append(peak_value(acc[floor], method, absolute=True, **peak_kwargs))
    return float(np.max(peaks)) if peaks else float("nan")


def get_max_acceleration_by_recurrence_period(
    container: ResultContainer, **kwargs
) -> dict[float, float]:
    """``{recurrence_period: max acceleration}`` for the comfort-vs-return plot.

    Groups the container by recurrence period and applies
    :func:`get_max_acceleration` to each group (``kwargs`` forwarded).
    """
    return {
        rp: get_max_acceleration(sub, **kwargs)
        for rp, sub in join_by_recurrence_period(container).items()
    }


def get_stats_forces_effective(
    response: PointsDataSource,
    stats_type: StatType,
    *,
    feq_fields: tuple[str, str, str] = ("feq_x", "feq_y", "meq_z"),
) -> dict[str, np.ndarray]:
    """Per-floor ``min`` / ``max`` / ``mean`` of the static-equivalent loads.

    Reduces the three static-equivalent floor-load fields
    (``feq_x`` / ``feq_y`` / ``meq_z``, each ``(n_floors, n_t)``) over time and
    returns per-floor arrays keyed ``{"x", "y", "z"}``. ``min`` / ``max`` are the
    signed time extrema (the two load envelopes), ``mean`` the time average --
    not absolute values, so the sign of the governing load is preserved for the
    downstream load-case tables.
    """
    reducer = _STAT_REDUCERS.get(stats_type)
    if reducer is None:
        raise ValueError(f"unknown stats_type {stats_type!r}; expected min / max / mean")
    fx, fy, mz = feq_fields
    return {
        "x": reducer(np.asarray(response.fields.read(fx), dtype=np.float64)),
        "y": reducer(np.asarray(response.fields.read(fy), dtype=np.float64)),
        "z": reducer(np.asarray(response.fields.read(mz), dtype=np.float64)),
    }


def get_global_peaks_by_direction(
    container: ResultContainer,
    *,
    feq_fields: tuple[str, str, str] = ("feq_x", "feq_y", "meq_z"),
) -> dict[str, pd.DataFrame]:
    """Per-direction global (base) static-equivalent force / moment stats.

    For each wind direction the three static-equivalent load fields are summed
    over floors into a global time history, then reduced to ``min`` / ``max`` /
    ``mean``. Returns ``{"forces_static_eq": df, "moments_static_eq": df}`` where
    each frame has a ``direction`` column plus ``min_{d}`` / ``max_{d}`` /
    ``mean_{d}`` columns (``d in {x, y}`` for forces, ``z`` for moments), sorted
    by direction -- the exact contract consumed by
    :func:`cfdmod.dynamics.plotting.plot_global_stats_per_direction` and
    :func:`export_global_stats_per_direction_csv`.

    Exactly one case per direction is required; pre-filter the container (e.g.
    :func:`filter_by_xi`) so each direction maps to a single case.
    """
    fx, fy, mz = feq_fields
    by_direction = join_by_direction(container)
    force_rows: list[dict[str, float]] = []
    moment_rows: list[dict[str, float]] = []
    for direction in sorted(by_direction):
        sub = by_direction[direction]
        if len(sub) != 1:
            raise ValueError(
                f"direction {direction} maps to {len(sub)} cases; pre-filter the container "
                "to a single case per direction (e.g. one xi / recurrence period)"
            )
        response = next(iter(sub.values()))
        # global (base) history = sum of the floor loads at each timestep
        global_hist = {
            "x": np.sum(np.asarray(response.fields.read(fx), dtype=np.float64), axis=0),
            "y": np.sum(np.asarray(response.fields.read(fy), dtype=np.float64), axis=0),
            "z": np.sum(np.asarray(response.fields.read(mz), dtype=np.float64), axis=0),
        }
        f_row: dict[str, float] = {"direction": float(direction)}
        m_row: dict[str, float] = {"direction": float(direction)}
        for stat, np_reduce in (("min", np.min), ("max", np.max), ("mean", np.mean)):
            f_row[f"{stat}_x"] = float(np_reduce(global_hist["x"]))
            f_row[f"{stat}_y"] = float(np_reduce(global_hist["y"]))
            m_row[f"{stat}_z"] = float(np_reduce(global_hist["z"]))
        force_rows.append(f_row)
        moment_rows.append(m_row)
    return {
        "forces_static_eq": pd.DataFrame(force_rows),
        "moments_static_eq": pd.DataFrame(moment_rows),
    }
