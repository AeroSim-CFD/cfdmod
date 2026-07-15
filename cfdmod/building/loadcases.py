"""Per-floor structural-handoff load-case tables.

Reduces a multi-direction result container (see :mod:`cfdmod.dynamics.cases`)
into the two deliverables the structural engineer receives, faithful to the
consulting ``hfpi_analysis`` notebook:

1. ``effective_load_stats`` -- the per-floor Fx / Fy / Mz tables, one column per
   wind direction, for ``peak`` / ``min`` / ``max`` (and ``mean``). ``min`` /
   ``max`` are the signed per-floor effective-load envelopes; ``peak`` is the
   *governing* envelope (whichever of min / max has the larger mean magnitude
   per direction / axis -- the notebook's ``r_peak`` selection).
2. ``generate_load_cases`` + ``invert_load_cases`` -- the Eberick load cases:
   for each principal axis the critical direction is picked, then the principal
   sign (max / min) is combined with all four companion-axis sign combinations
   (the companion-load method). ``invert_load_cases`` reorganises the resulting
   cases into per-axis DataFrames (it does *not* negate).

"Effective" loads combine the dynamic static-equivalent loads with the applied
quasi-static loads (see :func:`cfdmod.dynamics.get_stats_forces_effective`); the
fan-out driver attaches the applied loads (``fs_*`` / ``ms_z``) so the
combination happens automatically. This is the fuller reducer;
:func:`cfdmod.dynamics.plotting.effective_peak_loads_per_direction` is the
presentation-layer sibling that reads ``feq_*`` alone (no applied-static
combination) and is what the current consulting notebooks call. Both default to
the same exact tonne-force divisor ``1 / 9806.65`` (1000 kgf * standard gravity
9.80665 m/s^2); pass your own to byte-match an old deliverable that used a
rounded ``9800`` / ``9806``.
"""

from __future__ import annotations

__all__ = [
    "LoadStat",
    "directional_envelopes",
    "effective_load_stats",
    "generate_load_cases",
    "invert_load_cases",
    "save_load_case_tables",
]

import pathlib
from typing import Literal

import numpy as np
import pandas as pd

from cfdmod.dynamics.cases import get_stats_forces_effective, join_by_direction

LoadStat = Literal["peak", "min", "max", "mean"]

# static-equivalent field -> structural load name
_LOAD_NAMES: tuple[str, str, str] = ("Fx", "Fy", "Mz")
_AXES: tuple[str, str, str] = ("x", "y", "z")

# N -> tonne-force (1 tf = 1000 kgf * 9.80665 m/s^2); moments N.m -> tf.m
_N_PER_TF = 9806.65


def directional_envelopes(
    container,
    *,
    feq_fields: tuple[str, str, str] = ("feq_x", "feq_y", "meq_z"),
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Per-direction effective ``max`` / ``min`` envelopes ``{direction: {axis: arr}}``.

    Exactly one case per direction is required; pre-filter the container (e.g.
    ``cfdmod.dynamics.filter_by_xi``) so each direction maps to a single case.
    """
    max_dict: dict[str, dict[str, np.ndarray]] = {}
    min_dict: dict[str, dict[str, np.ndarray]] = {}
    for direction, sub in join_by_direction(container).items():
        if len(sub) != 1:
            raise ValueError(
                f"direction {direction} maps to {len(sub)} cases; pre-filter the container "
                "to a single case per direction (e.g. one xi / recurrence period)"
            )
        response = next(iter(sub.values()))
        max_dict[direction] = get_stats_forces_effective(response, "max", feq_fields=feq_fields)
        min_dict[direction] = get_stats_forces_effective(response, "min", feq_fields=feq_fields)
    return max_dict, min_dict


def effective_load_stats(
    container,
    *,
    feq_fields: tuple[str, str, str] = ("feq_x", "feq_y", "meq_z"),
    stats: tuple[LoadStat, ...] = ("peak", "min", "max"),
    unit_conversion: float = 1.0 / _N_PER_TF,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Per-direction, per-floor Fx / Fy / Mz effective-load tables per statistic.

    Returns ``{stat: {"Fx": df, "Fy": df, "Mz": df}}`` where each frame has
    rows = floors and one column per direction (label ``f"{direction:.1f}"``).
    ``min`` / ``max`` are the signed effective envelopes, ``mean`` the average,
    and ``peak`` the governing envelope (larger |mean| of min / max per
    direction / axis). All values scaled by ``unit_conversion``.
    """
    need_mean = "mean" in stats
    tables: dict[str, dict[str, dict[str, np.ndarray]]] = {
        stat: {name: {} for name in _LOAD_NAMES} for stat in stats
    }
    # Partition by direction once; reduce each single-case response inline.
    for direction, sub in join_by_direction(container).items():
        if len(sub) != 1:
            raise ValueError(
                f"direction {direction} maps to {len(sub)} cases; pre-filter the container "
                "to a single case per direction (e.g. one xi / recurrence period)"
            )
        response = next(iter(sub.values()))
        r_max = get_stats_forces_effective(response, "max", feq_fields=feq_fields)
        r_min = get_stats_forces_effective(response, "min", feq_fields=feq_fields)
        r_mean = (
            get_stats_forces_effective(response, "mean", feq_fields=feq_fields)
            if need_mean
            else None
        )
        col = f"{float(direction):.1f}"
        for name, axis in zip(_LOAD_NAMES, _AXES):
            mx, mn = r_max[axis], r_min[axis]
            r_peak = mn if abs(mn.mean()) > abs(mx.mean()) else mx
            per_stat = {"max": mx, "min": mn, "peak": r_peak}
            if need_mean:
                per_stat["mean"] = r_mean[axis]
            for stat in stats:
                tables[stat][name][col] = per_stat[stat] * unit_conversion

    return {
        stat: {name: pd.DataFrame(cols) for name, cols in loads.items()}
        for stat, loads in tables.items()
    }


def generate_load_cases(
    max_dict: dict[str, dict[str, np.ndarray]],
    min_dict: dict[str, dict[str, np.ndarray]],
    *,
    unit_conversion: float = 1.0 / _N_PER_TF,
) -> dict[int, dict[str, np.ndarray]]:
    """Eberick companion-load cases from the per-direction envelopes.

    For each principal axis the critical direction is the one with the largest
    mean ``max`` load; the principal sign (max / min) is then combined with all
    four companion-axis sign combinations. Returns ``{case_id: {"Fx","Fy","Mz":
    per-floor array}}`` (loads scaled by ``unit_conversion``). Mirrors the
    ``hfpi_analysis`` notebook's ``generate_load_cases``.
    """
    axes = list(_AXES)
    companion_combinations = [("max", "max"), ("max", "min"), ("min", "max"), ("min", "min")]
    picked = {"max": max_dict, "min": min_dict}

    def axis_name(axis: str) -> str:
        return ("F" + axis) if axis in ("x", "y") else ("M" + axis)

    load_cases: dict[int, dict[str, np.ndarray]] = {}
    case_id = 0
    for principal in axes:
        theta_star = max(max_dict.keys(), key=lambda d: max_dict[d][principal].mean())
        b, c = [a for a in axes if a != principal]
        for principal_sign in ("max", "min"):
            for comb in companion_combinations:
                load: dict[str, np.ndarray] = {}
                load[axis_name(principal)] = (
                    picked[principal_sign][theta_star][principal] * unit_conversion
                )
                for axis, sign in zip((b, c), comb):
                    load[axis_name(axis)] = picked[sign][theta_star][axis] * unit_conversion
                load_cases[case_id] = load
                case_id += 1
    return load_cases


def invert_load_cases(cases: dict[int, dict[str, np.ndarray]]) -> dict[str, pd.DataFrame]:
    """Reorganise load cases into per-axis DataFrames (rows = floors, cols = case).

    Mirrors the notebook's ``invert_load_cases`` -- a transpose of
    ``{case_id: {axis: arr}}`` into ``{axis: DataFrame}``; it does not negate.
    """
    data: dict[str, dict[int, np.ndarray]] = {axis: {} for axis in _LOAD_NAMES}
    for case_id, load in cases.items():
        for axis in _LOAD_NAMES:
            data[axis][case_id] = load[axis]
    return {axis: pd.DataFrame(data[axis]) for axis in _LOAD_NAMES}


def save_load_case_tables(
    stats: dict[str, dict[str, pd.DataFrame]],
    writer,
    *,
    deliverable: bool = True,
    floor_heights: np.ndarray | None = None,
    prefix: str = "loadcase",
    skip_if_exists: bool = False,
) -> dict[str, pathlib.Path]:
    """Write each ``{prefix}_{stat}_{Fx|Fy|Mz}.csv`` via ``writer.save_csv``.

    A leading ``floor`` column (and ``z`` when ``floor_heights`` is given) is
    materialized because ``DebugWriter.save_csv`` defaults to ``index=False``.
    Returns ``{csv name: written path}``.
    """
    written: dict[str, pathlib.Path] = {}
    for stat, loads in stats.items():
        for name, df in loads.items():
            out = df.copy()
            out.insert(0, "floor", np.arange(len(out)))
            if floor_heights is not None:
                if len(floor_heights) != len(out):
                    raise ValueError(
                        f"floor_heights has {len(floor_heights)} entries; expected {len(out)}"
                    )
                out.insert(1, "z", np.asarray(floor_heights, dtype=np.float64))
            csv_name = f"{prefix}_{stat}_{name}.csv"
            written[csv_name] = writer.save_csv(
                out, csv_name, deliverable=deliverable, skip_if_exists=skip_if_exists
            )
    return written
