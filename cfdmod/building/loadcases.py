"""Per-floor structural-handoff load-case tables.

Reduces a multi-direction result container (see
:mod:`cfdmod.dynamics.cases`) into the Fx / Fy / Mz load tables the structural
engineer applies: for every wind direction the per-floor static-equivalent
loads are reduced to ``peak`` / ``min`` / ``max`` envelopes, assembled into
load cases, and written to CSV via :class:`cfdmod.report.DebugWriter`.

Open questions (best-guess implementation, flagged for review):

- ``effective_load_stats`` takes the *independent* per-axis envelope: each of
  Fx / Fy / Mz is reduced on its own (the signed time extrema), matching the
  existing :func:`cfdmod.dynamics.plotting.effective_peak_loads_per_direction`.
  The source notebook may instead use *concurrent companion loads* (the
  governing peak on one axis, with the concurrent values on the other two axes
  at the governing time index). Confirm against the notebook before locking.
- ``unit_conversion`` defaults to ``1 / 9806.65`` (N -> tonne-force, N.m ->
  tf.m). The notebook divides by ``9800``; confirm the exact literal and that
  moments use the same divisor.
"""

from __future__ import annotations

__all__ = [
    "LoadStat",
    "effective_load_stats",
    "generate_load_cases",
    "invert_load_cases",
    "save_load_case_tables",
]

import pathlib
from typing import Literal

import numpy as np
import pandas as pd

from cfdmod.building.peaks import PeakMethod, peak_value
from cfdmod.dynamics.cases import get_stats_forces_effective, join_by_direction

LoadStat = Literal["peak", "min", "max", "mean"]

# static-equivalent field -> structural load name
_LOAD_NAMES: tuple[str, str, str] = ("Fx", "Fy", "Mz")
_AXES: tuple[str, str, str] = ("x", "y", "z")

# N -> tonne-force (1 tf = 1000 kgf * 9.80665 m/s^2); moments N.m -> tf.m
_N_PER_TF = 9806.65


def _peak_per_floor(field: np.ndarray, method: PeakMethod, **peak_kwargs) -> np.ndarray:
    """Design peak of |series| for each floor row of a ``(n_floors, n_t)`` field."""
    return np.array(
        [peak_value(row, method, absolute=True, **peak_kwargs) for row in field],
        dtype=np.float64,
    )


def effective_load_stats(
    container,
    *,
    feq_fields: tuple[str, str, str] = ("feq_x", "feq_y", "meq_z"),
    stats: tuple[LoadStat, ...] = ("peak", "min", "max"),
    unit_conversion: float = 1.0 / _N_PER_TF,
    peak_method: PeakMethod = "max",
    peak_kwargs: dict | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Per-direction, per-floor Fx / Fy / Mz envelopes for each statistic.

    ``container`` maps case parameters (with a ``direction`` attribute) to
    building-response data sources carrying the static-equivalent floor loads.
    Exactly one case per direction is required; pre-filter the container (e.g.
    ``cfdmod.dynamics.filter_by_xi``) so each direction maps to a single case.

    Returns ``{stat: {"Fx": df, "Fy": df, "Mz": df}}`` where each frame has
    rows = floors and one column per direction (label ``f"{direction:.1f}"``).
    ``min`` / ``max`` / ``mean`` are the signed time reductions; ``peak`` is the
    design peak of the absolute series (``peak_method``). All values scaled by
    ``unit_conversion``.
    """
    peak_kwargs = peak_kwargs or {}
    tables: dict[str, dict[str, dict[str, np.ndarray]]] = {
        stat: {name: {} for name in _LOAD_NAMES} for stat in stats
    }
    for direction, sub in join_by_direction(container).items():
        if len(sub) != 1:
            raise ValueError(
                f"direction {direction} maps to {len(sub)} cases; pre-filter the container "
                "to a single case per direction (e.g. one xi / recurrence period)"
            )
        response = next(iter(sub.values()))
        col = f"{float(direction):.1f}"
        for stat in stats:
            if stat == "peak":
                per_axis = {
                    name: _peak_per_floor(
                        np.asarray(response.fields.read(field), dtype=np.float64),
                        peak_method,
                        **peak_kwargs,
                    )
                    for name, field in zip(_LOAD_NAMES, feq_fields)
                }
            else:
                reduced = get_stats_forces_effective(response, stat, feq_fields=feq_fields)
                per_axis = {name: reduced[axis] for name, axis in zip(_LOAD_NAMES, _AXES)}
            for name, arr in per_axis.items():
                tables[stat][name][col] = arr * unit_conversion

    return {
        stat: {name: pd.DataFrame(cols) for name, cols in loads.items()}
        for stat, loads in tables.items()
    }


def generate_load_cases(
    stats: dict[str, dict[str, pd.DataFrame]],
    *,
    senses: tuple[LoadStat, LoadStat] = ("max", "min"),
) -> pd.DataFrame:
    """Assemble the envelope tables into a long-form load-case table.

    One logical load case per (direction, sense); for each floor it carries the
    concurrent (Fx, Fy, Mz) taken from that sense's envelope. Columns:
    ``direction``, ``sense``, ``floor``, ``Fx``, ``Fy``, ``Mz``.

    Note: this uses the independent per-axis envelope (each of Fx/Fy/Mz from its
    own signed extremum). If the notebook uses concurrent companion loads, the
    per-axis selection changes but the table shape does not.
    """
    rows: list[dict[str, float | str | int]] = []
    for sense in senses:
        if sense not in stats:
            raise ValueError(f"sense {sense!r} not in stats (have {sorted(stats)})")
        fx, fy, mz = stats[sense]["Fx"], stats[sense]["Fy"], stats[sense]["Mz"]
        n_floors = len(fx)
        for direction in fx.columns:
            for floor in range(n_floors):
                rows.append(
                    {
                        "direction": float(direction),
                        "sense": sense,
                        "floor": floor,
                        "Fx": float(fx.iloc[floor][direction]),
                        "Fy": float(fy.iloc[floor][direction]),
                        "Mz": float(mz.iloc[floor][direction]),
                    }
                )
    return pd.DataFrame(rows, columns=["direction", "sense", "floor", "Fx", "Fy", "Mz"])


def invert_load_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Sign-flipped companion of each load case (reversed-sense wind).

    Negates Fx / Fy / Mz and tags the sense with a trailing ``_inv``. Applying
    it twice restores the original loads (``invert(invert(x)) == x`` on the
    load columns).
    """
    out = cases.copy()
    out[["Fx", "Fy", "Mz"]] = -out[["Fx", "Fy", "Mz"]]
    out["sense"] = (
        out["sense"]
        .astype(str)
        .apply(lambda s: s[: -len("_inv")] if s.endswith("_inv") else s + "_inv")
    )
    return out


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
