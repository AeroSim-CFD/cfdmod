"""Read an Eberick per-floor (rigid-diaphragm) modal export.

Eberick models each storey as a rigid diaphragm with one master node, so
its dynamic-analysis results are already per-floor -- unlike TQS, no nodal
aggregation is needed. The reader consumes a workbook (``.xlsx``) with
three tables (default Portuguese sheet/column names, all overridable via
:class:`EberickColumns`):

- floors sheet (``Pavimentos``): ``Pavimento, Cota, Massa, Inercia`` and
  optionally ``Xcg, Ycg`` (centre-of-mass offset; defaults to 0).
- modes sheet (``Modos``): ``Modo, Periodo`` (seconds).
- shapes sheet (``Formas``), long form: ``Pavimento, Modo, DX, DY, RZ``.

No real Eberick export is on file yet, so the exact default column names
are a documented convention; point the reader at a real export by passing
an :class:`EberickColumns` with the true labels.
"""

from __future__ import annotations

__all__ = ["EberickColumns", "read_eberick"]

import pathlib

import numpy as np
import pandas as pd
from pydantic import BaseModel

from cfdmod.dynamics.structural import BuildingStructuralData, mass_normalize_mode_shapes


class EberickColumns(BaseModel):
    """Sheet and column names for an Eberick modal workbook (all overridable)."""

    sheet_floors: str = "Pavimentos"
    sheet_modes: str = "Modos"
    sheet_shapes: str = "Formas"

    floor_name: str = "Pavimento"
    elevation: str = "Cota"
    mass: str = "Massa"
    inertia: str = "Inercia"
    xcg: str = "Xcg"
    ycg: str = "Ycg"

    mode: str = "Modo"
    period: str = "Periodo"

    shape_floor: str = "Pavimento"
    shape_mode: str = "Modo"
    dx: str = "DX"
    dy: str = "DY"
    rz: str = "RZ"


def _require(df: pd.DataFrame, cols: list[str], sheet: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Eberick sheet {sheet!r} is missing columns {missing}; got {list(df.columns)}"
        )


def read_eberick(
    source: str | pathlib.Path,
    *,
    columns: EberickColumns | None = None,
    active_modes: list[int] | None = None,
) -> BuildingStructuralData:
    """Read an Eberick per-floor modal workbook into a :class:`BuildingStructuralData`.

    Args:
        source: Path to the ``.xlsx`` workbook.
        columns: Sheet/column-name overrides (defaults to the documented
            Portuguese layout).
        active_modes: 1-based mode numbers to keep (``None`` keeps all).

    Returns:
        Per-floor structural data (floors ordered by ascending elevation,
        mass-normalized mode shapes).
    """
    c = columns or EberickColumns()
    source = pathlib.Path(source)
    sheets = pd.read_excel(source, sheet_name=[c.sheet_floors, c.sheet_modes, c.sheet_shapes])
    floors = sheets[c.sheet_floors]
    modes = sheets[c.sheet_modes]
    shapes = sheets[c.sheet_shapes]

    _require(floors, [c.floor_name, c.elevation, c.mass, c.inertia], c.sheet_floors)
    _require(modes, [c.mode, c.period], c.sheet_modes)
    _require(shapes, [c.shape_floor, c.shape_mode, c.dx, c.dy, c.rz], c.sheet_shapes)

    floors = floors.sort_values(c.elevation).reset_index(drop=True)
    floor_order = list(floors[c.floor_name])
    floor_row = {name: i for i, name in enumerate(floor_order)}
    n_floors = len(floor_order)

    elevations = floors[c.elevation].to_numpy(dtype=np.float64)
    floors_mass = floors[c.mass].to_numpy(dtype=np.float64)
    inertia = floors[c.inertia].to_numpy(dtype=np.float64)
    floors_radius = np.sqrt(inertia / np.where(floors_mass > 0, floors_mass, np.nan))
    xcg = floors[c.xcg].to_numpy(dtype=np.float64) if c.xcg in floors else np.zeros(n_floors)
    ycg = floors[c.ycg].to_numpy(dtype=np.float64) if c.ycg in floors else np.zeros(n_floors)

    all_modes = sorted(int(m) for m in modes[c.mode].unique())
    keep = all_modes if active_modes is None else [m for m in all_modes if m in set(active_modes)]
    mode_row = {m: i for i, m in enumerate(keep)}
    period_by_mode = dict(zip(modes[c.mode].astype(int), modes[c.period].astype(float)))
    periods = np.asarray([period_by_mode[m] for m in keep], dtype=np.float64)

    phi = np.zeros((n_floors, len(keep), 3), dtype=np.float64)
    for _, row in shapes.iterrows():
        mode_no = int(row[c.shape_mode])
        if mode_no not in mode_row:
            continue
        fi = floor_row[row[c.shape_floor]]
        phi[fi, mode_row[mode_no]] = (float(row[c.dx]), float(row[c.dy]), float(row[c.rz]))

    phi = mass_normalize_mode_shapes(phi, floors_mass, floors_radius)
    wp = 2.0 * np.pi / periods

    return BuildingStructuralData(
        mode_shapes=phi,
        natural_frequencies=wp,
        floor_points=np.column_stack([xcg, ycg, elevations]),
        cm_positions=np.column_stack([xcg, ycg]),
        floors_mass=floors_mass,
        floors_radius=floors_radius,
    )
