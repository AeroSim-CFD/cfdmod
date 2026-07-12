"""Read an Eberick (AltoQi) per-floor modal export.

Eberick delivers the dynamic-analysis model as a pair of spreadsheets
(one row per storey -- Eberick works with rigid floor diaphragms, so no
nodal aggregation is needed):

- ``DISTRIBUICAO_DAS_MASSAS_DOS_PAVIMENTOS.xlsx`` -- a per-floor table
  ``Pavimento | Altura | Elevacao (cm) | Massa | Momento de inercia | Xcg (cm) | Ycg (cm)``.
- ``FORMAS_MODAIS_DOS_PAVIMENTOS.xlsx`` -- one block per mode, headed by
  ``Modo N`` and ``Frequencia (Hz): <f>``, then a per-floor table
  ``Pavimento | Dx (cm) | Dy (cm) | Rz (rad)``.

Both sheets carry a project-identifying header block (OBRA / Cliente /
Endereco) and an AltoQi footer, which the reader skips. Eberick uses
centimetre lengths and technical mass units (``tf.s^2/cm``); the defaults
in :class:`EberickUnits` convert those to metres and kilograms. A third
"sistema de referencia" sheet only documents the axes and the damping
ratio and is not required here (pass the damping to
:meth:`BuildingStructuralData.to_config`).
"""

from __future__ import annotations

__all__ = ["EberickUnits", "read_eberick"]

import pathlib

import numpy as np
import openpyxl
from pydantic import BaseModel

from cfdmod.dynamics.imports._textnum import norm_text as _norm
from cfdmod.dynamics.imports._textnum import to_float
from cfdmod.dynamics.structural import BuildingStructuralData, mass_normalize_mode_shapes

# 1 tf.s^2/cm = 1 tonne-force . s^2 / cm = 9806.65 N / 0.01 m . s^2 = 980665 kg.
_TF_S2_PER_CM_TO_KG = 9806.65 / 0.01


class EberickUnits(BaseModel):
    """Unit conversions for an Eberick export (defaults: cm -> m, tf.s^2/cm -> kg)."""

    length_to_m: float = 0.01
    mass_to_kg: float = _TF_S2_PER_CM_TO_KG


def _cell_float(c) -> float:
    """Read a cell that may be a number or a comma-decimal string."""
    if c is None:
        return float("nan")
    if isinstance(c, (int, float)):
        return float(c)
    return to_float(str(c))


def _rows(path: pathlib.Path) -> list[tuple]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(wb.active.iter_rows(values_only=True))
    finally:
        wb.close()


def _resolve(source: pathlib.Path, *needles: str) -> pathlib.Path:
    """Find the file in ``source`` whose name contains all ``needles`` (normalized)."""
    for p in sorted(source.iterdir()):
        name = _norm(p.name)
        if p.suffix.lower() in {".xlsx", ".xls"} and all(n in name for n in needles):
            return p
    raise FileNotFoundError(f"no Eberick file matching {needles} in {source}")


def _is_footer(name: str) -> bool:
    return name.startswith("altoqi") or name == ""


def _read_masses(rows: list[tuple]) -> dict[str, tuple[float, float, float, float, float]]:
    """{floor name: (elevation, mass, inertia, xcg, ycg)} from the masses sheet.

    Column positions follow the fixed Eberick layout (validated by checking the
    header row carries 'pavimento' / 'eleva' / 'massa'). The centre-of-mass
    sub-header ('Xcg'/'Ycg') sits one row below the main header, so data starts
    two rows after it.
    """
    header_i = None
    for i, row in enumerate(rows):
        cells = [_norm(c) for c in row]
        if (
            "pavimento" in cells
            and any("eleva" in c for c in cells)
            and any("massa" in c for c in cells)
        ):
            header_i = i
            break
    if header_i is None:
        raise ValueError("masses sheet: could not find the 'Pavimento ... Massa' header row")

    out: dict[str, tuple[float, float, float, float, float, float]] = {}
    for row in rows[header_i + 2 :]:
        name = row[0]
        if _is_footer(_norm(name)):
            if out:
                break
            continue
        # cols: 0 Pavimento, 1 Altura, 2 Elevacao, 3 Massa, 4 Inercia, 5 Xcg, 6 Ycg
        out[str(name).strip()] = (
            _cell_float(row[2]),  # elevation
            _cell_float(row[3]),  # mass
            _cell_float(row[4]),  # inertia
            _cell_float(row[5]),  # xcg
            _cell_float(row[6]),  # ycg
            _cell_float(row[1]),  # altura (storey height) -> metadata
        )
    return out


def _read_formas(rows: list[tuple]) -> list[tuple[float, dict[str, tuple[float, float, float]]]]:
    """[(frequency_hz, {floor: (Dx, Dy, Rz)})] per mode, in file order."""
    modes: list[tuple[float, dict[str, tuple[float, float, float]]]] = []
    freq: float | None = None
    cols: dict[str, int] | None = None
    current: dict[str, tuple[float, float, float]] | None = None

    def flush():
        if freq is not None and current:
            modes.append((freq, current))

    for row in rows:
        joined = _norm(" ".join(str(c) for c in row if c is not None))
        cells = [_norm(c) for c in row]

        if joined.startswith("modo "):
            flush()
            freq, cols, current = None, None, None
            continue
        if "frequencia (hz)" in joined:
            freq = to_float(joined.split(":")[-1])
            continue
        if "pavimento" in cells and any(c == "dx (cm)" or c.startswith("dx") for c in cells):
            cols = {}
            for j, c in enumerate(cells):
                if c == "pavimento":
                    cols["name"] = j
                elif c.startswith("dx"):
                    cols["dx"] = j
                elif c.startswith("dy"):
                    cols["dy"] = j
                elif c.startswith("rz"):
                    cols["rz"] = j
            current = {}
            continue
        if cols is not None and current is not None:
            name = row[cols["name"]]
            if _is_footer(_norm(name)):
                continue
            current[str(name).strip()] = (
                _cell_float(row[cols["dx"]]),
                _cell_float(row[cols["dy"]]),
                _cell_float(row[cols["rz"]]),
            )
    flush()
    return modes


def read_eberick(
    source: str | pathlib.Path,
    *,
    masses_file: str | pathlib.Path | None = None,
    formas_file: str | pathlib.Path | None = None,
    units: EberickUnits | None = None,
    active_modes: list[int] | None = None,
) -> BuildingStructuralData:
    """Read an Eberick export directory into a :class:`BuildingStructuralData`.

    Args:
        source: Directory holding the ``DISTRIBUICAO_DAS_MASSAS...`` and
            ``FORMAS_MODAIS...`` workbooks (matched case/accent-insensitively).
        masses_file / formas_file: Explicit workbook paths overriding the
            in-directory lookup (for renamed files).
        units: Unit conversions (default cm -> m, tf.s^2/cm -> kg).
        active_modes: 1-based mode numbers to keep (``None`` keeps all).

    Returns:
        Per-floor structural data (floors ascending by elevation, mass-
        normalized mode shapes; storey names in ``floor_labels`` and the
        storey heights in ``floor_metadata['altura_cm']``).
    """
    u = units or EberickUnits()
    source = pathlib.Path(source)
    mass_p = pathlib.Path(masses_file) if masses_file else _resolve(source, "distribui", "massa")
    formas_p = (
        pathlib.Path(formas_file) if formas_file else _resolve(source, "formas", "pavimentos")
    )
    mass_rows = _rows(mass_p)
    formas_rows = _rows(formas_p)

    masses = _read_masses(mass_rows)
    modes = _read_formas(formas_rows)
    if active_modes is not None:
        modes = [modes[m - 1] for m in active_modes]
    if not masses or not modes:
        raise ValueError("Eberick export parsed no floors or no modes")

    # Floor order: ascending elevation from the masses table.
    names = sorted(masses, key=lambda n: masses[n][0])
    elev = np.array([masses[n][0] for n in names]) * u.length_to_m
    mass = np.array([masses[n][1] for n in names]) * u.mass_to_kg
    inertia = np.array([masses[n][2] for n in names])
    xcg = np.array([masses[n][3] for n in names]) * u.length_to_m
    ycg = np.array([masses[n][4] for n in names]) * u.length_to_m
    # radius of gyration = sqrt(I/M); I/M is an area in cm^2 -> convert to m.
    radius = np.sqrt(inertia / np.array([masses[n][1] for n in names])) * u.length_to_m

    phi = np.zeros((len(names), len(modes), 3), dtype=np.float64)
    for mi, (_, shape) in enumerate(modes):
        for fi, name in enumerate(names):
            dx, dy, rz = shape.get(name, (0.0, 0.0, 0.0))
            phi[fi, mi] = (dx * u.length_to_m, dy * u.length_to_m, rz)
    phi = mass_normalize_mode_shapes(phi, mass, radius)

    periods = np.array([1.0 / f for f, _ in modes], dtype=np.float64)
    wp = 2.0 * np.pi / periods

    return BuildingStructuralData(
        mode_shapes=phi,
        natural_frequencies=wp,
        floor_points=np.column_stack([xcg, ycg, elev]),
        cm_positions=np.column_stack([xcg, ycg]),
        floors_mass=mass,
        floors_radius=radius,
        floor_labels=list(names),
        floor_metadata={"altura_cm": [float(masses[n][5]) for n in names]},
    )
