"""Read a TQS "Portico" per-floor modal export (the PAVIMENTO variant).

Some TQS deliveries ship a per-floor summary instead of (or alongside) the
nodal ``PORTELS(SE)`` set, so no nodal aggregation is needed. Files (TAB
separated, decimal point, Latin-1; floor names may contain spaces):

- ``PORTICO_MASSAS_PAVIMENTO.TXT`` -- one row per floor:
  ``Pavimento | Elevacao (cm) | Massa X | Massa Y | Massa Z |
  Momento de inercia da massa | Xcg (cm) | Ycg (cm)``.
- ``PORTICO_MODOS_PAVIMENTO.TXT`` -- a ``//Modo ... DX DY RZ`` header, then
  per mode a single-token mode-number line followed by per-floor
  ``Pavimento | DX (cm) | DY (cm) | RZ (rad)`` rows.
- ``modes.csv`` -- ``mode,period[,wp,freq]`` (the natural periods; the
  MODOS file carries only shapes).

Lengths (cm) and technical mass (``tf.s^2/cm``) are converted to m / kg via
:class:`~cfdmod.dynamics.imports.eberick.EberickUnits` (shared with the
Eberick reader, which uses the same units).
"""

from __future__ import annotations

__all__ = ["read_tqs_portico"]

import csv
import pathlib

import numpy as np

from cfdmod.dynamics.imports._textnum import iter_data_rows, norm_text, to_float
from cfdmod.dynamics.imports.eberick import EberickUnits
from cfdmod.dynamics.structural import BuildingStructuralData, mass_normalize_mode_shapes


def _resolve(source: pathlib.Path, *needles: str, ext: str) -> pathlib.Path:
    for p in sorted(source.iterdir()):
        if p.suffix.lower() == ext and all(n in norm_text(p.name) for n in needles):
            return p
    raise FileNotFoundError(f"no {ext} file matching {needles} in {source}")


def _read_masses(path: pathlib.Path) -> dict[str, tuple[float, float, float, float, float]]:
    """{floor: (elevation, mass_x, inertia, xcg, ycg)} from MASSAS_PAVIMENTO."""
    rows = list(iter_data_rows(path, sep="\t"))
    header_i = next(
        (
            i
            for i, r in enumerate(rows)
            if norm_text(r[0]) == "pavimento" and any("massa" in norm_text(c) for c in r)
        ),
        None,
    )
    if header_i is None:
        raise ValueError(f"{path.name}: no 'Pavimento ... Massa' header row")
    out: dict[str, tuple[float, float, float, float, float]] = {}
    for r in rows[header_i + 1 :]:
        if len(r) < 8 or not r[0]:
            continue
        # 0 name, 1 elev, 2 MassaX, 3 MassaY, 4 MassaZ, 5 inercia, 6 Xcg, 7 Ycg
        out[r[0]] = (
            to_float(r[1]),
            to_float(r[2]),
            to_float(r[5]),
            to_float(r[6]),
            to_float(r[7]),
        )
    return out


def _read_modos(path: pathlib.Path) -> dict[int, dict[str, tuple[float, float, float]]]:
    """{mode_number: {floor: (DX, DY, RZ)}} from MODOS_PAVIMENTO (blocks)."""
    blocks: dict[int, dict[str, tuple[float, float, float]]] = {}
    current: dict[str, tuple[float, float, float]] | None = None
    for r in iter_data_rows(path, sep="\t"):
        if len(r) == 1:
            current = {}
            blocks[int(r[0])] = current
        elif len(r) >= 4 and current is not None:
            current[r[0]] = (to_float(r[1]), to_float(r[2]), to_float(r[3]))
    return blocks


def _read_periods(path: pathlib.Path) -> dict[int, float]:
    """{mode_number: period} from a modes.csv (columns mode, period)."""
    out: dict[int, float] = {}
    with path.open("r", encoding="latin-1", newline="") as fh:
        for row in csv.DictReader(fh):
            keys = {k.strip().lower(): k for k in row}
            out[int(float(row[keys["mode"]]))] = float(row[keys["period"]])
    return out


def read_tqs_portico(
    source: str | pathlib.Path,
    *,
    masses_file: str | pathlib.Path | None = None,
    modos_file: str | pathlib.Path | None = None,
    modes_file: str | pathlib.Path | None = None,
    units: EberickUnits | None = None,
    active_modes: list[int] | None = None,
) -> BuildingStructuralData:
    """Read a TQS Portico per-floor export into a :class:`BuildingStructuralData`.

    Args:
        source: Directory containing the ``PORTICO_*_PAVIMENTO.TXT`` files and
            ``modes.csv``.
        masses_file / modos_file / modes_file: Explicit paths overriding the
            in-directory lookup (for renamed files).
        units: Unit conversions (default cm -> m, tf.s^2/cm -> kg).
        active_modes: 1-based mode numbers to keep (``None`` keeps all).

    Returns:
        Per-floor structural data (floors ascending by elevation,
        mass-normalized mode shapes).
    """
    u = units or EberickUnits()
    source = pathlib.Path(source)
    masses_p = (
        pathlib.Path(masses_file)
        if masses_file
        else _resolve(source, "massas", "pavimento", ext=".txt")
    )
    modos_p = (
        pathlib.Path(modos_file)
        if modos_file
        else _resolve(source, "modos", "pavimento", ext=".txt")
    )
    modes_p = pathlib.Path(modes_file) if modes_file else _resolve(source, "modes", ext=".csv")

    masses = _read_masses(masses_p)
    modos = _read_modos(modos_p)
    periods_by_mode = _read_periods(modes_p)

    mode_nos = sorted(m for m in modos if m in periods_by_mode)
    if active_modes is not None:
        mode_nos = [m for m in mode_nos if m in set(active_modes)]
    if not masses or not mode_nos:
        raise ValueError("Portico export parsed no floors or no usable modes")

    names = sorted(masses, key=lambda n: masses[n][0])
    elev = np.array([masses[n][0] for n in names]) * u.length_to_m
    mass = np.array([masses[n][1] for n in names]) * u.mass_to_kg
    inertia = np.array([masses[n][2] for n in names])
    xcg = np.array([masses[n][3] for n in names]) * u.length_to_m
    ycg = np.array([masses[n][4] for n in names]) * u.length_to_m
    radius = np.sqrt(inertia / np.array([masses[n][1] for n in names])) * u.length_to_m

    phi = np.zeros((len(names), len(mode_nos), 3), dtype=np.float64)
    for mi, mode_no in enumerate(mode_nos):
        block = modos[mode_no]
        for fi, name in enumerate(names):
            dx, dy, rz = block.get(name, (0.0, 0.0, 0.0))
            phi[fi, mi] = (dx * u.length_to_m, dy * u.length_to_m, rz)
    phi = mass_normalize_mode_shapes(phi, mass, radius)

    wp = 2.0 * np.pi / np.array([periods_by_mode[m] for m in mode_nos], dtype=np.float64)

    return BuildingStructuralData(
        mode_shapes=phi,
        natural_frequencies=wp,
        floor_points=np.column_stack([xcg, ycg, elev]),
        cm_positions=np.column_stack([xcg, ycg]),
        floors_mass=mass,
        floors_radius=radius,
        floor_labels=list(names),
    )
