"""Read a TQS "Portico Espacial" modal export.

TQS exports the spatial-frame modal analysis as a set of Latin-1,
TAB-separated, comma-decimal text files with ``//`` comment lines. The
file-name prefix is ``PORTELS_`` (older) or ``PORTELSSE_`` (newer); the
reader matches by the ``_<SUFFIX>.TXT`` tail so either works:

- ``*_MODOS.TXT``  -- ``Modo; Periodo(s); Freq angular(rad/s); Freq(Hz)``,
  preceded by a single line with the mode count.
- ``*_NOS.TXT``    -- ``No; X; Y; Z`` (m), preceded by the node count.
- ``*_MASSAS.TXT`` -- ``No; Massa X; Massa Y; Massa Z`` (lumped nodal mass).
- ``*_FORMAS2.TXT``-- per-mode blocks ``// Modo / <n> / No; DX; DY; RZ``
  (nodal mode shape carrying the ``RZ`` rotation). ``*_FORMAS.TXT`` is the
  same but with ``DZ`` instead of ``RZ`` -- we use ``FORMAS2``.
- ``*_PISOS.TXT``  -- optional ``Piso; Nome; Nivel(m)`` floor table (newer
  exports). When present it defines the real slab elevations, so the many
  intermediate FE node levels (beams, landings) collapse onto actual floors.

:func:`read_tqs_portels` parses these into a :class:`NodalModel` and
aggregates it per floor via :func:`aggregate_to_building`.
"""

from __future__ import annotations

__all__ = ["read_tqs_portels", "PORTELS_FILES"]

import pathlib

import numpy as np

from cfdmod.dynamics.imports._textnum import iter_data_rows, to_float
from cfdmod.dynamics.imports.nodal import NodalModel, aggregate_to_building
from cfdmod.dynamics.structural import BuildingStructuralData

# Role -> file-name suffix. TQS names these ``PORTELS_<SUFFIX>.TXT`` (older) or
# ``PORTELSSE_<SUFFIX>.TXT`` (newer), so we match by the ``_<SUFFIX>.TXT`` tail
# rather than a fixed prefix.
PORTELS_FILES = {
    "modes": "MODOS",
    "nodes": "NOS",
    "masses": "MASSAS",
    "shapes": "FORMAS2",
}
# Newer exports carry a floor table too; read for validation when present.
PISOS_SUFFIX = "PISOS"


def _resolve(source: str | pathlib.Path, suffix: str, *, required: bool = True):
    """Find the ``*_<suffix>.TXT`` file in ``source`` (case-insensitive).

    Matches both the ``PORTELS_`` and ``PORTELSSE_`` prefixes (preferring the
    newer ``PORTELSSE_`` when both exist). Returns ``None`` when ``required``
    is false and no file matches.
    """
    source = pathlib.Path(source)
    tail = f"_{suffix.lower()}.txt"
    matches = [p for p in source.iterdir() if p.is_file() and p.name.lower().endswith(tail)]
    if not matches:
        if required:
            raise FileNotFoundError(f"no *{tail.upper()} file in PORTELS export dir {source}")
        return None
    # Prefer PORTELSSE_ (newer), then PORTELS_, then anything, then shortest name.
    matches.sort(
        key=lambda p: (
            0
            if p.name.upper().startswith("PORTELSSE_")
            else 1
            if p.name.upper().startswith("PORTELS_")
            else 2,
            len(p.name),
        )
    )
    return matches[0]


def _read_modes(path: pathlib.Path) -> np.ndarray:
    """Periods (s), one per mode, ordered by mode number."""
    rows = [r for r in iter_data_rows(path) if len(r) >= 4]
    # Rows have 4 columns; the lone count line (1 token) is filtered out above.
    modes = sorted((int(r[0]), to_float(r[1])) for r in rows)
    return np.asarray([period for _, period in modes], dtype=np.float64)


def _read_nodes(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Node ids and ``(n, 3)`` coordinates, filtered to 4-column data rows."""
    ids, coords = [], []
    for r in iter_data_rows(path):
        if len(r) < 4:
            continue  # the count line
        ids.append(int(r[0]))
        coords.append([to_float(r[1]), to_float(r[2]), to_float(r[3])])
    return np.asarray(ids, dtype=np.int64), np.asarray(coords, dtype=np.float64)


def _read_masses(path: pathlib.Path) -> dict[int, float]:
    """Map node id -> translational mass (the X-direction lumped mass)."""
    out: dict[int, float] = {}
    for r in iter_data_rows(path):
        if len(r) < 2:
            continue
        out[int(r[0])] = to_float(r[1])
    return out


def _read_shapes(path: pathlib.Path) -> dict[int, dict[int, tuple[float, float, float]]]:
    """Parse per-mode blocks -> {mode_number: {node_id: (DX, DY, RZ)}}.

    Blocks are delimited by a single-token line carrying the mode number
    (the ``// Modo`` comment above it is stripped by the row iterator);
    subsequent 4-column lines are that mode's nodal shape rows.
    """
    blocks: dict[int, dict[int, tuple[float, float, float]]] = {}
    current: dict[int, tuple[float, float, float]] | None = None
    for r in iter_data_rows(path):
        if len(r) == 1:
            current = {}
            blocks[int(r[0])] = current
        elif len(r) >= 4 and current is not None:
            current[int(r[0])] = (to_float(r[1]), to_float(r[2]), to_float(r[3]))
    return blocks


def _read_pisos(path: pathlib.Path) -> tuple[list[float], list[str]]:
    """(levels, names) from a PISOS table (``Piso; Nome; Nivel(m)``).

    The level is the *last* token and the piso index the first; the name is
    everything in between (it may contain spaces), rejoined with a space.
    """
    levels: list[float] = []
    names: list[str] = []
    for r in iter_data_rows(path):
        if len(r) >= 3 and r[0].isdigit():
            levels.append(to_float(r[-1]))
            names.append(" ".join(r[1:-1]))
    return levels, names


def read_tqs_portels(
    source: str | pathlib.Path,
    *,
    active_modes: list[int] | None = None,
    tol_z: float = 0.05,
) -> BuildingStructuralData:
    """Read a TQS PORTELS export directory into a :class:`BuildingStructuralData`.

    Handles both the older ``PORTELS_*.TXT`` and newer ``PORTELSSE_*.TXT``
    file names (matched by suffix). When a ``*_PISOS.TXT`` floor table is
    present it is used to sanity-check the recovered floor count.

    Args:
        source: Directory containing the ``PORTELS(SE)_*.TXT`` files.
        active_modes: 1-based mode numbers to keep (``None`` keeps all).
        tol_z: Slab elevation clustering tolerance (m).

    Returns:
        Per-floor structural data ready for the building dynamic recipe.
    """
    modes_p = _resolve(source, PORTELS_FILES["modes"])
    nodes_p = _resolve(source, PORTELS_FILES["nodes"])
    masses_p = _resolve(source, PORTELS_FILES["masses"])
    shapes_p = _resolve(source, PORTELS_FILES["shapes"])

    periods = _read_modes(modes_p)
    node_ids, coords = _read_nodes(nodes_p)
    mass_by_id = _read_masses(masses_p)
    shape_blocks = _read_shapes(shapes_p)

    n_modes = periods.shape[0]
    mode_numbers = sorted(shape_blocks)
    if len(mode_numbers) < n_modes:
        raise ValueError(
            f"{shapes_p.name} has {len(mode_numbers)} mode blocks but "
            f"{modes_p.name} declares {n_modes} modes"
        )

    mass = np.asarray([mass_by_id.get(int(nid), 0.0) for nid in node_ids], dtype=np.float64)
    shapes = np.zeros((len(node_ids), n_modes, 3), dtype=np.float64)
    for mi, mode_no in enumerate(mode_numbers[:n_modes]):
        block = shape_blocks[mode_no]
        for ni, nid in enumerate(node_ids):
            dx, dy, rz = block.get(int(nid), (0.0, 0.0, 0.0))
            shapes[ni, mi] = (dx, dy, rz)

    # A PISOS floor table (newer exports), when present, defines the real slab
    # levels -- the FE model otherwise has many intermediate node elevations
    # (beams, landings) that naive Z-clustering would mistake for floors.
    pisos_p = _resolve(source, PISOS_SUFFIX, required=False)
    floor_levels, floor_labels = _read_pisos(pisos_p) if pisos_p is not None else (None, None)

    model = NodalModel(coords=coords, mass=mass, periods=periods, shapes=shapes, node_ids=node_ids)
    sd = aggregate_to_building(
        model,
        tol_z=tol_z,
        floor_levels=floor_levels,
        floor_labels=floor_labels,
        active_modes=active_modes,
    )

    if floor_levels is not None and sd.n_floors != len(floor_levels):
        import warnings

        warnings.warn(
            f"{pisos_p.name} lists {len(floor_levels)} floors but {sd.n_floors} carry mass "
            f"(zero-mass levels such as foundation/roof are dropped).",
            stacklevel=2,
        )
    return sd
