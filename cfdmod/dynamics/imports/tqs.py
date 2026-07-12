"""Read a TQS "Portico Espacial" (PORTELS) modal export.

TQS exports the spatial-frame modal analysis as a set of Latin-1,
TAB-separated, comma-decimal text files with ``//`` comment lines:

- ``PORTELS_MODOS.TXT``  -- ``Modo; Periodo(s); Freq angular(rad/s); Freq(Hz)``,
  preceded by a single line with the mode count.
- ``PORTELS_NOS.TXT``    -- ``No; X; Y; Z`` (m), preceded by the node count.
- ``PORTELS_MASSAS.TXT`` -- ``No; Massa X; Massa Y; Massa Z`` (lumped nodal mass).
- ``PORTELS_FORMAS2.TXT``-- per-mode blocks ``// Modo / <n> / No; DX; DY; RZ``
  (nodal mode shape carrying the ``RZ`` rotation). ``PORTELS_FORMAS.TXT``
  is the same but with ``DZ`` instead of ``RZ`` -- we use ``FORMAS2``.

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

# Default file names inside a PORTELS export directory (case-insensitive match).
PORTELS_FILES = {
    "modes": "PORTELS_MODOS.TXT",
    "nodes": "PORTELS_NOS.TXT",
    "masses": "PORTELS_MASSAS.TXT",
    "shapes": "PORTELS_FORMAS2.TXT",
}


def _resolve(source: str | pathlib.Path, name: str) -> pathlib.Path:
    """Find ``name`` inside ``source`` (a directory), case-insensitively."""
    source = pathlib.Path(source)
    direct = source / name
    if direct.exists():
        return direct
    lowered = name.lower()
    for p in source.iterdir():
        if p.name.lower() == lowered:
            return p
    raise FileNotFoundError(f"{name} not found in PORTELS export dir {source}")


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


def read_tqs_portels(
    source: str | pathlib.Path,
    *,
    active_modes: list[int] | None = None,
    tol_z: float = 0.05,
) -> BuildingStructuralData:
    """Read a TQS PORTELS export directory into a :class:`BuildingStructuralData`.

    Args:
        source: Directory containing the ``PORTELS_*.TXT`` files.
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

    model = NodalModel(coords=coords, mass=mass, periods=periods, shapes=shapes, node_ids=node_ids)
    return aggregate_to_building(model, tol_z=tol_z, active_modes=active_modes)
