"""Convert a line-probe H5 file into the legacy CSV layout.

The AeroSim solver now writes line probes as a single H5 file:

    <line>.h5
        Geometry      (N, 3) float  -- x, y, z of each probe point
        Connectivity  (M,)   int    -- line connectivity (not exported)
        ux / uy / uz / pressure     -- one group per field, each holding
                                       one dataset per time step, named
                                       "t<time>" and shaped (N,)

The old post-processing tools expected instead a set of CSV files:

    <line>.points.csv   idx,x,y,z                (one row per point)
    <line>.ux.csv       time_step,0,1,2,...,N-1  (one row per time step,
                                                  one column per point)

and equivalently for uy, uz and pressure. This script reproduces that CSV
layout from the H5 file so the legacy CSV-based scripts keep working
unchanged.

HOW TO USE: edit the CONFIG block right below, then run the file:

    python scripts/h5_line_to_csv.py
"""

from __future__ import annotations

import csv
import os

import h5py
import numpy as np

# =============================================================================
# CONFIG -- edit these values, then just run the file.
# =============================================================================

# Input line-probe H5 file.
H5_PATH = r"/home/waine/Downloads/line.line_line_roof.h5"

# Directory to write the CSV files into.
OUT_DIR = r"/home/waine/Downloads/h5_postpro_out"

# Fields to export. A field that is absent from the H5 is skipped with a
# warning (e.g. some line probes have no "pressure").
FIELDS = ["ux", "uy", "uz", "pressure"]

# =============================================================================
# End of CONFIG. You normally do not need to edit below this line.
# =============================================================================


def _parse_time(key: str) -> float:
    """Turn a time-step dataset name ("t2000.638062") into a float."""
    return float(key.lstrip("t"))


def _sorted_time_keys(group: h5py.Group) -> list[str]:
    """Return the time-step dataset names of a field group, sorted by time."""
    keys = list(group.keys())
    return sorted(keys, key=_parse_time)


def write_points_csv(geometry: np.ndarray, path: str) -> None:
    """Write the point coordinates as idx,x,y,z (matches points.csv)."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "x", "y", "z"])
        for idx, (x, y, z) in enumerate(geometry):
            writer.writerow([idx, x, y, z])
    print(f"[INFO] wrote {path}  ({geometry.shape[0]} points)")


def write_field_csv(group: h5py.Group, n_points: int, path: str) -> None:
    """Write one field as time_step,0,1,...,N-1 (matches the ux.csv layout).

    Rows are time steps (sorted by time), columns are point indices. The
    values are streamed one time step at a time to avoid loading the whole
    field into memory at once.
    """
    time_keys = _sorted_time_keys(group)
    header = ["time_step", *[str(i) for i in range(n_points)]]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for key in time_keys:
            values = group[key][:]
            writer.writerow([_parse_time(key), *values.tolist()])
    print(f"[INFO] wrote {path}  ({len(time_keys)} time steps x {n_points} points)")


def convert(h5_path: str, out_dir: str, fields: list[str]) -> None:
    """Convert every requested field of a line H5 file to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    # Base name mirrors the H5 file (drop only the .h5 extension), so the
    # output matches the historical "<line>.points.csv" / "<line>.ux.csv".
    base = os.path.basename(h5_path)
    if base.lower().endswith(".h5"):
        base = base[:-3]

    with h5py.File(h5_path, "r") as h5:
        if "Geometry" not in h5:
            raise KeyError(f"{h5_path}: no 'Geometry' dataset found")
        geometry = h5["Geometry"][:]
        n_points = geometry.shape[0]

        write_points_csv(geometry, os.path.join(out_dir, f"{base}.points.csv"))

        for field in fields:
            if field not in h5:
                print(f"[WARN] field '{field}' not in {h5_path}; skipping")
                continue
            group = h5[field]
            if not isinstance(group, h5py.Group):
                print(f"[WARN] '{field}' is not a time-step group; skipping")
                continue
            write_field_csv(group, n_points, os.path.join(out_dir, f"{base}.{field}.csv"))


def main() -> None:
    convert(H5_PATH, OUT_DIR, FIELDS)


if __name__ == "__main__":
    main()
