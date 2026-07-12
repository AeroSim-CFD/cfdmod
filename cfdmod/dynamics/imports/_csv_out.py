"""Write a BuildingStructuralData back to the internal modal CSVs.

The inverse of :meth:`BuildingStructuralData.from_csvs`: emit
``modes.csv`` (``mode,period``), ``floors.csv`` (``Z,M,I,XR,YR``) and one
``phi{m}.csv`` (``DX,DY,RZ``) per mode. Re-reading them with ``from_csvs``
reproduces the same model -- the stored shapes are already
mass-normalized (unit generalized mass), so ``from_csvs``'s normalization
is idempotent.
"""

from __future__ import annotations

__all__ = ["write_structural_csvs"]

import pathlib

import numpy as np
import pandas as pd

from cfdmod.dynamics.structural import BuildingStructuralData


def write_structural_csvs(
    sd: BuildingStructuralData, out_dir: str | pathlib.Path
) -> list[pathlib.Path]:
    """Write ``sd`` as ``modes.csv`` / ``floors.csv`` / ``phi{m}.csv``.

    Returns the list of written paths.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.asarray(sd.floor_points, dtype=np.float64)[:, 2]
    mass = np.asarray(sd.floors_mass, dtype=np.float64)
    radius = np.asarray(sd.floors_radius, dtype=np.float64)
    cm = np.asarray(sd.cm_positions, dtype=np.float64)
    phi = np.asarray(sd.mode_shapes, dtype=np.float64)  # (n_floors, n_modes, 3)
    wp = np.asarray(sd.natural_frequencies, dtype=np.float64)

    written: list[pathlib.Path] = []

    modes_path = out_dir / "modes.csv"
    pd.DataFrame({"mode": np.arange(1, len(wp) + 1), "period": 2.0 * np.pi / wp}).to_csv(
        modes_path, index=False
    )
    written.append(modes_path)

    floors_path = out_dir / "floors.csv"
    pd.DataFrame(
        {"Z": z, "M": mass, "I": mass * radius**2, "XR": cm[:, 0], "YR": cm[:, 1]}
    ).to_csv(floors_path, index=False)
    written.append(floors_path)

    for m in range(phi.shape[1]):
        p = out_dir / f"phi{m + 1}.csv"
        pd.DataFrame({"DX": phi[:, m, 0], "DY": phi[:, m, 1], "RZ": phi[:, m, 2]}).to_csv(
            p, index=False
        )
        written.append(p)

    return written
