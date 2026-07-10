"""Structural-data ingest for the building dynamic-response recipe.

These are out-of-paradigm *inputs* (like the S1 standard-profile
catalog): they read mode / floor / mode-shape data from CSV and build the
mass-normalized numpy arrays that
:func:`cfdmod.core.recipes.dynamic.build_building_dynamic_response`
consumes. They are not pipeline stages.

CSV layouts (ported from the legacy HFPI readers):

- modes CSV: columns ``mode``, ``period`` (frequency ``= 1/period``,
  angular ``wp = 2*pi*frequency``).
- floors CSV: columns ``Z, M, I, XR, YR`` (radius of gyration
  ``R = sqrt(I/M)`` is derived).
- floor mode-shape CSV (one per mode): columns ``DX, DY, RZ``.
"""

from __future__ import annotations

__all__ = [
    "read_modes_csv",
    "read_floors_csv",
    "read_mode_shape_csv",
    "mass_normalize_mode_shapes",
    "BuildingStructuralData",
]

import pathlib
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from cfdmod import utils


def read_modes_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read the modes CSV. Adds ``frequency = 1/period`` and ``wp = 2*pi*frequency``."""
    df = pd.read_csv(csv_path, index_col=None)
    req_keys = ["mode", "period"]
    if not utils.validate_keys_df(df, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in modes CSV "
            f"{csv_path.as_posix()}. Found only keys {list(df.columns)}"
        )
    df = df[req_keys].copy()
    df["frequency"] = 1 / df["period"]
    df["wp"] = 2 * np.pi * df["frequency"]
    df.sort_values(by="mode", inplace=True)
    return df.reset_index(drop=True)


def read_floors_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read the floors CSV. Derives radius of gyration ``R = sqrt(I/M)``."""
    df = pd.read_csv(csv_path, index_col=None)
    req_keys = ["Z", "M", "I", "XR", "YR"]
    if not utils.validate_keys_df(df, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in floors CSV "
            f"{csv_path.as_posix()}. Found only keys {list(df.columns)}"
        )
    df = df[req_keys].copy()
    df["R"] = (df["I"] / df["M"]) ** 0.5
    df.sort_values(by="Z", inplace=True)
    return df.reset_index(drop=True)


def read_mode_shape_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read a single mode's floor mode-shape CSV (columns ``DX, DY, RZ``)."""
    df = pd.read_csv(csv_path, index_col=None)
    req_keys = ["DX", "DY", "RZ"]
    if not utils.validate_keys_df(df, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in mode-shape CSV "
            f"{csv_path.as_posix()}. Found only keys {list(df.columns)}"
        )
    return df[req_keys].copy()


def mass_normalize_mode_shapes(
    mode_shapes: np.ndarray, floors_mass: np.ndarray, floors_radius: np.ndarray
) -> np.ndarray:
    """Mass-normalize mode shapes so each mode has unit generalized mass.

    For each mode the generalized mass is::

        M_gen = sum_floor M * (DX^2 + DY^2 + (R*RZ)^2)

    and the mode's ``[DX, DY, RZ]`` are divided by ``sqrt(M_gen)``. A
    zero-generalized-mass mode is zeroed (matches the legacy behavior).
    This is the precondition the SDOF solver assumes.

    Args:
        mode_shapes: ``(n_floors, n_modes, 3)`` array of ``[DX, DY, RZ]``.
        floors_mass: ``(n_floors,)`` floor masses.
        floors_radius: ``(n_floors,)`` floor radii of gyration.

    Returns:
        Normalized ``(n_floors, n_modes, 3)`` array (new array; input untouched).
    """
    phi = np.asarray(mode_shapes, dtype=np.float64)
    m = np.asarray(floors_mass, dtype=np.float64)[:, None]  # (n_floors, 1)
    r = np.asarray(floors_radius, dtype=np.float64)[:, None]
    dx, dy, rz = phi[:, :, 0], phi[:, :, 1], phi[:, :, 2]

    # Generalized mass per mode -> (n_modes,)
    m_gen = (m * (dx**2 + dy**2 + (r * rz) ** 2)).sum(axis=0)
    out = phi.copy()
    nonzero = m_gen > 0
    out[:, nonzero, :] /= np.sqrt(m_gen[nonzero])[None, :, None]
    out[:, ~nonzero, :] = 0.0
    return out


class BuildingStructuralData(BaseModel):
    """Assembled structural inputs for the building dynamic-response recipe.

    Holds numpy arrays ready to pass to
    :class:`~cfdmod.core.recipes.dynamic.BuildingDynamicConfig`. Mode
    shapes are stored mass-normalized.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode_shapes: Any  # (n_floors, n_modes, 3), mass-normalized
    natural_frequencies: Any  # (n_modes,) angular wp = 2*pi*f
    floor_points: Any  # (n_floors, 3)
    cm_positions: Any  # (n_floors, 2) [XR, YR]
    floors_mass: Any  # (n_floors,)
    floors_radius: Any  # (n_floors,)

    @property
    def n_modes(self) -> int:
        return int(np.asarray(self.mode_shapes).shape[1])

    @property
    def n_floors(self) -> int:
        return int(np.asarray(self.mode_shapes).shape[0])

    @classmethod
    def from_csvs(
        cls,
        modes_csv: pathlib.Path,
        floors_csv: pathlib.Path,
        mode_shape_csvs: list[pathlib.Path],
        *,
        active_modes: list[int] | None = None,
    ) -> "BuildingStructuralData":
        """Build from the modes / floors / per-mode mode-shape CSVs.

        Args:
            active_modes: 1-based mode numbers to keep. ``None`` keeps
                every mode that has a mode-shape CSV.
        """
        df_modes = read_modes_csv(modes_csv)
        df_floors = read_floors_csv(floors_csv)
        shapes = [read_mode_shape_csv(p) for p in mode_shape_csvs]

        n_floors = len(df_floors)
        for i, s in enumerate(shapes):
            if len(s) != n_floors:
                raise ValueError(
                    f"mode-shape CSV index {i} has {len(s)} floors; floors CSV has {n_floors}"
                )
        if len(df_modes) < len(shapes):
            raise ValueError("fewer modes than mode-shape CSVs provided")

        keep = (
            list(range(len(shapes)))
            if active_modes is None
            else [m - 1 for m in active_modes]  # 1-based -> 0-based
        )
        phi = np.stack(
            [np.column_stack([shapes[i]["DX"], shapes[i]["DY"], shapes[i]["RZ"]]) for i in keep],
            axis=1,
        )  # (n_floors, n_modes, 3)

        floors_mass = df_floors["M"].to_numpy(dtype=np.float64)
        floors_radius = df_floors["R"].to_numpy(dtype=np.float64)
        phi = mass_normalize_mode_shapes(phi, floors_mass, floors_radius)

        heights = df_floors["Z"].to_numpy(dtype=np.float64)
        floor_points = np.column_stack([np.zeros(n_floors), np.zeros(n_floors), heights])

        return cls(
            mode_shapes=phi,
            natural_frequencies=df_modes["wp"].to_numpy(dtype=np.float64)[keep],
            floor_points=floor_points,
            cm_positions=df_floors[["XR", "YR"]].to_numpy(dtype=np.float64),
            floors_mass=floors_mass,
            floors_radius=floors_radius,
        )
