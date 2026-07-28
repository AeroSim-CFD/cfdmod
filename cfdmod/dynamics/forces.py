"""Force-coefficient ingest for the building dynamic-response recipe.

Out-of-paradigm inputs (like :mod:`cfdmod.dynamics.structural`): read the
per-floor force / moment coefficient timeseries from H5, apply the
dimensional scaling, and build the floor-load
:class:`~cfdmod.core.data_source.PointsDataSource` that
:func:`cfdmod.core.recipes.dynamic.build_building_dynamic_response`
consumes.

H5 layout (matches the legacy HFPI force file): an h5py file with a 2-D
``/forces`` dataset ``(n_time, n_cols)`` and a ``/columns`` dataset of
UTF-8 column names including ``time_normalized``. A legacy single-key
pandas ``HDFStore`` is still read (with a deprecation warning).
"""

from __future__ import annotations

__all__ = [
    "DimensionalData",
    "read_force_h5",
    "write_force_h5",
    "build_floor_load_source",
]

import pathlib
import warnings

import h5py
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology

# Air dynamic-pressure coefficient q = 0.613 * U^2 (SI, sea level).
_DYN_PRESSURE_COEFF = 0.613


class DimensionalData(BaseModel):
    """Dimensional scaling for a building HFPI model.

    **The time axis must be re-dimensionalised with the same length that
    non-dimensionalised it.** The Cp stage writes ``time_normalized =
    t_solver / (Lc_simul / U_simul)``; turning that back into seconds needs
    ``Lc_simul``, which is an independent choice from the ``base`` used by the
    force normalisation (``F = Cf * base * height * q``). Real case configs do
    set them differently -- e.g. ``simul_characteristic_length = H`` with
    ``base = B`` -- and de-normalising with ``base`` then stretches or
    compresses the whole record by ``base / Lc_simul``, sliding the forcing
    spectrum off the structure's natural frequencies and silently changing the
    resonant amplification. Pass ``simul_characteristic_length`` so the round
    trip closes; the legacy fallback to ``base`` warns.

    With the length round trip closed the physical time step reduces to
    ``dt_phys = dt_solver * (U_simul / U_H) * ISM`` -- see
    :attr:`time_scale_factor` -- which carries no characteristic length at all.

    Attributes:
        U_H: Design wind speed at building height (m/s).
        height: Building height (m).
        base: Building base dimension (m), for the force / moment normalisation.
        integral_scale_multiplier: Turbulence integral-scale multiplier applied
            to the time normalisation (``1.0`` = no stretch).
        simul_characteristic_length: The length the Cp stage used to
            non-dimensionalise time (its ``simul_characteristic_length``), in m.
            Defaults to ``base`` with a warning -- correct only when the two
            genuinely coincide.
        simul_U_H: The speed the Cp stage used to non-dimensionalise time (its
            ``simul_U_H``), in m/s. Only needed for :attr:`time_scale_factor`.
    """

    model_config = ConfigDict(frozen=True)

    U_H: float
    height: float
    base: float
    integral_scale_multiplier: float
    simul_characteristic_length: float | None = None
    simul_U_H: float | None = None

    @property
    def dynamic_pressure(self) -> float:
        return _DYN_PRESSURE_COEFF * (self.U_H**2)

    @property
    def CST(self) -> float:
        """Characteristic structural time (base / U_H)."""
        return self.base / self.U_H

    @property
    def time_normalization_length(self) -> float:
        """Length that re-dimensionalises the normalised time axis (m).

        ``simul_characteristic_length`` when given; otherwise ``base``, with a
        warning -- that fallback is the legacy behaviour and is only correct
        when the Cp stage normalised time by ``base``.
        """
        if self.simul_characteristic_length is None:
            warnings.warn(
                "DimensionalData has no simul_characteristic_length; falling back to "
                "base for the time de-normalisation. That is only correct if the Cp "
                "stage normalised time by the same length. Pass the Cp config's "
                "simul_characteristic_length explicitly.",
                UserWarning,
                stacklevel=3,
            )
            return self.base
        return self.simul_characteristic_length

    @property
    def time_normalization_factor(self) -> float:
        """``time_normalized`` -> seconds."""
        return self.time_normalization_length / self.U_H * self.integral_scale_multiplier

    @property
    def time_scale_factor(self) -> float:
        """Solver seconds -> full-scale seconds (``U_simul / U_H * ISM``).

        The characteristic length cancels out of the round trip, so this is the
        scaling a geometrically full-scale run actually needs. Requires
        ``simul_U_H``.
        """
        if self.simul_U_H is None:
            raise ValueError(
                "time_scale_factor needs simul_U_H (the speed the Cp stage normalised by)"
            )
        return self.simul_U_H / self.U_H * self.integral_scale_multiplier

    @property
    def force_normalization_factor(self) -> float:
        return self.base * self.height * self.dynamic_pressure

    @property
    def moments_normalization_factor(self) -> float:
        return self.base * self.base * self.height * self.dynamic_pressure


def write_force_h5(df: pd.DataFrame, hdf_path: pathlib.Path) -> None:
    """Write a force DataFrame to the h5py-based force layout."""
    with h5py.File(hdf_path, "w") as f:
        f.create_dataset("forces", data=df.to_numpy(dtype=np.float64))
        f.create_dataset("columns", data=np.array([str(c).encode() for c in df.columns]))


def read_force_h5(hdf_path: pathlib.Path) -> pd.DataFrame:
    """Read a force H5 into a DataFrame (new h5py layout or legacy HDFStore)."""
    try:
        with h5py.File(hdf_path, "r") as f:
            if "forces" in f and "columns" in f:
                forces = f["forces"][:]
                columns = [c.decode() if isinstance(c, bytes) else c for c in f["columns"][:]]
                return pd.DataFrame(forces, columns=columns)
    except OSError:
        # A pandas HDFStore is not a valid h5py file in some configurations.
        pass

    warnings.warn(
        f"Reading legacy pandas-HDFStore force file {hdf_path}. Convert to the new "
        "layout (h5py with a 2-D /forces dataset and /columns names) via write_force_h5.",
        DeprecationWarning,
        stacklevel=2,
    )
    return pd.read_hdf(hdf_path)


def _floor_matrix(df: pd.DataFrame, n_floors: int, force_factor: float) -> np.ndarray:
    """Extract a scaled ``(n_floors, n_t)`` matrix from a force DataFrame.

    Numeric floor columns are ordered ascending; missing floors are filled
    with zeros. Values are multiplied by ``force_factor``.
    """
    label_by_floor: dict[int, str | int] = {}
    for col in df.columns:
        if isinstance(col, str):
            if col.isnumeric():
                label_by_floor[int(col)] = col
        elif isinstance(col, (int, np.integer)):
            label_by_floor[int(col)] = col

    n_t = len(df)
    out = np.zeros((n_floors, n_t), dtype=np.float64)
    for floor in range(n_floors):
        if floor in label_by_floor:
            out[floor, :] = df[label_by_floor[floor]].to_numpy(dtype=np.float64) * force_factor
    return out


def build_floor_load_source(
    cf_x_h5: pathlib.Path,
    cf_y_h5: pathlib.Path,
    cm_z_h5: pathlib.Path,
    dim_data: DimensionalData,
    n_floors: int,
) -> PointsDataSource:
    """Read + scale the three force H5 files into a floor-load data source.

    The output carries fields ``cf_x`` / ``cf_y`` / ``cm_z`` as
    ``(n_floors, n_t)`` arrays and a :class:`TimeAxis` whose timestep is the
    scaled physical ``dt``. Forces are scaled by
    ``force_normalization_factor`` (``cf_x`` / ``cf_y``) and
    ``moments_normalization_factor`` (``cm_z``); time is scaled by
    ``time_normalization_factor``.
    """
    df_x = read_force_h5(cf_x_h5)
    df_y = read_force_h5(cf_y_h5)
    df_z = read_force_h5(cm_z_h5)
    for name, df in (("cf_x", df_x), ("cf_y", df_y), ("cm_z", df_z)):
        if "time_normalized" not in df.columns:
            raise KeyError(f"force file for {name} is missing the 'time_normalized' column")
    if not (len(df_x) == len(df_y) == len(df_z)):
        raise ValueError("force files have mismatched lengths")

    # Physical time from the normalized time column (start at 0).
    t_norm = df_x["time_normalized"].to_numpy(dtype=np.float64)
    order = np.argsort(t_norm)
    t = (t_norm[order] - t_norm[order].min()) * dim_data.time_normalization_factor
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0

    def scaled(df: pd.DataFrame, factor: float) -> np.ndarray:
        return _floor_matrix(df.iloc[order].reset_index(drop=True), n_floors, factor)

    cf_x = scaled(df_x, dim_data.force_normalization_factor)
    cf_y = scaled(df_y, dim_data.force_normalization_factor)
    cm_z = scaled(df_z, dim_data.moments_normalization_factor)

    pts = np.zeros((n_floors, 3), dtype=np.float64)
    fields = {"cf_x": cf_x, "cf_y": cf_y, "cm_z": cm_z}
    return PointsDataSource(
        time=TimeAxis(initial_time=float(t[0]), timestep_size=dt, n_timesteps=len(t)),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(fields),
        field_meta={k: FieldMeta(name=k) for k in fields},
    )
