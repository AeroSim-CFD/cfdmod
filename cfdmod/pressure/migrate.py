"""Migration utilities: convert old pandas HDFStore H5 to new XDMF+H5 format."""

from __future__ import annotations

__all__ = ["migrate_body_h5", "migrate_probe_h5"]

import pathlib
from typing import Literal

import h5py
import numpy as np
import pandas as pd
from lnas import LnasFormat

from cfdmod.io.xdmf import (
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
    write_temporal_xdmf,
)


def migrate_body_h5(
    old_h5: pathlib.Path,
    mesh_path: pathlib.Path,
    output_h5: pathlib.Path,
    macroscopic_type: Literal["rho", "pressure"] = "pressure",
) -> None:
    """Convert old pandas HDFStore body H5 to new XDMF+H5 format.

    Old format: pandas HDFStore with /step{:07} keys, each key is a DataFrame
    with columns ["time_step", "0", "1", ..., str(n_tri-1)] + other metadata.

    New format: h5py with pressure/t{T} datasets of shape (n_tri,), plus
    /Triangles, /Geometry, and /meta datasets.

    Applies cs^2=1/3 scaling if macroscopic_type='rho'.

    Args:
        old_h5 (pathlib.Path): Old pandas HDFStore H5 file
        mesh_path (pathlib.Path): LNAS mesh file to provide geometry
        output_h5 (pathlib.Path): Output H5 file path
        macroscopic_type: "rho" or "pressure"
    """
    if output_h5.exists():
        output_h5.unlink()

    mesh = LnasFormat.load(mesh_path)
    write_timeseries_geometry(
        output_h5, mesh.geometry.triangles, mesh.geometry.vertices
    )

    multiplier = 1.0 / 3.0 if macroscopic_type == "rho" else 1.0

    time_steps_arr: list[float] = []

    with pd.HDFStore(old_h5, mode="r") as store:
        keys = sorted(store.keys())
        for store_key in keys:
            df: pd.DataFrame = store.get(store_key)
            numeric_cols = [col for col in df.columns if col.isnumeric()]

            for _, row in df.iterrows():
                t_val = float(row["time_step"])
                t_key = f"t{t_val}"
                pressure_data = row[numeric_cols].to_numpy().astype(np.float64) * multiplier
                write_timeseries_step(output_h5, "pressure", t_key, pressure_data)
                time_steps_arr.append(t_val)

    time_steps = np.array(sorted(set(time_steps_arr)))
    time_normalized = time_steps
    write_timeseries_meta(output_h5, time_steps, time_normalized)

    xdmf_path = output_h5.with_suffix(".xdmf")
    write_temporal_xdmf(output_h5, xdmf_path, "pressure")


def migrate_probe_h5(
    old_h5: pathlib.Path,
    output_h5: pathlib.Path,
    macroscopic_type: Literal["rho", "pressure"] = "pressure",
) -> None:
    """Convert old pandas HDFStore probe H5 to new XDMF+H5 format.

    Probe H5 stores one pressure value per timestep (shape (1,) per timestep).

    Args:
        old_h5 (pathlib.Path): Old pandas HDFStore H5 file
        output_h5 (pathlib.Path): Output H5 file path
        macroscopic_type: "rho" or "pressure"
    """
    if output_h5.exists():
        output_h5.unlink()

    trivial_triangles = np.array([[0, 0, 0]], dtype=np.int32)
    trivial_vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    write_timeseries_geometry(output_h5, trivial_triangles, trivial_vertices)

    multiplier = 1.0 / 3.0 if macroscopic_type == "rho" else 1.0
    time_steps_arr: list[float] = []

    with pd.HDFStore(old_h5, mode="r") as store:
        for store_key in sorted(store.keys()):
            df: pd.DataFrame = store.get(store_key)
            numeric_cols = [col for col in df.columns if col.isnumeric()]

            for _, row in df.iterrows():
                t_val = float(row["time_step"])
                t_key = f"t{t_val}"
                pressure_val = np.array([row[numeric_cols[0]] * multiplier], dtype=np.float64)
                write_timeseries_step(output_h5, "pressure", t_key, pressure_val)
                time_steps_arr.append(t_val)

    time_steps = np.array(sorted(set(time_steps_arr)))
    write_timeseries_meta(output_h5, time_steps, time_steps)

    xdmf_path = output_h5.with_suffix(".xdmf")
    write_temporal_xdmf(output_h5, xdmf_path, "pressure")
