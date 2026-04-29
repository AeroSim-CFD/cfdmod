"""One-off: convert legacy pressure test fixtures to the new XDMF+H5 layout.

Converts:
  new.bodies.galpao.data.resampled.h5    -> bodies.galpao.h5(+.xdmf)
  new.points.static_pressure.data.resampled.h5 -> points.static_pressure.h5(+.xdmf)
  cp_t.normalized.h5 (long-form chunks)  -> cp_t.normalized.h5 (cp/t{T} layout)

After conversion, the legacy result file `new.cp_t.normalized.h5` can be deleted.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pandas as pd
from lnas import LnasFormat

from cfdmod.io.xdmf import (
    write_temporal_xdmf,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
)
from cfdmod.pressure.migrate import migrate_body_h5, migrate_probe_h5

DATA = pathlib.Path("fixtures/tests/pressure/data")
MESH = pathlib.Path("fixtures/tests/pressure/galpao/galpao.normalized.lnas")


def convert_long_form_cp(old: pathlib.Path, new: pathlib.Path, mesh_path: pathlib.Path) -> None:
    """Convert long-format cp result (range_*; columns: time_normalized, point_idx, cp)
    to the new XDMF+H5 layout with /cp/t{T} datasets."""
    if new.exists():
        new.unlink()

    mesh = LnasFormat.from_file(mesh_path)
    write_timeseries_geometry(new, mesh.geometry.triangles, mesh.geometry.vertices)

    frames: list[pd.DataFrame] = []
    with pd.HDFStore(old, mode="r") as store:
        for k in sorted(store.keys()):
            frames.append(store.get(k))
    long = pd.concat(frames, ignore_index=True)

    long.sort_values(["time_normalized", "point_idx"], inplace=True)
    wide = long.pivot(index="time_normalized", columns="point_idx", values="cp")
    wide.sort_index(axis=1, inplace=True)

    time_steps: list[float] = []
    for t_val, row in wide.iterrows():
        t_key = f"t{float(t_val)}"
        write_timeseries_step(new, "cp", t_key, row.to_numpy(dtype=np.float64))
        time_steps.append(float(t_val))

    write_timeseries_meta(
        new, np.array(time_steps, dtype=np.float64), np.array(time_steps, dtype=np.float64)
    )
    write_temporal_xdmf(new, new.with_suffix(".xdmf"), "cp")


def main() -> None:
    body_old = DATA / "new.bodies.galpao.data.resampled.h5"
    probe_old = DATA / "new.points.static_pressure.data.resampled.h5"
    cp_old = DATA / "cp_t.normalized.h5"

    body_new = DATA / "bodies.galpao.h5"
    probe_new = DATA / "points.static_pressure.h5"
    cp_tmp = DATA / "_cp_t.normalized.new.h5"

    print(f"-> migrating body  {body_old.name} -> {body_new.name}")
    migrate_body_h5(body_old, MESH, body_new)

    print(f"-> migrating probe {probe_old.name} -> {probe_new.name}")
    migrate_probe_h5(probe_old, probe_new)

    print(f"-> converting cp   {cp_old.name} -> (new layout)")
    convert_long_form_cp(cp_old, cp_tmp, MESH)
    cp_old.unlink()
    cp_old.with_suffix(".xdmf").unlink(missing_ok=True)
    cp_tmp.rename(cp_old)
    cp_tmp.with_suffix(".xdmf").rename(cp_old.with_suffix(".xdmf"))

    for stale in [
        body_old,
        body_old.with_suffix(".xdmf"),
        probe_old,
        probe_old.with_suffix(".xdmf"),
        DATA / "new.cp_t.normalized.h5",
    ]:
        if stale.exists():
            print(f"-> removing legacy {stale.name}")
            stale.unlink()

    print("done.")
    for p in sorted(DATA.iterdir()):
        with h5py.File(p, "r") if p.suffix == ".h5" else open(p, "rb") as f:
            if isinstance(f, h5py.File):
                top = list(f.keys())
                print(f"   {p.name}: {top[:5]}")
            else:
                print(f"   {p.name}: <xdmf>")


if __name__ == "__main__":
    main()
