"""Facade Cp snapshots via the config-driven cfdmod.snapshot (ParaView/VTK) path.

This is the proper replacement for the removed 3-D matplotlib facade render: it
drives ``cfdmod.snapshot.take_snapshot`` from a ``snapshot_params.yaml`` that
fixes the per-face unfold layout, colormap, camera and compass overlays, and
repoints every projection at the case's Cp stats polydata per statistic. Mirrors
the consulting snapshot setup (068 CST) on the current ``cfdmod.snapshot`` API.

Run headless on the in-repo galpao fixture (writes PNGs to ``_run/``):

    uv run python examples/facade_snapshot/render_facade_snapshots.py

Point at a real case (e.g. Secco 070) with environment variables -- see the
README. Rendering is off-screen; on a box with no X server wrap the call in
``xvfb-run`` (offscreen VTK segfaults without a virtual display on some hosts).
"""

from __future__ import annotations

import os
import pathlib

import numpy as np

from cfdmod import mesh_field
from cfdmod.snapshot import SnapshotConfig
from cfdmod.snapshot.snapshot import take_snapshot

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Statistic -> (legend label, active scalar in the .vtp). The scalar names match
# what mesh_field.write_field_vtp writes below; a real case's stats .vtp uses the
# nested Cp names (e.g. "cp/base_cp/mean") -- override via CFDMOD_FS_SCALARS.
STATS = {
    "mean": ("Mean pressure coefficient", "cp_mean"),
    "min": ("Minimum pressure coefficient", "cp_min"),
    "max": ("Maximum pressure coefficient", "cp_max"),
}


def _galpao_cp_vtp(out_dir: pathlib.Path) -> tuple[pathlib.Path, dict[str, np.ndarray]]:
    """Compute per-triangle Cp stats on the galpao fixture and write them to a .vtp."""
    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage
    from cfdmod.building import cp_from_pressure, example_building_case

    fix = REPO / "fixtures" / "tests" / "pressure"
    mesh = str(fix / "galpao" / "galpao.normalized.lnas")
    storage = XdmfH5Storage(fix / "data")
    body = storage.read_data_source(pathlib.Path("bodies.galpao"))
    p_ref = storage.read_data_source(pathlib.Path("points.static_pressure"))
    case = example_building_case(mesh, n_floors=3)
    cp = cp_from_pressure(body, p_ref, case)
    series = np.asarray(cp.fields.read("cp"))
    fields = {
        "cp_mean": np.nanmean(series, axis=1),
        "cp_min": np.nanmin(series, axis=1),
        "cp_max": np.nanmax(series, axis=1),
    }
    geom = mesh_field.load_geometry(mesh)
    vtp = out_dir / "galpao_cp.vtp"
    if not mesh_field.write_field_vtp(geom, fields, vtp):
        raise SystemExit("the [vtk] extra (pyvista) is required to write the Cp polydata")
    return vtp, fields


def main() -> None:
    config_path = pathlib.Path(os.environ.get("CFDMOD_FS_CONFIG", HERE / "snapshot_params.yaml"))
    out_dir = pathlib.Path(os.environ.get("CFDMOD_FS_OUTPUT", HERE / "_run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    base = SnapshotConfig.from_file(config_path)

    vtp_env = os.environ.get("CFDMOD_FS_VTP")
    if vtp_env:
        # Real case: point at an existing Cp stats .vtp; scalars via CFDMOD_FS_SCALARS
        # ("mean=cp/base_cp/mean,max=cp/base_cp/max"), ranges auto from the config.
        vtp = pathlib.Path(vtp_env)
        pairs = os.environ.get("CFDMOD_FS_SCALARS", "mean=cp/base_cp/mean")
        stats = {}
        for pair in pairs.split(","):
            key, _, scalar = pair.partition("=")
            stats[key.strip()] = (f"{key.strip()} pressure coefficient", scalar.strip())
        ranges = None
    else:
        vtp, fields = _galpao_cp_vtp(out_dir)
        stats = STATS
        ranges = {
            k: (float(np.nanmin(fields[s])), float(np.nanmax(fields[s])))
            for k, (_, s) in stats.items()
        }

    written = []
    for key, (label, scalar) in stats.items():
        value_range = ranges[key] if ranges else base.legend_config.range
        cfg = base.retarget(vtp, scalar, label=label, value_range=value_range)
        image_path = out_dir / f"facade_cp_{key}.png"
        take_snapshot(image_path, cfg, off_screen=True)
        written.append(image_path)
        print(f"  [ok] {key}: {image_path} ({image_path.stat().st_size} bytes)")

    print(f"wrote {len(written)} facade snapshot(s) to {out_dir}")


if __name__ == "__main__":
    main()
