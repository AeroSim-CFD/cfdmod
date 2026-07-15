"""Building facade Cp snapshots via config-driven cfdmod.snapshot (ParaView/VTK).

The proper replacement for the removed 3-D matplotlib facade render: the four
tower walls (N/E/S/W) are unfolded side by side as upright vertical strips with
the roof on top, via `cfdmod.snapshot.building_facade_config`, and rendered with
`take_snapshot`. Produces, per statistic, a full-height image AND one image per
height band (walls clipped per band, so a tall tower reads at each level), with
a color scale shared per statistic. Mirrors the consulting snapshot setup
(068 CST) on the current cfdmod.snapshot API.

Run headless on the in-repo galpao fixture (writes PNGs to `_run/`):

    uv run --extra snapshot python examples/facade_snapshot/render_facade_snapshots.py

Rendering is off-screen; on a host with no X server wrap it in `xvfb-run`
(offscreen VTK falls back to software rendering, or needs a virtual display on
some boxes). Point at a real case (e.g. Secco 070) with the env vars below.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np

from cfdmod import mesh_field
from cfdmod.snapshot import building_facade_config
from cfdmod.snapshot.snapshot import take_snapshot

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

STATS = {"mean": "Mean pressure coefficient", "max": "Peak (max) pressure coefficient"}


def _galpao_cp_vtp(out_dir: pathlib.Path):
    """Per-triangle Cp stats on the galpao fixture -> .vtp; returns (vtp, bbox, ranges)."""
    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage
    from cfdmod.building import cp_from_pressure, example_building_case

    fix = REPO / "fixtures" / "tests" / "pressure"
    mesh = str(fix / "galpao" / "galpao.normalized.lnas")
    storage = XdmfH5Storage(fix / "data")
    body = storage.read_data_source(pathlib.Path("bodies.galpao"))
    p_ref = storage.read_data_source(pathlib.Path("points.static_pressure"))
    case = example_building_case(mesh, n_floors=3)
    cp = cp_from_pressure(body, p_ref, case, statistics=list(STATS))
    geom = mesh_field.load_geometry(mesh)
    fields, ranges = {}, {}
    for k in STATS:
        a = np.asarray(cp.fields.read(k), dtype=np.float64)
        fields[f"cp_{k}"] = a
        ranges[k] = (
            float(np.floor(np.nanmin(a) * 10) / 10),
            float(np.ceil(np.nanmax(a) * 10) / 10),
        )
    vtp = out_dir / "galpao_cp.vtp"
    if not mesh_field.write_field_vtp(geom, fields, vtp):
        raise SystemExit("the [snapshot] extra (pyvista) is required to write the Cp polydata")
    v = np.asarray(geom.vertices, dtype=np.float64)
    return vtp, (v.min(0), v.max(0)), ranges


def _bands(bbox_lo, bbox_hi, n=3):
    """Simple equal-height bands over the mesh z-range (demo). Real towers pass floor edges."""
    z0, z1 = float(bbox_lo[2]), float(bbox_hi[2])
    edges = np.linspace(z0, z1, n + 1)
    return [
        (f"band{i}_{edges[i]:.0f}-{edges[i + 1]:.0f}", edges[i], edges[i + 1]) for i in range(n)
    ]


def main() -> None:
    out_dir = pathlib.Path(os.environ.get("CFDMOD_FS_OUTPUT", HERE / "_run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    vtp, (lo, hi), ranges = _galpao_cp_vtp(out_dir)
    bands = _bands(lo, hi, n=int(os.environ.get("CFDMOD_FS_BANDS", "3")))

    written = 0
    for k, label in STATS.items():
        vr = ranges[k]  # shared per statistic (one case here; loop directions for a real study)
        # full height (walls + roof)
        cfg = building_facade_config(lo, hi, legend_label=label, value_range=vr)
        take_snapshot(
            out_dir / f"facade_cp_{k}.png", cfg.retarget(vtp, f"cp_{k}"), off_screen=True
        )
        written += 1
        # per-band (walls clipped to the band, shared scale, no roof)
        for blabel, z_lo, z_hi in bands:
            cfgb = building_facade_config(
                lo, hi, legend_label=f"{label} | {blabel}", value_range=vr, z_band=(z_lo, z_hi)
            )
            take_snapshot(
                out_dir / f"facade_cp_{k}_{blabel}.png",
                cfgb.retarget(vtp, f"cp_{k}"),
                off_screen=True,
            )
            written += 1
    print(f"wrote {written} facade snapshot(s) to {out_dir}")


if __name__ == "__main__":
    main()
