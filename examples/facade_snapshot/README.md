# Building facade Cp snapshots (config-driven, ParaView/VTK)

Renders building-facade pressure-coefficient snapshots with the
`cfdmod.snapshot` tooling. The four tower walls (N / E / S / W) are unfolded
**side by side** as upright vertical strips with the **roof on top** -- the
layout wind engineers use for facade deliverables. It replaces the earlier
per-triangle 3-D matplotlib render, which was illegible for tall/slender
buildings.

## How it works

1. `cfdmod.snapshot.building_facade_config(bbox_lo, bbox_hi, ...)` builds the
   layout from the building's bounding box: the four walls unfolded side by side
   (upright), roof on top, compass labels, colormap, camera. Pass `z_band` to
   clip every wall to a height band (per-floor bands), and a shared
   `value_range` per statistic so colors compare across directions/bands.
2. `SnapshotConfig.retarget(vtp, scalar, ...)` points every projection at the
   case's Cp polydata + active scalar.
3. `take_snapshot(...)` renders it.

Per statistic (mean / peak) the driver writes a **full-height** image and one
image **per height band** (a tall tower is unreadable as one strip; bands make
each level legible). Rendered mean + peak only -- no suction/min -- per the
facade deliverable convention. The color scale is shared per statistic; for a
multi-direction study, compute the global min/max across directions first and
pass the same `value_range` to every direction and band.

## Run (in-repo fixture, headless)

```bash
uv run --extra snapshot python examples/facade_snapshot/render_facade_snapshots.py
```

Writes `facade_cp_{mean,max}.png` (full) and `facade_cp_{mean,max}_band*.png` to
`examples/facade_snapshot/_run/`, computing Cp on the galpao fixture. Rendering
is off-screen; on a host with no X server wrap it in `xvfb-run` (offscreen VTK
falls back to software rendering, or needs a virtual display on some boxes):

```bash
xvfb-run -a uv run --extra snapshot python examples/facade_snapshot/render_facade_snapshots.py
```

## Point at a real case (e.g. Secco 070)

Compute the case's per-triangle Cp **stats** on the body mesh, write them to a
`.vtp` (`cfdmod.mesh_field.write_field_vtp` from a Cp `DataSource`; consulting
cases also ship a `stats_transformed.vtp`), then loop wind directions:

```python
from cfdmod.snapshot import building_facade_config
from cfdmod.snapshot.snapshot import take_snapshot

# 1) global range per statistic across ALL directions (shared color scale)
# 2) per direction/statistic:
cfg = building_facade_config(bbox_lo, bbox_hi, legend_label="Mean Cp",
                             value_range=shared_range)          # full height
take_snapshot(out / "facade_mean.png", cfg.retarget(vtp, "cp/base_cp/mean"))
for z_lo, z_hi in floor_bands:                                  # per 10 floors
    cfgb = building_facade_config(bbox_lo, bbox_hi, value_range=shared_range,
                                  z_band=(z_lo, z_hi))
    take_snapshot(out / f"band_{z_lo:.0f}.png", cfgb.retarget(vtp, "cp/base_cp/mean"))
```

Per-face `translate` offsets and the unfold are derived from the bbox; the wall
rotations bring each outward face upright. Tune `gap` / `zoom` / `window_size`
per building if needed. Full-scale render needs the `[snapshot]` extra and, on a
headless GPU box, an xvfb-capable environment.
