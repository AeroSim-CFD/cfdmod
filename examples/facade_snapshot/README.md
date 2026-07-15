# Facade Cp snapshots (config-driven, ParaView/VTK)

Renders building-facade pressure-coefficient snapshots with the
`cfdmod.snapshot` tooling, driven by a `snapshot_params.yaml`. This is the
proper facade deliverable: each building face is unfolded into the image plane
(roof in the centre, the four sides around it) with a shared colormap, legend
and compass overlays -- the same setup the consulting snapshot pipeline uses.

It replaces the earlier per-triangle 3-D matplotlib render, which was illegible
for tall/slender buildings (the equal-aspect 3-D box collapsed a tower to a
sliver; a near-planar facade face-on collapsed to a line).

## How it works

1. `snapshot_params.yaml` fixes the per-case **layout**: which projections exist
   (`top` / `front` / `back` / `left` / `right`) and how each face is
   transformed (translate / rotate / scale) to unfold the box, plus the
   colormap divisions, camera, and the N/S/E/W text overlays. The Cp data file
   and scalar are left blank here.
2. The driver loads it with `SnapshotConfig.from_file(...)`, then for each
   statistic (mean / min / max) repoints every projection at the case's Cp
   stats polydata and sets the legend with `SnapshotConfig.retarget(vtp,
   scalar, label=..., value_range=..., n_divs=...)`, and calls
   `take_snapshot(...)`.

`retarget` returns a fresh copy each call, so the base config is reused across
directions and statistics.

## Run (in-repo fixture, headless)

```bash
uv run --extra snapshot python examples/facade_snapshot/render_facade_snapshots.py
```

Writes `facade_cp_{mean,min,max}.png` to `examples/facade_snapshot/_run/`,
computing Cp on the galpao fixture. Rendering is off-screen; on a host with no X
server, wrap it in `xvfb-run` (offscreen VTK segfaults without a virtual display
on some boxes):

```bash
xvfb-run -a uv run --extra snapshot python examples/facade_snapshot/render_facade_snapshots.py
```

## Point at a real case (e.g. Secco 070)

Copy `snapshot_params.yaml` next to the case, retune the per-face `translate`
offsets to roughly half the building extent along each axis (as the consulting
snapshot configs do per building), and set:

| Variable | Meaning |
|---|---|
| `CFDMOD_FS_CONFIG` | path to the case's `snapshot_params.yaml` |
| `CFDMOD_FS_VTP` | path to the case's Cp **stats** polydata (`.vtp`) |
| `CFDMOD_FS_SCALARS` | `stat=scalar` pairs, e.g. `mean=cp/base_cp/mean,max=cp/base_cp/max` |
| `CFDMOD_FS_OUTPUT` | output directory for the PNGs |

The Cp stats `.vtp` is the per-triangle mean/min/max Cp on the body mesh
(`cfdmod.mesh_field.write_field_vtp` writes one from a Cp `DataSource`; consulting
cases already produce a `stats_transformed.vtp`). Legend ranges default to the
config's `legend_config.range` for real cases -- set per statistic there.

Loop the driver over the case's wind directions (one `.vtp` per direction) to
produce the per-direction facade set.
