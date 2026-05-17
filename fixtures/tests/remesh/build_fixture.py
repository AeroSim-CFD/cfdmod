"""Build the remesh test fixture from the in-repo output/sim_results/SN data.

Selects 6 container groups (2 from cc0, 2 from cc1, 2 from the last cc), keeps
only their parent triangles, subsets the body+probe H5 to the first 100
timesteps, and writes:

- fixtures/tests/remesh/bodies.h5
- fixtures/tests/remesh/points.h5
- fixtures/tests/remesh/manifest.yaml
"""

from __future__ import annotations

import pathlib
import re

import h5py
import numpy as np
import yaml

from cfdmod import (
    ByConnectivityGrouping,
    BySizeRoundedPerComponent,
    RegroupConfig,
    mesh_from_h5,
)
from cfdmod.geometry.grouping import GroupingResult
from cfdmod.regroup import build_regroup_mapping, expand_regroup_chain

SRC_DIR = pathlib.Path("output/sim_results/results_container/SN").resolve()
DEST_DIR = pathlib.Path("fixtures/tests/remesh").resolve()
DEST_DIR.mkdir(parents=True, exist_ok=True)

N_TIMESTEPS = 100

# Container-pack regroup parameters used by the notebook.
MIN_TRIS = 4
TARGET_SIZE_X = 6.34
TARGET_SIZE_Y = 2.58
TARGET_SIZE_Z = 2.6

body_h5_src = sorted(SRC_DIR.glob("bodies.*.h5"))[0]
probe_h5_src = sorted(SRC_DIR.glob("points.*.h5"))[0]
print(f"body  : {body_h5_src.name}")
print(f"probe : {probe_h5_src.name}")

mesh = mesh_from_h5(body_h5_src)
print(f"original mesh: {mesh.geometry.triangles.shape[0]:,} triangles")

# Run only the cheap grouping path (no slicing); we just need container -> parent indices.
regroup_cfg = RegroupConfig(
    groupings=[
        ByConnectivityGrouping(name_template="cc{idx}", min_triangles=MIN_TRIS),
        BySizeRoundedPerComponent(
            target_size_x=TARGET_SIZE_X,
            target_size_y=TARGET_SIZE_Y,
            target_size_z=TARGET_SIZE_Z,
            name_template="{parent}_c{idx}",
        ),
    ],
    aggregation="sliced",
    timeseries_group="cp",
    output_geometry_format="lnas",
    unassigned_policy="drop",
)
expanded, consumed, _, _ = expand_regroup_chain(
    regroup_cfg.groupings, mesh, regroup_cfg.transformation
)
grouping = build_regroup_mapping(mesh, expanded, regroup_cfg.transformation)
if consumed:
    grouping = GroupingResult(
        parent_n_triangles=grouping.parent_n_triangles,
        groups={n: i for n, i in grouping.groups.items() if n not in consumed},
    )

# Containers are named "cc{idx}_c{idx}" -- split by the cc component.
by_component: dict[int, list[tuple[int, str]]] = {}
pattern = re.compile(r"cc(\d+)_c(\d+)")
for name in grouping.groups:
    m = pattern.match(name)
    if not m:
        continue
    cc_idx = int(m.group(1))
    c_idx = int(m.group(2))
    by_component.setdefault(cc_idx, []).append((c_idx, name))
for cc in by_component:
    by_component[cc].sort()  # ascending by c_idx

component_ids = sorted(by_component)
print(f"components: {len(component_ids)} (cc{component_ids[0]} .. cc{component_ids[-1]})")

first_cc = component_ids[0]
second_cc = component_ids[1]
last_cc = component_ids[-1]

selected: list[str] = []
selected += [name for _, name in by_component[first_cc][:2]]
selected += [name for _, name in by_component[second_cc][:2]]
selected += [name for _, name in by_component[last_cc][-2:]]
print("selected containers:")
for n in selected:
    print(f"  {n}  ({grouping.groups[n].size} parent triangles)")

# Union of selected parents.
parent_idxs = np.unique(np.concatenate([grouping.groups[n] for n in selected]))
print(f"total selected parent triangles: {parent_idxs.size}")

# --- Build subset body H5 ----------------------------------------------------
with h5py.File(body_h5_src, "r") as src:
    src_tris = src["Triangles"][:]
    src_geom = src["Geometry"][:]
    tkeys_sorted = sorted(src["pressure"].keys(), key=lambda k: float(k[1:]))

sub_tris_parent = src_tris[parent_idxs]
used_verts = np.unique(sub_tris_parent)
vertex_remap = np.full(src_geom.shape[0], -1, dtype=np.int64)
vertex_remap[used_verts] = np.arange(used_verts.size, dtype=np.int64)
sub_tris = vertex_remap[sub_tris_parent].astype(src_tris.dtype)
sub_verts = src_geom[used_verts].astype(src_geom.dtype)
print(f"subset geometry: {sub_verts.shape[0]} verts, {sub_tris.shape[0]} triangles")

kept_keys = tkeys_sorted[:N_TIMESTEPS]
print(f"kept timesteps: {len(kept_keys)} ({kept_keys[0]} .. {kept_keys[-1]})")

body_dst = DEST_DIR / "bodies.h5"
if body_dst.exists():
    body_dst.unlink()
with h5py.File(body_h5_src, "r") as src, h5py.File(body_dst, "w") as dst:
    dst.create_dataset("Triangles", data=sub_tris)
    dst.create_dataset("Geometry", data=sub_verts)
    pgrp = dst.create_group("pressure")
    for k in kept_keys:
        pgrp.create_dataset(k, data=src["pressure"][k][:][parent_idxs])
print(f"wrote {body_dst} ({body_dst.stat().st_size / 1024:.1f} KB)")

# --- Build subset probe H5 (timesteps only; probe data is mesh-independent) --
probe_dst = DEST_DIR / "points.h5"
if probe_dst.exists():
    probe_dst.unlink()
with h5py.File(probe_h5_src, "r") as src, h5py.File(probe_dst, "w") as dst:
    if "Triangles" in src:
        dst.create_dataset("Triangles", data=src["Triangles"][:])
    if "Geometry" in src:
        dst.create_dataset("Geometry", data=src["Geometry"][:])
    pgrp = dst.create_group("pressure")
    pkeys_sorted = sorted(src["pressure"].keys(), key=lambda k: float(k[1:]))
    for k in pkeys_sorted[:N_TIMESTEPS]:
        pgrp.create_dataset(k, data=src["pressure"][k][:])
print(f"wrote {probe_dst} ({probe_dst.stat().st_size / 1024:.1f} KB)")

# --- Manifest -----------------------------------------------------------------
manifest = {
    "source": {
        "body_h5": str(body_h5_src.relative_to(pathlib.Path.cwd())),
        "probe_h5": str(probe_h5_src.relative_to(pathlib.Path.cwd())),
    },
    "selection": {
        "n_containers": len(selected),
        "components_picked": {
            f"first (cc{first_cc})": "first 2",
            f"second (cc{second_cc})": "first 2",
            f"last (cc{last_cc})": "last 2",
        },
        "container_names": selected,
        "n_parent_triangles": int(parent_idxs.size),
    },
    "regroup_config": {
        "min_triangles": MIN_TRIS,
        "target_size_x": TARGET_SIZE_X,
        "target_size_y": TARGET_SIZE_Y,
        "target_size_z": TARGET_SIZE_Z,
    },
    "timesteps": {
        "n_kept": len(kept_keys),
        "first": kept_keys[0],
        "last": kept_keys[-1],
    },
    "expected_outputs": {
        "n_containers_after_regroup": "~6 (depends on connectivity in the subset)",
        "n_per_face_buckets": "<= 36 (6 containers x up to 6 axis-aligned faces)",
        "n_coarse_triangles": "~12 (2 per face, modulo coplanar merge collapse)",
    },
}
manifest_path = DEST_DIR / "manifest.yaml"
with manifest_path.open("w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
print(f"wrote {manifest_path}")

print("\nFixture build complete.")
