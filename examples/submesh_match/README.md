# Match a region .stl onto the reference body mesh

The simulation wrote its per-triangle time series against one big **body** mesh.
The region you have to report on arrives separately as its own **.stl** exported
from CAD -- a facade, a roof panel, a canopy. The two files share no index; only
the geometry ties them together.

`cfdmod.geometry.match_triangles(reference, target)` returns, for every triangle
of the region file, the index of the same triangle in the reference mesh -- or a
missing marker where the region file has a triangle the reference does not.

Everything below is `.stl -> .stl`. `.lnas` files and in-memory
`(n_tri, 3, 3)` vertex arrays are accepted in the same arguments, but nothing
here needs them.

## The one call

```python
from cfdmod.geometry import match_triangles

match = match_triangles("body.stl", "facade_north.stl")

match.indices        # (n_target,) int64, in facade_north.stl triangle order; -1 where missing
match.as_float()     # same thing as float64 with NaN instead of -1
match.found          # bool mask of the triangles that matched
match.n_missing      # how many did not
match.is_injective   # False if two region triangles landed on the same body triangle
match.ambiguous      # True where the body mesh has duplicated/coincident triangles there
```

Reading a reference-mesh field on the region:

```python
region_cp = np.full(match.indices.size, np.nan)
region_cp[match.found] = body_cp[match.indices[match.found]]
```

Or refuse to proceed with an incomplete match:

```python
ids = match.require_complete()   # raises if any region triangle is unmatched
region_cp = body_cp[ids]
```

## The same thing as a pipeline op

`filter_by_submesh` does the match and slices a whole `DataSource` -- topology,
element metadata, groupings and every field -- down to the region, **in the
region file's triangle order**:

```python
from cfdmod.core.ops.data_source_create import FilterBySubmeshParams, filter_by_submesh

facade = filter_by_submesh(
    body_ds,
    FilterBySubmeshParams(mesh="facade_north.stl", reference_mesh="body.stl"),
)
```

- `reference_mesh` is optional: omitted, the reference triangles are taken from
  `body_ds.topology`, which is already the body mesh in the normal case.
- `on_missing="error"` (default) rejects a region triangle with no counterpart;
  `on_missing="drop"` keeps only the matched ones.
- Registered in the YAML template registry as `filter_by_submesh`, so it also
  works as a declarative pipeline step.

## How the matching works, and the two knobs

- Triangles are matched by **centroid** within a tolerance, then cross-checked
  by **area**. Centroids are immune to the vertex rotation and winding flip an
  STL round-trip introduces, so a re-exported region file still matches.
- `rtol` (default `1e-6`) is relative to the reference length scale, taken as
  `max(bounding-box diagonal, largest absolute coordinate)`. The second term is
  what matters for STL: the format stores `float32`, so its absolute round-trip
  error grows with the distance from the origin, not with the size of the model
  -- a mesh sitting at `z = 900` quantises to ~1e-4 no matter how small it is.
- `atol` overrides `rtol` with an absolute tolerance in model units.
- `area_rtol` (default `1e-2`) rejects a nearest-centroid candidate whose area
  disagrees -- that is a different triangle that happens to share a centroid,
  not the same one. `None` disables the check.
- A `RuntimeWarning` fires when the tolerance is not clearly smaller than the
  closest pair of reference centroids: at that point a match is ambiguous by
  construction and the mesh (or the tolerance) needs a look.

## Run the demo

```bash
uv run python examples/submesh_match/match_sub_stl.py
```

Writes `_run/<region>.ids.csv` (`target_triangle,reference_triangle`, `nan`
where unmatched) for three regions. The demo body `.stl` is assembled from the
20 galpao region fixtures, so a genuine sub-mesh pair exists in the repo; the
third region is deliberately half off-body to show the NaN path.

On your own files:

```bash
uv run python examples/submesh_match/match_sub_stl.py \
    --reference body.stl --target facade_north.stl --target roof.stl
```
