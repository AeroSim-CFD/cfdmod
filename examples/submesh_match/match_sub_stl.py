"""Match a region .stl (a sub-mesh) back onto the reference body .stl.

The everyday situation: the simulation wrote its per-triangle time series
against one big body mesh, and the region you want to report on arrives later
as a separate .stl exported from CAD -- a facade, a roof panel, a canopy. The
two files share no index; only the geometry ties them together.

`cfdmod.geometry.match_triangles` closes that gap: it returns, for every
triangle of the region .stl, the index of the same triangle in the reference
.stl (or `NaN` / `-1` where the region file has a triangle the reference does
not). Everything here is .stl -> .stl; no .lnas anywhere.

Run the built-in demo (writes to `_run/`):

    uv run python examples/submesh_match/match_sub_stl.py

Run it on your own pair of files:

    uv run python examples/submesh_match/match_sub_stl.py \
        --reference body.stl --target facade_north.stl --target roof.stl

The demo reference body is assembled from the 20 galpao region .stl fixtures,
so a real "the region file is a sub-mesh of the body file" pair exists in the
repo without shipping another binary.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from cfdmod.geometry import as_triangle_vertices, match_triangles
from cfdmod.io.geometry import export_stl, read_stl

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
GALPAO = REPO / "fixtures" / "tests" / "pressure" / "galpao"


def build_demo_reference_stl(out_dir: pathlib.Path) -> tuple[pathlib.Path, list[pathlib.Path]]:
    """Concatenate the galpao region .stl fixtures into one body .stl."""
    regions = sorted(p for p in GALPAO.glob("*.stl") if not p.name.startswith("galpao"))
    tris = np.concatenate([read_stl(p)[0] for p in regions])
    normals = np.concatenate([read_stl(p)[1] for p in regions])
    reference = out_dir / "galpao_body.stl"
    export_stl(reference, tris, normals)
    return reference, regions


def build_demo_partial_stl(out_dir: pathlib.Path) -> pathlib.Path:
    """A region .stl half of whose triangles are not on the reference body.

    Stands in for the usual accident: the CAD export sits in a different
    coordinate system, or the region file mixes in geometry the simulated body
    never had. Those triangles come back as NaN instead of a wrong id.
    """
    on_body, normals = read_stl(GALPAO / "p1_xp.stl")
    off_body = on_body + np.float32(1000.0)
    tris = np.concatenate([on_body, off_body])
    path = out_dir / "half_off_body.stl"
    export_stl(path, tris, np.concatenate([normals, normals]))
    return path


def report_match(reference: pathlib.Path, target: pathlib.Path, out_dir: pathlib.Path) -> int:
    """Match one target .stl against the reference and write the id table."""
    match = match_triangles(reference=reference, target=target)

    ids = match.as_float()  # float64, NaN where the triangle is not in the reference
    csv = out_dir / f"{target.stem}.ids.csv"
    np.savetxt(
        csv,
        np.column_stack([np.arange(ids.size, dtype=np.float64), ids]),
        delimiter=",",
        header="target_triangle,reference_triangle",
        comments="",
        fmt="%.0f",
    )

    print(f"{target.name}: {match.n_found}/{ids.size} matched, {match.n_missing} missing")
    max_dist = match.distances[match.found].max() if match.n_found else float("nan")
    print(f"  centroid tolerance {match.tolerance:.3e}, max distance {max_dist:.3e}")
    print(f"  first ids {match.indices[:8].tolist()}  ->  {csv.relative_to(REPO)}")
    if match.ambiguous.any():
        print(f"  WARNING {int(match.ambiguous.sum())} ambiguous (duplicated reference triangles)")

    # The point of the ids: read a reference-mesh field on the region. Stand-in
    # for the real per-triangle field (cp_mean, cp_peak, ...) read from the h5.
    n_ref = as_triangle_vertices(reference).shape[0]
    reference_field = np.arange(n_ref, dtype=np.float64)
    region_field = np.full(ids.size, np.nan)
    region_field[match.found] = reference_field[match.indices[match.found]]
    if match.n_found:
        print(
            f"  region field: mean {np.nanmean(region_field):.1f} over {match.n_found} triangles"
        )
    if match.n_missing:
        first = int(np.flatnonzero(match.missing)[0])
        print(
            f"  ids {match.n_missing} NaN (first at target triangle {first}) -- "
            "left as NaN here; pass on_missing='error' to filter_by_submesh to reject instead"
        )
    return match.n_missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", type=pathlib.Path, help="reference body .stl")
    ap.add_argument(
        "--target",
        type=pathlib.Path,
        action="append",
        default=[],
        help="region .stl to locate in the reference (repeatable)",
    )
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "_run")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    reference, targets = args.reference, args.target
    if reference is None:
        reference, regions = build_demo_reference_stl(out_dir)
        print(
            f"demo reference: {reference.relative_to(REPO)} "
            f"({as_triangle_vertices(reference).shape[0]} triangles, "
            f"assembled from {len(regions)} region files)"
        )
        targets = targets or [
            GALPAO / "t1_ym.stl",
            GALPAO / "L2_yp.stl",
            build_demo_partial_stl(out_dir),
        ]
    if not targets:
        ap.error("--reference given without any --target")

    missing = sum(report_match(reference, t, out_dir) for t in targets)
    print(f"\nid tables in {out_dir.relative_to(REPO)}; {missing} unmatched triangle(s) in total")


if __name__ == "__main__":
    main()
