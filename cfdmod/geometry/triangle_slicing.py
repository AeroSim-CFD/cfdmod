"""Pure axis-aligned triangle slicing + fragment geometry helpers.

Pure numpy -- no ``lnas``, no ``cfdmod.io`` (and therefore none of the
heavy scientific stack). This keeps the helpers importable from the v3
op layer, whose contract is a dependency-light import path, while the
legacy ``cfdmod.regroup`` pipeline and ``cfdmod.io.geometry.region_meshing``
re-export them for their own callers.

The cut is the "Ce-style 90-degree" slicing: for each fixed axis-aligned
plane, a triangle that straddles the plane is split into fragments, each
fragment tracks the parent triangle it came from (so per-timestep fields
can be gathered), and triangles parallel to / entirely on one side of a
plane pass through unchanged.
"""

from __future__ import annotations

__all__ = [
    "triangulate_tri",
    "slice_triangle",
    "slice_one_triangle",
    "slice_triangles_with_parents",
    "bin_centroid_to_cell",
    "build_geometry_from_fragments",
]

import numpy as np


def triangulate_tri(sorted_vertices: np.ndarray, insertion_indices: list[int]) -> np.ndarray:
    """Triangulates a point cloud of a triangle.

    Vertices are ordered according to the original triangle normal.
    If there is one vertex inserted, the original triangle splits into two.
    If there are two vertices inserted, it splits into three.

    Args:
        sorted_vertices (np.ndarray): Triangle vertices ordered.
        insertion_indices (list[int]): Indices of the vertices inserted in the slice.

    Returns:
        np.ndarray: Array of triangles.
    """
    tri_indexes = []
    if len(insertion_indices) == 1:
        i = insertion_indices[0]
        tri_indexes.append([i - 1, i, (i + 2) % 4])
        tri_indexes.append([i, (i + 1) % 4, (i + 2) % 4])
    elif len(insertion_indices) == 2:
        i, j = insertion_indices[0], insertion_indices[1]
        tri_indexes.append([4, 0, 1])
        if j == 3:
            tri_indexes.append([1, 2, 3])
            tri_indexes.append([3, 4, 1])
        else:
            tri_indexes.append([1, 2, 4])
            tri_indexes.append([2, 3, 4])
    else:
        tri_indexes.append([0, 1, 2])

    return sorted_vertices[np.array(tri_indexes, dtype=np.uint32)].astype(np.float32)


def slice_triangle(tri_verts: np.ndarray, axis: int, axis_value: float) -> np.ndarray:
    """Slice a triangle from a given plane.

    If the plane intersects any edge of the triangle, new vertices are
    generated and the triangle is re-triangulated into smaller triangles.

    Args:
        tri_verts (np.ndarray): Vertices of the triangle to slice.
        axis (int): Axis index (x=0, y=1, z=2).
        axis_value (float): Value of the interval.

    Returns:
        np.ndarray: Array of triangle vertices resulting from slicing.
    """
    intersected_pts = tri_verts.copy()
    insertion_indices = []

    for i in range(3):
        if len(intersected_pts) > 4:
            # Sliced all possible lines
            continue
        else:
            p1, p2 = tri_verts[i], tri_verts[(i + 1) % 3]

            if (p1[axis] < axis_value and p2[axis] > axis_value) or (
                p1[axis] > axis_value and p2[axis] < axis_value
            ):
                t = (axis_value - p1[axis]) / (p2[axis] - p1[axis])
                intersect_pt = p1 + t * (p2 - p1)

                insert_idx = i + 1 + intersected_pts.shape[0] // 4
                insertion_indices.append(insert_idx)

                intersected_pts = np.insert(intersected_pts, insert_idx, intersect_pt, axis=0)

    return triangulate_tri(intersected_pts, sorted(insertion_indices))


def slice_one_triangle(
    tri_verts: np.ndarray,
    tri_normal: np.ndarray,
    axis: int,
    axis_value: float,
) -> np.ndarray:
    """Slice a single triangle along an axis-aligned plane.

    Returns ``(N, 3, 3)`` post-slice triangle vertex arrays; ``N >= 1``.
    Skips slicing when the triangle is parallel to the cut plane (its
    normal is dominantly along ``axis``) or lies entirely on one side.
    """
    if np.abs(tri_normal).max() == np.abs(tri_normal)[axis]:
        return tri_verts.reshape(1, 3, 3).astype(np.float64)
    if tri_verts[:, axis].max() < axis_value or tri_verts[:, axis].min() > axis_value:
        return tri_verts.reshape(1, 3, 3).astype(np.float64)
    return slice_triangle(tri_verts, axis, axis_value).astype(np.float64)


def _apply_axis_cut(
    cur_verts: np.ndarray,
    cur_normals: np.ndarray,
    cur_parents: np.ndarray,
    axis: int,
    v: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice all current fragments along a single ``(axis, v)`` plane.

    Vectorises the pass-through decision so only the fragments that actually
    straddle the plane run the per-triangle slicing arithmetic. Output
    ordering matches the naive per-fragment loop (fragments emitted in the
    original row order ``i``, and in ``slice_triangle`` order within a row),
    so the result is identical to slicing each triangle one at a time.

    A fragment is passed through unchanged when its normal is dominantly
    along ``axis`` (parallel to the cut plane) or it lies entirely on one
    side of ``v`` -- mirroring :func:`slice_one_triangle`.
    """
    n = cur_verts.shape[0]
    if n == 0:
        return cur_verts, cur_normals, cur_parents

    abs_normals = np.abs(cur_normals)
    normal_parallel = abs_normals.max(axis=1) == abs_normals[:, axis]
    axis_coord = cur_verts[:, :, axis]
    entirely_below = axis_coord.max(axis=1) < v
    entirely_above = axis_coord.min(axis=1) > v
    keep = normal_parallel | entirely_below | entirely_above

    straddle_idx = np.flatnonzero(~keep)
    if straddle_idx.size == 0:
        # No fragment crosses this plane; nothing to do (identity).
        return cur_verts, cur_normals, cur_parents

    # Only the straddling fragments run the (Python) per-triangle cut.
    counts = np.ones(n, dtype=np.int64)
    straddle_fragments: dict[int, np.ndarray] = {}
    for i in straddle_idx.tolist():
        fragments = slice_triangle(cur_verts[i], axis, v).astype(np.float64)
        straddle_fragments[i] = fragments
        counts[i] = fragments.shape[0]

    total = int(counts.sum())
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    out_verts = np.empty((total, 3, 3), dtype=np.float64)
    out_normals = np.empty((total, 3), dtype=np.float64)
    out_parents = np.empty(total, dtype=np.int64)

    # Kept fragments (exactly one output each) filled in bulk by index.
    keep_pos = offsets[:-1][keep]
    out_verts[keep_pos] = cur_verts[keep]
    out_normals[keep_pos] = cur_normals[keep]
    out_parents[keep_pos] = cur_parents[keep]

    # Straddling fragments expand into their contiguous slice, inheriting
    # the parent's normal and parent index.
    for i, fragments in straddle_fragments.items():
        start = int(offsets[i])
        end = int(offsets[i + 1])
        out_verts[start:end] = fragments
        out_normals[start:end] = cur_normals[i]
        out_parents[start:end] = cur_parents[i]

    return out_verts, out_normals, out_parents


def slice_triangles_with_parents(
    tri_verts: np.ndarray,
    tri_normals: np.ndarray,
    parent_idxs: np.ndarray,
    intervals: tuple[list[float], list[float], list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice triangles along axis-aligned planes, tracking parent indices.

    Args:
        tri_verts: ``(n, 3, 3)`` per-triangle vertex array.
        tri_normals: ``(n, 3)`` per-triangle outward normal.
        parent_idxs: ``(n,)`` int64 - the parent (input-mesh) triangle each
            row of ``tri_verts`` came from.
        intervals: per-axis interval edges; non-finite values (``inf`` /
            ``-inf``) are skipped (sentinels for "no binning on this axis").

    Returns:
        ``(verts, normals, parent_per_fragment)``:
        - ``verts``: ``(m, 3, 3)`` post-slice triangle vertex arrays.
        - ``normals``: ``(m, 3)`` per-fragment normals (inherited from parent).
        - ``parent_per_fragment``: ``(m,)`` int64 - parent triangle index
          for each fragment (== input ``parent_idxs[i]`` for any fragment
          derived from input row ``i``).
    """
    cur_verts = tri_verts.astype(np.float64).copy()
    cur_normals = tri_normals.astype(np.float64).copy()
    cur_parents = np.asarray(parent_idxs, dtype=np.int64).copy()

    for axis in range(3):
        for v in intervals[axis]:
            if not np.isfinite(v):
                continue
            cur_verts, cur_normals, cur_parents = _apply_axis_cut(
                cur_verts, cur_normals, cur_parents, axis, float(v)
            )

    return cur_verts, cur_normals, cur_parents


def bin_centroid_to_cell(
    centroid: np.ndarray,
    intervals: tuple[list[float], list[float], list[float]],
) -> tuple[int, int, int]:
    """Return ``(ix, iy, iz)`` cell index of a centroid; -1 if outside any axis."""
    out = []
    for axis in range(3):
        edges = intervals[axis]
        idx = -1
        for j in range(len(edges) - 1):
            lo = edges[j]
            hi = edges[j + 1]
            if lo <= centroid[axis] < hi:
                idx = j
                break
        out.append(idx)
    return tuple(out)  # type: ignore[return-value]


def build_geometry_from_fragments(
    fragments_verts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate vertices and return ``(vertices, triangles)`` arrays."""
    if fragments_verts.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
        )
    n = fragments_verts.shape[0]
    flat = fragments_verts.reshape(n * 3, 3).astype(np.float64)
    unique_verts, inverse = np.unique(flat, axis=0, return_inverse=True)
    triangles = inverse.reshape(n, 3).astype(np.int32)
    return unique_verts, triangles
