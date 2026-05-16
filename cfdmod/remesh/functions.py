"""Core algorithms for the remesh module.

Two operations, both per-sub-mesh (one named surface == one sub-mesh):

- :func:`merge_coplanar`: connected coplanar fans are collapsed to the minimum
  triangulation of their boundary polygon. Exact (every output vertex either
  was an input vertex or lies on the original surface). The typical post-
  ``regroup`` ``sliced`` group -- a flat rectangle subdivided into many
  fragments -- comes out as 2 triangles.
- :func:`decimate_qem`: thin wrapper around ``fast_simplification.simplify``.
  Lossy in general; intended for curved groups where the coplanar pass has
  nothing to collapse. Mesh boundaries (which, for a per-group sub-mesh, are
  the group boundary) are preserved implicitly by the underlying algorithm.

:func:`remesh_per_group` dispatches both over the surfaces of an
``LnasFormat`` and restitches the per-group outputs into a fresh
``LnasFormat`` whose surfaces map one-to-one to the input's.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from lnas import LnasFormat, LnasGeometry

__all__ = [
    "merge_coplanar",
    "decimate_qem",
    "remesh_per_group",
]


def _triangle_planes(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-triangle (unit normal, plane offset n.v0, valid mask).

    Degenerate triangles (zero-area) get a zero normal and are marked invalid.
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(cross, axis=1)
    valid = norm > 0.0
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / norm[valid, None]
    plane_d = np.einsum("ij,ij->i", normals, v0)
    return normals, plane_d, valid


def _coplanar_components(
    triangles: np.ndarray,
    normals: np.ndarray,
    plane_d: np.ndarray,
    valid: np.ndarray,
    normal_tol: float,
    plane_tol: float,
) -> list[list[int]]:
    """Union-find over edge-adjacent triangles that share a plane.

    Triangles are merged if they share an edge AND their unit normals agree
    within ``1 - normal_tol`` (cosine) AND their plane offsets match within
    ``plane_tol``. Normals are taken at face value (no anti-parallel handling)
    -- for the intended ``regroup`` outputs all triangles in a group inherit a
    consistent orientation from their parent.
    """
    n = triangles.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edge_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ti in range(n):
        t = triangles[ti]
        for a, b in ((int(t[0]), int(t[1])), (int(t[1]), int(t[2])), (int(t[2]), int(t[0]))):
            key = (a, b) if a < b else (b, a)
            edge_map[key].append(ti)

    cos_threshold = 1.0 - normal_tol
    for tris in edge_map.values():
        if len(tris) < 2:
            continue
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                t1, t2 = tris[i], tris[j]
                if not (valid[t1] and valid[t2]):
                    continue
                cos = float(np.dot(normals[t1], normals[t2]))
                if cos < cos_threshold:
                    continue
                if abs(float(plane_d[t1] - plane_d[t2])) > plane_tol:
                    continue
                union(t1, t2)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _extract_boundary_loops(
    component: Iterable[int],
    triangles: np.ndarray,
) -> list[list[int]] | None:
    """Walk the boundary of a coplanar component into closed vertex loops.

    Returns ``None`` if the boundary half-edges cannot be chained into closed
    loops (malformed input). For a single topological disk this returns a
    one-element list containing the outer loop.
    """
    directed: dict[tuple[int, int], int] = defaultdict(int)
    for ti in component:
        t = triangles[ti]
        for a, b in ((int(t[0]), int(t[1])), (int(t[1]), int(t[2])), (int(t[2]), int(t[0]))):
            directed[(a, b)] += 1

    boundary_he: set[tuple[int, int]] = set()
    for (a, b), count in directed.items():
        if count > 0 and directed.get((b, a), 0) == 0:
            boundary_he.add((a, b))

    out_map: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_he:
        out_map[a].append(b)

    loops: list[list[int]] = []
    remaining = set(boundary_he)
    while remaining:
        start_a, start_b = next(iter(remaining))
        loop = [start_a]
        curr = start_b
        remaining.discard((start_a, start_b))
        while curr != start_a:
            loop.append(curr)
            advanced = False
            for nxt in out_map[curr]:
                edge = (curr, nxt)
                if edge in remaining:
                    remaining.discard(edge)
                    curr = nxt
                    advanced = True
                    break
            if not advanced:
                return None
        loops.append(loop)
    return loops


def _project_to_plane_2d(points_3d: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project 3D points onto a 2D orthonormal frame on the plane perpendicular to ``normal``."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - n * float(np.dot(ref, n))
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return np.stack([points_3d @ u, points_3d @ v], axis=1)


def _point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Strict-interior barycentric test in 2D; vertex hits count as inside."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d02 = float(np.dot(v0, v2))
    d12 = float(np.dot(v1, v2))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return False
    s = (d11 * d02 - d01 * d12) / denom
    t = (d00 * d12 - d01 * d02) / denom
    return s >= 0.0 and t >= 0.0 and s + t <= 1.0


def _drop_collinear_loop_vertices(
    loop_indices: list[int],
    vertices: np.ndarray,
    tol: float = 1e-12,
) -> list[int]:
    """Drop polygon vertices that sit on the straight edge between their two
    neighbors. The boundary walk of a coplanar fan inevitably picks up
    interior-of-original-edge vertices (e.g., the mid-edge vertices of a
    subdivided square); they make the polygon look as if it has many corners
    when the minimum triangulation only needs the true corners.
    """
    if len(loop_indices) <= 3:
        return list(loop_indices)
    loop = list(loop_indices)
    while True:
        n = len(loop)
        if n <= 3:
            return loop
        drop_at: int | None = None
        for i in range(n):
            a = vertices[loop[(i - 1) % n]]
            b = vertices[loop[i]]
            c = vertices[loop[(i + 1) % n]]
            cross = np.linalg.norm(np.cross(b - a, c - b))
            if cross < tol:
                drop_at = i
                break
        if drop_at is None:
            return loop
        loop.pop(drop_at)


def _earclip_loop(
    loop_indices: list[int],
    vertices: np.ndarray,
    normal: np.ndarray,
) -> list[tuple[int, int, int]] | None:
    """Ear-clip a single simple polygon loop. Output triangles are CCW in
    ``normal``'s frame, matching the input orientation when the loop was the
    natural boundary of a CCW-oriented coplanar component.

    Returns ``None`` if the algorithm cannot make progress (non-simple polygon).
    """
    n = len(loop_indices)
    if n < 3:
        return []
    points_2d = _project_to_plane_2d(vertices[loop_indices], normal)

    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += float(points_2d[i, 0] * points_2d[j, 1] - points_2d[j, 0] * points_2d[i, 1])
    signed_area *= 0.5
    if signed_area < 0:
        loop_indices = list(reversed(loop_indices))
        points_2d = points_2d[::-1].copy()

    if n == 3:
        return [(loop_indices[0], loop_indices[1], loop_indices[2])]

    indices = list(range(n))
    triangles_out: list[tuple[int, int, int]] = []
    guard = 0
    max_iter = n * n + 1

    while len(indices) > 3 and guard < max_iter:
        ear_found = False
        for i in range(len(indices)):
            prev_pos = indices[(i - 1) % len(indices)]
            curr_pos = indices[i]
            next_pos = indices[(i + 1) % len(indices)]
            a = points_2d[prev_pos]
            b = points_2d[curr_pos]
            c = points_2d[next_pos]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0.0:
                continue
            blocked = False
            for jpos in indices:
                if jpos in (prev_pos, curr_pos, next_pos):
                    continue
                if _point_in_triangle_2d(points_2d[jpos], a, b, c):
                    blocked = True
                    break
            if blocked:
                continue
            triangles_out.append(
                (loop_indices[prev_pos], loop_indices[curr_pos], loop_indices[next_pos])
            )
            indices.pop(i)
            ear_found = True
            break
        if not ear_found:
            return None
        guard += 1

    if len(indices) == 3:
        triangles_out.append(
            (loop_indices[indices[0]], loop_indices[indices[1]], loop_indices[indices[2]])
        )
    return triangles_out


def merge_coplanar(
    vertices: np.ndarray,
    triangles: np.ndarray,
    normal_tol: float = 1e-6,
    plane_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse coplanar adjacent triangles into the minimum triangulation of their region.

    Within each connected component of edge-adjacent triangles that share a
    plane (orientation and offset within tolerance), the interior triangulation
    is replaced by a fresh ear-clipped triangulation of the component's
    boundary loop. Components with multiple boundary loops (annular topology)
    or for which ear-clipping fails to make progress are kept as-is.

    Args:
        vertices: ``(V, 3)`` input vertex array.
        triangles: ``(T, 3)`` input triangle array of vertex indices.
        normal_tol: Max angular deviation (as ``1 - cos(theta)``) for two
            adjacent triangles to be considered coplanar.
        plane_tol: Max absolute deviation of plane offsets (``n . v0``) for
            two adjacent triangles to be considered coplanar.

    Returns:
        ``(new_vertices, new_triangles)``. Unused vertices are dropped; the
        remaining vertex order matches the surviving input vertex order.
        ``new_triangles`` has dtype ``int32``.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    if triangles.shape[0] == 0:
        return vertices.copy(), triangles.astype(np.int32)

    normals, plane_d, valid = _triangle_planes(vertices, triangles)
    components = _coplanar_components(triangles, normals, plane_d, valid, normal_tol, plane_tol)

    out_triangles: list[tuple[int, int, int]] = []
    for comp in components:
        if len(comp) == 1:
            t = triangles[comp[0]]
            out_triangles.append((int(t[0]), int(t[1]), int(t[2])))
            continue

        ref_normal = None
        for ti in comp:
            if valid[ti]:
                ref_normal = normals[ti]
                break
        if ref_normal is None:
            for ti in comp:
                t = triangles[ti]
                out_triangles.append((int(t[0]), int(t[1]), int(t[2])))
            continue

        loops = _extract_boundary_loops(comp, triangles)
        if loops is None or len(loops) != 1:
            for ti in comp:
                t = triangles[ti]
                out_triangles.append((int(t[0]), int(t[1]), int(t[2])))
            continue

        loop = _drop_collinear_loop_vertices(loops[0], vertices)
        retri = _earclip_loop(loop, vertices, ref_normal)
        if retri is None:
            for ti in comp:
                t = triangles[ti]
                out_triangles.append((int(t[0]), int(t[1]), int(t[2])))
            continue
        out_triangles.extend(retri)

    new_tris = np.asarray(out_triangles, dtype=np.int64)
    if new_tris.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
        )

    used = np.unique(new_tris)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    new_vertices = vertices[used].copy()
    new_tris = remap[new_tris]
    return new_vertices, new_tris.astype(np.int32)


def decimate_qem(
    vertices: np.ndarray,
    triangles: np.ndarray,
    target_reduction: float,
    aggressiveness: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """QEM decimation via ``fast-simplification``.

    Mesh boundaries (vertices and edges on the boundary of the input
    sub-mesh) are preserved implicitly by the underlying algorithm and are
    never collapsed; this means a per-surface call leaves the group boundary
    intact and adjacent groups still match exactly at their shared edges
    after each is decimated independently.

    Args:
        vertices: ``(V, 3)`` input vertex array.
        triangles: ``(T, 3)`` input triangle array of vertex indices.
        target_reduction: Fraction of triangles to remove (``0.9`` keeps 10%).
            ``<= 0`` returns the input unchanged.
        aggressiveness: ``agg`` parameter passed through to
            ``fast_simplification.simplify`` (default 7 matches the library).

    Returns:
        ``(new_vertices, new_triangles)``. ``new_triangles`` has dtype
        ``int32``.

    Raises:
        ImportError: if ``fast-simplification`` is not installed. Install it
            via ``pip install 'aerosim-cfdmod[remesh]'``.
    """
    try:
        import fast_simplification
    except ImportError as exc:
        raise ImportError(
            "decimate_qem requires the 'fast-simplification' package. "
            "Install the optional extra: pip install 'aerosim-cfdmod[remesh]'."
        ) from exc

    vertices_arr = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    triangles_arr = np.ascontiguousarray(np.asarray(triangles, dtype=np.int32))

    if target_reduction <= 0.0 or triangles_arr.shape[0] <= 1:
        return vertices_arr.copy(), triangles_arr.copy()

    new_v, new_t = fast_simplification.simplify(
        vertices_arr,
        triangles_arr,
        target_reduction=float(target_reduction),
        agg=float(aggressiveness),
    )
    return np.asarray(new_v, dtype=np.float64), np.asarray(new_t, dtype=np.int32)


def _extract_subgroup(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a sub-mesh for a contiguous block of triangles.

    Returns ``(sub_vertices, sub_triangles, vertex_remap_inverse)`` where
    ``vertex_remap_inverse[i]`` is the index in the parent ``vertices`` array
    of ``sub_vertices[i]`` -- used at restitching time to keep the original
    vertex when no decimation changed it.
    """
    sub_tris_parent = triangles[tri_indices]
    used = np.unique(sub_tris_parent)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    sub_vertices = vertices[used].astype(np.float64).copy()
    sub_triangles = remap[sub_tris_parent].astype(np.int32)
    return sub_vertices, sub_triangles, used


def remesh_per_group(
    mesh: LnasFormat,
    coplanar_merge: bool = True,
    target_reduction: float = 0.0,
    aggressiveness: float = 7.0,
    normal_tol: float = 1e-6,
    plane_tol: float = 1e-9,
) -> LnasFormat:
    """Per-surface remesh of an ``LnasFormat``.

    For each surface in ``mesh.surfaces``, extract its triangles into a
    sub-mesh, run ``merge_coplanar`` (if ``coplanar_merge``) and then
    ``decimate_qem`` (if ``target_reduction > 0``) on it, and restitch the
    per-surface outputs into a fresh ``LnasFormat`` with the same surface
    names.

    With the defaults (``coplanar_merge=True``, ``target_reduction=0.0``) the
    operation is geometrically lossless: every output vertex is either an
    input vertex or lies exactly on the input surface. A flat ``NxN``-
    subdivided square inside one surface comes out as 2 triangles; a curved
    patch comes out unchanged.

    Args:
        mesh: Input ``LnasFormat`` whose ``surfaces`` map names to triangle
            index arrays.
        coplanar_merge: Run :func:`merge_coplanar` per surface. Default True.
        target_reduction: If ``> 0``, run :func:`decimate_qem` per surface
            after the coplanar pass with this reduction fraction.
        aggressiveness: Forwarded to ``decimate_qem``.
        normal_tol, plane_tol: Forwarded to ``merge_coplanar``.

    Returns:
        A fresh ``LnasFormat`` with one named surface per input surface
        (insertion order preserved). Empty surfaces (no triangles) and
        surfaces that fully collapse during merge are kept as empty
        ``surfaces`` entries to preserve the name mapping.
    """
    parent_vertices = np.asarray(mesh.geometry.vertices, dtype=np.float64)
    parent_triangles = np.asarray(mesh.geometry.triangles, dtype=np.int64)

    out_vertices_chunks: list[np.ndarray] = []
    out_triangles_chunks: list[np.ndarray] = []
    out_surfaces: dict[str, np.ndarray] = {}
    vertex_cursor = 0
    triangle_cursor = 0

    for name, tri_idx_arr in mesh.surfaces.items():
        tri_idx = np.asarray(tri_idx_arr, dtype=np.int64)
        if tri_idx.size == 0:
            out_surfaces[name] = np.zeros(0, dtype=np.int32)
            continue

        sub_v, sub_t, _ = _extract_subgroup(parent_vertices, parent_triangles, tri_idx)

        if coplanar_merge:
            sub_v, sub_t = merge_coplanar(sub_v, sub_t, normal_tol=normal_tol, plane_tol=plane_tol)
        if target_reduction > 0.0 and sub_t.shape[0] > 1:
            sub_v, sub_t = decimate_qem(
                sub_v, sub_t, target_reduction=target_reduction, aggressiveness=aggressiveness
            )

        if sub_t.shape[0] == 0:
            out_surfaces[name] = np.zeros(0, dtype=np.int32)
            continue

        n_v_sub = sub_v.shape[0]
        n_t_sub = sub_t.shape[0]
        out_vertices_chunks.append(sub_v.astype(np.float64))
        out_triangles_chunks.append(sub_t.astype(np.int64) + vertex_cursor)
        out_surfaces[name] = np.arange(triangle_cursor, triangle_cursor + n_t_sub, dtype=np.int32)
        vertex_cursor += n_v_sub
        triangle_cursor += n_t_sub

    if out_vertices_chunks:
        new_vertices = np.concatenate(out_vertices_chunks, axis=0)
        new_triangles = np.concatenate(out_triangles_chunks, axis=0).astype(np.int32)
    else:
        new_vertices = np.zeros((0, 3), dtype=np.float64)
        new_triangles = np.zeros((0, 3), dtype=np.int32)

    new_vertices, new_triangles = _dedupe_vertices(new_vertices, new_triangles)

    new_geom = LnasGeometry(vertices=new_vertices, triangles=new_triangles)
    return LnasFormat(
        version=mesh.version,
        geometry=new_geom,
        surfaces=out_surfaces,
    )


def _dedupe_vertices(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge identical vertices across the stitched-together per-surface chunks.

    Each surface's sub-mesh keeps its own copy of any shared boundary vertex;
    deduplicating here lets neighbouring surfaces share those vertices in the
    output. Uses ``np.unique`` on the full vertex array (exact-equality match).
    """
    if vertices.shape[0] == 0:
        return vertices, triangles
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    new_triangles = inverse[triangles].astype(np.int32)
    return unique, new_triangles
