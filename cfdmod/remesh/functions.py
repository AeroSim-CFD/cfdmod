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

API convention:

- :func:`merge_coplanar` and :func:`decimate_qem` take **raw**
  ``(vertices, triangles)`` arrays -- they operate on a single sub-mesh and
  know nothing about surfaces. Use them when you have one extracted region
  in hand and want to coarsen it.
- :func:`remesh_per_group` takes a full :class:`lnas.LnasFormat` with named
  surfaces, dispatches the two array-level operations over each surface, and
  restitches the per-surface outputs back into a fresh ``LnasFormat``.

All three functions are exported from ``cfdmod.remesh`` and re-exported at
the top-level ``cfdmod`` package.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Iterable

import numpy as np
from lnas import LnasFormat, LnasGeometry

from cfdmod.logger import logger

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

    Triangles are merged if they share an edge AND their normals are
    parallel within ``normal_tol`` (cosine, allowing anti-parallel) AND
    their plane offsets match within ``plane_tol`` (with the offset sign
    flipped when the two normals are anti-parallel, so a flipped triangle
    on the same physical plane is still recognised).
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
                if abs(cos) < cos_threshold:
                    continue
                # Anti-parallel normals describe the same physical plane
                # when d1 + d2 ~= 0 (d2 is computed against -n1). Same-
                # direction normals require d1 - d2 ~= 0.
                if cos > 0:
                    plane_diff = abs(float(plane_d[t1] - plane_d[t2]))
                else:
                    plane_diff = abs(float(plane_d[t1] + plane_d[t2]))
                if plane_diff > plane_tol:
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

    Uses undirected edge counts to identify boundary edges, so inconsistent
    triangle winding across the component does not poison the boundary set.
    Returns ``None`` for closed components (no boundary edges) or for
    boundaries whose vertices have anything other than exactly two boundary
    neighbours (branching / pinched topology).
    """
    undirected: dict[tuple[int, int], int] = defaultdict(int)
    for ti in component:
        t = triangles[ti]
        for a, b in ((int(t[0]), int(t[1])), (int(t[1]), int(t[2])), (int(t[2]), int(t[0]))):
            key = (a, b) if a < b else (b, a)
            undirected[key] += 1

    boundary_edges = {k for k, v in undirected.items() if v == 1}
    if not boundary_edges:
        return None

    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    # A simple loop visits every boundary vertex with exactly two neighbours.
    for nbrs in neighbours.values():
        if len(nbrs) != 2:
            return None

    def edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    edges_in_order = sorted(boundary_edges)
    for start_edge in edges_in_order:
        if start_edge in visited_edges:
            continue
        start_a, start_b = start_edge
        loop = [start_a]
        prev = start_a
        curr = start_b
        visited_edges.add(start_edge)
        while curr != start_a:
            loop.append(curr)
            next_candidates = [
                n for n in neighbours[curr] if n != prev and edge_key(curr, n) not in visited_edges
            ]
            if not next_candidates:
                # Should never happen given the 2-neighbour invariant; defensive.
                return None
            nxt = next_candidates[0]
            visited_edges.add(edge_key(curr, nxt))
            prev = curr
            curr = nxt
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
    tol: float,
) -> list[int]:
    """Drop polygon vertices that sit on the straight edge between their two
    neighbours. The boundary walk of a coplanar fan inevitably picks up
    interior-of-original-edge vertices (e.g., the mid-edge vertices of a
    subdivided square); they make the polygon look as if it has many corners
    when the minimum triangulation only needs the true corners.

    ``tol`` is an absolute length threshold on the cross-product magnitude
    ``|(b - a) x (c - b)|`` (so it has units of [length]^2). Callers should
    scale it by the mesh's bbox diagonal so it stays meaningful in any unit
    system.
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


def _bbox_diagonal(vertices: np.ndarray) -> float:
    if vertices.shape[0] == 0:
        return 0.0
    diag = vertices.max(axis=0) - vertices.min(axis=0)
    return float(np.linalg.norm(diag))


def merge_coplanar(
    vertices: np.ndarray,
    triangles: np.ndarray,
    normal_tol: float = 1e-6,
    plane_tol: float = 1e-9,
    collinear_rel_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse coplanar adjacent triangles into the minimum triangulation of their region.

    Within each connected component of edge-adjacent triangles that share a
    plane (within ``normal_tol`` on the unit normal and ``plane_tol`` on the
    plane offset, with anti-parallel normals treated as the same plane), the
    interior triangulation is replaced by a fresh ear-clipped triangulation
    of the component's boundary loop. Components with multiple boundary loops
    (annular topology) or for which ear-clipping fails to make progress are
    kept as-is, and a ``logger.debug`` message is emitted so callers can see
    when fallback triggers.

    Args:
        vertices: ``(V, 3)`` input vertex array.
        triangles: ``(T, 3)`` input triangle array of vertex indices.
        normal_tol: Max angular deviation (as ``1 - |cos(theta)|``) for two
            adjacent triangles to be considered coplanar. Uses the absolute
            cosine so flipped (anti-parallel) triangles on the same physical
            plane are also merged.
        plane_tol: Max absolute deviation of plane offsets (``n . v0``) for
            two adjacent triangles to be considered coplanar.
        collinear_rel_tol: Relative tolerance for the collinear-vertex drop
            pass on the boundary loop. The absolute threshold is
            ``collinear_rel_tol * bbox_diagonal^2`` so the behaviour is
            independent of mesh units.

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

    # Scale the collinear tolerance by the mesh size so the threshold is
    # meaningful in any unit system. The cross product compared against this
    # threshold has units of [length]^2.
    bbox_diag = _bbox_diagonal(vertices)
    collinear_tol = max(collinear_rel_tol * bbox_diag * bbox_diag, 1e-18)

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
        if loops is None:
            logger.debug(
                "merge_coplanar: malformed or closed boundary for coplanar component "
                "of %d triangle(s); keeping originals",
                len(comp),
            )
            for ti in comp:
                t = triangles[ti]
                out_triangles.append((int(t[0]), int(t[1]), int(t[2])))
            continue
        if len(loops) != 1:
            logger.debug(
                "merge_coplanar: coplanar component of %d triangle(s) has %d "
                "boundary loops (annular topology not yet supported); keeping originals",
                len(comp),
                len(loops),
            )
            for ti in comp:
                t = triangles[ti]
                out_triangles.append((int(t[0]), int(t[1]), int(t[2])))
            continue

        loop = _drop_collinear_loop_vertices(loops[0], vertices, collinear_tol)
        retri = _earclip_loop(loop, vertices, ref_normal)
        if retri is None:
            logger.debug(
                "merge_coplanar: ear-clipping failed for coplanar component of "
                "%d triangle(s) with %d boundary vertices; keeping originals",
                len(comp),
                len(loop),
            )
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


def _has_open_boundary(triangles: np.ndarray) -> bool:
    """True if at least one undirected edge is incident to exactly one triangle."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for t in triangles:
        for a, b in ((int(t[0]), int(t[1])), (int(t[1]), int(t[2])), (int(t[2]), int(t[0]))):
            key = (a, b) if a < b else (b, a)
            counts[key] += 1
    return any(c == 1 for c in counts.values())


def decimate_qem(
    vertices: np.ndarray,
    triangles: np.ndarray,
    target_reduction: float,
    aggressiveness: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """QEM decimation via ``fast-simplification``.

    Mesh boundaries (vertices and edges on the boundary of the input
    sub-mesh) are preserved implicitly by the underlying algorithm and are
    never collapsed; a per-surface call therefore leaves the group boundary
    intact and adjacent groups still match exactly at their shared edges
    after each is decimated independently.

    **Closed surfaces** (sub-meshes with no boundary edges, e.g. a watertight
    sphere) have no boundary for the algorithm to protect, so a high
    ``target_reduction`` can collapse them aggressively. A ``RuntimeWarning``
    is emitted in that case; consider running ``fast_simplification.simplify``
    directly with ``lossless=True`` if you need a bounded-error path.

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

    if not _has_open_boundary(triangles_arr):
        warnings.warn(
            "decimate_qem: input sub-mesh has no boundary edges (closed surface); "
            "QEM has nothing to protect and may collapse it aggressively at high "
            "target_reduction. Consider fast_simplification.simplify(lossless=True) "
            "directly, or feed a sub-mesh with an open boundary.",
            RuntimeWarning,
            stacklevel=2,
        )

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
    seam_rel_tol: float = 1e-9,
) -> LnasFormat:
    """Per-surface remesh of an ``LnasFormat``.

    For each surface in ``mesh.surfaces``, extract its triangles into a
    sub-mesh, run :func:`merge_coplanar` (if ``coplanar_merge``) and then
    :func:`decimate_qem` (if ``target_reduction > 0``), and restitch the
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
        seam_rel_tol: Relative tolerance for merging shared boundary vertices
            between adjacent surfaces in the restitched output. The absolute
            threshold is ``seam_rel_tol * bbox_diagonal``. ``0`` disables and
            falls back to exact-equality dedup. Tolerance-based dedup matters
            once :func:`decimate_qem` is enabled because QEM can synthesise
            new vertex positions that drift below float-equality.

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

    # Scale the seam-dedup tolerance by the parent mesh's bbox so neighbouring
    # surfaces that share a boundary vertex still share it in the output even
    # after the per-surface processing (especially decimate_qem) perturbs the
    # exact coords slightly.
    dedup_tol = seam_rel_tol * _bbox_diagonal(parent_vertices) if seam_rel_tol > 0 else 0.0
    new_vertices, new_triangles = _dedupe_vertices(new_vertices, new_triangles, tol=dedup_tol)

    new_geom = LnasGeometry(vertices=new_vertices, triangles=new_triangles)
    return LnasFormat(
        version=mesh.version,
        geometry=new_geom,
        surfaces=out_surfaces,
    )


def _dedupe_vertices(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge identical vertices across the stitched-together per-surface chunks.

    Each surface's sub-mesh keeps its own copy of any shared boundary vertex;
    deduplicating here lets neighbouring surfaces share those vertices in the
    output. When ``tol > 0`` vertices are quantised to a grid of that size
    before the unique-merge -- enough to absorb the sub-float-precision drift
    that :func:`decimate_qem` can introduce on a shared boundary. The
    quantisation has the standard grid-cell limitation: in the worst case
    two vertices straddling a grid boundary can stay distinct even when they
    are separated by less than ``tol``, but any pair separated by more than
    ``sqrt(3) * tol`` is guaranteed to remain distinct. For the documented
    use case (absorbing sub-1e-9 drift on a metre-scale mesh) the worst-case
    miss is well below any meaningful feature. When ``tol == 0`` an
    exact-equality unique is used (the coplanar-merge-only path always
    produces bit-identical seam coords, so this is the cheaper default).
    """
    if vertices.shape[0] == 0:
        return vertices, triangles
    if tol > 0.0:
        # Quantise to the nearest multiple of `tol`, then unique on the
        # quantised values. ``return_index`` gives the first-occurrence index
        # of each unique row in one shot -- preserves the caller's precision
        # by reusing the original (un-quantised) coords for the survivors.
        quantised = np.round(vertices / tol)
        _, first_idx, inverse = np.unique(
            quantised, axis=0, return_index=True, return_inverse=True
        )
        new_vertices = vertices[first_idx]
        new_triangles = inverse[triangles].astype(np.int32)
        return new_vertices, new_triangles
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    new_triangles = inverse[triangles].astype(np.int32)
    return unique, new_triangles
