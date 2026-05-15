"""Group triangles by connected component of the (sub)mesh.

Connectivity is defined by **shared edges**: two triangles are adjacent
when they share a vertex pair. Components are extracted with a
union-find over the triangle set restricted by ``allowed`` (or all
parent triangles when ``allowed is None``). Edges to triangles outside
the allowed set are ignored, as documented on the spec.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field


class ByConnectivityGrouping(BaseModel):
    """Group triangles by connected component (shared-edge adjacency).

    Args:
        kind: Discriminator literal, always ``"by_connectivity"``.
        name_template: Format string for group names. Available
            placeholder: ``{idx}`` (component index, 0-based; components
            are ordered by descending triangle count so ``cc0`` is the
            largest).
        min_triangles: Components smaller than this are dropped.
        restrict_to: Optional list of earlier group names; when set, only
            triangles in (the union of) those groups participate in the
            connectivity analysis. Edges to triangles outside the
            restriction are ignored.
    """

    kind: Literal["by_connectivity"] = "by_connectivity"
    name_template: Annotated[
        str,
        Field("cc{idx}", description="Format string for group names; placeholder: {idx}"),
    ]
    min_triangles: Annotated[
        int,
        Field(1, ge=1, description="Drop components with fewer triangles than this"),
    ]
    restrict_to: Annotated[
        list[str] | None,
        Field(None, description="Optional list of earlier group names to restrict to."),
    ]


def _connected_components(
    triangles: np.ndarray,
    candidate_idxs: np.ndarray,
) -> list[np.ndarray]:
    """Return connected components (lists of parent triangle indices) by shared edge.

    Args:
        triangles: ``(n_parent_tri, 3)`` vertex indices for the parent mesh.
        candidate_idxs: ``(n_cand,)`` parent triangle indices to consider.
            Edges to triangles outside this set are ignored.

    Returns:
        List of int64 arrays of parent triangle indices, one per
        component. Order is undefined (the caller sorts).
    """
    if candidate_idxs.size == 0:
        return []

    n = candidate_idxs.size
    # Union-find over candidate positions [0, n).
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

    # Build the (sorted vertex pair) -> list of candidate positions map.
    cand_tris = triangles[candidate_idxs]  # (n, 3)
    edges_a = np.stack([cand_tris[:, 0], cand_tris[:, 1], cand_tris[:, 2]], axis=0)  # (3, n)
    edges_b = np.stack([cand_tris[:, 1], cand_tris[:, 2], cand_tris[:, 0]], axis=0)  # (3, n)
    lo = np.minimum(edges_a, edges_b).astype(np.int64)  # (3, n)
    hi = np.maximum(edges_a, edges_b).astype(np.int64)  # (3, n)

    # 3*n flat list of (lo, hi, candidate_position).
    edge_lo = lo.reshape(-1)  # (3n,)
    edge_hi = hi.reshape(-1)  # (3n,)
    cand_pos = np.tile(np.arange(n, dtype=np.int64), 3)  # (3n,)

    # Sort by (lo, hi) so identical edges land next to each other.
    key = edge_lo.astype(np.int64) * (int(edge_hi.max()) + 2) + edge_hi.astype(np.int64)
    order = np.argsort(key, kind="stable")
    edge_lo = edge_lo[order]
    edge_hi = edge_hi[order]
    cand_pos = cand_pos[order]

    # Walk runs of identical edges and union all triangles sharing it.
    i = 0
    m = edge_lo.size
    while i < m:
        j = i + 1
        while j < m and edge_lo[j] == edge_lo[i] and edge_hi[j] == edge_hi[i]:
            j += 1
        if j - i > 1:
            base = int(cand_pos[i])
            for k in range(i + 1, j):
                union(base, int(cand_pos[k]))
        i = j

    # Group candidate positions by root.
    roots = np.array([find(int(x)) for x in range(n)], dtype=np.int64)
    components: dict[int, list[int]] = {}
    for pos, root in enumerate(roots):
        components.setdefault(int(root), []).append(pos)

    return [
        np.sort(candidate_idxs[np.asarray(positions, dtype=np.int64)])
        for positions in components.values()
    ]


def apply_by_connectivity(
    spec: ByConnectivityGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Connected-components grouping. See module docstring."""
    triangles = np.asarray(mesh.geometry.triangles, dtype=np.int64)
    n_parent = triangles.shape[0]

    if allowed is not None:
        candidate_idxs = np.unique(np.asarray(allowed, dtype=np.int64))
    else:
        candidate_idxs = np.arange(n_parent, dtype=np.int64)

    components = _connected_components(triangles, candidate_idxs)
    components.sort(key=lambda c: (-c.size, int(c[0]) if c.size else 0))

    out: dict[str, np.ndarray] = {}
    keep_idx = 0
    for comp in components:
        if comp.size < spec.min_triangles:
            continue
        name = spec.name_template.format(idx=keep_idx)
        if name in out:
            raise ValueError(
                f"ByConnectivityGrouping: name_template {spec.name_template!r} produced "
                f"duplicate group name {name!r}; include {{idx}} for uniqueness"
            )
        out[name] = comp
        keep_idx += 1
    return out
