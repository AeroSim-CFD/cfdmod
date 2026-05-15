"""Module-level callbacks for ``CustomGrouping`` tests.

These need to live in an importable module (not the test file's local
scope) so that the dotted-path resolution and serialisation paths can
be exercised end-to-end.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lnas import LnasFormat


def split_by_threshold(
    mesh: LnasFormat,
    candidate_idxs: np.ndarray,
    params: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Split candidates by a centroid-axis threshold from ``params``.

    Required params: ``axis`` ("x"/"y"/"z") and ``threshold`` (float).
    """
    axis_index = {"x": 0, "y": 1, "z": 2}[params["axis"]]
    threshold = float(params["threshold"])
    centroids = np.mean(mesh.geometry.triangle_vertices[candidate_idxs], axis=1)
    coords = centroids[:, axis_index]
    out: dict[str, np.ndarray] = {}
    below = candidate_idxs[coords < threshold]
    above = candidate_idxs[coords >= threshold]
    if below.size:
        out["below"] = below
    if above.size:
        out["above"] = above
    return out


def first_n(
    mesh: LnasFormat,
    candidate_idxs: np.ndarray,
    params: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Place the first ``n`` candidates in a single named group.

    Required params: ``n`` (int) and ``name`` (str).
    """
    n = int(params.get("n", 0))
    name = str(params["name"])
    return {name: candidate_idxs[:n]}
