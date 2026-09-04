from collections.abc import Sequence

import numpy as np

from cfdmod.roughness.parameters import (
    ElementParams,
    OnMissingSurface,
    check_on_missing_surface,
)
from cfdmod.roughness.surface_sampler import (
    DEFAULT_MAX_SAMPLE_POINTS,
    SurfaceInput,
    build_surface_sampler,
)

__all__ = [
    "radial_pattern",
]


def _generate_positions(
    r_start: float,
    r_end: float,
    radial_spacing: float,
    arc_spacing: float,
    ring_offset_distance: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Ring positions and fin orientations for the radial pattern.

    Args:
        r_start (float): Inner radius of the roughness band.
        r_end (float): Outer radius of the roughness band.
        radial_spacing (float): Distance between rings.
        arc_spacing (float): Target arc-length spacing between fins per ring.
        ring_offset_distance (float): Arc-length stagger for alternating rings.
        center (tuple[float, float]): XY center of the radial pattern.

    Returns:
        np.ndarray: (N, 3) array of (x, y, theta).
    """
    rings = np.arange(r_start, r_end + radial_spacing * 0.5, radial_spacing)
    if len(rings) == 0:
        return np.empty((0, 3), dtype=np.float64)

    n_fins = np.maximum(1, (2.0 * np.pi * rings / arc_spacing).astype(np.int64))
    base_angles = (ring_offset_distance / rings) * (np.arange(len(rings)) % 2)

    # Index of each fin within its own ring, without a Python loop over rings.
    ring_starts = np.concatenate(([0], np.cumsum(n_fins)[:-1]))
    fin_index = np.arange(n_fins.sum()) - np.repeat(ring_starts, n_fins)

    theta = (2.0 * np.pi / np.repeat(n_fins, n_fins)) * fin_index + np.repeat(base_angles, n_fins)
    r = np.repeat(rings, n_fins)

    return np.stack([center[0] + r * np.cos(theta), center[1] + r * np.sin(theta), theta], axis=1)


def radial_pattern(
    element_params: ElementParams,
    r_start: float,
    r_end: float,
    radial_spacing: float,
    arc_spacing: float,
    ring_offset_distance: float,
    center: tuple[float, float],
    surfaces: Sequence[SurfaceInput] | None = None,
    max_points: int | None = DEFAULT_MAX_SAMPLE_POINTS,
    on_missing_surface: OnMissingSurface = "drop",
    *,
    surface_paths: Sequence[SurfaceInput] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate radially placed roughness fins above a set of surfaces.

    Each fin is oriented with its face normal pointing radially outward from center.
    Fins are arranged in rings with arc-length-based angular spacing and optional
    staggering between alternating rings. A fin whose position falls outside the
    sampled region (the XY convex hull of the pooled surface vertices) has no
    height to sit on: by default it is dropped, and
    ``on_missing_surface="keep"`` leaves it at z = 0 instead.

    Args:
        element_params (ElementParams): Height and width of each fin.
        r_start (float): Inner radius of the roughness band.
        r_end (float): Outer radius of the roughness band.
        radial_spacing (float): Distance between rings.
        arc_spacing (float): Target arc-length spacing between fins per ring.
        ring_offset_distance (float): Arc-length stagger for alternating rings (angle = offset/r).
        center (tuple[float, float]): XY center of the radial pattern.
        surfaces (Sequence[SurfaceInput] | None): Surfaces for Z sampling, as
            LNAS/STL paths or in-memory ``LnasFormat`` / ``LnasGeometry`` /
            vertex arrays.
        max_points (int | None, optional): Cap on the number of surface points
            used for interpolation. None disables thinning. Defaults to
            ``DEFAULT_MAX_SAMPLE_POINTS``.
        on_missing_surface (OnMissingSurface, optional): What to do with a fin
            that lands over no surface: "drop" removes it, "keep" leaves it at
            z = 0. Defaults to "drop".
        surface_paths (Sequence[SurfaceInput] | None): Deprecated alias for
            ``surfaces``, kept for existing callers.

    Returns:
        tuple[np.ndarray, np.ndarray]: Triangles and normals arrays (STL representation).
    """
    if surfaces is None:
        surfaces = surface_paths
    elif surface_paths is not None:
        raise ValueError("Pass either `surfaces` or the deprecated `surface_paths`, not both")
    if surfaces is None:
        raise ValueError("`surfaces` is required")
    check_on_missing_surface(on_missing_surface)

    sampler = build_surface_sampler(surfaces, max_points=max_points)
    positions = _generate_positions(
        r_start, r_end, radial_spacing, arc_spacing, ring_offset_distance, center
    )

    z_heights = sampler.sample(positions[:, :2])
    missing = np.isnan(z_heights)
    if on_missing_surface == "drop":
        positions = positions[~missing]
        z_heights = z_heights[~missing]
    else:
        z_heights = np.where(missing, 0.0, z_heights)

    h = element_params.height
    w = element_params.width
    base_verts = np.array(
        [[0.0, 0.0, 0.0], [0.0, w, 0.0], [0.0, w, h], [0.0, 0.0, h]],
        dtype=np.float64,
    )

    theta = positions[:, 2]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    n_fins = len(positions)
    rotations = np.zeros((n_fins, 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = cos_t
    rotations[:, 0, 1] = -sin_t
    rotations[:, 1, 0] = sin_t
    rotations[:, 1, 1] = cos_t
    rotations[:, 2, 2] = 1.0

    rotated = np.einsum("nij,vj->nvi", rotations, base_verts)
    translation = np.stack(
        [
            positions[:, 0] + (w / 2.0) * sin_t,
            positions[:, 1] - (w / 2.0) * cos_t,
            z_heights,
        ],
        axis=1,
    )
    verts = (rotated + translation[:, None, :]).astype(np.float32)

    triangles = np.stack([verts[:, [0, 1, 2]], verts[:, [0, 2, 3]]], axis=1).reshape(-1, 3, 3)
    normals = np.repeat(
        np.stack([cos_t, sin_t, np.zeros(n_fins)], axis=1).astype(np.float32), 2, axis=0
    )

    return triangles, normals
