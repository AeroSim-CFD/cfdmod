from collections.abc import Sequence

import numpy as np

from cfdmod.roughness.build_element import build_single_element
from cfdmod.roughness.linear_pattern import linear_pattern
from cfdmod.roughness.parameters import (
    BoundingBox,
    ElementParams,
    GenerationParams,
    OnMissingSurface,
    SpacingParams,
    check_on_missing_surface,
)
from cfdmod.roughness.surface_sampler import (
    DEFAULT_MAX_SAMPLE_POINTS,
    SurfaceInput,
    build_surface_sampler,
    load_surface_vertices,
)

__all__ = [
    "position_pattern",
    "clip_bounding_box",
    "generation_params_for_box",
]

TRIANGLES_PER_ELEMENT = 2


def clip_bounding_box(bounding_box: BoundingBox, vertices: Sequence[np.ndarray]) -> np.ndarray:
    """Intersect the configured bounding box with the surfaces' own XY extent.

    Args:
        bounding_box (BoundingBox): Configured spawn volume.
        vertices (Sequence[np.ndarray]): One (N, 3) vertex array per surface.

    Returns:
        np.ndarray: Array of shape (2, 2) with the clipped [[x_start, y_start],
            [x_end, y_end]].
    """
    mesh_min = np.min([v.min(axis=0)[:2] for v in vertices], axis=0)
    mesh_max = np.max([v.max(axis=0)[:2] for v in vertices], axis=0)

    start = np.maximum(np.array(bounding_box.start[:2], dtype=np.float64), mesh_min)
    end = np.minimum(np.array(bounding_box.end[:2], dtype=np.float64), mesh_max)
    return np.array([start, end])


def generation_params_for_box(
    element_params: ElementParams,
    spacing_params: SpacingParams,
    clipped_box: np.ndarray,
) -> GenerationParams:
    """Derive the element counts that fill a clipped XY box.

    The counts are chosen so the whole array, staggering included, stays inside
    the box.

    Args:
        element_params (ElementParams): Height and width of a single element.
        spacing_params (SpacingParams): Spacing, line offset and offset direction.
        clipped_box (np.ndarray): (2, 2) XY box as returned by
            :func:`clip_bounding_box`.

    Returns:
        GenerationParams: Generation parameters for the linear pattern.
    """
    lx, ly = clipped_box[1] - clipped_box[0]
    spacing_x, spacing_y = spacing_params.spacing
    if spacing_x <= 0:
        raise ValueError(
            f"Spacing in X must be positive to space the element lines, got {spacing_x}"
        )
    line_pitch = element_params.width + spacing_y

    if spacing_params.offset_direction == "x":
        n_x = int((lx - spacing_params.line_offset) // spacing_x + 1)
        n_y = int((ly + spacing_y) // line_pitch)
    else:
        n_x = int(lx // spacing_x + 1)
        n_y = int((ly + spacing_y - spacing_params.line_offset) // line_pitch)

    if n_x <= 0 or n_y <= 0:
        raise ValueError(
            f"Bounding box clipped to the surfaces is too small for a single element: "
            f"lx={lx:.6g}, ly={ly:.6g} gives N_elements_x={n_x}, N_elements_y={n_y}. "
            "Widen the bounding box or reduce the spacing / element width."
        )

    return GenerationParams(
        N_elements_x=n_x,
        N_elements_y=n_y,
        element_params=element_params,
        spacing_params=spacing_params,
    )


def _build_element_array(cfg: GenerationParams) -> tuple[np.ndarray, np.ndarray]:
    """Build the full flat element array at the origin.

    Args:
        cfg (GenerationParams): Generation parameters.

    Returns:
        tuple[np.ndarray, np.ndarray]: Triangles and normals (STL representation).
    """
    triangles, normals = build_single_element(cfg.element_params)

    line_triangles, line_normals = linear_pattern(
        triangles,
        normals,
        direction=cfg.spacing_params.offset_direction,
        n_repeats=cfg.single_line_elements,
        spacing_value=cfg.single_line_spacing,
    )
    return linear_pattern(
        line_triangles,
        line_normals,
        direction=cfg.perpendicular_direction,
        n_repeats=cfg.multi_line_elements,
        spacing_value=cfg.multi_line_spacing,
        offset_value=cfg.spacing_params.line_offset,
    )


def _footprint_centroids(triangles: np.ndarray) -> np.ndarray:
    """XY centroid of each element's footprint, where the surface is sampled.

    Args:
        triangles (np.ndarray): (2 * n_elements, 3, 3) triangles, two consecutive
            triangles per element.

    Returns:
        np.ndarray: (n_elements, 2) query positions.
    """
    per_element = triangles.reshape(-1, TRIANGLES_PER_ELEMENT * 3, 3)
    return per_element[:, :, :2].mean(axis=1)


def position_pattern(
    element_params: ElementParams,
    spacing_params: SpacingParams,
    bounding_box: BoundingBox,
    surfaces: Sequence[SurfaceInput],
    max_points: int | None = DEFAULT_MAX_SAMPLE_POINTS,
    on_missing_surface: OnMissingSurface = "drop",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linear roughness array draped onto a set of surfaces.

    The array fills the intersection of ``bounding_box`` and the surfaces' own
    XY extent, then every element is lifted to sit on the surface below it. The
    lift is the surface Z at the element footprint's centroid: one sample per
    element.

    An element whose centroid falls outside the sampled region has no height to
    be seated on. By default it is dropped; ``on_missing_surface="keep"`` leaves
    it unlifted instead, with its base at z = 0. The sampled region is the XY
    convex hull of the pooled surface vertices, so an element over a hole
    between two surfaces is seated on the height interpolated across the gap and
    is never the missing case.

    Only the X and Y components of ``bounding_box`` constrain the placement. The
    Z of each element comes from the surface it is draped onto, so the Z
    components of the box are not used.

    Args:
        element_params (ElementParams): Height and width of a single element.
        spacing_params (SpacingParams): Spacing, line offset and offset direction.
        bounding_box (BoundingBox): Volume in which to spawn elements (XY used).
        surfaces (Sequence[SurfaceInput]): Surfaces to drape onto, as paths or
            in-memory geometry.
        max_points (int | None, optional): Cap on the number of surface points
            used for interpolation. None disables thinning. Defaults to
            ``DEFAULT_MAX_SAMPLE_POINTS``.
        on_missing_surface (OnMissingSurface, optional): What to do with an
            element that lands over no surface: "drop" removes it, "keep" leaves
            it unlifted at z = 0. Defaults to "drop".

    Returns:
        tuple[np.ndarray, np.ndarray]: Triangles and normals arrays (STL
            representation).
    """
    check_on_missing_surface(on_missing_surface)

    vertices = load_surface_vertices(surfaces)
    clipped_box = clip_bounding_box(bounding_box, vertices)
    cfg = generation_params_for_box(element_params, spacing_params, clipped_box)

    triangles, normals = _build_element_array(cfg)
    triangles[:, :, 0] += clipped_box[0][0]
    triangles[:, :, 1] += clipped_box[0][1]

    sampler = build_surface_sampler(vertices, max_points=max_points)
    z_offset = sampler.sample(_footprint_centroids(triangles))
    on_surface = ~np.isnan(z_offset)

    if on_missing_surface == "drop":
        keep = on_surface
    else:
        keep = np.ones(len(on_surface), dtype=bool)
        # An element over nothing is left where it was built, base at z = 0.
        z_offset = np.where(on_surface, z_offset, 0.0)

    element_triangles = triangles.reshape(-1, TRIANGLES_PER_ELEMENT, 3, 3)[keep]
    element_normals = normals.reshape(-1, TRIANGLES_PER_ELEMENT, 3)[keep]
    element_triangles[:, :, :, 2] += z_offset[keep, None, None].astype(np.float32)

    return (
        element_triangles.reshape(-1, 3, 3).astype(np.float32),
        element_normals.reshape(-1, 3).astype(np.float32),
    )
