import pathlib

import numpy as np
from lnas import LnasFormat
from scipy.interpolate import LinearNDInterpolator

from cfdmod.use_cases.roughness_gen.parameters import ElementParams

__all__ = [
    "radial_pattern",
]


def _build_z_interpolator(surface_paths: list[pathlib.Path]) -> LinearNDInterpolator:
    all_verts_list = []
    for path in surface_paths:
        geom = LnasFormat.from_file(path).geometry
        all_verts_list.append(geom.vertices.astype(np.float64))
    all_verts = np.unique(np.concatenate(all_verts_list), axis=0)
    return LinearNDInterpolator(all_verts[:, :2], all_verts[:, 2])


def _generate_positions(
    r_start: float,
    r_end: float,
    radial_spacing: float,
    arc_spacing: float,
    ring_offset_distance: float,
    center: tuple[float, float],
) -> np.ndarray:
    center_arr = np.array(center)
    rings = np.arange(r_start, r_end + radial_spacing * 0.5, radial_spacing)
    positions = []
    for ring_idx, r in enumerate(rings):
        n_fins = max(1, int(2.0 * np.pi * r / arc_spacing))
        base_angle = (ring_offset_distance / r) * (ring_idx % 2)
        angles = np.linspace(0.0, 2.0 * np.pi, n_fins, endpoint=False) + base_angle
        for theta in angles:
            x = center_arr[0] + r * np.cos(theta)
            y = center_arr[1] + r * np.sin(theta)
            positions.append((x, y, theta))
    return np.array(positions)


def radial_pattern(
    element_params: ElementParams,
    r_start: float,
    r_end: float,
    radial_spacing: float,
    arc_spacing: float,
    ring_offset_distance: float,
    center: tuple[float, float],
    surface_paths: list[pathlib.Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate radially placed roughness fins above a set of surfaces.

    Each fin is oriented with its face normal pointing radially outward from center.
    Fins are arranged in rings with arc-length-based angular spacing and optional
    staggering between alternating rings.

    Args:
        element_params (ElementParams): Height and width of each fin.
        r_start (float): Inner radius of the roughness band.
        r_end (float): Outer radius of the roughness band.
        radial_spacing (float): Distance between rings.
        arc_spacing (float): Target arc-length spacing between fins per ring.
        ring_offset_distance (float): Arc-length stagger for alternating rings (angle = offset/r).
        center (tuple[float, float]): XY center of the radial pattern.
        surface_paths (list[pathlib.Path]): LNAS surface files for Z sampling.

    Returns:
        tuple[np.ndarray, np.ndarray]: Triangles and normals arrays (STL representation).
    """
    z_interp = _build_z_interpolator(surface_paths)
    positions = _generate_positions(r_start, r_end, radial_spacing, arc_spacing, ring_offset_distance, center)

    z_heights = z_interp(positions[:, 0], positions[:, 1])
    valid_mask = ~np.isnan(z_heights)

    h = element_params.height
    w = element_params.width
    base_verts = np.array(
        [[0.0, 0.0, 0.0], [0.0, w, 0.0], [0.0, w, h], [0.0, 0.0, h]],
        dtype=np.float64,
    )

    all_triangles = []
    all_normals = []

    for i, (x_pos, y_pos, theta) in enumerate(positions):
        if not valid_mask[i]:
            continue

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        R = np.array(
            [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]]
        )
        rotated = (R @ base_verts.T).T

        translation = np.array(
            [x_pos + (w / 2.0) * sin_t, y_pos - (w / 2.0) * cos_t, z_heights[i]]
        )
        verts = (rotated + translation).astype(np.float32)

        n = np.array([cos_t, sin_t, 0.0], dtype=np.float32)

        all_triangles.append(verts[[0, 1, 2]])
        all_triangles.append(verts[[0, 2, 3]])
        all_normals.append(n)
        all_normals.append(n)

    return np.array(all_triangles, dtype=np.float32), np.array(all_normals, dtype=np.float32)
