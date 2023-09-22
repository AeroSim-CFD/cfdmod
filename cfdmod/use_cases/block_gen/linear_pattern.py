from typing import Literal

import numpy as np

__all__ = [
    "linear_pattern",
]


def linear_pattern(
    triangles: np.ndarray,
    normals: np.ndarray,
    direction: Literal["x", "y"],
    n_repeats: int,
    spacing_value: float,
    offset_value: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies linear pattern to geometry objects

    Args:
        triangles (np.ndarray): Array of triangles vertices.
        normals (np.ndarray): Array of triangles normals.
        direction (Literal): Direction of the linear pattern (x or y).
        n_repeats (int): Number of times to copy the pattern.
        spacing_value (float): Spacing value for the linear pattern.
        offset_value (float, optional): Offset value for the linear pattern perpendicular to the pattern direction. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: Replicated geometry in STL representation (triangles, normals)
    """
    full_triangles = np.tile(triangles, (n_repeats + 1, 1, 1))
    full_normals = np.tile(normals, (n_repeats + 1, 1))

    spacing_array = np.array(
        [
            spacing_value if direction == "x" else 0,
            spacing_value if direction == "y" else 0,
            0,
        ]
    )
    offset_array = np.array(
        [
            offset_value if direction != "x" else 0,
            offset_value if direction != "y" else 0,
            0,
        ]
    )

    for i in range(1, n_repeats+1):
        # Iterations starts at the first row to be replicated, original is skipped.
        # In that way, the row index is i+1
        # For each replication, calculate the spacing between the rows.
        # For odd rows, apply the offset value.
        full_triangles[
            i * triangles.shape[0] : (i + 1) * triangles.shape[0], :, :
        ] += spacing_array * i + offset_array * (i % 2)

    return full_triangles, full_normals
