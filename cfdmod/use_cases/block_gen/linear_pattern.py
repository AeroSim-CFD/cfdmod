from typing import Literal

import numpy as np


def linear_pattern(
    vertices: np.ndarray,
    triangles: np.ndarray,
    direction: Literal["x", "y"],
    n_repeats: int,
    spacing_value: float,
    offset_value: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies linear pattern to vertices

    Args:
        vertices (np.ndarray): Array of vertices
        triangles (np.ndarray): Array of triangles
        direction (Literal): Direction of the linear pattern (x or y)
        n_repeats (int): Number of times to repeat the pattern
        spacing_value (float): Spacing value for the linear pattern
        offset_value (float, optional): Offset value for the linear pattern perpendicular to the pattern direction. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """

    full_vertices = vertices.copy()
    full_triangles = triangles.copy()

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

    number_of_vertices_to_replicate = max(triangles.flatten()) + 1

    for i in range(n_repeats):
        new_v = vertices.copy() + spacing_array * (i + 1)
        new_v = new_v + offset_array if i % 2 == 0 else new_v
        new_t = triangles.copy() + np.repeat(number_of_vertices_to_replicate * (i + 1), 3)
        full_vertices = np.concatenate((full_vertices, new_v))
        full_triangles = np.concatenate((full_triangles, new_t))

    return full_vertices, full_triangles
