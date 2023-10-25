import numpy as np

from cfdmod.use_cases.roughness_gen import ElementParams

__all__ = [
    "build_single_element",
]


def build_single_element(element_params: ElementParams) -> tuple[np.ndarray, np.ndarray]:
    """Builds a single element

    Args:
        element_params (ElementParams): Object with element parameters

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with triangles and normals (STL representation)
    """
    triangles, normals = _generate_square(element_params.width, element_params.height)

    return triangles, normals


def _get_triangle_normal(t: np.ndarray):
    u, v = t[1] - t[0], t[2] - t[0]
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    return n


def _generate_square(width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate square with fiven width and height

    Args:
        vertices (np.ndarray): Array of element vertices
        width (float): Width of element (y)
        height (float): Height of element (z)

    Returns:
        tuple[np.ndarray, np.ndarray]: STL representation of the element (triangles, normals)
    """

    triangles = np.empty((2, 3, 3), dtype=np.float32)
    normals = np.empty((2, 3), dtype=np.float32)

    f_normal = np.array([-1, 0, 0])
    vertices = np.array(
        [[0, 0, 0], [0, width, 0], [0, width, height], [0, 0, height]], dtype=np.float32
    )

    t0 = vertices[np.array([0, 1, 2])].copy()
    n0 = _get_triangle_normal(t0)
    if not np.all(n0 == f_normal):
        p1, p2 = t0[1].copy(), t0[2].copy()
        t0[1] = p2
        t0[2] = p1
        n0 = _get_triangle_normal(t0)

    t1 = vertices[np.array([0, 2, 3])].copy()
    n1 = _get_triangle_normal(t1)
    if not np.all(n1 == f_normal):
        p1, p2 = t1[1].copy(), t1[2].copy()
        t1[1] = p2
        t1[2] = p1
        n1 = _get_triangle_normal(t1)

    triangles[0] = t0
    triangles[1] = t1

    normals[0] = n0
    normals[1] = n1

    return triangles, normals
