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
    vertices = []
    x = 0
    y = np.linspace(0, element_params.width, 2)
    z = np.linspace(0, element_params.height, 2)
    xv, yv, zv = np.meshgrid(x, y, z)

    for coordinate in list(zip(xv.flatten(), yv.flatten(), zv.flatten())):
        vertices.append(coordinate)

    vertices = np.array(vertices)

    triangles, normals = _triangulate_element(vertices, element_params)

    return triangles, normals


def _get_triangle_normal(t: np.ndarray):
    u, v = t[1] - t[0], t[2] - t[0]
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    return n


def _triangulate_element(
    vertices: np.ndarray, element_params: ElementParams
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulates the given vertices of a roughness element

    Args:
        vertices (np.ndarray): Array of element vertices
        element_params (BlockParams): Object with element parameters

    Returns:
        tuple[np.ndarray, np.ndarray]: STL representation of the element (triangles, normals)
    """
    n_triangles = 2  # Number of triangles in one face. Each face has 2 triangles

    triangles = np.empty((n_triangles, 3, 3), dtype=np.float32)
    normals = np.empty((n_triangles, 3), dtype=np.float32)

    f_normal = np.array([-1, 0, 0])

    t0 = vertices[0:3]
    n0 = _get_triangle_normal(t0)
    if not np.all(n0 == f_normal):
        p1, p2 = t0[1].copy(), t0[2].copy()
        t0[1] = p2
        t0[2] = p1
        n0 = _get_triangle_normal(t0)

    t0 = t0 if np.all(n0 == f_normal) else t0[::-1]

    t1 = vertices[1:4]
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
