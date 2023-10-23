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

    t_0 = vertices[0:3]
    n_0 = np.cross(t_0[1] - t_0[0], t_0[2] - t_0[1])
    n_0 /= np.linalg.norm(n_0)
    t_0 = t_0 if np.all(n_0 == f_normal) else t_0[::-1]

    t_1 = vertices[1:4]
    n_1 = np.cross(t_1[1] - t_1[0], t_1[2] - t_1[1])
    n_1 /= np.linalg.norm(n_1)
    t_1 = t_1 if np.all(n_1 == f_normal) else t_1[::-1]

    triangles[0] = t_0
    triangles[1] = t_1

    normals[0] = n_0
    normals[1] = n_1

    return triangles, normals
