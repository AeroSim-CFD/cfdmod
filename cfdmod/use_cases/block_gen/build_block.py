import numpy as np

from cfdmod.use_cases.block_gen import BlockParams

__all__ = [
    "build_single_block",
]


def _find_triangle_indices(vertices: np.ndarray, triangle_vertices: np.ndarray) -> list[int]:
    """Find the indices of vertices in the given triangle

    Args:
        vertices (np.ndarray): List of vertices in the block
        triangle_vertices (np.ndarray): List of vertices in the triangle

    Returns:
        list[int]: List of indices of vertices matched
    """
    indices: list[int] = []
    for vertex in triangle_vertices:
        idx = np.where(np.all(vertices == vertex, axis=1))[0]
        if idx.size > 0:
            indices.append(idx[0])
        else:
            return None
    return indices


def _triangulate_block(vertices: np.ndarray, block_params: BlockParams) -> np.ndarray:
    """Triangulates the given vertices of a block

    Args:
        vertices (np.ndarray): Array of block vertices
        block_params (BlockParams): Object with block parameters

    Returns:
        np.ndarray: Array of triangles
    """
    triangles = []
    for face_normal in [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
    ]:
        normal_idx = [i for i, component in enumerate(face_normal) if component != 0][0]
        match normal_idx:
            case 0:
                size = block_params.length
            case 1:
                size = block_params.width
            case _:
                size = block_params.height
        plane_origin = size if face_normal[normal_idx] > 0 else 0
        face_v = list(filter(lambda v: v[normal_idx] == plane_origin, vertices))
        face_v.sort(key=lambda v: (v[0], v[1], v[2]))

        t_0 = face_v[0:3]
        n_0 = np.cross(t_0[1] - t_0[0], t_0[2] - t_0[1])
        n_0 /= np.linalg.norm(n_0)
        t_0 = t_0 if np.all(n_0 == face_normal) else t_0[::-1]

        t_1 = face_v[1:4]
        n_1 = np.cross(t_1[1] - t_1[0], t_1[2] - t_1[1])
        n_1 /= np.linalg.norm(n_1)
        t_1 = t_1 if np.all(n_1 == face_normal) else t_1[::-1]

        triangles.append(_find_triangle_indices(vertices, t_0))
        triangles.append(_find_triangle_indices(vertices, t_1))
    triangles = np.array(triangles)

    return triangles


def build_single_block(block_params: BlockParams) -> tuple[np.ndarray, np.ndarray]:
    """Builds a single block

    Args:
        block_params (BlockParams): Object with block parameters

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with vertices and triangles
    """
    vertices = []
    x = np.linspace(0, block_params.length, 2)
    y = np.linspace(0, block_params.width, 2)
    z = np.linspace(0, block_params.height, 2)
    xv, yv, zv = np.meshgrid(x, y, z)

    for coordinate in list(zip(xv.flatten(), yv.flatten(), zv.flatten())):
        vertices.append(coordinate)

    vertices = np.array(vertices)

    triangles = _triangulate_block(vertices, block_params)

    return vertices, triangles
