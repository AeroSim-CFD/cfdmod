import numpy as np

from cfdmod.use_cases.block_gen import BlockParams

__all__ = [
    "build_single_block",
]


def build_single_block(block_params: BlockParams) -> tuple[np.ndarray, np.ndarray]:
    """Builds a single block

    Args:
        block_params (BlockParams): Object with block parameters

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with triangles and normals (STL representation)
    """
    vertices = []
    x = np.linspace(0, block_params.length, 2)
    y = np.linspace(0, block_params.width, 2)
    z = np.linspace(0, block_params.height, 2)
    xv, yv, zv = np.meshgrid(x, y, z)

    for coordinate in list(zip(xv.flatten(), yv.flatten(), zv.flatten())):
        vertices.append(coordinate)

    vertices = np.array(vertices)

    triangles, normals = _triangulate_block(vertices, block_params)

    return triangles, normals


def _triangulate_block(
    vertices: np.ndarray, block_params: BlockParams
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulates the given vertices of a block

    Args:
        vertices (np.ndarray): Array of block vertices
        block_params (BlockParams): Object with block parameters

    Returns:
        tuple[np.ndarray, np.ndarray]: STL representation of the block (triangles, normals)
    """
    n_triangles = 10  # Number of triangles in 5 faces. Each face has 2 triangles

    triangles = np.empty((n_triangles, 3, 3), dtype=np.float32)
    normals = np.empty((n_triangles, 3), dtype=np.float32)
    face_normals = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
    ]

    for f_idx, f_normal in enumerate(face_normals):
        normal_idx = [i for i, component in enumerate(f_normal) if component != 0][0]
        match normal_idx:
            case 0:
                size = block_params.length
            case 1:
                size = block_params.width
            case _:
                size = block_params.height

        plane_origin = size if f_normal[normal_idx] > 0 else 0
        face_verts = list(filter(lambda v: v[normal_idx] == plane_origin, vertices))
        face_verts.sort(key=lambda v: (v[0], v[1], v[2]))

        t_0 = face_verts[0:3]
        n_0 = np.cross(t_0[1] - t_0[0], t_0[2] - t_0[1])
        n_0 /= np.linalg.norm(n_0)
        t_0 = t_0 if np.all(n_0 == f_normal) else t_0[::-1]

        t_1 = face_verts[1:4]
        n_1 = np.cross(t_1[1] - t_1[0], t_1[2] - t_1[1])
        n_1 /= np.linalg.norm(n_1)
        t_1 = t_1 if np.all(n_1 == f_normal) else t_1[::-1]

        # Each face has 2 triangles so it must be accounted when indexing
        triangles[f_idx * 2] = t_0
        triangles[f_idx * 2 + 1] = t_1

        normals[f_idx * 2] = n_0
        normals[f_idx * 2 + 1] = n_1

    return triangles, normals
