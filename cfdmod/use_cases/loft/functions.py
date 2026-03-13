from __future__ import annotations

import numpy as np

import lnas

def flatten_vertices_and_get_triangles_as_list_of_indexes(
    triangle_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Takes a set of faces defined as 3 vertices with coordinates and gives back the vertices separated in and the
    triangles specified by the vertices indexes

    Args:
        triangle_vertices[a,b,c] (np.ndarray): 3d numpy array. Coordinates ->
            0 - number of triangle (len = number of triangles)
            1 - number of vertice (len = 3)
            2 - coordinate of verice (len = 3)

    Returns:
        tuple[np.ndarray, np.ndarray]: Flattened vertices and triangles specified as vertices indexes:
            - flattened_vertices[d,c] -> number of vertices, vertice coordinate
            - triangles[a,c] -> number of triangle, number of triangle vertice (3)
    """

    def _get_float_as_int(v: float) -> int:
        return int(v * 10**decimals)

    def _get_as_key(v: np.ndarray) -> tuple[int]:
        return tuple(_get_float_as_int(vv) for vv in v)

    s = triangle_vertices.shape
    flattened_vertices = triangle_vertices.reshape((s[0] * s[1], 3))

    # Round for comparison
    decimals = 10

    flat_indexes = {_get_as_key(v): i for i, v in enumerate(flattened_vertices)}

    # Indexed as [t_idx, edge_idx] = (v0, v1)
    tri_index_matrix = np.empty((s[0], 3), dtype=np.uint32)

    for t_idx, tri in enumerate(triangle_vertices):
        v_idxs = []
        for v in tri:
            key = _get_as_key(v)
            val = flat_indexes[key]
            v_idxs.append(val)
        tri_index_matrix[t_idx] = v_idxs

    return flattened_vertices, tri_index_matrix


def find_borders(triangle_vertices: np.ndarray) -> np.ndarray:
    """Identify edges of border, based on repetition of edges

    Args:
        triangle_vertices (np.ndarray): vertices indexes

    Returns:
        set[tuple[int,int]]: selected set of edges, indetified by vertices indexes
    """
    n_triangles = len(triangle_vertices)
    triangles_edges = np.empty((n_triangles, 3, 2), dtype=np.uint32)
    for t_idx, tri in enumerate(triangle_vertices):
        triangles_edges[t_idx] = [
            tuple(sorted([tri[id_0], tri[id_1]])) for id_0, id_1 in [(0, 1), (0, 2), (1, 2)]
        ]
    flat_edges = triangles_edges.reshape(n_triangles * 3, 2)
    flat_edges_tp = [tuple(edge) for edge in flat_edges]

    unseen_edges = set(flat_edges_tp)
    unique_edges = set()
    for edge in flat_edges_tp:
        if edge in unseen_edges:
            unseen_edges.remove(edge)
            unique_edges.add(edge)
        elif edge in unique_edges:
            unique_edges.remove(edge)

    return np.array(list(unique_edges))


def remove_edges_of_internal_holes(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Remove border vertices comming from internal holes

    Args:
        border_verts (np.ndarray): All border vertices
        radius (float): Internal radius where to ignore edges

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated filtered border vertices and border edges
        that are not from internal holes
    """
    edges_by_vertex = [set() for v in range(0, vertices.shape[0])]
    vertices_to_analyze = set()
    for edge in edges:
        edges_by_vertex[edge[0]].add(tuple(edge))
        vertices_to_analyze.add(int(edge[0]))
        edges_by_vertex[edge[1]].add(tuple(edge))
        vertices_to_analyze.add(int(edge[1]))
    groups = []
    while len(vertices_to_analyze) > 0:
        group = {"vertices_id": set(), "edges": set()}
        current_vertice = vertices_to_analyze.pop()
        while len(edges_by_vertex[current_vertice]) > 0:
            current_edge = edges_by_vertex[current_vertice].pop()
            group["vertices_id"].add(current_vertice)
            group["edges"].add(current_edge)
            next_vertice = (
                current_edge[0] if current_edge[1] == current_vertice else current_edge[1]
            )
            edges_by_vertex[next_vertice].remove(current_edge)
            vertices_to_analyze.discard(next_vertice)
            current_vertice = next_vertice
        groups.append(group)

    # find group with biggest diameter
    groups_diameter = []
    for group in groups:
        vertices_id = list(group["vertices_id"])
        x = [vertices[v, 0] for v in vertices_id]
        y = [vertices[v, 1] for v in vertices_id]
        z = [vertices[v, 2] for v in vertices_id]
        x_ampl = max(x) - min(x)
        y_ampl = max(y) - min(y)
        z_ampl = max(z) - min(z)
        groups_diameter.append(x_ampl**2 + y_ampl**2 + z_ampl**2)
    max_diam = max(groups_diameter)
    max_diam_group_id = groups_diameter.index(max_diam)
    biggest_group = groups[max_diam_group_id]

    return np.array(list(biggest_group["edges"]))

def generate_loft_triangles(
    vertices: np.ndarray,
    edges: np.ndarray,
    loft_radius: float,
    mesh_center: np.ndarray,
    loft_z_pos: float,
) -> lnas.LnasGeometry:
    """Generates STL of loft

    Note that terrain and loft vertices are assumed to be ordered and aligned correctly

    Args:
        border_profile (np.ndarray): Vertices in terrain
        loft_vertices (np.ndarray): Vertices generated for loft

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with loft surface triangles and normals
    """

    def get_distance_from_center_in_the_projection_direction(
        vertices: np.ndarray, mesh_center: np.ndarray, projection_diretion: np.ndarray
    ) -> np.ndarray:
        vert_dir_from_center = vertices - mesh_center
        return np.dot(vert_dir_from_center, projection_diretion)

    def normal_of_triangles(triangles: np.ndarray) -> np.ndarray:
        v0 = triangles[:, 0, :].squeeze()
        v1 = triangles[:, 1, :].squeeze()
        v2 = triangles[:, 2, :].squeeze()
        u = v1 - v0
        v = v2 - v0
        return np.cross(u, v)

    num_edges_on_border = edges.shape[0]
    vertices_on_boder_id = edges.reshape(num_edges_on_border * 2)
    vertices_on_border = vertices[vertices_on_boder_id, :]
    loft_distantce_start = np.max(
        get_distance_from_center_in_the_projection_direction(
            vertices=vertices_on_border,
            mesh_center=mesh_center,
            projection_diretion=projection_diretion,
        )
    )
    loft_distantce_end = loft_distantce_start + loft_length

    edge_verts = []
    edge_verts.append(vertices[edges[:, 0]])
    edge_verts.append(vertices[edges[:, 1]])
    edge_verts_projection = []
    for vertices in edge_verts:
        distance_of_vertices_from_center = get_distance_from_center_in_the_projection_direction(
            vertices=vertices,
            mesh_center=mesh_center,
            projection_diretion=projection_diretion,
        )
        distance_to_project = loft_distantce_end - distance_of_vertices_from_center

        projected_vertices = vertices + distance_to_project[:, np.newaxis] * projection_diretion
        projected_vertices[:, 2] = loft_z_pos
        edge_verts_projection.append(projected_vertices)

    triangles_0 = np.stack([edge_verts[0], edge_verts[1], edge_verts_projection[1]], axis=1)
    triangles_1 = np.stack(
        [edge_verts[0], edge_verts_projection[1], edge_verts_projection[0]], axis=1
    )
    full_triangles = np.concatenate([triangles_0, triangles_1], axis=0)
    full_normals = normal_of_triangles(full_triangles)
    mask_inverted_nomals = (full_normals[:, 2]).squeeze() < 0
    corrected_triangles = full_triangles.copy()
    corrected_triangles[mask_inverted_nomals, 0, :] = full_triangles[mask_inverted_nomals, 1, :]
    corrected_triangles[mask_inverted_nomals, 1, :] = full_triangles[mask_inverted_nomals, 0, :]
    corrected_normals = normal_of_triangles(corrected_triangles)

    lnas_geom = lnas.LnasFormat.from_triangles(triangles=corrected_triangles, normals=corrected_normals).geometry
    return lnas_geom


def generate_loft_surface(
    geom: lnas.LnasGeometry,
    projection_angle_rad: float,
    loft_radius: float,
    loft_z_pos: float,
) -> lnas.LnasGeometry:
    """Generate loft surface (triangles and normals)

    Args:
        triangle_vertices (np.ndarray): Mesh triangle vertices
        projection_diretion (np.ndarray): Direction of loft projection
        loft_length (float): Minimum length of loft
        loft_z_pos (float): Target z position
        filter_radius (float): Internal filter radius to ignore edges

    Returns:
        tuple[np.ndarray, np.ndarray]: Loft triangles and normals
    """

    geom = geom.copy()
    center = np.array(
        [
            (geom.triangle_vertices[:, 0].max() + geom.triangle_vertices[:, 0].min()) / 2,
            (geom.triangle_vertices[:, 1].max() + geom.triangle_vertices[:, 1].min()) / 2,
            (geom.triangle_vertices[:, 2].max() + geom.triangle_vertices[:, 2].min()) / 2,
        ]
    )

    transf = lnas.transformations.TransformationsMatrix(angle=np.array((0, 0, projection_angle_rad)), fixed_point=center)
    geom.apply_transformation(transf=transf)

    flattened_vertices, tri_index_matrix = flatten_vertices_and_get_triangles_as_list_of_indexes(
        triangle_vertices=geom.triangle_vertices,
    )

    border_edges = find_borders(triangle_vertices=tri_index_matrix)

    border_edges = remove_edges_of_internal_holes(
        vertices=flattened_vertices,
        edges=border_edges,
    )

    center = np.array(
        [
            (flattened_vertices[:, 0].max() + flattened_vertices[:, 0].min()) / 2,
            (flattened_vertices[:, 1].max() + flattened_vertices[:, 1].min()) / 2,
            (flattened_vertices[:, 2].max() + flattened_vertices[:, 2].min()) / 2,
        ]
    )

    loft_geom = generate_loft_triangles(
        vertices=flattened_vertices,
        edges=border_edges,
        loft_radius=loft_radius,
        loft_z_pos=loft_z_pos,
        mesh_center=center,
    )

    return loft_geom
