import pathlib
from lnas import LnasFormat
import numpy as np
import pandas as pd
import trimesh

lnas_path = "output/low_rise_building_Rabat_Sale.lnas"
points_path = "output/maroco_points.csv"

lnas_fmt = LnasFormat.from_file(pathlib.Path(lnas_path))
lnas_geo = lnas_fmt.geometry

tri_up = lnas_geo.normals[:, 2] > 0.5
tri_down = lnas_geo.normals[:, 2] < -0.5
tri_up_idxs = np.arange(len(tri_up))[tri_up]
tri_down_idxs = np.arange(len(tri_down))[tri_down]

lnas_up = lnas_fmt.filter_triangles(tri_up)
lnas_down = lnas_fmt.filter_triangles(tri_down)

points = pd.read_csv(points_path, sep=",")
arr_points = np.array([points[d].to_numpy() for d in ["x", "y", "z"]], dtype=np.float32)
arr_points = arr_points.swapaxes(0, 1)

mesh_up = trimesh.Trimesh(vertices=lnas_up.geometry.vertices.copy(), faces=lnas_up.geometry.triangles.copy())
mesh_down = trimesh.Trimesh(vertices=lnas_down.geometry.vertices.copy(), faces=lnas_down.geometry.triangles.copy())

def check_intersection(mesh, ray_origin, ray_direction):
    # Create a ray from start_point to end_point
    ray = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    intersects, index_tri, index_ray = ray.intersects_location([ray_origin], [ray_direction])

    if intersects.any():
        # If any intersection is found, return the indices of triangles intersected
        return intersects, index_tri, index_ray
    raise ValueError("not interecpts")

normals = []

for idx, p in enumerate(arr_points):
    mesh_use = mesh_up
    if(p[2] == 10):
        mesh_use = mesh_down
        normals.append([0, 0, 1])
    else:
        normals.append([0, 0, -1])
    ray_origin = [p[0], p[1], 0]
    ray_direction = [0, 0, 1]

    ints, tri_int, index_ray = check_intersection(mesh_use, ray_origin, ray_direction)

    arr_points[idx, 2] = ints[0, 2]

normals = np.array(normals, dtype=np.float32)

df_save = points.copy()
df_save["x"] = arr_points[:, 0]
df_save["y"] = arr_points[:, 1]
df_save["z"] = arr_points[:, 2]
df_save["nx"] = normals[:, 0]
df_save["ny"] = normals[:, 1]
df_save["nz"] = normals[:, 2]


df_save.to_csv("output/maroco_real_points.csv", index=False)


