from nassu.lnas import LagrangianGeometry

from cfdmod.use_cases.pressure.shape.meshing_steps import (
    clean_order_contour_verts,
    combine_and_sort_full_vertices,
    define_cutting_points,
    generate_polyline_edges,
    get_mesh_contour_polyline,
    project_grid_intersection_points,
    triangulate_point_cloud,
)
from cfdmod.use_cases.pressure.shape.regions import ZoningModel


def create_regions_mesh(
    input_mesh: LagrangianGeometry, regions_intervals: ZoningModel, sfc_name
) -> LagrangianGeometry:
    """Create a mesh to represent regions projected into an input mesh

    Args:
        input_mesh (LagrangianGeometry): LNAS Mesh object containing mesh points and triangles
        regions_intervals (ZoningModel): Object to describe regions in x, y and z intervals

    Returns:
        LagrangianGeometry: A new mesh with regions projected into the input mesh
    """
    # polyline_vertices, mesh_bbox = get_mesh_contour_polyline(mesh=input_mesh)
    (polyline_vertices, polyline_edges), mesh_bbox = get_mesh_contour_polyline(mesh=input_mesh)

    polyline_vertices.tofile(f"{sfc_name}.polyline")
    polyline_edges.tofile(f"{sfc_name}.polyline_edges")
    ordered_outline_vertices, normal_index = clean_order_contour_verts(
        mesh_points=input_mesh.vertices,
        mesh_triangles=input_mesh.triangles,
        polyline_vertices=polyline_vertices,
    )
    ordered_outline_vertices.tofile(f"{sfc_name}.outline")

    return input_mesh

    # primitive_poly_edges = generate_polyline_edges(
    #     ordered_outline_vertices=ordered_outline_vertices
    # )
    # points_interceptor = define_cutting_points(
    #     bb_min=mesh_bbox[0],
    #     bb_max=mesh_bbox[1],
    #     poly_edges=primitive_poly_edges,
    #     regions_intervals=regions_intervals,
    #     ordered_outline_vertices=ordered_outline_vertices,
    # )
    # grid_intersection_points = project_grid_intersection_points(points_interceptor)

    # sorted_vertices = combine_and_sort_full_vertices(
    #     intersecting_vertices=points_interceptor.get_all_interception_points(),
    #     ordered_outline_vertices=ordered_outline_vertices,
    #     grid_intersection_points=grid_intersection_points,
    #     normal_index=normal_index,
    # )
    # triangles = triangulate_point_cloud(sorted_vertices, normal_index)
    # # return VTPMesh(points=sorted_vertices, triangles=triangles)
    # return LagrangianGeometry(vertices=sorted_vertices, triangles=triangles)
