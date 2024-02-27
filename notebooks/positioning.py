import numpy as np
import vtk
import sys
import pathlib

from nassu.lnas import LagrangianFormat, LagrangianGeometry
from cfdmod.api.vtk.write_vtk import _mkVtkIdList


def join_multiple_geometries(filenames: list[pathlib.Path]) -> LagrangianGeometry:
    geometries: list[LagrangianGeometry] = []
    for f in filenames:
        lnas_fmt = LagrangianFormat.from_file(f)
        geometries.append(lnas_fmt.geometry.copy())

    geometry_ret = LagrangianGeometry(
        vertices=geometries[0].vertices, triangles=geometries[0].triangles
    )
    for g in geometries[1:]:
        idx_offset = len(geometry_ret.vertices)
        new_triangles = g.triangles + idx_offset
        geometry_ret.vertices = np.concatenate((geometry_ret.vertices, g.vertices), axis=0)
        geometry_ret.triangles = np.concatenate((geometry_ret.triangles, new_triangles), axis=0)
    return geometry_ret


def create_polydata(mesh: LagrangianGeometry) -> vtk.vtkPolyData:
    """Creates a vtkPolyData for cell data combined with mesh description

    Args:
        data (pd.DataFrame): Compiled cell data
        mesh (LagrangianGeometry): Mesh description

    Returns:
        vtkPolyData: Extracted polydata
    """

    # We'll create the building blocks of polydata including data attributes.
    polyData = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()

    # Load the point, cell, and data attributes.
    for i, xi in enumerate(mesh.vertices):
        points.InsertPoint(i, xi)
    for pt in mesh.triangles:
        polys.InsertNextCell(_mkVtkIdList(pt))

    # We now assign the pieces to the vtkPolyData.
    polyData.SetPoints(points)
    polyData.SetPolys(polys)

    return polyData


def get_connected_vertices(geometry: LagrangianGeometry) -> list[set[int]]:
    verts = geometry.vertices
    triangles = geometry.triangles

    # Set in which vertice is z
    vertices_set: list[int | None] = [None for _ in range(len(verts))]
    # Groups of sets of vertices
    sets_found: dict[int, set[int]] = {}
    n_sets = 0

    def join_sets(v1: int, v2: int):
        nonlocal n_sets, sets_found, vertices_set
        s1 = vertices_set[v1]
        s2 = vertices_set[v2]
        # Already joined
        if s1 == s2 and s1 is not None:
            return
        set1 = sets_found.pop(s1) if s1 is not None else set((v1,))
        set2 = sets_found.pop(s2) if s2 is not None else set((v2,))
        new_set = set1 | set2
        sets_found[n_sets] = new_set
        for v in new_set:
            vertices_set[v] = n_sets
        n_sets += 1

    for t in triangles:
        for i in range(1, 3):
            join_sets(t[i - 1], t[i])

    list_sets = list(sets_found.values())
    return list_sets


def get_line_xy(xy: tuple[float, float]) -> vtk.vtkPolyData:
    # from https://examples.vtk.org/site/Python/GeometricObjects/PolyLine/
    p0 = (xy[0], xy[1], -10_000)
    p1 = (xy[0], xy[1], 10_000)
    points = vtk.vtkPoints()
    points.InsertNextPoint(p0)
    points.InsertNextPoint(p1)

    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(2)
    for i in range(2):
        polyLine.GetPointIds().SetId(i, i)

    # Create a cell array to store the lines in and add the lines to it
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyLine)

    # Create a polydata to store everything in
    polyData = vtk.vtkPolyData()

    # Add the points to the dataset
    polyData.SetPoints(points)

    # Add the lines to the dataset
    polyData.SetLines(cells)

    return polyData


def intersection_xy_line_polydata(line: vtk.vtkPolyData, terrain: vtk.vtkPolyData) -> float | None:
    filter_use = vtk.vtkIntersectionPolyDataFilter()
    filter_use.AddInputData(line)
    filter_use.AddInputData(terrain)
    filter_use.Update()

    output = filter_use.GetOutput()
    return None


def main():
    filename_elements = pathlib.Path("./output/my-rough/cube.lnas")
    filenames_terrain = [
        pathlib.Path("./fixtures/tests/roughness_gen/disk/disk.lnas"),
        pathlib.Path("./fixtures/tests/roughness_gen/loft/loft.lnas"),
    ]

    terrain = join_multiple_geometries(filenames_terrain)
    elems = LagrangianFormat.from_file(filename_elements).geometry

    vtk_terrain = create_polydata(terrain)
    vtk_elems = create_polydata(elems)

    connected_verts = get_connected_vertices(elems)
    print(len(connected_verts))
    print([len(s) for s in connected_verts])

    # points = [(10, 10), (100, 100), (0, 0)]
    # for p in points:
    #     line = get_line_xy(p)
    #     p_intersec = intersection_xy_line_polydata(line, vtk_terrain)
    #     print("p", p_intersec)


if __name__ == "__main__":
    main()
