import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry
from vtk import vtkPolyData

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.geometry import (
    GeometryData,
    create_NaN_polydata,
    tabulate_geometry_data,
)
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


@pytest.fixture()
def mesh():
    vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    geometry = LnasGeometry(vertices=vertices, triangles=triangles)
    yield LnasFormat(
        version="",
        geometry=geometry,
        surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])},
    )


def test_no_excluded_surfaces(mesh):
    sfc_list = []
    geom, idx = mesh.geometry_from_list_surfaces(surfaces_names=sfc_list)

    assert len(geom.triangles) == 0
    assert len(idx) == 0


def test_some_excluded_surfaces(mesh):
    sfc_list = ["sfc2"]
    geometry, idx = mesh.geometry_from_list_surfaces(surfaces_names=sfc_list)
    surface_use = mesh.geometry_from_surface(sfc_list[0])

    assert isinstance(geometry, LnasGeometry)  # Expecting a LnasGeometry object
    np.testing.assert_equal(geometry.triangle_vertices, surface_use.triangle_vertices)


def test_create_NaN_polydata(mesh):
    column_labels = ["col1", "col2"]
    result_polydata = create_NaN_polydata(mesh.geometry, column_labels)

    cell_data = result_polydata.GetCellData()
    num_cell_arrays = cell_data.GetNumberOfArrays()
    cell_scalar_names = [cell_data.GetArrayName(i) for i in range(num_cell_arrays)]

    assert isinstance(result_polydata, vtkPolyData)
    assert all(column in cell_scalar_names for column in column_labels)


def test_filter_geometry_from_list(mesh):
    surface_list = ["sfc1"]

    result_geometry, result_triangle_idxs = mesh.geometry_from_list_surfaces(
        surfaces_names=surface_list
    )

    assert isinstance(result_geometry, LnasGeometry)
    assert isinstance(result_triangle_idxs, np.ndarray)

    assert result_triangle_idxs == np.array([0])


def test_combine_geometries():
    geometry_list = [
        LnasGeometry(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            triangles=np.array([[0, 1, 2]]),
        ),
        LnasGeometry(
            vertices=np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]]),
            triangles=np.array([[0, 1, 2]]),
        ),
    ]

    result_geometry = geometry_list[0].copy()
    result_geometry.join(geometries_list=geometry_list[1:])

    assert isinstance(result_geometry, LnasGeometry)
    assert (result_geometry.triangles == np.array([[0, 1, 2], [3, 4, 5]])).all()
    assert len(result_geometry.vertices) == 6


def test_tabulate_geometry(mesh):
    zoning = ZoningModel(x_intervals=[0, 5, 10])
    zoning.offset_limits(0.1)

    geom_dict = {
        "sfc1": GeometryData(
            mesh=mesh.geometry, zoning_to_use=zoning, triangles_idxs=np.array([0, 1])
        )
    }
    transformation = TransformationConfig()
    geometry_df = tabulate_geometry_data(
        geom_dict,
        mesh_areas=mesh.geometry.areas,
        mesh_normals=mesh.geometry.normals,
        transformation=transformation,
    )
    expected_columns = ["region_idx", "point_idx", "area", "n_x", "n_y", "n_z"]

    assert all([prop in geometry_df.columns for prop in expected_columns])
