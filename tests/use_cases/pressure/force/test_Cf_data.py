import numpy as np
import pandas as pd
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.force.Cf_data import transform_Cf
from cfdmod.use_cases.pressure.geometry import GeometryData, tabulate_geometry_data
from cfdmod.use_cases.pressure.statistics import BasicStatisticModel
from cfdmod.use_cases.pressure.zoning.processing import calculate_statistics
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel
from cfdmod.utils import convert_dataframe_into_matrix


@pytest.fixture()
def cp_data():
    cp_data = pd.DataFrame(
        {
            "point_idx": [0, 0, 0, 1, 1, 1],
            "cp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "time_normalized": [0, 1, 2, 0, 1, 2],
        }
    )
    yield convert_dataframe_into_matrix(
        cp_data, row_data_label="time_normalized", value_data_label="cp"
    )


@pytest.fixture()
def body_geom():
    vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    yield LnasGeometry(vertices, triangles)


def test_transform_to_Cf(body_geom, cp_data):
    geom_data = GeometryData(
        mesh=body_geom, zoning_to_use=ZoningModel(), triangles_idxs=np.array([0, 1])
    )
    geometry_dict = {"body": geom_data}

    geometry_df = tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=body_geom.areas,
        mesh_normals=body_geom.normals,
        transformation=TransformationConfig(),
    )
    cf_data = transform_Cf(cp_data, geometry_df, body_geom, nominal_area=10)

    assert (
        len(cf_data) == cp_data.time_normalized.nunique() * geometry_df.region_idx.nunique()
    )  # Three timesteps x 1 region
    assert all([f"Cf{var}" in cf_data.columns for var in ["x", "y", "z"]])


def test_liquid_coefficients(body_geom):
    cp_data = pd.DataFrame(
        {
            "point_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "cp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            "time_normalized": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )

    upper_mesh = LnasFormat(version="", geometry=body_geom, surfaces={"sfc1": np.array([0, 1])})
    lower_mesh = upper_mesh.copy()
    geometry_negative = lower_mesh.geometry
    geometry_negative.triangles = np.flip(geometry_negative.triangles, 1)
    geometry_negative._full_update()
    geometry_negative.vertices = (
        geometry_negative.vertices + geometry_negative.vertices_normals * 0.1
    )
    upper_mesh.join(lnas_fmts=[lower_mesh], surfaces_suffixes=["_lower"])
    geom_data = GeometryData(
        mesh=upper_mesh.geometry,
        zoning_to_use=ZoningModel(),
        triangles_idxs=np.array([0, 1, 2, 3]),
    )
    geometry_dict = {"body": geom_data}

    geometry_df = tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=upper_mesh.geometry.areas,
        mesh_normals=upper_mesh.geometry.normals,
        transformation=TransformationConfig(),
    )
    cp_data = convert_dataframe_into_matrix(
        cp_data, row_data_label="time_normalized", value_data_label="cp"
    )
    cf_data = transform_Cf(cp_data, geometry_df, upper_mesh.geometry, nominal_area=10)
    cf_data = convert_dataframe_into_matrix(
        cf_data,
        row_data_label="time_normalized",
        column_data_label="region_idx",
        value_data_label="Cfz",
    )
    calculate_statistics(
        historical_data=cf_data,
        statistics_to_apply=[
            BasicStatisticModel(stats="mean"),
            BasicStatisticModel(stats="rms"),
            BasicStatisticModel(stats="skewness"),
            BasicStatisticModel(stats="kurtosis"),
        ],
    )
