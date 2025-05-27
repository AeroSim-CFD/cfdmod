import numpy as np
import pandas as pd
import pytest
from lnas import LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.geometry import GeometryData, tabulate_geometry_data
from cfdmod.use_cases.pressure.moment.Cm_data import add_lever_arm_to_geometry_df, transform_Cm
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel
from cfdmod.utils import convert_dataframe_into_matrix


@pytest.fixture()
def body_data():
    body_data = pd.DataFrame(
        {
            "cp": [0.1, 0.2, 0.3, 0.4],
            "time_normalized": [0, 0, 1, 1],
            "point_idx": [0, 1, 0, 1],
        }
    )
    yield convert_dataframe_into_matrix(
        body_data, row_data_label="time_normalized", value_data_label="cp"
    )


@pytest.fixture()
def body_geom():
    vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    yield LnasGeometry(vertices, triangles)


@pytest.fixture()
def geom_data(body_geom):
    yield GeometryData(
        mesh=body_geom, zoning_to_use=ZoningModel(), triangles_idxs=np.array([0, 1])
    )


@pytest.fixture()
def geometry_df(geom_data, body_geom):
    geometry_dict = {"body": geom_data}
    yield tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=body_geom.areas,
        mesh_normals=body_geom.normals,
        transformation=TransformationConfig(),
    )


def test_add_lever_arm(geom_data, geometry_df):
    geometry_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        lever_origin=[0, 0, 10],
        geometry_df=geometry_df,
    )

    assert all([f"r{dir}" in geometry_df.columns for dir in ["x", "y", "z"]])


def test_transform_Cm(geom_data, body_data, body_geom, geometry_df):
    geometry_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        lever_origin=[0, 0, 10],
        geometry_df=geometry_df,
    )
    Cm_data = transform_Cm(raw_cp=body_data, geometry_df=geometry_df, geometry=body_geom, nominal_volume=10)

    assert Cm_data.notna().all().all()
    assert all([f"Cm{dir}" in Cm_data.columns for dir in ["x", "y", "z"]])
