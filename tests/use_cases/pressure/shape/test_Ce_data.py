import numpy as np
import pandas as pd
import pytest
from lnas import LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.geometry import GeometryData, tabulate_geometry_data
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig, ZoningConfig
from cfdmod.use_cases.pressure.shape.Ce_data import (
    calculate_statistics,
    get_region_definition_dataframe,
    process_surfaces,
    transform_Ce,
)
from cfdmod.use_cases.pressure.statistics import BasicStatisticModel
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel
from cfdmod.utils import convert_dataframe_into_matrix


@pytest.fixture()
def mesh():
    yield LnasGeometry(
        vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
        triangles=np.array([[0, 1, 2], [2, 1, 3]]),
    )


@pytest.fixture()
def cp_data():
    yield pd.DataFrame(
        {
            "point_idx": [0, 0, 0, 1, 1, 1],
            "cp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "time_normalized": [0, 1, 2, 0, 1, 2],
        }
    )


@pytest.fixture()
def matrix_cp_data(cp_data):
    yield convert_dataframe_into_matrix(
        cp_data, row_data_label="time_normalized", value_data_label="cp"
    )


@pytest.fixture()
def zoning():
    zoning = ZoningModel(x_intervals=[0, 5, 10])
    zoning.offset_limits(0.1)
    yield zoning


def test_get_region_definition_dataframe(mesh, zoning):
    geom_dict = {
        "sfc1": GeometryData(mesh=mesh, zoning_to_use=zoning, triangles_idxs=np.array([0, 1]))
    }
    region_df = get_region_definition_dataframe(geom_dict)

    assert [
        f"{i}-{sfc_id}" in region_df["region_idx"]
        for i in range(len(zoning.get_regions()))
        for sfc_id in geom_dict.keys()
    ]


def test_transform_Ce(matrix_cp_data, zoning, mesh, cp_data):
    geom_dict = {
        "sfc1": GeometryData(mesh=mesh, zoning_to_use=zoning, triangles_idxs=np.array([0, 1]))
    }
    geometry_df = tabulate_geometry_data(
        geom_dict,
        mesh_areas=mesh.areas,
        mesh_normals=mesh.normals,
        transformation=TransformationConfig(),
    )
    ce_data = transform_Ce(matrix_cp_data, geometry_df, mesh)

    assert (
        len(ce_data) == cp_data["time_normalized"].nunique() * cp_data.point_idx.nunique()
    )  # Three timesteps x 2 triangle
    assert "Ce" in ce_data.columns


def test_process_surfaces(mesh, zoning):
    geom_dict = {
        "sfc1": GeometryData(mesh=mesh, zoning_to_use=zoning, triangles_idxs=np.array([0, 1]))
    }
    region_data = convert_dataframe_into_matrix(
        pd.DataFrame(
            {
                "region_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "time_normalized": [0, 1, 2, 3, 0, 1, 2, 3],
                "Ce": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            }
        ),
        row_data_label="time_normalized",
        column_data_label="region_idx",
        value_data_label="Ce",
    )
    cfg = CeConfig(
        statistics=[
            BasicStatisticModel(stats="mean"),
            BasicStatisticModel(stats="rms"),
            BasicStatisticModel(stats="skewness"),
            BasicStatisticModel(stats="kurtosis"),
        ],
        zoning=ZoningConfig(global_zoning=zoning),
        sets={},
        transformation=TransformationConfig(),
    )
    ce_stats = calculate_statistics(
        historical_data=region_data, statistics_to_apply=cfg.statistics
    )
    processed_sfcs, processed_df = process_surfaces(
        geometry_dict=geom_dict, cfg=cfg, ce_stats=ce_stats
    )

    assert len(processed_sfcs) == len(geom_dict)
