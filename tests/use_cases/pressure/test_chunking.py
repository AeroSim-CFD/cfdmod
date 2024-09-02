import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from lnas import LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.chunking import (
    HDFGroupInterface,
    process_timestep_groups,
    split_into_chunks,
)
from cfdmod.use_cases.pressure.geometry import GeometryData, tabulate_geometry_data
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel
from cfdmod.utils import convert_matrix_into_dataframe


def _mock_processing_function(
    cp_df: pd.DataFrame, _geom_df: pd.DataFrame, _geom: LnasGeometry
) -> pd.DataFrame:
    cols_points = [c for c in cp_df.columns if c != "time_step"]
    result = cp_df.copy()
    result[cols_points] *= 2

    return result


@pytest.fixture()
def sample_df():
    time_series_data = {
        "time_step": np.repeat(range(100), 100),
        "value": range(10000),
        "point_idx": np.tile(range(100), 100),
    }
    yield pd.DataFrame(time_series_data)


@pytest.fixture()
def geometry():
    yield LnasGeometry(
        vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
        triangles=np.array([[0, 1, 2], [2, 1, 3]]),
    )


@pytest.fixture()
def output_path():
    output_path = pathlib.Path("./output/chunking.h5")
    if output_path.exists():
        output_path.unlink()
    yield output_path


@pytest.fixture()
def zoning():
    zoning = ZoningModel(x_intervals=[0, 5, 10])
    zoning.offset_limits(0.1)
    yield zoning


@pytest.mark.parametrize("number_of_chunks", [2, 4, 5])
def test_raises_error_if_dataframe_lacks_time_step_column(output_path, number_of_chunks):
    time_series_df = pd.DataFrame({"value": [10, 20, 30]})

    with pytest.raises(ValueError) as context:
        split_into_chunks(time_series_df, number_of_chunks, output_path)

    assert str(context.value) == "Time series dataframe must have a time_step column to be chunked"


def test_split_more_than_possible(sample_df, output_path):
    with pytest.raises(ValueError) as context:
        # sample_df has 10 time_step values and it is impossible to split
        # into 6 chunks with at least two steps each
        split_into_chunks(sample_df, 10000, output_path)

    assert str(context.value) == "There must be at least two steps in each chunk"


@pytest.mark.parametrize("number_of_chunks", [2, 4, 5])
def test_split_into_chunks(sample_df, number_of_chunks, output_path):
    time_arr = sample_df.time_step.unique()
    step = math.ceil(len(time_arr) / number_of_chunks)
    expected_keys = [
        HDFGroupInterface.time_key(time_arr[i * step]) for i in range(number_of_chunks)
    ]

    split_into_chunks(sample_df, number_of_chunks, output_path)
    assert output_path.exists()

    with pd.HDFStore(output_path, "r") as store:
        assert len(store.keys()) == number_of_chunks
        assert list(store.keys()) == expected_keys

    output_path.unlink()


@pytest.mark.parametrize("number_of_chunks", [2, 4, 5])
def test_process_timestep_groups(number_of_chunks, geometry, output_path, zoning, sample_df):
    split_into_chunks(sample_df, number_of_chunks, output_path)

    geom_dict = {
        "sfc1": GeometryData(mesh=geometry, zoning_to_use=zoning, triangles_idxs=np.array([0, 1]))
    }
    geometry_df = tabulate_geometry_data(
        geom_dict=geom_dict,
        mesh_areas=geometry.areas,
        mesh_normals=geometry.normals,
        transformation=TransformationConfig(),
    )

    result_df = process_timestep_groups(
        output_path,
        geometry_df,
        geometry,
        _mock_processing_function,
        data_label="value",
        time_column_label="time_step",
    )
    sample_df.sort_values(by=["time_step"], inplace=True)
    result_df = convert_matrix_into_dataframe(result_df, value_data_label="value")
    result_df.sort_values(by=["time_step"], inplace=True)

    assert (result_df.value.to_numpy() == sample_df.value.to_numpy() * 2).all()
    assert result_df.time_step.nunique() == sample_df.time_step.nunique()

    output_path.unlink()


@pytest.mark.parametrize("number_of_chunks", [2, 4, 5])
def test_process_groups_statistics(geometry, zoning, sample_df, output_path, number_of_chunks):
    geom_dict = {
        "sfc1": GeometryData(mesh=geometry, zoning_to_use=zoning, triangles_idxs=np.array([0, 1]))
    }
    geometry_df = tabulate_geometry_data(
        geom_dict=geom_dict,
        mesh_areas=geometry.areas,
        mesh_normals=geometry.normals,
        transformation=TransformationConfig(),
    )
    sample_df.sort_values(by=["time_step", "point_idx"], inplace=True)

    split_into_chunks(sample_df, number_of_chunks, output_path)
    first_df = process_timestep_groups(
        output_path,
        geometry_df,
        geometry,
        _mock_processing_function,
        data_label="value",
        time_column_label="time_step",
    )
    first_df = convert_matrix_into_dataframe(first_df, value_data_label="value")
    avg = first_df.value.mean()
    std = first_df.value.std()
    min_val, max_val = first_df.value.min(), first_df.value.max()

    output_path.unlink()
    sample_df.sort_values(by=["time_step", "point_idx"], inplace=True)

    split_into_chunks(sample_df, number_of_chunks + 2, output_path)
    second_df = process_timestep_groups(
        output_path,
        geometry_df,
        geometry,
        _mock_processing_function,
        data_label="value",
        time_column_label="time_step",
    )
    second_df = convert_matrix_into_dataframe(second_df, value_data_label="value")

    assert avg == second_df.value.mean()
    assert std == second_df.value.std()
    assert min_val == second_df.value.min()
    assert max_val == second_df.value.max()

    output_path.unlink()
