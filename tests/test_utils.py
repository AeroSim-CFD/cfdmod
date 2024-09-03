import numpy as np
import pandas as pd
import pytest

from cfdmod.utils import convert_dataframe_into_matrix, convert_matrix_into_dataframe


@pytest.fixture()
def dataframe_timeseries():
    n_points, n_timesteps = 200, 100
    df_dict = {
        "point_idx": np.tile(np.array(range(n_points), dtype=np.int32), n_timesteps),
        "rho": np.random.uniform(0.99, 1.01, n_timesteps * n_points),
        "time_step": np.repeat(np.linspace(0, 100, n_timesteps), n_points),
    }
    yield pd.DataFrame(data=df_dict)


def test_conversion(dataframe_timeseries):
    matrix_from_dataframe = convert_dataframe_into_matrix(dataframe_timeseries)
    dataframe_from_matrix = convert_matrix_into_dataframe(matrix_from_dataframe)
    matrix_from_converted_dataframe = convert_dataframe_into_matrix(dataframe_from_matrix)

    assert dataframe_from_matrix.equals(dataframe_timeseries)
    assert matrix_from_converted_dataframe.equals(matrix_from_dataframe)
