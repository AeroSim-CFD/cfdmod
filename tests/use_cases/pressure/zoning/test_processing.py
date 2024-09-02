import numpy as np
import pandas as pd
import pytest
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.zoning.processing import (
    combine_stats_data_with_mesh,
    get_indexing_mask,
)
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


@pytest.fixture()
def zoning():
    zoning = ZoningModel(x_intervals=[0, 5, 10], y_intervals=[0, 10], z_intervals=[0, 10])
    zoning.offset_limits(0.1)
    yield zoning


@pytest.fixture()
def mesh():
    yield LnasGeometry(
        vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
        triangles=np.array([[0, 1, 2], [2, 1, 3]]),
    )


@pytest.fixture()
def stats():
    region_idx_values = np.array([0, 3], dtype=np.int32)
    data = {
        "region_idx": region_idx_values,
        "mean": [0.5, 0.7],
        "rms": [0.1, 0.15],
        "max": [1.2, 0.08],
        "min": [-0.9, -1.5],
    }
    yield pd.DataFrame(data)


def test_get_indexing_mask(zoning, mesh):
    df_regions = zoning.get_regions_df()
    region_mask = get_indexing_mask(mesh, df_regions)

    assert len(region_mask) == len(mesh.triangles)
    assert region_mask[0] == 0
    assert region_mask[1] == 1


def test_combine_stats_data_with_mesh(zoning, mesh, stats):
    df_regions = zoning.get_regions_df()
    idx_arr = get_indexing_mask(mesh, df_regions)
    result = combine_stats_data_with_mesh(mesh, idx_arr, stats)

    vars_match = ["mean", "rms", "min", "max"]
    points_match = [i in result.point_idx for i in range(len(mesh.triangles))]

    assert all(vars_match)
    assert all(points_match)
