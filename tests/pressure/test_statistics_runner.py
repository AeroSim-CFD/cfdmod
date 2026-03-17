"""Tests for calculate_statistics_from_h5 (streaming stats)."""

import pathlib

import numpy as np
import pytest

from cfdmod.pressure.parameters import (
    BasicStatisticModel,
    ExtremeAbsoluteParamsModel,
    ParameterizedStatisticModel,
)
from cfdmod.pressure.statistics_runner import calculate_statistics_from_h5

BODY_H5 = pathlib.Path("fixtures/tests/pressure/xdmf/bodies.building.h5")


def test_calculate_statistics_from_h5_basic():
    statistics = [
        BasicStatisticModel(stats="mean"),
        BasicStatisticModel(stats="rms"),
    ]
    stats_df = calculate_statistics_from_h5(
        h5_path=BODY_H5,
        group="pressure",
        statistics=statistics,
    )
    assert "mean" in stats_df.columns
    assert "rms" in stats_df.columns
    assert len(stats_df) == 51593
    assert not stats_df.isnull().values.any()


def test_calculate_statistics_from_h5_with_range():
    statistics = [BasicStatisticModel(stats="mean")]
    all_stats = calculate_statistics_from_h5(
        h5_path=BODY_H5, group="pressure", statistics=statistics
    )
    partial_stats = calculate_statistics_from_h5(
        h5_path=BODY_H5,
        group="pressure",
        statistics=statistics,
        timestep_range=(125.0, 128.0),
    )
    assert len(all_stats) == len(partial_stats)
    # Partial range mean will differ from full range
    assert not np.allclose(all_stats["mean"].values, partial_stats["mean"].values)


def test_calculate_statistics_from_h5_with_absolute():
    statistics = [
        BasicStatisticModel(stats="mean"),
        ParameterizedStatisticModel(stats="min", params=ExtremeAbsoluteParamsModel()),
        ParameterizedStatisticModel(stats="max", params=ExtremeAbsoluteParamsModel()),
    ]
    stats_df = calculate_statistics_from_h5(
        h5_path=BODY_H5,
        group="pressure",
        statistics=statistics,
        timestep_range=(125.0, 128.0),
    )
    assert "mean" in stats_df.columns
    assert "min" in stats_df.columns
    assert "max" in stats_df.columns
    assert (stats_df["min"] <= stats_df["mean"]).all()
    assert (stats_df["mean"] <= stats_df["max"]).all()
