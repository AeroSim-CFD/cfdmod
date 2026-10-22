import pathlib

import numpy as np
import pandas as pd
import pytest

from cfdmod.inflow import (
    InflowData,
    NormalizationParameters,
    calculate_autocorrelation,
    calculate_mean_velocity,
    calculate_spectral_density,
    calculate_turbulence_intensity,
    spectral_density_function,
)

pytestmark = pytest.mark.integration


@pytest.fixture()
def inflow_data_dict():
    inflow_data = {}
    folder_path = pathlib.Path("./fixtures/tests/inflow/pitot_inlet")
    inflow_data["old_version"] = InflowData.from_files(
        hist_series_path=folder_path / "hist_series.csv",
        points_path=folder_path / "points.csv",
    )
    folder_path = pathlib.Path("./fixtures/tests/inflow/autocorrelacao_10_h5")
    inflow_data["new_version"] = InflowData.from_files(
        hist_series_path=folder_path / "hist_series.h5",
        points_path=folder_path / "points.csv",
    )
    yield inflow_data


def test_calculate_mean_velocity(inflow_data_dict):
    for inflow_data in inflow_data_dict.values():
        result = calculate_mean_velocity(
            inflow_data=inflow_data, for_components=["ux", "uy", "uz"]
        )
        assert all([f"{c}_mean" in result.columns for c in ["ux", "uy", "uz"]])


def test_calculate_turbulence_intensity(inflow_data_dict):
    for inflow_data in inflow_data_dict.values():
        result = calculate_turbulence_intensity(
            inflow_data=inflow_data, for_components=["ux", "uy", "uz"]
        )
        assert all([f"I_{c}" in result.columns for c in ["ux", "uy", "uz"]])


def test_calculate_spectral_density(inflow_data_dict):
    normalization_params = NormalizationParameters(
        reference_velocity=1.0, characteristic_length=1.0
    )
    for inflow_data in inflow_data_dict.values():
        result = calculate_spectral_density(
            inflow_data=inflow_data,
            target_index=0,
            for_components=["ux", "uy", "uz"],
            normalization_params=normalization_params,
        )
        assert all([f"S ({c})" in result.columns for c in ["ux", "uy", "uz"]])
        assert all([f"f ({c})" in result.columns for c in ["ux", "uy", "uz"]])


def test_calculate_autocorrelation(inflow_data_dict):
    for inflow_data in inflow_data_dict.values():
        result = calculate_autocorrelation(
            inflow_data=inflow_data, anchor_point_idx=0, for_components=["ux", "uy", "uz"]
        )
        assert all([f"coef_{c}" in result.columns for c in ["ux", "uy", "uz"]])


def test_spectral_density():
    timestamps = np.linspace(0, 10, 1000)
    velocity_signal = np.sin(2 * np.pi * 1 * timestamps) + 0.5 * np.random.randn(1000)
    xf, yf = spectral_density_function(velocity_signal, timestamps, 1.0, 1.0)

    assert len(xf) == len(yf)
    assert isinstance(xf, np.ndarray)
    assert isinstance(yf, np.ndarray)


@pytest.mark.unit
def test_calculate_spectral_density_columns_not_swapped():
    t = np.arange(2000) * 0.1
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"time_step": t, "point_idx": 0, "ux": 10 + rng.standard_normal(t.size)})
    inflow = InflowData(
        data=data, points=pd.DataFrame({"idx": [0], "x": [0.0], "y": [0.0], "z": [10.0]})
    )
    norm = NormalizationParameters(reference_velocity=10.0, characteristic_length=100.0)

    result = calculate_spectral_density(
        inflow, target_index=0, for_components=["ux"], normalization_params=norm
    )
    expected_f, expected_S = spectral_density_function(data["ux"].to_numpy(), t, 10.0, 100.0)

    np.testing.assert_allclose(result["f (ux)"], expected_f)
    np.testing.assert_allclose(result["S (ux)"], expected_S)
    assert np.all(np.diff(result["f (ux)"]) > 0)
