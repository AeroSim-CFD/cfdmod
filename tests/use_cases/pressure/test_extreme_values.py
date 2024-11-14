import numpy as np
import pytest

from cfdmod.use_cases.pressure.extreme_values import (
    fit_gumbel_model,
    gumbel_extreme_values,
    moving_average_extreme_values,
    peak_extreme_values,
)
from cfdmod.use_cases.pressure.statistics import (
    ExtremeGumbelParamsModel,
    ExtremeMovingAverageParamsModel,
    ExtremePeakParamsModel,
)


@pytest.fixture()
def timestep_arr():
    yield np.linspace(0, 10, 100)


@pytest.fixture()
def hist_series():
    yield np.linspace(0.1, 0.6, 100)


@pytest.fixture()
def gumbel_params():
    yield ExtremeGumbelParamsModel(
        peak_duration=3,
        event_duration=60,
        non_exceedance_probability=0.78,
        full_scale_U_H=40,
        full_scale_characteristic_length=22.4,
    )


@pytest.fixture()
def moving_avg_params():
    yield ExtremeMovingAverageParamsModel(
        window_size_interval=2, full_scale_U_H=40, full_scale_characteristic_length=22.4
    )


@pytest.fixture()
def peak_params():
    yield ExtremePeakParamsModel(peak_factor=2)


def test_fit_gumbel_model(hist_series, timestep_arr, gumbel_params):
    result = fit_gumbel_model(hist_series, gumbel_params, timestep_arr[-1])
    assert round(result, ndigits=3) == 1.184


def test_calculate_gumbel_extreme_values(gumbel_params, timestep_arr, hist_series):
    result = gumbel_extreme_values(gumbel_params, timestep_arr, hist_series)

    assert len(result) == 2
    assert round(result[0], ndigits=3) == -0.07
    assert round(result[1], ndigits=3) == 0.768


def test_calculate_moving_avg_extreme_values(moving_avg_params, hist_series):
    result = moving_average_extreme_values(moving_avg_params, hist_series)

    assert len(result) == 2
    assert round(result[0], ndigits=3) == 0.108
    assert round(result[1], ndigits=3) == 0.592


def test_calculate_peak_extreme_values(peak_params, hist_series):
    result = peak_extreme_values(peak_params, hist_series)

    assert len(result) == 2
    assert round(result[0], ndigits=3) == 0.058
    assert round(result[1], ndigits=3) == 0.642
