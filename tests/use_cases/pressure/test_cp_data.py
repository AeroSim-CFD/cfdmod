import pandas as pd
import pytest

from cfdmod.use_cases.pressure.cp_data import filter_data, transform_to_cp


@pytest.fixture()
def press_data():
    yield pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "0": [1, 1, 1, 1, 1]})


@pytest.fixture()
def body_data():
    yield pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "0": [1.1, 1.2, 1.3, 1.4, 1.5]})


def test_read_and_slice_data(press_data, body_data):
    press_data_filtered = filter_data(press_data, (2, 4))
    body_data_filtered = filter_data(body_data, (2, 4))

    assert press_data_filtered["time_step"].tolist() == [2, 3, 4]
    assert body_data_filtered["time_step"].tolist() == [2, 3, 4]


def test_transform_rho(press_data, body_data):
    transformed_data = transform_to_cp(
        press_data=press_data,
        body_data=body_data,
        reference_vel=0.05,
        fluid_density=1,
        macroscopic_type="rho",
        characteristic_length=1,
        columns_drop=None,
        columns_process=None,
    )

    assert "0" in transformed_data.columns
    assert len(transformed_data.iloc[0]) == len(body_data.iloc[0])


def test_transform_pressure(press_data, body_data):
    transformed_data = transform_to_cp(
        press_data=press_data,
        body_data=body_data,
        reference_vel=10,
        fluid_density=1,
        macroscopic_type="pressure",
        characteristic_length=1,
        columns_drop=None,
        columns_process=None,
    )

    assert "0" in transformed_data.columns
    assert len(transformed_data.iloc[0]) == len(body_data.iloc[0])
