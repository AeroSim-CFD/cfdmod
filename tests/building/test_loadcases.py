"""Per-floor load-case tables from a multi-direction result container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.building import (
    effective_load_stats,
    generate_load_cases,
    invert_load_cases,
    save_load_case_tables,
)
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.container import Container
from cfdmod.dynamics import BuildingCaseParameters, build_cases, solve_building_cases
from cfdmod.report import DebugWriter

pytestmark = pytest.mark.unit

N_FLOORS = 2
N_T = 4


def _response(feq_x, feq_y, meq_z) -> PointsDataSource:
    pts = np.zeros((N_FLOORS, 3))
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=N_T),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(
            {
                "feq_x": np.asarray(feq_x, dtype=float),
                "feq_y": np.asarray(feq_y, dtype=float),
                "meq_z": np.asarray(meq_z, dtype=float),
            }
        ),
    )


def _two_direction_container() -> Container:
    # floor 0: +1..-3 ; floor 1: +2..-2 -> max_x=[1,2] min_x=[-3,-2]
    feq_x = np.array([[1.0, 0.0, -3.0, 0.0], [2.0, 0.0, -2.0, 0.0]])
    feq_y = np.array([[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    meq_z = np.array([[10.0, -10.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    def solve_fn(case):
        scale = 1.0 if case.direction == 0.0 else 2.0
        return _response(scale * feq_x, scale * feq_y, scale * meq_z)

    cases = build_cases(directions=[0.0, 90.0], xis=[0.01], recurrence_periods=[10.0])
    return solve_building_cases(cases, solve_fn)


def test_effective_load_stats_shape_and_values():
    container = _two_direction_container()
    stats = effective_load_stats(container, unit_conversion=1.0)

    assert set(stats) == {"peak", "min", "max"}
    for stat in stats:
        assert set(stats[stat]) == {"Fx", "Fy", "Mz"}
        for df in stats[stat].values():
            assert list(df.index) == [0, 1]  # rows = floors
            assert set(df.columns) == {"0.0", "90.0"}  # cols = direction labels

    # direction 0 signed envelopes (unit_conversion = 1.0)
    np.testing.assert_allclose(stats["max"]["Fx"]["0.0"].to_numpy(), [1.0, 2.0])
    np.testing.assert_allclose(stats["min"]["Fx"]["0.0"].to_numpy(), [-3.0, -2.0])
    # peak = max(|series|) per floor -> floor 0 = 3, floor 1 = 2
    np.testing.assert_allclose(stats["peak"]["Fx"]["0.0"].to_numpy(), [3.0, 2.0])
    # direction 90 is the x2 case
    np.testing.assert_allclose(stats["max"]["Fx"]["90.0"].to_numpy(), [2.0, 4.0])


def test_effective_load_stats_unit_conversion():
    container = _two_direction_container()
    raw = effective_load_stats(container, unit_conversion=1.0)
    conv = effective_load_stats(container, unit_conversion=1.0 / 9806.65)
    np.testing.assert_allclose(
        conv["max"]["Fx"]["0.0"].to_numpy(),
        raw["max"]["Fx"]["0.0"].to_numpy() / 9806.65,
    )


def test_effective_load_stats_rejects_multiple_cases_per_direction():
    resp = _response(
        np.zeros((N_FLOORS, N_T)), np.zeros((N_FLOORS, N_T)), np.zeros((N_FLOORS, N_T))
    )
    container = Container(
        items={
            BuildingCaseParameters(direction=0.0, xi=0.01, recurrence_period=10.0): resp,
            BuildingCaseParameters(direction=0.0, xi=0.02, recurrence_period=10.0): resp,
        }
    )
    with pytest.raises(ValueError):
        effective_load_stats(container)


def test_generate_and_invert_load_cases():
    container = _two_direction_container()
    stats = effective_load_stats(container, unit_conversion=1.0)
    cases = generate_load_cases(stats, senses=("max", "min"))

    assert list(cases.columns) == ["direction", "sense", "floor", "Fx", "Fy", "Mz"]
    # 2 senses x 2 directions x 2 floors
    assert len(cases) == 8
    assert set(cases["sense"]) == {"max", "min"}

    # inverting twice restores the load columns
    twice = invert_load_cases(invert_load_cases(cases))
    pd.testing.assert_frame_equal(
        twice[["Fx", "Fy", "Mz"]].reset_index(drop=True),
        cases[["Fx", "Fy", "Mz"]].reset_index(drop=True),
    )
    # a single inversion negates the loads
    inv = invert_load_cases(cases)
    np.testing.assert_allclose(inv["Fx"].to_numpy(), -cases["Fx"].to_numpy())
    assert set(inv["sense"]) == {"max_inv", "min_inv"}


def test_save_load_case_tables(tmp_path):
    container = _two_direction_container()
    stats = effective_load_stats(container, unit_conversion=1.0)
    writer = DebugWriter(tmp_path, "loadcases", "v1")
    written = save_load_case_tables(
        stats, writer, deliverable=True, floor_heights=np.array([3.0, 6.0])
    )

    assert len(written) == len(stats) * 3  # {peak,min,max} x {Fx,Fy,Mz}
    for name, path in written.items():
        assert path.exists()
    round_trip = pd.read_csv(written["loadcase_max_Fx.csv"])
    assert list(round_trip.columns[:2]) == ["floor", "z"]
    assert "0.0" in round_trip.columns and "90.0" in round_trip.columns
    np.testing.assert_allclose(round_trip["z"].to_numpy(), [3.0, 6.0])
