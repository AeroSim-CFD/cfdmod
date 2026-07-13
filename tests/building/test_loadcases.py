"""Per-floor load-case tables from a multi-direction result container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.building import (
    directional_envelopes,
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


def _response(feq_x, feq_y, meq_z, static=None) -> PointsDataSource:
    fields = {
        "feq_x": np.asarray(feq_x, dtype=float),
        "feq_y": np.asarray(feq_y, dtype=float),
        "meq_z": np.asarray(meq_z, dtype=float),
    }
    if static is not None:
        fields |= {k: np.asarray(v, dtype=float) for k, v in static.items()}
    pts = np.zeros((N_FLOORS, 3))
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=N_T),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(fields),
    )


def _two_direction_container(static=None) -> Container:
    # floor 0: +1..-3 ; floor 1: +2..-2 -> max_x=[1,2] min_x=[-3,-2]
    feq_x = np.array([[1.0, 0.0, -3.0, 0.0], [2.0, 0.0, -2.0, 0.0]])
    feq_y = np.array([[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    meq_z = np.array([[10.0, -10.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    def solve_fn(case):
        scale = 1.0 if case.direction == 0.0 else 2.0
        st = None if static is None else {k: scale * np.asarray(v) for k, v in static.items()}
        return _response(scale * feq_x, scale * feq_y, scale * meq_z, static=st)

    cases = build_cases(directions=[0.0, 90.0], xis=[0.01], recurrence_periods=[10.0])
    return solve_building_cases(cases, solve_fn)


def test_effective_load_stats_governing_peak():
    container = _two_direction_container()
    stats = effective_load_stats(container, unit_conversion=1.0)

    assert set(stats) == {"peak", "min", "max"}
    for stat in stats:
        assert set(stats[stat]) == {"Fx", "Fy", "Mz"}
        for df in stats[stat].values():
            assert list(df.index) == [0, 1]  # rows = floors
            assert set(df.columns) == {"0.0", "90.0"}  # cols = direction labels

    # direction 0 signed envelopes
    np.testing.assert_allclose(stats["max"]["Fx"]["0.0"].to_numpy(), [1.0, 2.0])
    np.testing.assert_allclose(stats["min"]["Fx"]["0.0"].to_numpy(), [-3.0, -2.0])
    # peak = governing envelope: |mean(min)|=2.5 > |mean(max)|=1.5 -> the min envelope
    np.testing.assert_allclose(stats["peak"]["Fx"]["0.0"].to_numpy(), [-3.0, -2.0])
    # Fy: max mean 4.5 > |min| 4.5 -> tie goes to max envelope
    np.testing.assert_allclose(stats["peak"]["Fy"]["0.0"].to_numpy(), [4.0, 5.0])


def test_effective_load_stats_combines_applied_static():
    # applied-static loads dominate the max envelope on floor 0
    static = {
        "fs_x": np.full((N_FLOORS, N_T), 0.0),
        "fs_y": np.full((N_FLOORS, N_T), 0.0),
        "ms_z": np.full((N_FLOORS, N_T), 0.0),
    }
    static["fs_x"][0] = 9.0  # applied Fx on floor 0 exceeds the dynamic max (1.0)
    container = _two_direction_container(static=static)
    stats = effective_load_stats(container, unit_conversion=1.0)
    # effective max_x floor 0 = max(dyn 1.0, applied 9.0) = 9.0
    assert stats["max"]["Fx"]["0.0"].to_numpy()[0] == pytest.approx(9.0)
    # floor 1 has no applied load -> unchanged dynamic max 2.0
    assert stats["max"]["Fx"]["0.0"].to_numpy()[1] == pytest.approx(2.0)


def test_generate_and_invert_load_cases():
    container = _two_direction_container()
    max_dict, min_dict = directional_envelopes(container)
    assert set(max_dict) == {0.0, 90.0}
    assert set(max_dict[0.0]) == {"x", "y", "z"}

    cases = generate_load_cases(max_dict, min_dict, unit_conversion=1.0)
    # 3 principal axes x 2 principal signs x 4 companion combos
    assert len(cases) == 3 * 2 * 4
    for load in cases.values():
        assert set(load) == {"Fx", "Fy", "Mz"}
        assert load["Fx"].shape == (N_FLOORS,)

    frames = invert_load_cases(cases)
    assert set(frames) == {"Fx", "Fy", "Mz"}
    # per-axis frame: rows = floors, cols = case ids
    assert frames["Fx"].shape == (N_FLOORS, len(cases))
    # column 0 of Fx frame equals case 0's Fx per-floor loads
    np.testing.assert_allclose(frames["Fx"][0].to_numpy(), cases[0]["Fx"])


def test_save_load_case_tables(tmp_path):
    container = _two_direction_container()
    stats = effective_load_stats(container, unit_conversion=1.0)
    writer = DebugWriter(tmp_path, "loadcases", "v1")
    written = save_load_case_tables(
        stats, writer, deliverable=True, floor_heights=np.array([3.0, 6.0])
    )

    assert len(written) == len(stats) * 3  # {peak,min,max} x {Fx,Fy,Mz}
    for path in written.values():
        assert path.exists()
    round_trip = pd.read_csv(written["loadcase_max_Fx.csv"])
    assert list(round_trip.columns[:2]) == ["floor", "z"]
    assert "0.0" in round_trip.columns and "90.0" in round_trip.columns
    np.testing.assert_allclose(round_trip["z"].to_numpy(), [3.0, 6.0])


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
