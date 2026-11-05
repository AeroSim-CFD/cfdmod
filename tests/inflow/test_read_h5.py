import pathlib
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

from cfdmod.inflow import InflowData

pytestmark = pytest.mark.unit

# Solver probe layout: no /meta group, physical time only in the key names,
# negative times during spin-up, keys deliberately written out of order.
TIMES = [-119.905098, -0.5, 0.0, 0.25, 10.0, 2.5]
N_POINTS = 4


def _value(comp_offset: float, t: float) -> np.ndarray:
    return comp_offset + t + np.arange(N_POINTS, dtype=np.float64) * 0.1


def _write_nassu_h5(path: pathlib.Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("Connectivity", data=np.arange(N_POINTS, dtype=np.int64))
        f.create_dataset("Geometry", data=np.zeros((N_POINTS, 3), dtype=np.float32))
        for comp, offset in (("ux", 100.0), ("uy", 200.0), ("uz", 300.0)):
            grp = f.create_group(comp)
            for t in TIMES:
                grp.create_dataset(f"t{t}", data=_value(offset, t).astype(np.float32))


@pytest.fixture()
def nassu_inflow(tmp_path: pathlib.Path) -> InflowData:
    h5_path = tmp_path / "line.inst.h5"
    _write_nassu_h5(h5_path)
    points_path = tmp_path / "points.csv"
    pd.DataFrame(
        {"idx": range(N_POINTS), "x": 0.0, "y": 0.0, "z": np.arange(N_POINTS, dtype=float)}
    ).to_csv(points_path, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return InflowData.from_files(h5_path, points_path)


def test_reads_nassu_layout_without_meta(nassu_inflow):
    data = nassu_inflow.data
    assert list(data.columns) == ["time_step", "point_idx", "ux", "uy", "uz"]
    assert len(data) == len(TIMES) * N_POINTS


def test_time_comes_from_keys_sorted_numerically(nassu_inflow):
    times = nassu_inflow.data["time_step"].drop_duplicates().to_numpy()
    np.testing.assert_allclose(times, sorted(TIMES))


def test_values_aligned_with_time_and_point(nassu_inflow):
    data = nassu_inflow.data
    for comp, offset in (("ux", 100.0), ("uy", 200.0), ("uz", 300.0)):
        for t in TIMES:
            got = data.loc[np.isclose(data["time_step"], t)].sort_values("point_idx")[comp]
            np.testing.assert_allclose(got.to_numpy(), _value(offset, t), rtol=1e-6)


def test_mismatched_component_timesteps_raise(tmp_path):
    h5_path = tmp_path / "bad.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_group("ux").create_dataset("t0.0", data=np.zeros(N_POINTS))
        f.create_group("uy").create_dataset("t1.0", data=np.zeros(N_POINTS))
    points_path = tmp_path / "points.csv"
    pd.DataFrame({"idx": [0], "x": [0.0], "y": [0.0], "z": [0.0]}).to_csv(points_path, index=False)
    with pytest.raises(ValueError, match="same timesteps"):
        InflowData.from_files(h5_path, points_path)


def test_legacy_hdfstore_still_warns():
    folder = pathlib.Path("./fixtures/tests/inflow/autocorrelacao_10_h5")
    with pytest.warns(DeprecationWarning, match="legacy"):
        InflowData.from_files(folder / "hist_series.h5", folder / "points.csv")
