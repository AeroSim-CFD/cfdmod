"""Tests for Cp functions (io/xdmf + process_xdmf_to_cp)."""

import h5py
import pytest

from cfdmod.io.xdmf import filter_keys_by_range, get_pressure_keys, read_step
from cfdmod.pressure import run_cp
from cfdmod.pressure.functions import process_xdmf_to_cp
from cfdmod.pressure.parameters import BasicStatisticModel, CpConfig
from tests.pressure.conftest import BUILDING_BODY_H5 as BODY_H5
from tests.pressure.conftest import BUILDING_PROBE_H5 as PROBE_H5
from tests.pressure.conftest import make_cp_cfg

pytestmark = pytest.mark.integration


def test_get_pressure_keys():
    keys = get_pressure_keys(BODY_H5, "pressure")
    assert len(keys) > 0
    times, key_strs = zip(*keys)
    assert list(times) == sorted(times)
    assert all(k.startswith("t") for k in key_strs)


def test_filter_keys_by_range():
    keys = get_pressure_keys(BODY_H5, "pressure")
    t_min = keys[10][0]
    t_max = keys[20][0]
    filtered = filter_keys_by_range(keys, (t_min, t_max))
    assert len(filtered) == 11
    assert all(t_min <= t <= t_max for t, _ in filtered)


def test_read_step():
    keys = get_pressure_keys(BODY_H5, "pressure")
    t_val, t_key = keys[0]
    arr = read_step(BODY_H5, t_key, "pressure")
    assert arr.ndim == 1
    assert arr.shape[0] == 51593


def test_process_xdmf_to_cp_pressure(tmp_path):
    cfg = CpConfig(
        timestep_range=(125.0, 130.0),
        macroscopic_type="pressure",
        simul_U_H=1.0,
        simul_characteristic_length=1.0,
        statistics=[BasicStatisticModel(stats="mean")],
    )
    output_h5 = tmp_path / "cp.h5"
    process_xdmf_to_cp(BODY_H5, PROBE_H5, output_h5, cfg)

    assert output_h5.exists()
    xdmf_path = output_h5.with_suffix(".xdmf")
    assert xdmf_path.exists()

    with h5py.File(output_h5, "r") as f:
        assert "cp" in f
        assert "Triangles" in f
        assert "Geometry" in f
        assert "meta" in f
        keys = list(f["cp"].keys())
        assert len(keys) > 0
        arr = f["cp"][keys[0]][:]
        assert arr.shape == (51593,)


def test_process_xdmf_to_cp_rho(tmp_path):
    cfg = CpConfig(
        timestep_range=(125.0, 130.0),
        macroscopic_type="rho",
        simul_U_H=1.0,
        simul_characteristic_length=1.0,
        statistics=[BasicStatisticModel(stats="mean")],
    )
    output_h5 = tmp_path / "cp_rho.h5"
    process_xdmf_to_cp(BODY_H5, None, output_h5, cfg)

    assert output_h5.exists()
    with h5py.File(output_h5, "r") as f:
        keys = list(f["cp"].keys())
        assert len(keys) > 0


def test_process_xdmf_to_cp_no_probe(tmp_path):
    cfg = CpConfig(
        timestep_range=(125.0, 128.0),
        macroscopic_type="pressure",
        simul_U_H=1.0,
        simul_characteristic_length=1.0,
        statistics=[BasicStatisticModel(stats="mean")],
    )
    output_h5 = tmp_path / "cp_no_probe.h5"
    process_xdmf_to_cp(BODY_H5, None, output_h5, cfg)
    assert output_h5.exists()


def test_process_xdmf_to_cp_does_not_modify_inputs(tmp_path):
    """Cp computation must produce a NEW file and leave body/probe untouched."""
    import shutil

    body_copy = tmp_path / "body.h5"
    probe_copy = tmp_path / "probe.h5"
    shutil.copy(BODY_H5, body_copy)
    shutil.copy(PROBE_H5, probe_copy)

    body_mtime_before = body_copy.stat().st_mtime_ns
    body_size_before = body_copy.stat().st_size
    probe_mtime_before = probe_copy.stat().st_mtime_ns
    probe_size_before = probe_copy.stat().st_size

    with h5py.File(body_copy, "r") as f:
        body_keys_before = sorted(f.keys())
    with h5py.File(probe_copy, "r") as f:
        probe_keys_before = sorted(f.keys())

    cfg = CpConfig(
        statistics=[BasicStatisticModel(stats="mean")],
        timestep_range=(0.0, 1e9),
        macroscopic_type="pressure",
        reference_pressure="average",
        simul_U_H=1.0,
        simul_characteristic_length=1.0,
        fluid_density=1.0,
    )
    output_h5 = tmp_path / "cp.time_series.h5"
    process_xdmf_to_cp(body_copy, probe_copy, output_h5, cfg)

    assert output_h5.exists()
    assert body_copy.stat().st_mtime_ns == body_mtime_before
    assert body_copy.stat().st_size == body_size_before
    assert probe_copy.stat().st_mtime_ns == probe_mtime_before
    assert probe_copy.stat().st_size == probe_size_before
    with h5py.File(body_copy, "r") as f:
        assert sorted(f.keys()) == body_keys_before
    with h5py.File(probe_copy, "r") as f:
        assert sorted(f.keys()) == probe_keys_before


def test_cpconfig_statistics_optional_default_empty():
    """Statistics is optional; default is an empty list (skips stats step)."""
    cfg = CpConfig(
        timestep_range=(0.0, 1.0),
        simul_U_H=1.0,
        simul_characteristic_length=1.0,
    )
    assert cfg.statistics == []


def test_run_cp_skips_stats_when_statistics_empty(tmp_path):
    """With statistics=[], run_cp writes the timeseries but no stats.h5/.xdmf."""
    cfg = make_cp_cfg(statistics=[])
    run_cp(body_h5=BODY_H5, probe_h5=PROBE_H5, cfg_path=cfg, output=tmp_path)
    assert (tmp_path / "cp.default.time_series.h5").exists()
    assert (tmp_path / "cp.default.time_series.xdmf").exists()
    assert not (tmp_path / "stats.h5").exists()
    assert not (tmp_path / "stats.xdmf").exists()


def test_run_cp_writes_stats_when_statistics_nonempty(tmp_path):
    """Sanity check: the stats step still runs when statistics is non-empty."""
    cfg = make_cp_cfg()  # default = [BasicStatisticModel(stats='mean')]
    run_cp(body_h5=BODY_H5, probe_h5=PROBE_H5, cfg_path=cfg, output=tmp_path)
    assert (tmp_path / "stats.h5").exists()
    assert (tmp_path / "stats.xdmf").exists()
