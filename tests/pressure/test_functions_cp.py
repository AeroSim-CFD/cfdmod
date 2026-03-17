"""Tests for Cp functions (io/xdmf + process_xdmf_to_cp)."""

import pathlib

import h5py
import numpy as np
import pytest

from cfdmod.io.xdmf import (
    filter_keys_by_range,
    get_pressure_keys,
    read_step,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
)
from cfdmod.pressure.functions import add_cp2xdmf, process_xdmf_to_cp
from cfdmod.pressure.parameters import CpConfig, BasicStatisticModel


BODY_H5 = pathlib.Path("fixtures/tests/pressure/xdmf/bodies.building.h5")
PROBE_H5 = pathlib.Path("fixtures/tests/pressure/xdmf/points.point0.h5")


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


def test_add_cp2xdmf(tmp_path):
    import shutil

    body_copy = tmp_path / "body.h5"
    shutil.copy(BODY_H5, body_copy)

    with h5py.File(body_copy, "r") as f:
        keys = list(f["pressure"].keys())

    add_cp2xdmf(
        body_h5=body_copy,
        atm_probe_h5=None,
        reference_vel=1.0,
        fluid_density=1.0,
    )

    with h5py.File(body_copy, "r") as f:
        assert "cp" in f
        assert list(f["cp"].keys()) == keys
