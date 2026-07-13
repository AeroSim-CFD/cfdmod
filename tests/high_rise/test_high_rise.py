"""Tests for the ``cfdmod.building`` post-processing helpers.

Promotes the checks in ``examples/high_rise/_validate_high_rise.py`` into the
pytest suite so the building glue (BuildingCase + the Cp / per-floor Cf/Cm
wiring) is covered by CI.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.integration

REPO = pathlib.Path(__file__).resolve().parents[2]

FIX = REPO / "fixtures" / "tests" / "pressure"
DATA = FIX / "data"
MESH = str(FIX / "galpao" / "galpao.normalized.lnas")

building = pytest.importorskip("cfdmod.building")


@pytest.fixture(scope="module")
def galpao_case():
    return building.example_building_case(MESH, n_floors=3)


@pytest.fixture(scope="module")
def cp_ds(galpao_case):
    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage

    storage = XdmfH5Storage(DATA)
    body = storage.read_data_source(pathlib.Path("bodies.galpao"))
    p_ref = storage.read_data_source(pathlib.Path("points.static_pressure"))
    return building.cp_from_pressure(body, p_ref, galpao_case)


def test_example_case_geometry(galpao_case):
    assert galpao_case.n_floors == 3
    assert galpao_case.nominal_area > 0
    assert len(galpao_case.floor_heights) == 4


def test_cp_from_pressure(cp_ds):
    assert "cp" in cp_ds.field_names
    cp = cp_ds.fields.read("cp")
    assert cp.ndim == 2 and cp.shape[1] > 1  # time-resolved
    assert np.isfinite(cp).all()


@pytest.mark.parametrize("method", ["face_cut", "centroid"])
def test_cf_per_floor_shapes(cp_ds, galpao_case, method):
    cf = building.cf_per_floor(cp_ds, MESH, galpao_case, directions=("x", "y"), method=method)
    assert cf.kind == "groups"
    assert 1 <= cf.n_elements <= galpao_case.n_floors
    cfx = cf.fields.read("cf_x")
    assert cfx.shape[1] > 1
    assert np.isfinite(cfx).all()


@pytest.mark.parametrize("method", ["face_cut", "centroid"])
def test_cm_per_floor_finite(cp_ds, galpao_case, method):
    cm = building.cm_per_floor(cp_ds, MESH, galpao_case, directions=("z",), method=method)
    cmz = cm.fields.read("cm_z")
    assert np.isfinite(cmz).all()


def test_face_cut_conserves_total_force(cp_ds, galpao_case):
    """Per-floor Cf summed over floors equals the whole-body Cf (exactness).

    face_cut partitions each triangle's area across floors, so summing the
    per-floor contributions recovers the single-region total. The cut core
    round-trips fragment vertices through float32, so parity holds to ~1e-7
    rather than to machine epsilon.
    """
    zmin = min(galpao_case.floor_heights)
    zmax = max(galpao_case.floor_heights)
    whole = galpao_case.model_copy(update={"floor_heights": [zmin, zmax + 1e-6]})

    per_floor = building.cf_per_floor(
        cp_ds, MESH, galpao_case, directions=("x",), method="face_cut"
    )
    single = building.cf_per_floor(cp_ds, MESH, whole, directions=("x",), method="face_cut")

    total_per_floor = per_floor.fields.read("cf_x").sum(axis=0)
    total_single = single.fields.read("cf_x").sum(axis=0)
    np.testing.assert_allclose(total_per_floor, total_single, rtol=1e-6, atol=1e-9)


_MINIMAL_PARAMS = """
pressure_coefficient:
  base:
    fluid_density: 1.225
    simul_U_H: 30.0
force_coefficient:
  fc:
    nominal_area: 100.0
    bodies:
      - name: building
        sub_bodies:
          z_intervals: [0.0, 50.0]
moment_coefficient:
  mc:
    nominal_volume: 1000.0
    bodies:
      - lever_origin: [0.0, 0.0, 0.0]
"""

_GLOBAL = '{"H": 70, "L": 6.95, "V0": 38, "analysis": {"body_name": "building"}}'


def _write_case_data(dir_path: pathlib.Path, alturas: str | None) -> pathlib.Path:
    cd = dir_path / "case_data"
    cd.mkdir(parents=True, exist_ok=True)
    (cd / "global_data.json").write_text(_GLOBAL)
    (cd / "params.yaml").write_text(_MINIMAL_PARAMS)
    if alturas is not None:
        (cd / "alturas.csv").write_text(alturas)
    return cd


def test_from_case_data_reads_floors_from_alturas(tmp_path):
    """alturas.csv is the floor source of truth: N storeys -> N floors."""
    rows = ["Pavimento,z_min,z_max,dz"]
    for i in range(5):
        rows.append(f"{i + 1},{i * 10.0},{(i + 1) * 10.0},10.0")
    cd = _write_case_data(tmp_path, "\n".join(rows) + "\n")
    case = building.BuildingCase.from_case_data(cd, "params.yaml")
    assert case.n_floors == 5
    assert case.floor_heights[0] == 0.0
    assert case.floor_heights[-1] == 50.0


def test_from_case_data_falls_back_to_yaml_when_alturas_empty(tmp_path):
    """A header-only (or missing) alturas.csv falls back to the yaml z_intervals."""
    header_only = _write_case_data(tmp_path, "Pavimento,z_min,z_max,dz\n")
    case = building.BuildingCase.from_case_data(header_only, "params.yaml")
    assert case.floor_heights == [0.0, 50.0]  # from the yaml anchor -> 1 floor
    assert case.n_floors == 1

    missing = _write_case_data(tmp_path / "nocsv", None)
    case2 = building.BuildingCase.from_case_data(missing, "params.yaml")
    assert case2.floor_heights == [0.0, 50.0]


@pytest.fixture(scope="module")
def body_ref():
    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage

    storage = XdmfH5Storage(DATA)
    body = storage.read_data_source(pathlib.Path("bodies.galpao"))
    p_ref = storage.read_data_source(pathlib.Path("points.static_pressure"))
    return body, p_ref


def test_cf_cm_per_floor_matches_separate(cp_ds, galpao_case):
    """The fused single-force-pass Cf/Cm equals the separate cf_/cm_per_floor."""
    cf, cm = building.cf_cm_per_floor(
        cp_ds, MESH, galpao_case, cf_directions=("x", "y"), cm_directions=("z",)
    )
    cf_ref = building.cf_per_floor(cp_ds, MESH, galpao_case, directions=("x", "y"))
    cm_ref = building.cm_per_floor(cp_ds, MESH, galpao_case, directions=("z",))
    for f in ("cf_x", "cf_y"):
        np.testing.assert_allclose(cf.fields.read(f), cf_ref.fields.read(f))
    np.testing.assert_allclose(cm.fields.read("cm_z"), cm_ref.fields.read("cm_z"))


def test_per_floor_loads_whole_matches_fused(body_ref, galpao_case):
    """per_floor_loads (whole-series) == cp_from_pressure -> cf_cm_per_floor."""
    body, p_ref = body_ref
    cf, cm = building.per_floor_loads(body, p_ref, MESH, galpao_case)
    cp = building.cp_from_pressure(body, p_ref, galpao_case)
    cf_ref, cm_ref = building.cf_cm_per_floor(cp, MESH, galpao_case)
    np.testing.assert_allclose(cf.fields.read("cf_x"), cf_ref.fields.read("cf_x"))
    np.testing.assert_allclose(cm.fields.read("cm_z"), cm_ref.fields.read("cm_z"))


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_per_floor_loads_chunk_parity(body_ref, galpao_case, chunk):
    """Time-chunked per_floor_loads matches the whole-series result exactly."""
    body, p_ref = body_ref
    cf_w, cm_w = building.per_floor_loads(body, p_ref, MESH, galpao_case, chunk_size=None)
    cf_c, cm_c = building.per_floor_loads(body, p_ref, MESH, galpao_case, chunk_size=chunk)
    assert cf_c.time.n_timesteps == cf_w.time.n_timesteps
    np.testing.assert_allclose(cf_c.fields.read("cf_x"), cf_w.fields.read("cf_x"))
    np.testing.assert_allclose(cf_c.fields.read("cf_y"), cf_w.fields.read("cf_y"))
    np.testing.assert_allclose(cm_c.fields.read("cm_z"), cm_w.fields.read("cm_z"))


def test_face_cut_and_centroid_agree_on_body_total(cp_ds, galpao_case):
    """Both methods integrate the same whole body, so the total over floors matches.

    Per-floor distributions differ (that is the point of face_cut), but the sum
    across all floors is the same whole-body force for either partition.
    """
    fc = building.cf_per_floor(cp_ds, MESH, galpao_case, directions=("x",), method="face_cut")
    ct = building.cf_per_floor(cp_ds, MESH, galpao_case, directions=("x",), method="centroid")
    np.testing.assert_allclose(
        fc.fields.read("cf_x").sum(axis=0),
        ct.fields.read("cf_x").sum(axis=0),
        rtol=1e-6,
        atol=1e-9,
    )
