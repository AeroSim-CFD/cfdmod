"""Tests for the high-rise ``pp/`` helper package.

Promotes the checks in ``notebooks/high_rise/_validate_pp.py`` into the pytest
suite so the notebook-side glue (HighRiseCase + the Cp / per-floor Cf/Cm
wiring) is covered by CI. The ``pp`` package lives under ``notebooks/`` and is
not installed, so it is put on the path here.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.integration

REPO = pathlib.Path(__file__).resolve().parents[2]
NB = REPO / "notebooks" / "high_rise"
if str(NB) not in sys.path:
    sys.path.insert(0, str(NB))

FIX = REPO / "fixtures" / "tests" / "pressure"
DATA = FIX / "data"
MESH = str(FIX / "galpao" / "galpao.normalized.lnas")

pp = pytest.importorskip("pp")


@pytest.fixture(scope="module")
def galpao_case():
    return pp.example_high_rise_case(MESH, n_floors=3)


@pytest.fixture(scope="module")
def cp_ds(galpao_case):
    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage

    storage = XdmfH5Storage(DATA)
    body = storage.read_data_source(pathlib.Path("bodies.galpao"))
    p_ref = storage.read_data_source(pathlib.Path("points.static_pressure"))
    return pp.cp_from_pressure(body, p_ref, galpao_case)


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
    cf = pp.cf_per_floor(cp_ds, MESH, galpao_case, directions=("x", "y"), method=method)
    assert cf.kind == "groups"
    assert 1 <= cf.n_elements <= galpao_case.n_floors
    cfx = cf.fields.read("cf_x")
    assert cfx.shape[1] > 1
    assert np.isfinite(cfx).all()


@pytest.mark.parametrize("method", ["face_cut", "centroid"])
def test_cm_per_floor_finite(cp_ds, galpao_case, method):
    cm = pp.cm_per_floor(cp_ds, MESH, galpao_case, directions=("z",), method=method)
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

    per_floor = pp.cf_per_floor(cp_ds, MESH, galpao_case, directions=("x",), method="face_cut")
    single = pp.cf_per_floor(cp_ds, MESH, whole, directions=("x",), method="face_cut")

    total_per_floor = per_floor.fields.read("cf_x").sum(axis=0)
    total_single = single.fields.read("cf_x").sum(axis=0)
    np.testing.assert_allclose(total_per_floor, total_single, rtol=1e-6, atol=1e-9)


def test_face_cut_and_centroid_agree_on_body_total(cp_ds, galpao_case):
    """Both methods integrate the same whole body, so the total over floors matches.

    Per-floor distributions differ (that is the point of face_cut), but the sum
    across all floors is the same whole-body force for either partition.
    """
    fc = pp.cf_per_floor(cp_ds, MESH, galpao_case, directions=("x",), method="face_cut")
    ct = pp.cf_per_floor(cp_ds, MESH, galpao_case, directions=("x",), method="centroid")
    np.testing.assert_allclose(
        fc.fields.read("cf_x").sum(axis=0),
        ct.fields.read("cf_x").sum(axis=0),
        rtol=1e-6,
        atol=1e-9,
    )
