"""Tests for cfdmod.building.pressure (Cp + per-floor Cf/Cm + per_floor_loads).

Promotes the checks in ``examples/high_rise/_validate_high_rise.py`` into the
pytest suite so the Cp / per-floor Cf/Cm wiring is covered by CI.
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


@pytest.fixture(scope="module")
def body_ref():
    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage

    storage = XdmfH5Storage(DATA)
    body = storage.read_data_source(pathlib.Path("bodies.galpao"))
    p_ref = storage.read_data_source(pathlib.Path("points.static_pressure"))
    return body, p_ref


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
    """per_floor_loads (whole-series) == cp_from_pressure -> cf_cm_per_floor.

    Mirror per_floor_loads' float32 source cast in the reference so the two
    compute in the same precision and the equality is exact.
    """
    body, p_ref = body_ref
    cf, cm = building.per_floor_loads(body, p_ref, MESH, galpao_case)

    def _f32(ds):
        return ds.with_field("pressure", np.asarray(ds.fields.read("pressure"), dtype="float32"))

    cp = building.cp_from_pressure(_f32(body), _f32(p_ref), galpao_case)
    cf_ref, cm_ref = building.cf_cm_per_floor(cp, MESH, galpao_case)
    np.testing.assert_allclose(cf.fields.read("cf_x"), cf_ref.fields.read("cf_x"))
    np.testing.assert_allclose(cm.fields.read("cm_z"), cm_ref.fields.read("cm_z"))


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_per_floor_loads_chunk_parity(body_ref, galpao_case, chunk):
    """Time-chunked per_floor_loads matches the whole-series result.

    Per-timestep values are computed identically regardless of windowing; the
    only difference is float32 summation-order rounding across window shapes
    (~1 ULP), hence the float32-appropriate tolerance rather than exact equality.
    """
    body, p_ref = body_ref
    cf_w, cm_w = building.per_floor_loads(body, p_ref, MESH, galpao_case, chunk_size=None)
    cf_c, cm_c = building.per_floor_loads(body, p_ref, MESH, galpao_case, chunk_size=chunk)
    assert cf_c.time.n_timesteps == cf_w.time.n_timesteps
    kw = dict(rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(cf_c.fields.read("cf_x"), cf_w.fields.read("cf_x"), **kw)
    np.testing.assert_allclose(cf_c.fields.read("cf_y"), cf_w.fields.read("cf_y"), **kw)
    np.testing.assert_allclose(cm_c.fields.read("cm_z"), cm_w.fields.read("cm_z"), **kw)


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


def test_static_loads_to_base_moments_end_to_end(body_ref, galpao_case):
    """per_floor_loads -> static_floor_loads -> base envelope, on real fixture data.

    The chain a consulting notebook runs. Asserts the two properties the
    downstream deliverable depends on: the loads scale with the *supplied*
    design speed (not the case's simulation speed), and the base overturning
    moments are real moments -- not a rescaled copy of the base forces.
    """
    from cfdmod.core.container import Container
    from cfdmod.dynamics import BuildingCaseParameters, get_global_peaks_by_direction

    body, p_ref = body_ref
    cf, cm = building.per_floor_loads(body, p_ref, MESH, galpao_case, method="centroid")

    design_u_h = 3.0 * galpao_case.simul_reference_velocity
    loads = building.static_floor_loads(cf, cm, galpao_case, reference_velocity=design_u_h)

    # loads referenced at the design speed, not the simulation one
    at_simul = building.static_floor_loads(
        cf, cm, galpao_case, reference_velocity=galpao_case.simul_reference_velocity
    )
    ratio = np.asarray(loads.fields.read("feq_x")) / np.asarray(at_simul.fields.read("feq_x"))
    np.testing.assert_allclose(ratio, 9.0, rtol=1e-9)

    # lever arms travel with the source, so the base moments are well-defined
    z = np.asarray(loads.elements.position)[:, 2]
    assert np.any(z)

    container = Container(
        items={BuildingCaseParameters(direction=0.0, xi=0.0, recurrence_period=50.0): loads}
    )
    frames = get_global_peaks_by_direction(container, variable_type="static")
    forces, moments = frames["forces_static"], frames["moments_static"]
    assert np.isfinite(forces.to_numpy().astype(float)).all()
    assert np.isfinite(moments.to_numpy().astype(float)).all()

    # Mx must not be Fy times a constant: rebuild it by hand from the arrays.
    feq_y = np.asarray(loads.fields.read("feq_y"), dtype=np.float64)
    np.testing.assert_allclose(
        moments["mean_x"].to_numpy(), -(feq_y * z[:, None]).sum(axis=0).mean(), rtol=1e-9
    )
    # ... and the arm is genuinely non-uniform across floors
    assert np.ptp(z) > 0
