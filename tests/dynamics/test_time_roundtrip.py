"""Regression cover for the Cp -> HFPI time-axis round trip.

The defect this guards, found on a real high-rise deliverable: the Cp stage
non-dimensionalises time by its own ``simul_characteristic_length`` while the
HFPI stage re-dimensionalised it with ``base``, the *force* normalisation
length. On a case where the two differ (``Lc_simul = H = 85.83`` m,
``base = B = 22.7`` m) the whole record was compressed by ``B / H = 0.264``:
a 600 s simulated event became 148 s, every forcing frequency moved up 3.8x,
and the resonant amplification of a 4.7 s-period tower came out roughly 40%
high on the base shear -- with no shape, unit or range check able to notice.

The invariant that makes it impossible: *the length that re-dimensionalises the
time axis must be the length that non-dimensionalised it*, and once it is, the
characteristic length cancels out of the round trip entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfdmod.dynamics import DimensionalData, build_floor_load_source, write_force_h5

N_FLOORS = 3
N_T = 200

# A case shaped like the real one: time normalised by H, forces by B.
H = 85.83
B = 22.7
SIMUL_U_H = 26.035
DESIGN_U_H = 27.859065215610023
DT_SOLVER = 0.075439


def _dim(**kwargs) -> DimensionalData:
    base = dict(
        U_H=DESIGN_U_H,
        height=H,
        base=B,
        integral_scale_multiplier=1.0,
        simul_characteristic_length=H,
        simul_U_H=SIMUL_U_H,
    )
    base.update(kwargs)
    return DimensionalData(**base)


@pytest.mark.unit
def test_time_denormalised_with_the_length_that_normalised_it():
    dim = _dim()
    assert dim.time_normalization_length == H
    assert dim.time_normalization_factor == pytest.approx(H / DESIGN_U_H)
    # NOT the force-normalisation length -- that was the bug.
    assert dim.time_normalization_factor != pytest.approx(B / DESIGN_U_H)


@pytest.mark.unit
def test_round_trip_cancels_the_characteristic_length():
    """Solver seconds -> full-scale seconds must not depend on Lc at all."""
    for lc in (B, H, 1.0, 250.0):
        dim = _dim(simul_characteristic_length=lc)
        dt_norm = DT_SOLVER / (lc / SIMUL_U_H)
        dt_phys = dt_norm * dim.time_normalization_factor
        assert dt_phys == pytest.approx(DT_SOLVER * dim.time_scale_factor, rel=1e-12)
        assert dt_phys == pytest.approx(DT_SOLVER * SIMUL_U_H / DESIGN_U_H, rel=1e-12)


@pytest.mark.unit
def test_integral_scale_multiplier_stretches_the_round_trip():
    dim = _dim(integral_scale_multiplier=1.71)
    assert dim.time_scale_factor == pytest.approx(1.71 * SIMUL_U_H / DESIGN_U_H)
    assert dim.time_normalization_factor == pytest.approx(1.71 * H / DESIGN_U_H)


@pytest.mark.unit
def test_a_600s_record_stays_a_600s_class_event():
    """The design speed is defined over ~10 min; the mapped record must match.

    Under the old ``base``-based de-normalisation this came out at 148 s.
    """
    dim = _dim()
    dt_norm = DT_SOLVER / (H / SIMUL_U_H)
    n_t = 7955
    duration = n_t * dt_norm * dim.time_normalization_factor
    assert 500.0 < duration < 700.0


@pytest.mark.unit
def test_missing_simul_characteristic_length_warns_and_falls_back():
    dim = DimensionalData(U_H=DESIGN_U_H, height=H, base=B, integral_scale_multiplier=1.0)
    with pytest.warns(UserWarning, match="simul_characteristic_length"):
        assert dim.time_normalization_length == B


@pytest.mark.unit
def test_time_scale_factor_requires_simul_u_h():
    dim = _dim(simul_U_H=None)
    with pytest.raises(ValueError, match="simul_U_H"):
        _ = dim.time_scale_factor


@pytest.mark.unit
def test_build_floor_load_source_closes_the_round_trip(tmp_path):
    """End to end from the force H5: dt must be dt_solver * U_simul / U_design."""
    dt_norm = DT_SOLVER / (H / SIMUL_U_H)
    rng = np.random.default_rng(2)
    paths = {}
    for name in ("cf_x", "cf_y", "cm_z"):
        df = pd.DataFrame({str(f): rng.standard_normal(N_T) for f in range(N_FLOORS)})
        df["time_normalized"] = np.arange(N_T) * dt_norm
        paths[name] = tmp_path / f"{name}.h5"
        write_force_h5(df, paths[name])

    src = build_floor_load_source(
        paths["cf_x"], paths["cf_y"], paths["cm_z"], _dim(), n_floors=N_FLOORS
    )

    assert src.time.timestep_size == pytest.approx(DT_SOLVER * SIMUL_U_H / DESIGN_U_H, rel=1e-12)
    # forces still normalised by the *force* length, independently of the above
    assert np.asarray(src.fields.read("cf_x")).shape == (N_FLOORS, N_T)
