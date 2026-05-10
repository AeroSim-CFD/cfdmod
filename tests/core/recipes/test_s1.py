"""Unit tests for the S1 recipe and profile_interpolation op."""

from __future__ import annotations

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.ops.data_source_create import (
    ProfileInterpolationParams,
    profile_interpolation,
)
from cfdmod.core.recipes import S1RecipeConfig, build_s1


def _profile(z: np.ndarray, u: np.ndarray) -> PointsDataSource:
    pos = np.zeros((z.size, 3))
    pos[:, 2] = z
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology.points(pos),
        elements=ElementMeta(position=pos),
        fields=MemoryFieldStore({"u": u.astype(np.float64)}),
    )


def test_profile_interpolation_lifts_to_new_heights():
    p = _profile(np.array([0.0, 10.0, 20.0]), np.array([0.0, 10.0, 20.0]))
    out = profile_interpolation(
        p, ProfileInterpolationParams(target_heights=np.array([5.0, 15.0]))
    )
    np.testing.assert_allclose(out.fields.read("u"), [5.0, 15.0])
    np.testing.assert_allclose(out.elements.position[:, 2], [5.0, 15.0])


def test_s1_recipe_against_identical_reference_yields_one():
    z = np.array([0.0, 5.0, 10.0, 20.0])
    u_cfd = np.array([0.0, 5.0, 8.0, 10.0])
    u_ref = u_cfd.copy()
    cfd = _profile(z, u_cfd)
    ref = _profile(z, u_ref)
    out = build_s1(cfd, ref, S1RecipeConfig())
    # Wall (z=0) is dropped by design.
    np.testing.assert_allclose(out.fields.read("s1"), 1.0)
    np.testing.assert_allclose(out.elements.position[:, 2], [5.0, 10.0, 20.0])


def test_s1_drops_below_threshold_reference_samples():
    z_ref = np.array([0.0, 1.0, 5.0, 10.0])
    u_ref = np.array([0.0, 1e-9, 5.0, 10.0])  # 2nd value is wall-like
    z_cfd = np.array([0.0, 1.0, 5.0, 10.0])
    u_cfd = np.array([0.0, 0.5, 4.5, 9.0])
    out = build_s1(_profile(z_cfd, u_cfd), _profile(z_ref, u_ref), S1RecipeConfig())
    # z=0 (wall) and z=1 (below threshold) both removed.
    np.testing.assert_allclose(out.elements.position[:, 2], [5.0, 10.0])
    np.testing.assert_allclose(out.fields.read("s1"), [4.5 / 5.0, 9.0 / 10.0])


def test_s1_interpolates_cfd_onto_reference_heights():
    z_ref = np.array([0.0, 5.0, 15.0])
    u_ref = np.array([0.0, 5.0, 15.0])
    z_cfd = np.array([0.0, 10.0, 20.0])  # different sampling
    u_cfd = np.array([0.0, 10.0, 20.0])
    out = build_s1(_profile(z_cfd, u_cfd), _profile(z_ref, u_ref), S1RecipeConfig())
    np.testing.assert_allclose(out.fields.read("s1"), 1.0)
