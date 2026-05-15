"""Tests for ByNormalGrouping."""

from __future__ import annotations

import pytest

from cfdmod.geometry import ByNormalGrouping, BySurfaceGrouping, apply_groupings


def test_grid_mesh_normals_all_plus_z(grid_mesh):
    # All grid_mesh triangles have outward normal +z.
    spec = ByNormalGrouping()
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"n_+z"}
    assert sorted(res.groups["n_+z"].tolist()) == [0, 1, 2, 3, 4, 5]


def test_cube_six_buckets_two_triangles_each(cube_mesh):
    # 12 triangles, 2 per face direction, default tolerance 45 deg.
    spec = ByNormalGrouping()
    res = apply_groupings(cube_mesh, [spec])
    assert set(res.groups) == {"n_+x", "n_-x", "n_+y", "n_-y", "n_+z", "n_-z"}
    for name in res.groups:
        assert res.groups[name].size == 2


def test_axes_subset_drops_unrequested_buckets(cube_mesh):
    # Only the +x and -x buckets are emitted; faces that best fit the
    # remaining cardinal directions exceed the 45-deg tolerance to +/-x
    # and so are not assigned anywhere.
    spec = ByNormalGrouping(axes=["+x", "-x"])
    res = apply_groupings(cube_mesh, [spec])
    assert set(res.groups) == {"n_+x", "n_-x"}
    assert sorted(res.groups["n_+x"].tolist()) == [0, 1]
    assert sorted(res.groups["n_-x"].tolist()) == [2, 3]


def test_tight_tolerance_excludes_oblique_or_offaxis(grid_mesh):
    # grid_mesh normals are exactly +z. Asking only for +x with a tight
    # tolerance excludes everything (90 deg off axis).
    spec = ByNormalGrouping(axes=["+x"], tolerance_deg=10.0)
    res = apply_groupings(grid_mesh, [spec])
    assert res.groups == {}


def test_restrict_to_filters_candidates(cube_mesh):
    # Take only the +x face triangles via BySurface, then bucket by normal:
    # they should all land in +x.
    surfs = BySurfaceGrouping(sets={"x_face": ["cube"]})
    nrm = ByNormalGrouping(restrict_to=["x_face"], axes=["+x"])
    res = apply_groupings(cube_mesh, [surfs, nrm])
    # Cube has 12 triangles in surface "cube"; only the 2 +x ones pass
    # through the normal filter into n_+x.
    assert res.groups["n_+x"].size == 2


def test_name_template_must_disambiguate(cube_mesh):
    spec = ByNormalGrouping(name_template="constant")
    with pytest.raises(ValueError, match="duplicate group name"):
        apply_groupings(cube_mesh, [spec])


def test_validation_rejects_empty_or_dup_axes():
    with pytest.raises(ValueError):
        ByNormalGrouping(axes=[])
    with pytest.raises(ValueError):
        ByNormalGrouping(axes=["+x", "+x"])


def test_validation_rejects_zero_or_huge_tolerance():
    with pytest.raises(ValueError):
        ByNormalGrouping(tolerance_deg=0.0)
    with pytest.raises(ValueError):
        ByNormalGrouping(tolerance_deg=120.0)
