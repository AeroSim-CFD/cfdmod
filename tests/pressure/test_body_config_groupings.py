"""Tests for the BodyConfig.groupings opt-in field (step 5, issue #128).

Verifies the back-compat shim:
- Legacy YAML (sub_bodies only) keeps producing the canonical chain.
- Explicit groupings replace the implicit chain.
- Setting both groupings and a non-default sub_bodies is rejected.
"""

from __future__ import annotations

import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.geometry import (
    ByConnectivityGrouping,
    BySurfaceGrouping,
    ByZoningGrouping,
)
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.pressure.geometry import build_geometry_data, get_geometry_data
from cfdmod.pressure.parameters import BodyConfig, MomentBodyConfig, ZoningModel

pytestmark = pytest.mark.unit


@pytest.fixture()
def simple_mesh() -> LnasFormat:
    """4-triangle mesh with two named surfaces (A: tris 0-1, B: tris 2-3)."""
    vertices = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.uint32
    )
    geometry = LnasGeometry(vertices=vertices, triangles=triangles)
    return LnasFormat(
        version="v1.0",
        geometry=geometry,
        surfaces={
            "A": np.array([0, 1], dtype=np.uint32),
            "B": np.array([2, 3], dtype=np.uint32),
        },
    )


def test_legacy_sub_bodies_yields_canonical_chain():
    body = BodyConfig(name="pack")
    chain = body.resolved_groupings(["A", "B"])

    assert len(chain) == 2
    assert isinstance(chain[0], BySurfaceGrouping)
    assert chain[0].sets == {"pack": ["A", "B"]}

    assert isinstance(chain[1], ByZoningGrouping)
    assert chain[1].name_template == "{idx}-pack"
    assert chain[1].restrict_to == ["pack"]


def test_explicit_groupings_replace_legacy(simple_mesh):
    custom = [
        BySurfaceGrouping(sets={"pack": ["A"], "wing": ["B"]}),
        ByConnectivityGrouping(restrict_to=["pack"]),
    ]
    body = BodyConfig(name="pack", groupings=custom)
    assert body.resolved_groupings(["A", "B"]) == custom


def test_groupings_with_default_sub_bodies_is_allowed():
    """Default sub_bodies (-inf..inf) does not conflict with explicit groupings."""
    custom = [BySurfaceGrouping(sets={"pack": ["A"]})]
    body = BodyConfig(name="pack", groupings=custom)
    assert body.resolved_groupings(["A"]) == custom


def test_groupings_with_non_default_sub_bodies_rejected():
    custom = [BySurfaceGrouping(sets={"pack": ["A"]})]
    with pytest.raises(ValueError, match="cannot set both 'groupings'"):
        BodyConfig(
            name="pack",
            sub_bodies=ZoningModel(x_intervals=[0.0, 5.0, 10.0]),
            groupings=custom,
        )


def test_get_geometry_data_honors_explicit_groupings(simple_mesh):
    """End-to-end: a BodyConfig with explicit groupings drives the chain
    that get_geometry_data applies."""
    body = BodyConfig(
        name="pack",
        groupings=[BySurfaceGrouping(sets={"pack": ["A"]})],
    )
    data = get_geometry_data(
        body_cfg=body,
        sfc_list=["A", "B"],
        mesh=simple_mesh,
        transformation=TransformationConfig(),
    )
    # The custom chain only mentions surface "A" -- the result must
    # have the "pack" group covering tris 0,1 only, and no zoning cell
    # group at all (the user dropped the canonical ByZoning step).
    assert "pack" in data.grouping.groups
    assert sorted(data.grouping.groups["pack"].tolist()) == [0, 1]
    assert all(not name.endswith("-pack") or name == "pack"
               for name in data.grouping.groups)


def test_moment_body_inherits_groupings_field():
    """MomentBodyConfig extends BodyConfig and must accept groupings too."""
    custom = [BySurfaceGrouping(sets={"pack": ["A"]})]
    body = MomentBodyConfig(name="pack", groupings=custom)
    assert body.resolved_groupings(["A"]) == custom


def test_yaml_fixture_loads_with_explicit_chain(tmp_path):
    """The new Cf_params_groupings.yaml fixture parses into a CfCaseConfig
    whose body carries the user-provided chain (no implicit sub_bodies).
    """
    import pathlib

    from cfdmod.pressure.parameters import CfCaseConfig

    fixtures = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests" / "pressure"
    cfg = CfCaseConfig.from_file(fixtures / "Cf_params_groupings.yaml")
    body = cfg.force_coefficient["measurement_explicit_chain"].bodies[0]
    assert body.name == "lanternim"
    assert body.groupings is not None
    assert len(body.groupings) == 3
    assert isinstance(body.groupings[0], BySurfaceGrouping)
    assert isinstance(body.groupings[1], ByZoningGrouping)
    assert isinstance(body.groupings[2], ByConnectivityGrouping)


def test_legacy_chain_round_trip_via_canonical_template(simple_mesh):
    """The synthesized canonical chain must produce exactly the same
    GeometryData as constructing one directly via build_geometry_data."""
    body = BodyConfig(name="pack", sub_bodies=ZoningModel(x_intervals=[0.0, 1.5, 3.0]))
    transformation = TransformationConfig()

    legacy = get_geometry_data(
        body_cfg=body, sfc_list=["A", "B"], mesh=simple_mesh, transformation=transformation
    )
    direct = build_geometry_data(
        body_label="pack",
        sfc_list=["A", "B"],
        zoning=body.sub_bodies,
        mesh=simple_mesh,
        transformation=transformation,
    )
    assert set(legacy.grouping.groups) == set(direct.grouping.groups)
    for name in legacy.grouping.groups:
        np.testing.assert_array_equal(
            legacy.grouping.groups[name], direct.grouping.groups[name]
        )
