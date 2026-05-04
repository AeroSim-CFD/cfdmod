"""Round-trip tests for grouping-chain serialization."""

from __future__ import annotations

import pytest

from cfdmod.geometry import (
    ByConnectivityGrouping,
    ByCylindricalGrouping,
    ByDivisionsGrouping,
    ByNormalGrouping,
    ByPercentileGrouping,
    ByPlaneGrouping,
    BySizeGrouping,
    BySurfaceGrouping,
    ByZoningGrouping,
    CustomGrouping,
    dump_groupings,
    load_groupings,
)


def test_round_trip_preserves_specs():
    chain = [
        BySurfaceGrouping(sets={"body": ["A", "B"]}),
        ByZoningGrouping(
            x_intervals=[0.0, 1.0, 2.0],
            name_template="r{idx}",
            restrict_to=["body"],
        ),
        ByConnectivityGrouping(min_triangles=2, restrict_to=["body"]),
    ]
    serialized = dump_groupings(chain)
    assert isinstance(serialized, list)
    assert all(isinstance(d, dict) and "kind" in d for d in serialized)

    rehydrated = load_groupings(serialized)
    assert rehydrated == chain


def test_kind_discriminator_routes_to_right_class():
    serialized = [
        {"kind": "by_surface", "sets": {"x": ["A"]}},
        {"kind": "by_zoning", "x_intervals": [0.0, 1.0]},
        {"kind": "by_divisions", "n_div_x": 4},
        {"kind": "by_size", "size_x": 0.5},
        {"kind": "by_connectivity", "min_triangles": 1},
        {"kind": "by_normal", "axes": ["+x", "-x"]},
        {"kind": "by_plane", "point": [0, 0, 0], "normal": [1, 0, 0]},
        {"kind": "by_percentile", "axis": "x", "n_quantiles": 4},
        {
            "kind": "by_cylindrical",
            "origin": [0, 0, 0],
            "axis": "z",
            "theta_intervals_deg": [0, 90, 180, 270, 360],
        },
        {
            "kind": "by_custom",
            "callback": "tests.geometry.grouping._custom_callbacks.first_n",
            "params": {"n": 3, "name": "head"},
        },
    ]
    chain = load_groupings(serialized)
    assert isinstance(chain[0], BySurfaceGrouping)
    assert isinstance(chain[1], ByZoningGrouping)
    assert isinstance(chain[2], ByDivisionsGrouping)
    assert isinstance(chain[3], BySizeGrouping)
    assert isinstance(chain[4], ByConnectivityGrouping)
    assert isinstance(chain[5], ByNormalGrouping)
    assert isinstance(chain[6], ByPlaneGrouping)
    assert isinstance(chain[7], ByPercentileGrouping)
    assert isinstance(chain[8], ByCylindricalGrouping)
    assert isinstance(chain[9], CustomGrouping)


def test_unknown_kind_raises():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_groupings([{"kind": "by_made_up", "foo": 1}])


def test_serialized_form_is_yaml_safe():
    """dump_groupings output must be plain Python types (no numpy/pydantic
    objects), so ``write_processing_metadata`` can YAML-serialize it.
    """
    chain = [
        BySurfaceGrouping(sets={"x": ["A"], "y": ["B"]}),
        ByZoningGrouping(x_intervals=[0.0, 1.0, 2.0], restrict_to=["x"]),
    ]
    serialized = dump_groupings(chain)
    # Walk the structure and verify every value is JSON/YAML-safe.
    safe_types = (str, int, float, bool, type(None), list, dict)
    stack = list(serialized)
    while stack:
        node = stack.pop()
        assert isinstance(node, safe_types), f"unsafe type: {type(node).__name__}"
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
