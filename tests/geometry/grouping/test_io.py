"""Round-trip tests for grouping-chain serialization."""

from __future__ import annotations

import pytest

from cfdmod.geometry import (
    ByConnectivityGrouping,
    BySurfaceGrouping,
    ByZoningGrouping,
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
        {"kind": "by_connectivity", "min_triangles": 1},
    ]
    chain = load_groupings(serialized)
    assert isinstance(chain[0], BySurfaceGrouping)
    assert isinstance(chain[1], ByZoningGrouping)
    assert isinstance(chain[2], ByConnectivityGrouping)


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
