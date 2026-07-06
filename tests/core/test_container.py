"""Unit tests for :class:`Container`."""

from __future__ import annotations

from cfdmod.core import Container


def test_with_item_and_without_key_are_functional():
    c = Container[str, int](items={"a": 1, "b": 2})
    c2 = c.with_item("c", 3)
    assert "c" not in c
    assert c2["c"] == 3
    c3 = c2.without_key("a")
    assert "a" in c2 and "a" not in c3


def test_filter_by_returns_subcontainer():
    c = Container[str, int](items={"a": 1, "b": 2, "ab": 3})
    sub = c.filter_by(lambda k: k.startswith("a"))
    assert sorted(sub.keys()) == ["a", "ab"]


def test_join_by_partitions_by_derived_key():
    c = Container[str, int](items={"a1": 1, "a2": 2, "b1": 3})
    parts = c.join_by(lambda k: k[0])
    assert sorted(parts.keys()) == ["a", "b"]
    assert sorted(parts["a"].keys()) == ["a1", "a2"]


def test_map_values_runs_func_on_each_value():
    c = Container[str, int](items={"a": 1, "b": 2, "c": 3})
    doubled = c.map_values(lambda v: v * 2)
    assert dict(doubled.items) == {"a": 2, "b": 4, "c": 6}


def test_map_values_uses_pool_when_provided():
    captured: list = []

    class FakePool:
        def map(self, func, iterable):
            vals = list(iterable)
            captured.append(vals)
            return [func(v) for v in vals]

    c = Container[str, int](items={"a": 1, "b": 2})
    out = c.map_values(lambda v: v + 10, pool=FakePool())
    assert dict(out.items) == {"a": 11, "b": 12}
    assert captured == [[1, 2]]


def test_merge_combines_two_containers():
    a = Container[str, int](items={"x": 1, "y": 2})
    b = Container[str, int](items={"y": 20, "z": 3})
    merged = a.merge(b)
    assert dict(merged.items) == {"x": 1, "y": 20, "z": 3}
