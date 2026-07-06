"""Public op catalog (issue #147).

The op registry must be populated eagerly on import, and ``list_ops`` /
``op_info`` must expose each op's contract plus its parameter JSON Schema
so a service consumer can enumerate the op set without running a template.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

_VALID_KINDS = {"surface", "volume", "points", "groups", "modes"}
_VALID_PRODUCES = _VALID_KINDS | {"same"}
_VALID_FAMILIES = {"time", "geometric", "source_create", "field"}


def test_registry_populated_on_fresh_import():
    """A fresh interpreter that only imports cfdmod sees a non-empty registry.

    Guards the eager ``_populate_default_registry()`` call: a consumer must
    be able to enumerate ops without first running a template.
    """
    code = "import cfdmod; assert len(cfdmod.OP_REGISTRY) >= 20, len(cfdmod.OP_REGISTRY)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_list_ops_covers_every_registered_kind():
    from cfdmod import OP_REGISTRY, list_ops

    infos = list_ops()
    assert {i.kind for i in infos} == set(OP_REGISTRY)
    assert [i.kind for i in infos] == sorted(i.kind for i in infos)


def test_op_info_fields_are_well_formed():
    from cfdmod import list_ops

    for info in list_ops():
        assert info.family in _VALID_FAMILIES, info.kind
        assert info.arity in {"unary", "binary"}, info.kind
        assert info.produces in _VALID_PRODUCES, (info.kind, info.produces)
        if info.consumes is not None:
            assert set(info.consumes) <= _VALID_KINDS, (info.kind, info.consumes)
        assert isinstance(info.params_schema, dict) and info.params_schema, info.kind
        # The params schema is the op's public parameter contract.
        assert "properties" in info.params_schema, info.kind


def test_op_info_single_and_unknown():
    from cfdmod import op_info

    stats = op_info("statistics")
    assert stats.kind == "statistics"
    assert stats.family == "source_create"
    assert stats.replaces_fields is True
    assert "kinds" in stats.params_schema["properties"]

    with pytest.raises(KeyError):
        op_info("does_not_exist")


def test_custom_op_appears_in_catalog():
    """A consumer-registered op is a first-class catalog citizen (#147).

    Registering a custom op with a contract-bearing OpParams subclass must
    surface it in list_ops / op_info with the declared contract and a
    parameter schema -- no different from a built-in.
    """
    from typing import ClassVar, Literal

    from cfdmod.core import DataSource, list_ops, op_info, register_op
    from cfdmod.core.ops import OpParams

    class ClipParams(OpParams):
        kind: Literal["clip_custom"] = "clip_custom"
        field: str = "cp"
        lo: float = 0.0
        hi: float = 1.0
        op_family: ClassVar[str] = "field"
        produces: ClassVar[str] = "same"

    def clip(ds: DataSource, p: ClipParams) -> DataSource:  # pragma: no cover - not run
        return ds

    register_op("clip_custom", clip, ClipParams, arity="unary")
    try:
        info = op_info("clip_custom")
        assert info.kind == "clip_custom"
        assert info.family == "field"  # explicit op_family wins over module path
        assert info.produces == "same"
        assert "lo" in info.params_schema["properties"]
        assert "clip_custom" in {i.kind for i in list_ops()}
    finally:
        from cfdmod.core.pipeline_yaml import OP_REGISTRY

        OP_REGISTRY.pop("clip_custom", None)


def test_contract_examples():
    """Spot-check a few contracts that downstream validation relies on."""
    from cfdmod import op_info

    assert set(op_info("force_contribution").requires_element_meta) == {"area", "normal"}
    assert set(op_info("mesh_attach").produces_element_meta) == {"area", "normal", "position"}
    assert op_info("field_series_for_groups").produces == "groups"
    assert op_info("field_series_for_groups").consumes == ["surface"]
    assert op_info("modal_recomposition").consumes == ["modes"]
