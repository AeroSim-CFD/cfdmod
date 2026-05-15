"""Serialization helpers for grouping chains.

The existing :func:`cfdmod.io.write_processing_metadata` takes a generic
``config: dict`` which it serializes to YAML inside the HDF5 file. To
record a grouping chain, callers pass ``{"groupings": [spec.model_dump()
for spec in chain]}`` (alongside ``"filters"``, etc.). The helpers in
this module wrap that idiom so call sites stay legible:

    write_processing_metadata(h5, "/", {"groupings": dump_groupings(chain)})

    md = read_processing_metadata(h5, "/")
    chain = load_groupings(md["config"]["groupings"])

Round-trip: ``load_groupings(dump_groupings(chain)) == chain`` for any
valid chain.
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from cfdmod.geometry.grouping.specs import GroupingSpec

_GROUPING_LIST_ADAPTER = TypeAdapter(list[GroupingSpec])


def dump_groupings(groupings: list[GroupingSpec]) -> list[dict[str, Any]]:
    """Serialize a chain of grouping specs to plain dicts (YAML/JSON-safe).

    Each entry retains its ``kind`` discriminator so :func:`load_groupings`
    can route it back to the correct spec class.

    Args:
        groupings: Chain of validated spec instances.

    Returns:
        ``list[dict]`` suitable for ``write_processing_metadata`` or any
        YAML/JSON serializer.
    """
    return [spec.model_dump(mode="python") for spec in groupings]


def load_groupings(serialized: list[dict[str, Any]]) -> list[GroupingSpec]:
    """Re-hydrate a chain of grouping specs from their dict form.

    Uses the ``GroupingSpec`` discriminated union, so each entry must
    carry a ``kind`` key matching one of the registered spec classes.

    Args:
        serialized: Output of :func:`dump_groupings` (or any list of
            dicts with a valid ``kind`` discriminator).

    Returns:
        Validated spec instances ready to feed to
        :func:`cfdmod.geometry.grouping.apply_groupings`.

    Raises:
        pydantic.ValidationError: If any entry is not a valid spec.
    """
    return _GROUPING_LIST_ADAPTER.validate_python(serialized)
