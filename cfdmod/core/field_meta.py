"""Per-field metadata: human name, unit, and scale factor.

Recipes that consume a :class:`DataSource` look at the field shapes via
the :class:`FieldStore` and at this metadata for unit handling. The
goal is *not* to do unit conversion automatically; a recipe that wants
to convert from Pa to Cp does so explicitly via
``cfdmod.core.algebra``. The metadata is for downstream consumers
(plots, exports) and for sanity checks during pipeline construction.
"""

from __future__ import annotations

__all__ = ["FieldMeta"]

from pydantic import BaseModel, ConfigDict


class FieldMeta(BaseModel):
    """Metadata for one field on a :class:`DataSource`.

    Attributes:
        name: Display name (e.g. ``"pressure"``, ``"u_x"``).
        unit: SI-ish unit string (e.g. ``"Pa"``, ``"m/s"``, ``"-"``).
        scale: Multiplicative scale relative to the unit. Defaults to
            1.0; nonzero values let a recipe carry "the field is in
            ``unit * scale``" without rewriting the array.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    unit: str = "-"
    scale: float = 1.0
