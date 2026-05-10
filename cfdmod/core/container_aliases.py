"""Typed aliases proving Container is the right shape for legacy collections.

The legacy ``cfdmod.hfpi.handler.HFPIAnalysisResults`` predates the
generic :class:`cfdmod.core.container.Container`. This module documents
the equivalence with a typed alias so consumers can write::

    from cfdmod.core.container_aliases import HFPIContainer

without depending on ``cfdmod.hfpi`` directly.

The legacy class is *not* replaced -- it carries disk-backed
loading/saving methods that downstream code depends on. The Phase 9
public API will surface :class:`Container` next to it; the actual
constructor swap is a v4 concern.
"""

from __future__ import annotations

__all__ = ["HFPIContainer"]

from typing import TYPE_CHECKING, Any

from cfdmod.core.container import Container

if TYPE_CHECKING:
    from cfdmod.hfpi.common import HFPICaseParameters


HFPIContainer = Container[Any, Any]
"""Generic-key generic-value container, parameterised at use-site as
``Container[HFPICaseParameters, ResultType]``.

Kept as a runtime alias rather than a parameterised type so import
order stays clean (the core layer must not depend on
``cfdmod.hfpi``)."""
