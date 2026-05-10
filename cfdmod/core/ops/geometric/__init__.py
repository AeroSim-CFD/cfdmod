"""Geometric ops -- modify the elements axis metadata of a data source.

Per issue #131:

- Rigid-body transformation (updates positions + normals).
- Rescale (updates areas + volumes).
- Assume position and size of a given geometry (same element count).
- Group-belonging index (just index assignment, no new geometry).

The first three are placeholders for now; only :func:`attach_grouping`
lands here in Phase 5 because the Cf/Cm/Ce recipes need it.
"""

from __future__ import annotations

__all__ = [
    "AttachGroupingParams",
    "attach_grouping",
]

from cfdmod.core.ops.geometric.attach_grouping import AttachGroupingParams, attach_grouping
