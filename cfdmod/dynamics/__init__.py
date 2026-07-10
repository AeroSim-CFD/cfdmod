"""Building dynamic-response domain module.

Out-of-paradigm inputs and presentation for the v3 dynamic-response
recipe (:func:`cfdmod.core.recipes.dynamic.build_building_dynamic_response`).
Mirrors the ``cfdmod.s1`` layout: the pure recipe/ops live in
``cfdmod.core``; the file ingest that builds the recipe inputs lives here.
"""

from __future__ import annotations

__all__ = [
    "read_modes_csv",
    "read_floors_csv",
    "read_mode_shape_csv",
    "mass_normalize_mode_shapes",
    "BuildingStructuralData",
]

from cfdmod.dynamics.structural import (
    BuildingStructuralData,
    mass_normalize_mode_shapes,
    read_floors_csv,
    read_mode_shape_csv,
    read_modes_csv,
)
