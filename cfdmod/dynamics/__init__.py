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
    "DimensionalData",
    "read_force_h5",
    "write_force_h5",
    "build_floor_load_source",
    "BuildingCaseParameters",
    "build_cases",
    "solve_building_cases",
    "filter_by_recurrence_period",
    "filter_by_xi",
    "filter_by_kd",
    "join_by_direction",
    "join_by_recurrence_period",
    "get_max_acceleration",
    "get_max_acceleration_by_recurrence_period",
    "get_stats_forces_effective",
    "get_global_peaks_by_direction",
    "read_tqs_portels",
    "read_tqs_portico",
    "read_eberick",
    "EberickUnits",
    "aggregate_to_building",
    "NodalModel",
]

from cfdmod.dynamics.cases import (
    BuildingCaseParameters,
    build_cases,
    filter_by_kd,
    filter_by_recurrence_period,
    filter_by_xi,
    get_global_peaks_by_direction,
    get_max_acceleration,
    get_max_acceleration_by_recurrence_period,
    get_stats_forces_effective,
    join_by_direction,
    join_by_recurrence_period,
    solve_building_cases,
)
from cfdmod.dynamics.forces import (
    DimensionalData,
    build_floor_load_source,
    read_force_h5,
    write_force_h5,
)
from cfdmod.dynamics.imports import (
    EberickUnits,
    NodalModel,
    aggregate_to_building,
    read_eberick,
    read_tqs_portels,
    read_tqs_portico,
)
from cfdmod.dynamics.structural import (
    BuildingStructuralData,
    mass_normalize_mode_shapes,
    read_floors_csv,
    read_mode_shape_csv,
    read_modes_csv,
)
