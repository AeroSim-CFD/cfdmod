"""Building wind-load post-processing.

Application-directed helpers that wire the core recipes/ops into the
per-floor pressure, force/moment and dynamic-response flow shared by
building wind studies (high-rise, low-rise, ...). Lower-level computation
lives in ``cfdmod.core`` (recipes/ops) and ``cfdmod.dynamics``; this package
is the building-specific glue on top.
"""

from __future__ import annotations

from cfdmod.building.case import BuildingCase, example_building_case
from cfdmod.building.comfort import (
    Occupancy,
    Standard,
    comfort_limit,
    melbourne1992_acceleration_limit,
    milli_g_to_mps2,
    mps2_to_milli_g,
    nbcc_acceleration_limit,
    nbr6123_acceleration_limit,
)
from cfdmod.building.dynamic import (
    example_building_structure,
    floor_accelerations,
    floor_load_source,
    peak_response_table,
    solve_building_response,
    structure_from_csvs,
)
from cfdmod.building.modes_report import (
    plot_floor_mass,
    plot_mode_shape,
    plot_natural_frequencies,
)
from cfdmod.building.peaks import PeakMethod, gust_peak_factor, peak_value
from cfdmod.building.pressure import cf_per_floor, cm_per_floor, cp_from_pressure

__all__ = [
    "BuildingCase",
    "example_building_case",
    "cp_from_pressure",
    "cf_per_floor",
    "cm_per_floor",
    "floor_load_source",
    "example_building_structure",
    "structure_from_csvs",
    "solve_building_response",
    "floor_accelerations",
    "peak_response_table",
    "PeakMethod",
    "gust_peak_factor",
    "peak_value",
    "Occupancy",
    "Standard",
    "comfort_limit",
    "nbr6123_acceleration_limit",
    "melbourne1992_acceleration_limit",
    "nbcc_acceleration_limit",
    "milli_g_to_mps2",
    "mps2_to_milli_g",
    "plot_mode_shape",
    "plot_floor_mass",
    "plot_natural_frequencies",
]
