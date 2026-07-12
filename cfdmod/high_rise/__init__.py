"""High-rise post-processing helpers.

These modules keep the presentation / orchestration / debug-IO logic out of the
example notebooks, so each stage notebook stays a thin driver. Lower-level
computational logic lives in the core library (recipes / ops); everything here
is high-rise-specific glue, plotting, and file layout.

Layout:
    case         -- HighRiseCase: aggregate the case_data config (global_data.json,
                    params yaml, floor heights, wind analysis).
    debug_io     -- DebugWriter: versioned debug/ and deliverables/ output roots.
    plotting     -- shared matplotlib style + savefig helper.
    inflow_report-- vertical-profile detection + inflow validation figures.
    pressure     -- v3 recipe/op wiring for Cp and per-floor Cf/Cm.
    dynamic      -- floor-load assembly + building dynamic-response recipe wiring.
    snapshots    -- facade / structure mesh-field renders (matplotlib + optional PyVista).
"""

from __future__ import annotations

from . import snapshots
from .case import HighRiseCase, example_high_rise_case
from .debug_io import DebugWriter
from .dynamic import (
    example_building_structure,
    floor_accelerations,
    floor_load_source,
    peak_response_table,
    solve_building_response,
    structure_from_csvs,
)
from .inflow_report import ProfileLine, detect_profiles, reference_velocity
from .pressure import cf_per_floor, cm_per_floor, cp_from_pressure

__all__ = [
    "HighRiseCase",
    "example_high_rise_case",
    "DebugWriter",
    "ProfileLine",
    "detect_profiles",
    "reference_velocity",
    "cp_from_pressure",
    "cf_per_floor",
    "cm_per_floor",
    "floor_load_source",
    "example_building_structure",
    "structure_from_csvs",
    "solve_building_response",
    "floor_accelerations",
    "peak_response_table",
    "snapshots",
]
