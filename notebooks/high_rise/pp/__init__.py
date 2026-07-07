"""High-rise post-processing helpers (notebook-side, not part of the cfdmod library).

These modules keep the presentation / orchestration / debug-IO logic out of the
notebooks themselves, so each stage notebook stays a thin driver. Computational
logic lives in the cfdmod library (recipes / ops); everything here is glue,
plotting, and file layout.

Layout:
    case         -- HighRiseCase: aggregate the case_data config (global_data.json,
                    params yaml, floor heights, wind analysis).
    debug_io     -- DebugWriter: versioned debug/ and deliverables/ output roots.
    plotting     -- shared matplotlib style + savefig helper.
    inflow_report-- vertical-profile detection + inflow validation figures.
    pressure     -- v3 recipe/op wiring for Cp and per-floor Cf/Cm.
"""

from __future__ import annotations

from pp.case import HighRiseCase, example_high_rise_case
from pp.debug_io import DebugWriter
from pp.inflow_report import ProfileLine, detect_profiles, reference_velocity
from pp.pressure import cf_per_floor, cm_per_floor, cp_from_pressure

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
]
