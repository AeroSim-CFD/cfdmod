"""Converters from structural-engineering exports to the internal modal format.

The building dynamic-response recipe consumes a
:class:`~cfdmod.dynamics.structural.BuildingStructuralData` (per-floor
mass / inertia / radius / centre-of-mass, modal periods, and per-floor
mode shapes). Structural engineers deliver that model in tool-specific
exports; this package converts them:

- :func:`read_tqs_portels` -- TQS "Portico Espacial" (PORTELS) nodal text export.
- :func:`read_eberick` -- Eberick per-floor (rigid-diaphragm) tabular export.
- :func:`aggregate_to_building` -- the shared nodal -> per-floor core.
"""

from __future__ import annotations

__all__ = [
    "NodalModel",
    "aggregate_to_building",
    "read_tqs_portels",
    "read_eberick",
    "EberickUnits",
]

from cfdmod.dynamics.imports.eberick import EberickUnits, read_eberick
from cfdmod.dynamics.imports.nodal import NodalModel, aggregate_to_building
from cfdmod.dynamics.imports.tqs import read_tqs_portels
