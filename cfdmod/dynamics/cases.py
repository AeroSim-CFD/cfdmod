"""Directional / parametric case orchestration for building dynamics.

Ports the legacy ``cfdmod.hfpi.handler`` multi-case orchestration onto the
generic :class:`cfdmod.core.container.Container`. A case is identified by a
frozen :class:`BuildingCaseParameters` key (direction, damping, recurrence
period, and the mass / frequency / integral-scale multipliers); the results
form a ``Container[BuildingCaseParameters, PointsDataSource]`` that groups
(``join_by``) and filters (``filter_by``) with no bespoke machinery.

The wind-profile-driven reference-speed lookup and per-direction force
selection stay in the shell (the caller supplies a ``solve_fn`` that turns a
case into a response), keeping this layer free of I/O -- exactly the split
the v3 paradigm asks for.
"""

from __future__ import annotations

__all__ = ["BuildingCaseParameters", "build_cases", "solve_building_cases"]

import itertools
from typing import Callable

from pydantic import BaseModel

from cfdmod.core.container import Container
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.protocols import Pool


class BuildingCaseParameters(BaseModel, frozen=True):
    """Identifies one building dynamic-analysis case (a Container key).

    Frozen and hashable so it can key a
    :class:`~cfdmod.core.container.Container`.
    """

    direction: float
    xi: float
    recurrence_period: float
    use_kd: bool = False
    frequency_multiplier: float = 1.0
    integral_scale_multiplier: float = 1.0
    mass_multiplier: float = 1.0


def build_cases(
    *,
    directions: list[float],
    xis: list[float],
    recurrence_periods: list[float],
    use_kd: list[bool] = [False],
    frequency_multipliers: list[float] = [1.0],
    integral_scale_multipliers: list[float] = [1.0],
    mass_multipliers: list[float] = [1.0],
) -> list[BuildingCaseParameters]:
    """Cartesian product of case knobs -> list of case parameters."""
    return [
        BuildingCaseParameters(
            direction=direction,
            xi=xi,
            use_kd=kd,
            recurrence_period=period,
            frequency_multiplier=fm,
            integral_scale_multiplier=ism,
            mass_multiplier=mm,
        )
        for direction, xi, kd, period, fm, ism, mm in itertools.product(
            directions,
            xis,
            use_kd,
            recurrence_periods,
            frequency_multipliers,
            integral_scale_multipliers,
            mass_multipliers,
        )
    ]


def solve_building_cases(
    cases: list[BuildingCaseParameters],
    solve_fn: Callable[[BuildingCaseParameters], PointsDataSource],
    *,
    pool: Pool | None = None,
) -> Container[BuildingCaseParameters, PointsDataSource]:
    """Solve every case and collect the responses in a Container.

    ``solve_fn`` maps a case to its response data source (the caller wires
    the direction -> forces / dimensional scaling / structural multipliers).
    With ``pool`` the fanout runs through ``pool.map``; otherwise it is
    sequential. Group the result with ``container.join_by(lambda c:
    c.direction)`` and slice it with ``container.filter_by(...)``.
    """
    if pool is None:
        results = [solve_fn(c) for c in cases]
    else:
        results = pool.map(solve_fn, cases)
    return Container(items=dict(zip(cases, results)))
