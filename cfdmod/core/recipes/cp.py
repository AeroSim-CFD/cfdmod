"""Cp recipe -- the literal odt formula.

::

    Cp = (p_body - p_ref) / dyn_pressure

The recipe assembles three primitive ops:

1. ``algebra.sub(body, p_ref)`` over the pressure field -- column-wise
   broadcast (rule 2) when ``p_ref`` is a single probe sharing the time
   axis; constant when it is a scalar.
2. ``field.scale(by = 1 / dyn_pressure)`` -- constant-broadcast (rule 1).
3. Optional ``rescale_time`` to convert solver time to convective time.
4. Optional ``compute_statistics`` -- collapses the time axis into
   ``mean / rms / peak_min / peak_max`` outputs.

Because steps 1 and 2 take a *second* operand (the reference data
source / scalar), the pipeline they form is not a strict
``Pipeline = Callable[[ds], ds]`` -- it is built as a closure that
binds the rhs at recipe-construction time. The convenience builder
:func:`build_cp` does exactly that.
"""

from __future__ import annotations

__all__ = ["CpRecipeConfig", "cp_pipeline", "build_cp"]

from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from cfdmod.core import algebra
from cfdmod.core.data_source import DataSource
from cfdmod.core.ops.data_source_create.statistics import (
    STAT_KINDS,
    StatisticsParams,
    compute_statistics,
)
from cfdmod.core.ops.time.rescale import RescaleTimeParams, rescale
from cfdmod.core.pipeline import Pipeline, compose


class CpRecipeConfig(BaseModel):
    """Cp pipeline parameters.

    Attributes:
        field: Source pressure field name. Defaults to ``"pressure"``.
        out: Output Cp field name. Defaults to ``"cp"``.
        dynamic_pressure: ``0.5 * rho * U_ref^2``; Cp's denominator.
            Required.
        time_rescale_factor: Optional scalar applied to the time axis
            (e.g. ``U_ref / L`` for convective time). ``None`` -> leave
            time axis untouched.
        statistics: Optional list of statistics to compute on the
            resulting Cp series. ``None`` -> no statistics step (the
            output keeps its full time axis).
    """

    model_config = ConfigDict(frozen=True)

    field: str = "pressure"
    out: str = "cp"
    dynamic_pressure: float = Field(gt=0)
    time_rescale_factor: float | None = None
    statistics: list[STAT_KINDS] | None = None


def cp_pipeline(cfg: CpRecipeConfig, p_ref: DataSource | float) -> Pipeline:
    """Assemble the Cp pipeline as a single-arg callable on the body data source.

    Args:
        cfg: Recipe parameters.
        p_ref: Either a constant (uniform reference) or a
            :class:`DataSource` whose ``cfg.field`` shares the time
            axis with the body source (column-wise broadcast).
    """
    inv_q = 1.0 / cfg.dynamic_pressure

    def step_subtract(ds: DataSource) -> DataSource:
        return algebra.sub(ds, p_ref, field=cfg.field, out=cfg.out)

    def step_scale(ds: DataSource) -> DataSource:
        return algebra.mul(ds, inv_q, field=cfg.out)

    steps: list[Callable[[DataSource], DataSource]] = [step_subtract, step_scale]

    if cfg.time_rescale_factor is not None:
        steps.append(lambda ds: rescale(ds, RescaleTimeParams(factor=cfg.time_rescale_factor)))

    if cfg.statistics:
        steps.append(
            lambda ds: compute_statistics(
                ds, StatisticsParams(kinds=cfg.statistics, field=cfg.out)
            )
        )

    return compose(*steps)


def build_cp(
    body: DataSource,
    p_ref: DataSource | float,
    cfg: CpRecipeConfig,
) -> DataSource:
    """One-shot convenience -- build the pipeline and run it."""
    return cp_pipeline(cfg, p_ref)(body)
