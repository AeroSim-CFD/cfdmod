"""Field ops -- overwrite or create same-shape fields on a data source.

Field ops never change the shape of the elements axis or the time axis;
they only rewrite a field's values. Numpy-style algebra (the four
broadcasting rules from issue #131) lives in
:mod:`cfdmod.core.algebra`; the temporal-filter family lives here.
"""

from __future__ import annotations

__all__ = [
    "MovingAverageParams",
    "moving_average",
    "DerivativeParams",
    "derivative",
    "FrequencyFilterParams",
    "frequency_filter",
    "AddParams",
    "SubParams",
    "MulParams",
    "DivParams",
    "ScaleParams",
    "add",
    "sub",
    "mul",
    "div",
    "scale",
    "ForceContributionParams",
    "force_contribution",
    "MomentContributionParams",
    "moment_contribution",
]

from cfdmod.core.ops.field.algebra import (
    AddParams,
    DivParams,
    MulParams,
    ScaleParams,
    SubParams,
    add,
    div,
    mul,
    scale,
    sub,
)
from cfdmod.core.ops.field.force_contribution import (
    ForceContributionParams,
    force_contribution,
)
from cfdmod.core.ops.field.moment_contribution import (
    MomentContributionParams,
    moment_contribution,
)
from cfdmod.core.ops.field.derivative import DerivativeParams, derivative
from cfdmod.core.ops.field.frequency_filter import FrequencyFilterParams, frequency_filter
from cfdmod.core.ops.field.moving_average import MovingAverageParams, moving_average
