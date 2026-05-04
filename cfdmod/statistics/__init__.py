"""Statistics for cfdmod timeseries.

A statistics chain reduces a per-timestep series to one summary value
per feature for each requested statistic. Statistics are not coupled
to any specific coefficient: the same dispatch that summarises a Cp
series can summarise any other timeseries the caller supplies.

Two entry points:

- :func:`apply_statistics` -- pure numpy. Pass a ``(n_time,)`` or
  ``(n_time, n_features)`` array, the matching ``time`` axis, and a
  list of stat specs; get back a DataFrame indexed by feature with
  one column per statistic. No file I/O.
- :func:`apply_statistics_h5` -- file-in wrapper around the H5
  layout. Picks a single-pass streaming path when only basic moments
  are requested; falls back to a full load when extreme-value methods
  (Gumbel / Peak / Absolute) or ``mean_eq`` are requested.

Spec types live in :mod:`cfdmod.statistics.specs` (currently re-exported
from ``cfdmod.pressure.parameters``; physical move slated for a follow-
up).
"""

from cfdmod.statistics.core import (
    apply_statistics,
    calculate_extreme_values,
    calculate_mean_equivalent,
    calculate_statistics,
    extreme_values_analysis,
    fit_gumbel_model,
    gumbel_extreme_values,
    peak_extreme_values,
)
from cfdmod.statistics.h5 import apply_statistics_h5
from cfdmod.statistics.specs import (
    BasicStatisticModel,
    ExtremeAbsoluteParamsModel,
    ExtremeGumbelParamsModel,
    ExtremeMethods,
    ExtremePeakParamsModel,
    MeanEquivalentParamsModel,
    ParameterizedStatisticModel,
    Statistics,
    StatisticsParamsModel,
)

__all__ = [
    # Specs
    "Statistics",
    "ExtremeMethods",
    "BasicStatisticModel",
    "ParameterizedStatisticModel",
    "StatisticsParamsModel",
    "ExtremeAbsoluteParamsModel",
    "ExtremeGumbelParamsModel",
    "ExtremePeakParamsModel",
    "MeanEquivalentParamsModel",
    # Entry points
    "apply_statistics",
    "apply_statistics_h5",
    # Lower-level helpers (DataFrame-based, used by pressure pipeline)
    "calculate_statistics",
    "calculate_extreme_values",
    "calculate_mean_equivalent",
    "extreme_values_analysis",
    "fit_gumbel_model",
    "gumbel_extreme_values",
    "peak_extreme_values",
]
