"""Pressure module public API.

The pressure pipeline is a single disk-first chain (run_cp -> run_cf /
run_cm / run_ce, optionally interleaved with apply_filters). The
in-memory transform helpers (process_xdmf_to_cp, process_Cf, ...) are
implementation details consumed by run_*; they are reachable as
``cfdmod.pressure.functions.process_*`` for unit tests but are not part
of the public surface.
"""

__all__ = [
    # Configs
    "BasePressureConfig",
    "CpConfig",
    "CpCaseConfig",
    "CfConfig",
    "CfCaseConfig",
    "CmConfig",
    "CmCaseConfig",
    "CeConfig",
    "CeCaseConfig",
    # Bodies / zoning
    "ZoningModel",
    "ZoningConfig",
    "ZoningBuilder",
    "BodyDefinition",
    "BodyConfig",
    "MomentBodyConfig",
    # Statistics models
    "BasicStatisticModel",
    "ParameterizedStatisticModel",
    "Statistics",
    "ExtremeMethods",
    "AxisDirections",
    # Pipeline entry points (the only way to drive the pipeline)
    "run_cp",
    "run_cf",
    "run_cm",
    "run_ce",
    # Filter chain (opt-in pipeline step on any timeseries H5).
    # NOTE: filters have moved to cfdmod.filters; these names are
    # re-exported here for back-compat. ``apply_filters`` here is the
    # H5 wrapper (the previous behaviour). For the pure-numpy core,
    # import ``cfdmod.filters.apply_filters`` or ``cfdmod.apply_filters``.
    "MovingAverageFilter",
    "FilterSpec",
    "apply_filters",
    "apply_filters_h5",
    # Stats reader (compute stats over an existing timeseries H5).
    # NOTE: stats have moved to cfdmod.statistics; ``calculate_statistics_from_h5``
    # is back-compat for ``apply_statistics_h5``.
    "calculate_statistics_from_h5",
    "apply_statistics_h5",
]

from cfdmod.filters import FilterSpec, MovingAverageFilter, apply_filters_h5

# Within the pressure namespace, `apply_filters` keeps its historical
# meaning -- the H5 file-in / file-out flow used by the pressure pipeline.
apply_filters = apply_filters_h5
from cfdmod.pressure.parameters import (
    AxisDirections,
    BasePressureConfig,
    BasicStatisticModel,
    BodyConfig,
    BodyDefinition,
    CeCaseConfig,
    CeConfig,
    CfCaseConfig,
    CfConfig,
    CmCaseConfig,
    CmConfig,
    CpCaseConfig,
    CpConfig,
    ExtremeMethods,
    MomentBodyConfig,
    ParameterizedStatisticModel,
    Statistics,
    ZoningBuilder,
    ZoningConfig,
    ZoningModel,
)
from cfdmod.pressure.run import run_ce, run_cf, run_cm, run_cp
from cfdmod.statistics import apply_statistics_h5

# Within the pressure namespace, ``calculate_statistics_from_h5`` keeps its
# historical meaning as the H5 file flow.
calculate_statistics_from_h5 = apply_statistics_h5
