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
    # Filter chain (opt-in pipeline step on any timeseries H5)
    "MovingAverageFilter",
    "FilterSpec",
    "apply_filters",
    # Stats reader (compute stats over an existing timeseries H5)
    "calculate_statistics_from_h5",
]

from cfdmod.pressure.filters import FilterSpec, MovingAverageFilter, apply_filters
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
from cfdmod.pressure.statistics_runner import calculate_statistics_from_h5
