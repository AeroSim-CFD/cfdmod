"""Pedestrian comfort recipe -- velocities at probes + climate -> per-probe stats.

Per the odt: a volume / point velocity field, evaluated at probe
positions, is then combined with climate data (Weibull / wind-rose
inputs from outside the pipeline) to compute comfort statistics on
points.

The recipe implemented here covers the *pipeline-internal* portion:

1. probe extraction from a source data source -> per-probe timeseries;
2. statistics (mean / rms / peak_max) on the per-probe series.

Climate ingestion is intentionally *not* a pipeline stage (per the
odt). A downstream consumer combines the per-probe statistics with a
``cfdmod.climate`` summary to produce comfort categories.
"""

from __future__ import annotations

__all__ = ["PedestrianComfortConfig", "build_pedestrian_comfort"]

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from cfdmod.core.data_source import DataSource, PointsDataSource
from cfdmod.core.ops.data_source_create.probe_extraction import (
    ProbeExtractionParams,
    probe_extraction,
)
from cfdmod.core.ops.data_source_create.statistics import (
    STAT_KINDS,
    StatisticsParams,
    compute_statistics,
)


class PedestrianComfortConfig(BaseModel):
    """Pedestrian comfort recipe parameters.

    Attributes:
        probes: ``(n_probes, 3)`` probe positions.
        field: Velocity field on the source (e.g. ``"u_mag"``).
        statistics: Statistics to compute per probe.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    probes: Any
    field: str = "u_mag"
    statistics: list[STAT_KINDS] = ["mean", "rms", "peak_max"]


def build_pedestrian_comfort(
    velocity_source: DataSource, cfg: PedestrianComfortConfig
) -> PointsDataSource:
    probes = np.asarray(cfg.probes, dtype=np.float64)
    extracted = probe_extraction(
        velocity_source,
        ProbeExtractionParams(probes=probes, field=cfg.field, mode="nearest"),
    )
    return compute_statistics(
        extracted,
        StatisticsParams(kinds=cfg.statistics, field=cfg.field),
    )
