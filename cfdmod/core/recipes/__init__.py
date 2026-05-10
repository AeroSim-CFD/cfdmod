"""High-level pipelines (Cp, Cf, Cm, Ce, S1, ...) composed from the ops.

Each recipe builds a :class:`cfdmod.core.pipeline.Pipeline` from
:mod:`cfdmod.core.ops` and a small Pydantic config. Recipes are pure --
no I/O, no logging. They are run by the shell (CLI / notebook), which
sources ``DataSource`` instances from a :class:`Storage` and writes the
results back.

Phase 4-7 deliverables:

- :mod:`cfdmod.core.recipes.cp` -- pressure coefficient.
- :mod:`cfdmod.core.recipes.cf`, :mod:`.cm`, :mod:`.ce` -- force,
  moment, shape coefficients (Phase 5).
- :mod:`cfdmod.core.recipes.s1`,
  :mod:`cfdmod.core.recipes.pedestrian_comfort` (Phase 7).

The recipes here always run on the small-data path (``MemoryStorage``).
The legacy disk-first paths in ``cfdmod/pressure/run.py`` and
``cfdmod/inflow.py`` remain as the production entry points until v3.0.
"""

from __future__ import annotations

__all__ = [
    "cp_pipeline",
    "CpRecipeConfig",
    "build_cp",
    "cf_pipeline",
    "CfRecipeConfig",
    "ce_pipeline",
    "CeRecipeConfig",
    "cm_pipeline",
    "CmRecipeConfig",
    "S1RecipeConfig",
    "s1_pipeline",
    "build_s1",
    "PedestrianComfortConfig",
    "build_pedestrian_comfort",
]

from cfdmod.core.recipes.ce import CeRecipeConfig, ce_pipeline
from cfdmod.core.recipes.cf import CfRecipeConfig, cf_pipeline
from cfdmod.core.recipes.cm import CmRecipeConfig, cm_pipeline
from cfdmod.core.recipes.cp import CpRecipeConfig, build_cp, cp_pipeline
from cfdmod.core.recipes.pedestrian_comfort import (
    PedestrianComfortConfig,
    build_pedestrian_comfort,
)
from cfdmod.core.recipes.s1 import S1RecipeConfig, build_s1, s1_pipeline
