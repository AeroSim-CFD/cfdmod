"""High-level pipelines (Cp, Cf, Cm, Ce, S1, ...) composed from the ops.

Each recipe builds a :class:`cfdmod.core.pipeline.Pipeline` from
:mod:`cfdmod.core.ops` and a small Pydantic config. Recipes are pure --
no I/O, no logging. They are run by the shell (CLI / notebook), which
sources ``DataSource`` instances from a :class:`Storage` and writes the
results back.

Available recipes:

- :mod:`cfdmod.core.recipes.cp` -- pressure coefficient.
- :mod:`cfdmod.core.recipes.cf`, :mod:`.cm`, :mod:`.ce` -- force,
  moment, shape coefficients.
- :mod:`cfdmod.core.recipes.s1`,
  :mod:`cfdmod.core.recipes.pedestrian_comfort`.
- :mod:`cfdmod.core.recipes.dynamic` -- modal-coordinate response.

End-to-end pipelines also assemble from YAML templates -- see
:func:`run_yaml` and the templates under
``fixtures/tests/pressure/templates/``. The legacy disk-first Python
entry points (``cfdmod.pressure.run.run_cp`` etc.) were removed in v3.
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
    "DynamicAnalysisConfig",
    "build_dynamic_response",
    "identity_solver",
    "run_yaml",
]

from cfdmod.core.recipes.ce import CeRecipeConfig, ce_pipeline
from cfdmod.core.recipes.cf import CfRecipeConfig, cf_pipeline
from cfdmod.core.recipes.cm import CmRecipeConfig, cm_pipeline
from cfdmod.core.recipes.cp import CpRecipeConfig, build_cp, cp_pipeline
from cfdmod.core.recipes.pedestrian_comfort import (
    PedestrianComfortConfig,
    build_pedestrian_comfort,
)
from cfdmod.core.recipes.dynamic import (
    DynamicAnalysisConfig,
    build_dynamic_response,
    identity_solver,
)
from cfdmod.core.recipes.run_yaml import run_yaml
from cfdmod.core.recipes.s1 import S1RecipeConfig, build_s1, s1_pipeline
