"""Cf recipe -- per-body directional force coefficient (own-area basis).

Two normalisation conventions produce a force coefficient, and they use
different aggregations:

- **Own-area basis** (this recipe): normalise by the body's own wetted
  area ``A_total``. Then

      Cf_i = (1 / A_total) * sum_k cp_k * area_k * n_k,i
           = area_weighted_mean_k(cp_k * n_k,i),

  because the ``area_k`` weights sum to ``A_total`` and cancel. So the
  **area-weighted mean** of the per-direction Cp component *is* the force
  coefficient on the own-area basis. This recipe takes pre-existing
  ``cp_<dir> = cp * n_<dir>`` fields and area-weight-means them per body.

- **Reference-area basis** (the ``fixtures/tests/pressure/templates/cf.yaml``
  template): normalise by an independent ``A_ref`` (e.g. a frontal area).
  Then ``A_ref != A_total`` does not cancel and Cf is a **sum** of
  ``force_contribution`` (``-cp*area*n / A_ref``) per body. Use this when
  a fixed reference area is required so Cf converts back to force
  unambiguously.

Contrast with :mod:`cfdmod.core.recipes.cm`, which always *sums* moment
contributions (there is no own-area cancellation for a moment). The caller
must supply the ``cp_<dir>`` fields for this recipe -- no built-in op
produces them; the reference-area path uses ``force_contribution`` instead.

Given a Cp data source with a ``"body"`` grouping (triangle -> body id)
and per-direction ``cp_<dir>`` fields, the recipe builds a
:class:`GroupsDataSource` with one output field per direction, all on the
same groups data source.
"""

from __future__ import annotations

__all__ = ["CfRecipeConfig", "cf_pipeline"]

from typing import Literal

from pydantic import BaseModel, ConfigDict

from cfdmod.core.data_source import DataSource, GroupsDataSource
from cfdmod.core.ops.data_source_create.field_series_for_groups import (
    FieldSeriesForGroupsParams,
    field_series_for_groups,
)


class CfRecipeConfig(BaseModel):
    """Cf recipe parameters.

    Attributes:
        grouping: Name of the grouping in ``ds.groupings`` that maps
            triangles to body ids.
        directions: Field-name suffixes per force direction. The recipe
            aggregates ``cp_<dir>`` for each entry. Defaults to the
            three Cartesian components.
        prefix: Source field prefix (``"cp"`` for the standard
            convention).
    """

    model_config = ConfigDict(frozen=True)

    grouping: str
    directions: list[Literal["x", "y", "z"]] = ["x", "y", "z"]
    prefix: str = "cp"


def cf_pipeline(cfg: CfRecipeConfig):
    """Return a single-arg callable producing a :class:`GroupsDataSource`."""

    def run(ds: DataSource) -> GroupsDataSource:
        result: GroupsDataSource | None = None
        for d in cfg.directions:
            field_in = f"{cfg.prefix}_{d}"
            field_out = f"cf_{d}"
            partial = field_series_for_groups(
                ds,
                FieldSeriesForGroupsParams(
                    grouping=cfg.grouping,
                    field=field_in,
                    agg="area_weighted_mean",
                    out=field_out,
                ),
            )
            if result is None:
                result = partial
            else:
                arr = partial.fields.read(field_out)
                meta = partial.field_meta[field_out]
                result = result.with_field(field_out, arr, meta=meta)
        assert result is not None
        return result

    return run
