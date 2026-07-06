"""Cf recipe -- per-body directional aggregation of Cp components.

.. note::

   This small-data recipe computes the **area-weighted mean** of
   pre-existing per-direction Cp fields (``cp_x``/``cp_y``/``cp_z``) over
   each body -- a directional mean pressure coefficient. It is *not* the
   summed force coefficient; the canonical Cf (``-cp*area*n`` summed per
   body via ``force_contribution`` + ``field_series_for_groups(sum)``) is
   the ``fixtures/tests/pressure/templates/cf.yaml`` template run through
   ``cfdmod run``. Unlike :mod:`cfdmod.core.recipes.cm` (which sums moment
   contributions), this recipe means the components; the two differ on
   purpose. The caller must supply the ``cp_<dir>`` fields -- no built-in
   op produces them.

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
