"""Static contract validation of pipeline templates (issue #147).

``validate_template`` runs a symbolic pass over the op catalog: it checks
each step's declared ``consumes`` kind, ``requires_element_meta``, and
field reads against the incoming binding, without running any op. These
tests cover the graph-wiring mistakes a node editor can produce, and
guard that valid templates (including the shipped fixtures) still pass.
"""

from __future__ import annotations

import pathlib

import pytest

from cfdmod.core import PipelineTemplate
from cfdmod.core.pipeline_yaml import load_template, validate_template

pytestmark = pytest.mark.unit

_FIXTURES = pathlib.Path(__file__).parents[2] / "fixtures/tests/pressure/templates"


@pytest.mark.parametrize("name", ["cp", "cf", "cm", "ce"])
def test_shipped_fixture_templates_validate(name):
    """The real cp/cf/cm/ce templates must pass the stricter linter."""
    load_template(_FIXTURES / f"{name}.yaml")


def test_force_contribution_before_mesh_attach_is_rejected():
    """cf math needs area+normal; running force_contribution first must fail."""
    tpl = PipelineTemplate(
        name="bad_cf",
        root="/tmp",
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {
                "id": "forces",
                "kind": "force_contribution",
                "source": "body",
                "nominal_area": 1.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="element metadata.*area.*normal|area.*normal"):
        validate_template(tpl)


def test_mesh_attach_then_force_contribution_is_accepted():
    tpl = PipelineTemplate(
        name="ok_cf",
        root="/tmp",
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {"id": "meshed", "kind": "mesh_attach", "source": "body", "mesh": "m.lnas"},
            {
                "id": "forces",
                "kind": "force_contribution",
                "source": "meshed",
                "nominal_area": 1.0,
            },
        ],
    )
    validate_template(tpl)  # no raise


def test_surface_only_op_on_points_binding_is_rejected():
    """field_series_for_groups consumes a surface; a points source must fail."""
    tpl = PipelineTemplate(
        name="bad_kind",
        root="/tmp",
        inputs={"probe": {"kind": "points", "path": "points.probe"}},
        pipeline=[
            {
                "id": "agg",
                "kind": "field_series_for_groups",
                "source": "probe",
                "grouping": "body",
            }
        ],
    )
    with pytest.raises(ValueError, match="consumes a .*surface.* data source"):
        validate_template(tpl)


def test_missing_declared_field_is_rejected():
    """When the input declares its field, reading a different one must fail."""
    tpl = PipelineTemplate(
        name="bad_field",
        root="/tmp",
        inputs={"body": {"kind": "surface", "path": "body", "field": "pressure"}},
        pipeline=[
            {
                "id": "stats",
                "kind": "statistics",
                "source": "body",
                "field": "not_there",
                "kinds": ["mean"],
            }
        ],
    )
    with pytest.raises(ValueError, match="reads field.*not_there"):
        validate_template(tpl)


def test_undeclared_field_is_permissive():
    """With no declared input field the linter must not reject a field read."""
    tpl = PipelineTemplate(
        name="ok_field",
        root="/tmp",
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {
                "id": "stats",
                "kind": "statistics",
                "source": "body",
                "field": "whatever",
                "kinds": ["mean"],
            }
        ],
    )
    validate_template(tpl)  # no raise: fields unknown -> permissive


def test_kind_propagates_through_produced_source():
    """A groups-producing step must expose a groups binding downstream.

    field_series_for_groups produces groups; feeding that into another
    surface-only op must be rejected on kind.
    """
    tpl = PipelineTemplate(
        name="chain",
        root="/tmp",
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {"id": "meshed", "kind": "mesh_attach", "source": "body", "mesh": "m.lnas"},
            {
                "id": "grp",
                "kind": "body_grouping",
                "source": "meshed",
                "mesh": "m.lnas",
                "bodies": {"building": []},
            },
            {
                "id": "agg",
                "kind": "field_series_for_groups",
                "source": "grp",
                "grouping": "body",
            },
            # regroup_topology also consumes a surface; agg is now groups.
            {"id": "re", "kind": "regroup_topology", "source": "agg", "mesh": "m.lnas"},
        ],
    )
    with pytest.raises(ValueError, match="consumes a .*surface.* data source"):
        validate_template(tpl)
