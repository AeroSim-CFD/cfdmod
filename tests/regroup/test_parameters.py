"""Tests for ``cfdmod.regroup.parameters``."""

from __future__ import annotations

import pathlib
import textwrap

import pytest
from pydantic import ValidationError

from cfdmod.geometry.grouping import ByConnectivityGrouping, ByZoningGrouping
from cfdmod.regroup.parameters import (
    BySizeRoundedPerComponent,
    RegroupConfig,
)


def _write_yaml(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    p = tmp_path / "regroup.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_round_trip_zoning_only(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        groupings:
          - kind: by_zoning
            x_intervals: [0.0, 1.0, 2.0]
        aggregation: per_triangle
        """,
    )
    cfg = RegroupConfig.from_file(yaml_path)
    assert len(cfg.groupings) == 1
    assert isinstance(cfg.groupings[0], ByZoningGrouping)
    assert cfg.aggregation == "per_triangle"
    assert cfg.timeseries_group == "cp"
    assert cfg.unassigned_policy == "drop"
    assert cfg.output_geometry_format == "lnas"


def test_round_trip_connectivity_then_size_per_component(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        groupings:
          - kind: by_connectivity
            name_template: "container_{idx}"
            min_triangles: 4
          - kind: by_size_rounded_per_component
            target_size_x: 2.0
            target_size_y: 3.0
            target_size_z: 6.0
            name_template: "{parent}_face_r{idx}"
        aggregation: area_weighted_mean
        timeseries_group: cp
        output_geometry_format: lnas_and_stl
        """,
    )
    cfg = RegroupConfig.from_file(yaml_path)
    assert isinstance(cfg.groupings[0], ByConnectivityGrouping)
    assert isinstance(cfg.groupings[1], BySizeRoundedPerComponent)
    assert cfg.groupings[1].target_size_x == 2.0
    assert cfg.aggregation == "area_weighted_mean"
    assert cfg.output_geometry_format == "lnas_and_stl"


def test_fan_out_must_be_last(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        groupings:
          - kind: by_size_rounded_per_component
            target_size_x: 2.0
          - kind: by_zoning
            x_intervals: [0.0, 1.0]
        """,
    )
    with pytest.raises(ValidationError, match="must be the last spec"):
        RegroupConfig.from_file(yaml_path)


def test_fan_out_requires_at_least_one_target(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        groupings:
          - kind: by_connectivity
          - kind: by_size_rounded_per_component
        """,
    )
    with pytest.raises(ValidationError, match="at least one of"):
        RegroupConfig.from_file(yaml_path)


def test_groupings_cannot_be_empty(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        groupings: []
        """,
    )
    with pytest.raises(ValidationError):
        RegroupConfig.from_file(yaml_path)


def test_unknown_kind_rejected(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        groupings:
          - kind: not_a_real_kind
        """,
    )
    with pytest.raises(ValidationError):
        RegroupConfig.from_file(yaml_path)
