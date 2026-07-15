"""Tests for cfdmod.building.case (BuildingCase + case_data loading)."""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MESH = str(REPO / "fixtures" / "tests" / "pressure" / "galpao" / "galpao.normalized.lnas")

building = pytest.importorskip("cfdmod.building")


@pytest.fixture(scope="module")
def galpao_case():
    return building.example_building_case(MESH, n_floors=3)


def test_example_case_geometry(galpao_case):
    assert galpao_case.n_floors == 3
    assert galpao_case.nominal_area > 0
    assert len(galpao_case.floor_heights) == 4


_MINIMAL_PARAMS = """
pressure_coefficient:
  base:
    fluid_density: 1.225
    simul_U_H: 30.0
force_coefficient:
  fc:
    nominal_area: 100.0
    bodies:
      - name: building
        sub_bodies:
          z_intervals: [0.0, 50.0]
moment_coefficient:
  mc:
    nominal_volume: 1000.0
    bodies:
      - lever_origin: [0.0, 0.0, 0.0]
"""

_GLOBAL = '{"H": 70, "L": 6.95, "V0": 38, "analysis": {"body_name": "building"}}'


def _write_case_data(dir_path: pathlib.Path, alturas: str | None) -> pathlib.Path:
    cd = dir_path / "case_data"
    cd.mkdir(parents=True, exist_ok=True)
    (cd / "global_data.json").write_text(_GLOBAL)
    (cd / "params.yaml").write_text(_MINIMAL_PARAMS)
    if alturas is not None:
        (cd / "alturas.csv").write_text(alturas)
    return cd


def test_from_case_data_reads_floors_from_alturas(tmp_path):
    """alturas.csv is the floor source of truth: N storeys -> N floors."""
    rows = ["Pavimento,z_min,z_max,dz"]
    for i in range(5):
        rows.append(f"{i + 1},{i * 10.0},{(i + 1) * 10.0},10.0")
    cd = _write_case_data(tmp_path, "\n".join(rows) + "\n")
    case = building.BuildingCase.from_case_data(cd, "params.yaml")
    assert case.n_floors == 5
    assert case.floor_heights[0] == 0.0
    assert case.floor_heights[-1] == 50.0


def test_from_case_data_falls_back_to_yaml_when_alturas_empty(tmp_path):
    """A header-only (or missing) alturas.csv falls back to the yaml z_intervals."""
    header_only = _write_case_data(tmp_path, "Pavimento,z_min,z_max,dz\n")
    case = building.BuildingCase.from_case_data(header_only, "params.yaml")
    assert case.floor_heights == [0.0, 50.0]  # from the yaml anchor -> 1 floor
    assert case.n_floors == 1

    missing = _write_case_data(tmp_path / "nocsv", None)
    case2 = building.BuildingCase.from_case_data(missing, "params.yaml")
    assert case2.floor_heights == [0.0, 50.0]
