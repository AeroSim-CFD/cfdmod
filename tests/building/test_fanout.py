"""Per-direction/body/config fan-out driver (global_data.json -> Container)."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from cfdmod.building import (
    FanoutPlan,
    StaticCaseKey,
    build_static_keys,
    build_static_solve_fn,
    example_building_case,
    run_fanout,
)
from cfdmod.report import DebugWriter

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests"
MESH = str(FIX / "pressure" / "galpao" / "galpao.normalized.lnas")
DATA = FIX / "pressure" / "data"


def test_fanout_plan_from_global_data(tmp_path):
    (tmp_path / "global_data.json").write_text(
        json.dumps(
            {
                "H": 70,
                "L": 6.95,
                "V0": 38,
                "analysis": {
                    "batch_name": "b1",
                    "categories": ["III"],
                    "directions_catIII": ["0", "45", "90"],
                    "body_name": "galpao",
                },
            }
        )
    )
    plan = FanoutPlan.from_global_data(tmp_path)
    assert plan.batch_name == "b1"
    assert plan.categories == ["III"]
    assert plan.directions == ["0", "45", "90"]
    assert plan.bodies == ["galpao"]
    assert plan.cp_configs == ["base"]
    assert len(build_static_keys(plan)) == 3  # 3 dir x 1 body x 1 config


def test_fanout_plan_multi_category_union(tmp_path):
    (tmp_path / "global_data.json").write_text(
        json.dumps(
            {
                "H": 70,
                "L": 7,
                "V0": 38,
                "analysis": {
                    "categories": ["III", "IV"],
                    "directions_catIII": ["0", "45"],
                    "directions_catIV": ["45", "90"],
                    "body_name": "galpao",
                },
            }
        )
    )
    plan = FanoutPlan.from_global_data(tmp_path, cp_configs=["base", "alt"])
    assert plan.directions == ["0", "45", "90"]  # order-preserving union
    assert len(build_static_keys(plan)) == 3 * 1 * 2


@pytest.mark.integration
def test_run_fanout_produces_directional_container(tmp_path):
    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage

    storage = XdmfH5Storage(DATA)
    case = example_building_case(MESH, n_floors=3)

    # the galpao fixture is stored under flat keys; ignore the direction axis
    def key_for(kind, key):
        return "bodies.galpao" if kind == "body" else "points.static_pressure"

    plan = FanoutPlan(
        batch_name="b1",
        categories=[""],
        directions_by_category={"": ["0", "90"]},
        bodies=["galpao"],
    )
    solve_fn = build_static_solve_fn(case, storage, MESH, key_for=key_for)
    writer = DebugWriter(tmp_path, "cf_cm", "b1")

    container = run_fanout(plan, solve_fn, writer=writer, case=case)

    assert len(container) == 2  # 2 directions x 1 body x 1 config
    assert all(isinstance(k, StaticCaseKey) for k in container.keys())

    by_direction = container.join_by(lambda k: k.direction)
    assert set(by_direction) == {"0", "90"}

    for response in container.values():
        feq_x = np.asarray(response.fields.read("feq_x"))
        acc_mag = np.asarray(response.fields.read("acc_mag"))
        assert feq_x.ndim == 2 and np.all(np.isfinite(feq_x))
        assert np.all(np.isfinite(acc_mag))

    # provenance written once beside the outputs
    assert (tmp_path / "debug" / "b1" / "cf_cm" / "region_info.csv").exists()
    assert (tmp_path / "debug" / "b1" / "cf_cm" / "config.yaml").exists()


@pytest.mark.integration
def test_run_fanout_pool_seam(tmp_path):
    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage

    class SyncPool:
        def map(self, func, iterable):
            return [func(x) for x in iterable]

    storage = XdmfH5Storage(DATA)
    case = example_building_case(MESH, n_floors=3)

    def key_for(kind, key):
        return "bodies.galpao" if kind == "body" else "points.static_pressure"

    plan = FanoutPlan(directions_by_category={"": ["0"]}, bodies=["galpao"])
    solve_fn = build_static_solve_fn(case, storage, MESH, key_for=key_for)

    container = run_fanout(plan, solve_fn, pool=SyncPool())
    assert len(container) == 1
    (response,) = container.values()
    assert np.all(np.isfinite(np.asarray(response.fields.read("feq_x"))))
