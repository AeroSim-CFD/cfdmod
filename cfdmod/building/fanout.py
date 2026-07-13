"""Per-direction / body / cp-config fan-out driver.

Real consulting cases process many wind directions (and bodies / Cp configs);
the v3 building path solves one case at a time. This driver reads the fan-out
axes from ``global_data.json``, maps each ``(direction, body, cp_config)`` key
through the Cp -> per-floor Cf/Cm -> dynamic-response pipeline, and collects the
results in a :class:`cfdmod.core.container.Container` -- the exact shape the
multi-direction reducers in :mod:`cfdmod.dynamics.cases` and the load-case
tables in :mod:`cfdmod.building.loadcases` consume.

Design notes / best-guess calls (flagged for review):

- The value type is the response ``PointsDataSource`` augmented with floor
  accelerations, so the container drops straight into
  ``cfdmod.dynamics.get_stats_forces_effective`` / ``get_max_acceleration`` and
  ``cfdmod.building.effective_load_stats``.
- ``StaticCaseKey.direction`` is a *string* (categories yield labels like
  ``"0"`` / ``"45"``, matching ``BuildingCase.directions``), not the ``float``
  of ``BuildingCaseParameters``. ``join_by(lambda k: k.direction)`` still works.
- The storage key template (how a ``(direction, body)`` maps to a stored body /
  reference-pressure DataSource) is the one hard external contract -- it is
  injectable via ``key_for`` so it can be pinned to the real server layout.
- A single ``case`` is used across cp_configs (per-config case variation is a
  future refinement); provenance (region info + config) is written once.
- With a real multiprocessing ``Pool`` the ``solve_fn`` must be picklable, so it
  re-reads its inputs from ``storage`` inside the worker rather than closing over
  large in-RAM DataSources.
"""

from __future__ import annotations

__all__ = [
    "StaticCaseKey",
    "FanoutPlan",
    "build_static_keys",
    "default_storage_key",
    "build_static_solve_fn",
    "run_fanout",
    "dump_provenance",
]

import itertools
import json
import pathlib
from typing import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from cfdmod.building.case import BuildingCase
from cfdmod.building.dynamic import (
    example_building_structure,
    floor_accelerations,
    floor_load_source,
    solve_building_response,
)
from cfdmod.building.pressure import cf_per_floor, cm_per_floor, cp_from_pressure
from cfdmod.core.container import Container
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.protocols import Pool
from cfdmod.utils import save_yaml


class StaticCaseKey(BaseModel, frozen=True):
    """One fan-out case: a (direction, body, cp-config) triple (a Container key)."""

    direction: str
    body: str
    cp_config: str = "base"


class FanoutPlan(BaseModel, frozen=True):
    """The fan-out axes parsed from ``global_data.json`` (or built directly)."""

    batch_name: str = ""
    categories: list[str] = []
    directions_by_category: dict[str, list[str]] = {}
    bodies: list[str]
    cp_configs: list[str] = ["base"]

    @property
    def directions(self) -> list[str]:
        """Order-preserving union of directions across the selected categories."""
        cats = self.categories or list(self.directions_by_category)
        seen: list[str] = []
        for cat in cats:
            for d in self.directions_by_category.get(cat, []):
                if d not in seen:
                    seen.append(d)
        return seen

    @classmethod
    def from_global_data(
        cls,
        case_data_dir: str | pathlib.Path,
        *,
        bodies: list[str] | None = None,
        cp_configs: list[str] | None = None,
    ) -> "FanoutPlan":
        """Read ``analysis.categories`` / ``directions_cat*`` / ``body_name`` /
        ``batch_name`` from ``<case_data_dir>/global_data.json``.

        ``bodies`` / ``cp_configs`` override the single body / default config
        the JSON implies (real cases fan over several).
        """
        gd = json.loads((pathlib.Path(case_data_dir) / "global_data.json").read_text())
        analysis = gd.get("analysis", {})
        categories = [str(c) for c in analysis.get("categories", [])]
        directions_by_category = {
            cat: [str(d) for d in analysis.get(f"directions_cat{cat}", [])] for cat in categories
        }
        if not any(directions_by_category.values()) and analysis.get("directions"):
            directions_by_category = {"": [str(d) for d in analysis["directions"]]}
            categories = [""]
        body = analysis.get("body_name")
        resolved_bodies = bodies or ([body] if body else [])
        if not resolved_bodies:
            raise ValueError("no bodies in global_data.json analysis.body_name; pass bodies=[...]")
        return cls(
            batch_name=str(analysis.get("batch_name", "")),
            categories=categories,
            directions_by_category=directions_by_category,
            bodies=resolved_bodies,
            cp_configs=cp_configs or ["base"],
        )


def build_static_keys(plan: FanoutPlan) -> list[StaticCaseKey]:
    """Cartesian product of (direction, body, cp_config) -> fan-out keys."""
    return [
        StaticCaseKey(direction=d, body=b, cp_config=c)
        for d, b, c in itertools.product(plan.directions, plan.bodies, plan.cp_configs)
    ]


# Default storage-key templates. The real consulting layout encodes the
# direction in a case-name directory and nests the body under a probe path, e.g.
#   body:  "ClementePereira_{direction}/000/probes/hist_series/cp_analysis_{body}/bodies.{body}"
#   p_ref: "ClementePereira_{direction}/000/probes/hist_series/cp_analysis_{body}/points.point0"
# so the template is genuinely case-specific -- pass your own to
# ``build_static_solve_fn``. The defaults below are the minimal (direction, body)
# form. Placeholders: {direction} {body} {cp_config}.
DEFAULT_BODY_KEY = "{direction}/bodies.{body}"
DEFAULT_PREF_KEY = "{direction}/points.point0"


def default_storage_key(kind: str, key: StaticCaseKey) -> str:
    """Default ``(direction, body)`` -> storage key (the minimal template form).

    ``kind`` is ``"body"`` or ``"p_ref"``. Real cases have a nested, case-named
    layout -- pass ``body_key_template`` / ``pref_key_template`` (or a full
    ``key_for``) to :func:`build_static_solve_fn` to match it.
    """
    if kind == "body":
        return DEFAULT_BODY_KEY.format(
            direction=key.direction, body=key.body, cp_config=key.cp_config
        )
    if kind == "p_ref":
        return DEFAULT_PREF_KEY.format(
            direction=key.direction, body=key.body, cp_config=key.cp_config
        )
    raise ValueError(f"unknown storage key kind {kind!r}")


def build_static_solve_fn(
    case: BuildingCase,
    storage,
    mesh_path: str,
    *,
    structure=None,
    method: str = "face_cut",
    damping_ratio: float = 0.02,
    point: tuple[float, float] = (0.0, 0.0),
    cp_statistics: list[str] | None = None,
    body_key_template: str = DEFAULT_BODY_KEY,
    pref_key_template: str = DEFAULT_PREF_KEY,
    key_for: Callable[[str, StaticCaseKey], str] | None = None,
) -> Callable[[StaticCaseKey], PointsDataSource]:
    """Build the default per-key pipeline: storage -> Cp -> Cf/Cm -> response.

    Returns a ``solve_fn(key) -> response`` (a floor ``PointsDataSource`` carrying
    ``feq_*`` / ``meq_z``, ``acc_*`` / ``acc_mag`` and the applied-static floor
    loads ``fs_x`` / ``fs_y`` / ``ms_z`` so downstream *effective* stats can
    combine them). Reads its body / reference pressure from ``storage`` inside the
    call so it stays picklable for a multiprocessing ``Pool``.

    The ``(direction, body)`` -> storage-key mapping is case-specific: pass
    ``body_key_template`` / ``pref_key_template`` (``str.format`` templates with
    ``{direction}`` / ``{body}`` / ``{cp_config}`` placeholders) or a full
    ``key_for(kind, key)`` override.
    """
    if key_for is None:

        def key_for(kind: str, key: StaticCaseKey) -> str:
            template = body_key_template if kind == "body" else pref_key_template
            return template.format(direction=key.direction, body=key.body, cp_config=key.cp_config)

    def solve_fn(key: StaticCaseKey) -> PointsDataSource:
        body = storage.read_data_source(key_for("body", key))
        p_ref = storage.read_data_source(key_for("p_ref", key))
        cp = cp_from_pressure(body, p_ref, case, statistics=cp_statistics)
        cf = cf_per_floor(cp, mesh_path, case, method=method)
        cm = cm_per_floor(cp, mesh_path, case, method=method)
        load = floor_load_source(cf, cm, case)
        struct = (
            structure
            if structure is not None
            else example_building_structure(case, load.n_elements)
        )
        response = solve_building_response(load, struct, damping_ratio=damping_ratio)
        result = floor_accelerations(response, struct, point=point)
        # attach the applied-static floor loads so get_stats_forces_effective /
        # effective_load_stats combine them with the dynamic static-equivalent
        return (
            result.with_field("fs_x", load.fields.read("cf_x"))
            .with_field("fs_y", load.fields.read("cf_y"))
            .with_field("ms_z", load.fields.read("cm_z"))
        )

    return solve_fn


def run_fanout(
    plan: FanoutPlan,
    solve_fn: Callable[[StaticCaseKey], PointsDataSource],
    *,
    pool: Pool | None = None,
    writer=None,
    case: BuildingCase | None = None,
) -> Container[StaticCaseKey, PointsDataSource]:
    """Fan ``solve_fn`` out over every ``(direction, body, cp_config)`` key.

    Collects the responses in a ``Container[StaticCaseKey, PointsDataSource]``
    (group with ``container.join_by(lambda k: k.direction)``). With ``pool`` the
    fan-out runs through ``pool.map``. When both ``writer`` and ``case`` are
    given, the provenance dump (region info + resolved config) is written once
    beside the outputs.
    """
    keys = build_static_keys(plan)
    if pool is None:
        results = [solve_fn(k) for k in keys]
    else:
        results = pool.map(solve_fn, keys)
    container = Container(items=dict(zip(keys, results)))
    if writer is not None and case is not None:
        # actual computed floor count (pressure may drop empty floor slices)
        n_floors = next(iter(container.values())).n_elements if len(container) else None
        dump_provenance(writer, case, plan, n_floors=n_floors)
    return container


def dump_provenance(
    writer,
    case: BuildingCase,
    plan: FanoutPlan,
    *,
    n_floors: int | None = None,
) -> dict[str, pathlib.Path]:
    """Write the per-floor region info (z-edges) and the resolved config yaml.

    Mirrors the notebooks' ``save_region_info`` + ``save_yaml``: a
    ``region_info.csv`` (one row per floor with ``z_min`` / ``z_max``) and a
    ``config.yaml`` (case + plan dump), written to the writer's debug root.
    Returns ``{name: path}``.

    The case z-edges are used when they match the actual computed floor count
    (``n_floors``); if the pressure stage dropped empty floor slices they will
    not, so a plain floor-index ladder is written instead of misleading edges.
    """
    edges = np.asarray(case.floor_heights, dtype=np.float64)
    if n_floors is None or len(edges) == n_floors + 1:
        region = pd.DataFrame(
            {
                "floor": np.arange(len(edges) - 1),
                "z_min": edges[:-1],
                "z_max": edges[1:],
            }
        )
    else:
        region = pd.DataFrame({"floor": np.arange(n_floors)})
    region_path = writer.save_csv(region, "region_info.csv")
    config_path = writer.debug_path("config.yaml")
    save_yaml({"case": case.model_dump(), "plan": plan.model_dump()}, config_path)
    return {"region_info.csv": region_path, "config.yaml": config_path}
