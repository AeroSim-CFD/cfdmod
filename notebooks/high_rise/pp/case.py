"""HighRiseCase -- aggregate the per-case configuration.

A consulting case keeps its metadata in a ``case_data/`` directory:

    global_data.json     -- H, L, V0, batch name, body name, wind directions
    params_cat*.yaml     -- simulation U_H, fluid density, nominal area/volume,
                            floor heights (HEIGHTS), lever origin, coefficient blocks
    wind_analysis_*.csv  -- per-direction z0 / Kd / category (optional here)

``HighRiseCase`` reads those into one immutable object the notebooks share.
The high-rise sequence extracts the *simulation* mean velocity at the reference
height from the inflow profile and then updates the case with it
(:meth:`with_reference_velocity`) so the Cp step non-dimensionalises with the
measured value rather than the value guessed at config time.
"""

from __future__ import annotations

import json
import pathlib
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

_yaml = YAML(typ="safe")


class HighRiseCase(BaseModel):
    """Immutable aggregate of a high-rise case's post-processing inputs."""

    model_config = ConfigDict(frozen=True)

    name: str
    reference_height: Annotated[float, Field(gt=0, description="H, interest height [m]")]
    characteristic_length: Annotated[float, Field(gt=0, description="L / B [m]")]
    basic_wind_speed: Annotated[float, Field(gt=0, description="V0 [m/s]")]
    fluid_density: Annotated[float, Field(gt=0, description="rho [kg/m^3]")] = 1.225
    simul_reference_velocity: Annotated[
        float, Field(gt=0, description="U_H from the simulation profile [m/s]")
    ]
    reference_velocity: Annotated[
        float | None,
        Field(description="Measured U_H at reference height; overrides simul when set."),
    ] = None
    nominal_area: Annotated[float, Field(gt=0, description="reference area for Cf [m^2]")]
    nominal_volume: Annotated[float, Field(gt=0, description="reference volume for Cm [m^3]")]
    floor_heights: Annotated[
        list[float], Field(min_length=2, description="ascending floor Z edges [m]")
    ]
    lever_origin: list[float] = [0.0, 0.0, 0.0]
    directions: list[str] = []
    body_name: str = "building"

    # -- derived -----------------------------------------------------------

    @property
    def u_h(self) -> float:
        """Reference velocity actually used: measured if set, else simulation."""
        return (
            self.reference_velocity
            if self.reference_velocity is not None
            else (self.simul_reference_velocity)
        )

    @property
    def dynamic_pressure(self) -> float:
        """q = 0.5 * rho * U_H^2 (Pa)."""
        return 0.5 * self.fluid_density * self.u_h**2

    @property
    def n_floors(self) -> int:
        return len(self.floor_heights) - 1

    def with_reference_velocity(self, u_ref: float) -> "HighRiseCase":
        """Return a copy whose Cp normalisation uses the measured ``u_ref``."""
        return self.model_copy(update={"reference_velocity": float(u_ref)})

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_case_data(
        cls,
        case_data_dir: str | pathlib.Path,
        params_name: str,
        *,
        body_name: str | None = None,
    ) -> "HighRiseCase":
        """Build from a ``case_data/`` dir containing global_data.json + a params yaml.

        Parses the 067-style params layout (top-level ``anchors`` plus
        ``pressure_coefficient`` / ``force_coefficient`` / ``moment_coefficient``
        blocks). Missing optional fields fall back to sensible defaults.
        """
        case_data_dir = pathlib.Path(case_data_dir)
        gd = json.loads((case_data_dir / "global_data.json").read_text())
        with (case_data_dir / params_name).open() as fh:
            params = _yaml.load(fh)

        base_cp = _first_block(params.get("pressure_coefficient", {}))
        force = _first_block(params.get("force_coefficient", {}))
        moment = _first_block(params.get("moment_coefficient", {}))

        heights = _floor_heights(force)
        lever = _lever_origin(moment)
        analysis = gd.get("analysis", {})
        resolved_body = (
            body_name or analysis.get("body_name") or (force.get("bodies") or [{}])[0].get("name")
        )

        directions = analysis.get(f"directions_cat{_cat(analysis)}", []) or analysis.get(
            "directions", []
        )

        return cls(
            name=case_data_dir.parent.name
            if case_data_dir.name == "case_data"
            else case_data_dir.name,
            reference_height=float(gd["H"]),
            characteristic_length=float(gd.get("L", gd.get("L1", 1.0))),
            basic_wind_speed=float(gd["V0"]),
            fluid_density=float(base_cp.get("fluid_density", 1.225)),
            simul_reference_velocity=float(base_cp["simul_U_H"]),
            nominal_area=float(force["nominal_area"]),
            nominal_volume=float(moment["nominal_volume"]),
            floor_heights=[float(z) for z in heights],
            lever_origin=[float(v) for v in lever],
            directions=list(directions),
            body_name=resolved_body or "building",
        )


def example_high_rise_case(
    mesh_path: str | pathlib.Path,
    *,
    u_h: float = 0.05,
    rho: float = 1.0,
    n_floors: int = 3,
) -> HighRiseCase:
    """A self-contained HighRiseCase tuned to a mesh, for notebook demos/tests.

    Geometry fields are derived from the mesh bounding box (frontal area,
    volume, floor z-edges). Defaults ``u_h``/``rho`` match the galpao fixture's
    dynamic pressure (q = 0.00125) so Cp lands in the same range as the
    ``cp.yaml`` template. For real work use :meth:`HighRiseCase.from_case_data`.
    """
    from lnas import LnasFormat

    verts = LnasFormat.from_file(pathlib.Path(mesh_path)).geometry.vertices
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    lx, ly, lz = hi - lo
    return HighRiseCase(
        name=pathlib.Path(mesh_path).stem,
        reference_height=float(max(hi[2], 1e-6)),
        characteristic_length=float(max(lx, 1e-6)),
        basic_wind_speed=float(u_h),
        fluid_density=float(rho),
        simul_reference_velocity=float(u_h),
        nominal_area=float(max(lx * lz, 1e-6)),
        nominal_volume=float(max(lx * lz * ly, 1e-6)),
        floor_heights=[float(z) for z in np.linspace(lo[2], hi[2], n_floors + 1)],
    )


def _first_block(section: dict) -> dict:
    """Coefficient blocks are keyed by config version (e.g. base_cp); take the first."""
    if not section:
        return {}
    return next(iter(section.values()))


def _floor_heights(force_block: dict) -> list[float]:
    bodies = force_block.get("bodies") or [{}]
    sub = bodies[0].get("sub_bodies", {}) if bodies else {}
    return list(sub.get("z_intervals", []))


def _lever_origin(moment_block: dict) -> list[float]:
    bodies = moment_block.get("bodies") or [{}]
    return list(bodies[0].get("lever_origin", [0.0, 0.0, 0.0])) if bodies else [0.0, 0.0, 0.0]


def _cat(analysis: dict) -> str:
    cats = analysis.get("categories") or []
    return str(cats[0]) if cats else ""
