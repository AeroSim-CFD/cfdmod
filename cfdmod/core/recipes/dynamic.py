"""Dynamic-analysis recipe -- per-element loads -> modal solution -> physical response.

Per the odt::

    container of Cf + modes data -> solution as modal displacements
    container of solutions       -> displacement / acceleration / loads
                                    in original coordinates

This recipe is the small-data analogue of the legacy
``cfdmod.hfpi.dynamic`` pipeline. It wires three of the Phase 6
primitives together:

1. :func:`modal_projection` -- physical-space load timeseries ``f``
   times mode shapes ``phi`` -> generalized loads ``Q``.
2. A user-supplied modal solver (``Q -> q``) -- the linear ODE
   ``Mq'' + Cq' + Kq = Q`` is the SDOF case per mode and lives outside
   the algebra layer. We accept any callable that maps a
   :class:`ModesDataSource` to another :class:`ModesDataSource`.
3. :func:`modal_recomposition` -- modal coordinates back into the
   physical mesh.

A trivial ``identity`` solver is provided for tests and for the
"already solved" path (Q == q, e.g. quasi-static modal scaling).
"""

from __future__ import annotations

__all__ = [
    "DynamicAnalysisConfig",
    "build_dynamic_response",
    "identity_solver",
    "sdof_rk45_solver",
    "BuildingDynamicConfig",
    "build_building_dynamic_response",
]

from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy import integrate
from scipy.interpolate import interp1d

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource, ModesDataSource, PointsDataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops.data_source_create.generalized_building_load import (
    GeneralizedBuildingLoadParams,
    generalized_building_load,
)
from cfdmod.core.ops.data_source_create.modal_projection import (
    ModalProjectionParams,
    modal_projection,
)
from cfdmod.core.ops.data_source_create.modal_recomposition import (
    ModalRecompositionParams,
    modal_recomposition,
)
from cfdmod.core.topology import ElementMeta, Topology

ModalSolver = Callable[[ModesDataSource], ModesDataSource]


def identity_solver(modes: ModesDataSource) -> ModesDataSource:
    """Pass-through solver. Useful when the caller already has the modal
    response (e.g. quasi-static scaling) or for tests."""
    return modes


def _solve_sdof_rk45(gen_force: np.ndarray, dt: float, wp: float, xi: float) -> np.ndarray:
    """Integrate one mode's single-degree-of-freedom modal ODE with RK45.

    Solves the mass-normalized modal equation for the generalized
    displacement ``x(t)``::

        x'' + 2 * xi * wp * x' + wp^2 * x = Q(t)

    where ``Q`` is the (mass-normalized) generalized-load timeseries
    ``gen_force``. The equation assumes unit generalized mass -- the
    mode shapes feeding the projection must be mass-normalized (see
    :func:`sdof_rk45_solver`).

    Args:
        gen_force: Generalized-load history ``Q`` for one mode, shape ``(n_t,)``.
        dt: Timestep size (seconds).
        wp: Angular natural frequency ``wp = 2 * pi * f`` (rad/s).
        xi: Damping ratio (e.g. 0.01 - 0.02).

    Returns:
        Generalized-displacement history ``x`` for the mode, shape ``(n_t,)``.
    """
    end_step = (len(gen_force) - 1) * dt
    t_eval = np.linspace(0, end_step, len(gen_force))

    f_func = interp1d(t_eval, gen_force, kind="linear", fill_value="extrapolate")

    def system(t, y):
        f_t = f_func(t)
        x, v = y
        # x' = v ; v' = Q(t) - 2*xi*wp*v - wp^2*x
        return [v, f_t - 2 * xi * wp * v - wp**2 * x]

    # Seed the ODE near steady state to suppress a spurious startup transient:
    # x0 balances the mean forcing, v0 tracks the mean forcing rate.
    x0 = gen_force.mean() / (wp**2)
    dfdt = (gen_force[1:] - gen_force[:-1]).mean() / dt
    v0 = dfdt / (2 * xi * wp) if xi * wp != 0 else 0.0

    sol = integrate.solve_ivp(
        system, (t_eval[0], t_eval[-1]), [x0, v0], t_eval=t_eval, method="RK45"
    )
    return sol.y[0]


def sdof_rk45_solver(
    *,
    natural_frequencies: Any,
    damping_ratio: Any,
) -> ModalSolver:
    """Build a :class:`ModalSolver` that integrates each mode as an SDOF oscillator.

    The returned solver reads the generalized-load timeseries from field
    ``"q"`` of the :class:`ModesDataSource` (shape ``(n_modes, n_t)``),
    integrates the mass-normalized modal ODE per mode with
    :func:`_solve_sdof_rk45`, and returns the modes source with ``"q"``
    replaced by the generalized-displacement response.

    Precondition (not silently assumed): the mode shapes used to build
    the generalized load must be mass-normalized (unit generalized mass),
    since the SDOF ODE carries no explicit mass term. Build the modal
    load from mass-normalized shapes upstream.

    Args:
        natural_frequencies: Angular natural frequencies ``wp = 2*pi*f``
            (rad/s), one per mode. Length must equal ``n_modes``.
        damping_ratio: Damping ratio ``xi``. A scalar is broadcast across
            all modes; an array must have one entry per mode.

    Returns:
        A callable ``ModesDataSource -> ModesDataSource``.
    """
    wps = np.atleast_1d(np.asarray(natural_frequencies, dtype=np.float64))

    def solver(modes: ModesDataSource) -> ModesDataSource:
        q = np.asarray(modes.fields.read("q"), dtype=np.float64)
        if q.ndim != 2:
            raise ValueError(f"modes field 'q' must be 2-D (n_modes, n_t); got {q.shape}")
        n_modes = q.shape[0]
        if wps.shape[0] != n_modes:
            raise ValueError(
                f"natural_frequencies has {wps.shape[0]} entries; expected n_modes={n_modes}"
            )
        xi = np.broadcast_to(
            np.atleast_1d(np.asarray(damping_ratio, dtype=np.float64)), (n_modes,)
        )
        dt = float(modes.time.timestep_size)

        disp = np.empty_like(q)
        for i in range(n_modes):
            disp[i, :] = _solve_sdof_rk45(q[i, :], dt=dt, wp=float(wps[i]), xi=float(xi[i]))
        return modes.with_field("q", disp)

    return solver


class DynamicAnalysisConfig(BaseModel):
    """Dynamic-analysis recipe parameters.

    Attributes:
        mode_shapes: ``(n_load_elements, n_modes)`` mode-shape matrix at
            the load points (used to compute ``Q``). For most cases the
            same mode shapes also drive recomposition (see
            ``recomposition_mode_shapes``).
        recomposition_mode_shapes: Optional ``(n_target_elements,
            n_modes)`` matrix evaluated at the *target* coordinates. If
            ``None``, ``mode_shapes`` is reused (load and target
            coincide).
        target_points: ``(n_target_elements, 3)`` coordinates for the
            recomposed response.
        load_field: Field name carrying the load timeseries on the
            input data source. Defaults to ``"force"``.
        response_field: Field name on the output points data source.
            Defaults to ``"u"``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mode_shapes: Any
    target_points: Any
    recomposition_mode_shapes: Any | None = None
    load_field: str = "force"
    response_field: str = "u"


def build_dynamic_response(
    load_source: DataSource,
    cfg: DynamicAnalysisConfig,
    *,
    solver: ModalSolver = identity_solver,
) -> PointsDataSource:
    """Assemble the recipe end-to-end."""
    phi = np.asarray(cfg.mode_shapes, dtype=np.float64)
    modes = modal_projection(
        load_source,
        ModalProjectionParams(mode_shapes=phi, field=cfg.load_field, out="q"),
    )
    solved = solver(modes)
    phi_target = (
        phi
        if cfg.recomposition_mode_shapes is None
        else np.asarray(cfg.recomposition_mode_shapes, dtype=np.float64)
    )
    return modal_recomposition(
        solved,
        ModalRecompositionParams(
            mode_shapes=phi_target,
            target_points=np.asarray(cfg.target_points, dtype=np.float64),
            field="q",
            out=cfg.response_field,
        ),
    )


class BuildingDynamicConfig(BaseModel):
    """Building dynamic-response recipe parameters.

    Ports the legacy ``cfdmod.hfpi`` high-frequency-pressure-integration
    pipeline: per-floor force/moment coefficients -> generalized modal
    loads -> SDOF modal displacements -> physical floor displacements and
    static-equivalent floor forces.

    Attributes:
        mode_shapes: ``(n_floors, n_modes, 3)`` per-floor ``[DX, DY, RZ]``
            components. Must be mass-normalized (unit generalized mass).
        floor_points: ``(n_floors, 3)`` floor coordinates for the output.
        cm_positions: ``(n_floors, 2)`` CM offsets ``[XR, YR]`` per floor.
        floors_mass: ``(n_floors,)`` floor masses (for static-equivalent forces).
        floors_radius: ``(n_floors,)`` floor radii of gyration.
        natural_frequencies: ``(n_modes,)`` angular natural frequencies
            ``wp = 2*pi*f`` (rad/s).
        damping_ratio: Damping ratio ``xi``; scalar (broadcast) or per-mode array.
        field_x / field_y / field_mz: Load-coefficient field names on the input.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mode_shapes: Any
    floor_points: Any
    cm_positions: Any
    floors_mass: Any
    floors_radius: Any
    natural_frequencies: Any
    damping_ratio: Any = 0.02
    field_x: str = "cf_x"
    field_y: str = "cf_y"
    field_mz: str = "cm_z"


def build_building_dynamic_response(
    load_source: DataSource, cfg: BuildingDynamicConfig
) -> PointsDataSource:
    """Assemble the building dynamic-response recipe end-to-end.

    Returns a :class:`PointsDataSource` over the floors carrying six
    fields: floor displacements ``disp_x`` / ``disp_y`` / ``rot_z`` and
    static-equivalent floor loads ``feq_x`` / ``feq_y`` / ``meq_z``.
    """
    phi = np.asarray(cfg.mode_shapes, dtype=np.float64)
    wps = np.atleast_1d(np.asarray(cfg.natural_frequencies, dtype=np.float64))

    # 1. Physical loads -> generalized modal loads (CM lever arm).
    modes = generalized_building_load(
        load_source,
        GeneralizedBuildingLoadParams(
            mode_shapes=phi,
            cm_positions=np.asarray(cfg.cm_positions, dtype=np.float64),
            field_x=cfg.field_x,
            field_y=cfg.field_y,
            field_mz=cfg.field_mz,
            out="q",
        ),
    )

    # 2. Per-mode SDOF integration -> generalized modal displacements.
    solver = sdof_rk45_solver(natural_frequencies=wps, damping_ratio=cfg.damping_ratio)
    solved = solver(modes)

    # 3. Recompose physical floor response + static-equivalent loads.
    q = np.asarray(solved.fields.read("q"), dtype=np.float64)  # (n_modes, n_t)
    dx = phi[:, :, 0]  # (n_floors, n_modes)
    dy = phi[:, :, 1]
    rz = phi[:, :, 2]

    disp_x = dx @ q  # (n_floors, n_t)
    disp_y = dy @ q
    rot_z = rz @ q

    mass = np.asarray(cfg.floors_mass, dtype=np.float64)[:, None]
    radius = np.asarray(cfg.floors_radius, dtype=np.float64)[:, None]
    qw = (wps**2)[:, None] * q  # (n_modes, n_t)

    feq_x = mass * (dx @ qw)
    feq_y = mass * (dy @ qw)
    meq_z = mass * radius**2 * (rz @ qw)

    pts = np.asarray(cfg.floor_points, dtype=np.float64)
    fields = {
        "disp_x": disp_x,
        "disp_y": disp_y,
        "rot_z": rot_z,
        "feq_x": feq_x,
        "feq_y": feq_y,
        "meq_z": meq_z,
    }
    return PointsDataSource(
        time=solved.time,
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(fields),
        field_meta={k: FieldMeta(name=k) for k in fields},
    )
