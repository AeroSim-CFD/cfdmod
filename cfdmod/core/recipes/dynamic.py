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
   :class:`ModesDataSource` to another :class:`ModesDataSource`;
   :func:`sdof_exact_solver` is the one the building recipe uses.
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
    "sdof_exact_solver",
    "check_modal_sampling",
    "BuildingDynamicConfig",
    "build_building_dynamic_response",
    "ComfortConfig",
    "build_point_accelerations",
]

import warnings
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict

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
from cfdmod.core.ops.field.derivative import DerivativeParams, derivative
from cfdmod.core.topology import ElementMeta, Topology

ModalSolver = Callable[[ModesDataSource], ModesDataSource]


def identity_solver(modes: ModesDataSource) -> ModesDataSource:
    """Pass-through solver. Useful when the caller already has the modal
    response (e.g. quasi-static scaling) or for tests."""
    return modes


def _sdof_seed(gen_force: np.ndarray, dt: float, wp: float, xi: float) -> tuple[float, float]:
    """Near-steady-state seed suppressing the spurious startup transient.

    ``x0`` balances the mean forcing, ``v0`` tracks the mean forcing rate --
    the same seed both solvers use, so they stay comparable.
    """
    x0 = float(gen_force.mean()) / (wp**2)
    dfdt = float((gen_force[1:] - gen_force[:-1]).mean()) / dt if gen_force.size > 1 else 0.0
    v0 = dfdt / (2 * xi * wp) if xi * wp != 0 else 0.0
    return x0, v0


def _sdof_recurrence_coeffs(dt: float, wp: float, xi: float) -> tuple[float, ...]:
    """Nigam-Jennings coefficients for one mode (unit generalized mass).

    Exact step response of ``x'' + 2 xi wp x' + wp^2 x = Q(t)`` when ``Q`` varies
    linearly across a step -- which is exactly the interpolation the RK45 path
    assumes -- so this is not an approximation of that path, it is its closed
    form. Returns ``(A, B, C, D, Ap, Bp, Cp, Dp)`` for

        x[i+1] = A x[i] + B v[i] + C Q[i] + D Q[i+1]
        v[i+1] = Ap x[i] + Bp v[i] + Cp Q[i] + Dp Q[i+1]

    Reference: Nigam & Jennings (1969); Chopra, *Dynamics of Structures*,
    Table 5.2.1 (with mass 1 and stiffness ``k = wp^2``).
    """
    k = wp**2
    s = np.sqrt(1.0 - xi**2)
    wd = wp * s
    e = np.exp(-xi * wp * dt)
    sin, cos = np.sin(wd * dt), np.cos(wd * dt)

    a = e * (xi / s * sin + cos)
    b = e * (sin / wd)
    c = (
        1.0
        / k
        * (
            2 * xi / (wp * dt)
            + e * (((1 - 2 * xi**2) / (wd * dt) - xi / s) * sin - (1 + 2 * xi / (wp * dt)) * cos)
        )
    )
    d = (
        1.0
        / k
        * (
            1
            - 2 * xi / (wp * dt)
            + e * ((2 * xi**2 - 1) / (wd * dt) * sin + 2 * xi / (wp * dt) * cos)
        )
    )
    ap = -e * (wp / s * sin)
    bp = e * (cos - xi / s * sin)
    cp = 1.0 / k * (-1.0 / dt + e * ((wp / s + xi / (s * dt)) * sin + cos / dt))
    dp = 1.0 / (k * dt) * (1 - e * (xi / s * sin + cos))
    return a, b, c, d, ap, bp, cp, dp


def _solve_sdof_exact(
    gen_force: np.ndarray, dt: float, wp: np.ndarray, xi: np.ndarray
) -> np.ndarray:
    """Advance every mode together with the Nigam-Jennings recurrence.

    ``gen_force`` is ``(n_modes, n_t)``. The recurrence is sequential in time but
    vectorised across modes, so the Python loop runs once per timestep rather
    than once per (mode, timestep) -- orders of magnitude faster than an adaptive
    ODE integrator, with no step-size error.
    """
    q = np.asarray(gen_force, dtype=np.float64)
    n_modes, n_t = q.shape
    if n_t == 0:
        return q.copy()

    coef = np.array(
        [_sdof_recurrence_coeffs(dt, float(w), float(x)) for w, x in zip(wp, xi)], dtype=np.float64
    )
    a, b, c, d, ap, bp, cp, dp = (coef[:, i] for i in range(8))

    x = np.empty((n_modes, n_t), dtype=np.float64)
    xi_state = np.empty(n_modes, dtype=np.float64)
    vi_state = np.empty(n_modes, dtype=np.float64)
    for m in range(n_modes):
        xi_state[m], vi_state[m] = _sdof_seed(q[m], dt, float(wp[m]), float(xi[m]))
    x[:, 0] = xi_state

    for i in range(n_t - 1):
        q_i, q_n = q[:, i], q[:, i + 1]
        x_next = a * xi_state + b * vi_state + c * q_i + d * q_n
        vi_state = ap * xi_state + bp * vi_state + cp * q_i + dp * q_n
        xi_state = x_next
        x[:, i + 1] = x_next
    return x


def sdof_exact_solver(*, natural_frequencies: Any, damping_ratio: Any) -> ModalSolver:
    """:class:`ModalSolver` using the exact piecewise-linear recurrence.

    Solves the mass-normalized modal equation ``x'' + 2 xi wp x' + wp^2 x = Q(t)``
    in closed form per step, exactly for a ``Q`` interpolated linearly between
    samples. Requires sub-critical damping ``0 <= xi < 1`` and a uniform time
    step.
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
        xis = np.broadcast_to(
            np.atleast_1d(np.asarray(damping_ratio, dtype=np.float64)), (n_modes,)
        )
        if np.any(xis < 0) or np.any(xis >= 1):
            raise ValueError(f"the exact solver needs sub-critical damping 0 <= xi < 1; got {xis}")
        dt = float(modes.time.timestep_size)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"modal time step must be positive and finite; got {dt}")
        return modes.with_field("q", _solve_sdof_exact(q, dt, wps, xis))

    return solver


def check_modal_sampling(
    time_axis, natural_frequencies: Any, *, min_points_per_cycle: float = 8.0
) -> None:
    """Warn when the load's time axis cannot carry the modal response.

    A dynamic run is only as good as the time axis of its forcing: feed a record
    still on the solver's normalised time (or de-normalised with the wrong
    length) and the response is computed at the wrong frequencies while every
    array shape still looks right. Two cheap guards:

    * the highest natural frequency must be resolved (``min_points_per_cycle``
      samples per cycle, well inside Nyquist);
    * the record must span at least ~10 cycles of the fundamental, or the peak
      statistics are meaningless.

    Warns rather than raises -- a short record is sometimes deliberate (a smoke
    run), and the caller may be sampling a sub-window.
    """
    wps = np.atleast_1d(np.asarray(natural_frequencies, dtype=np.float64))
    if wps.size == 0:
        return
    dt = float(time_axis.timestep_size)
    n_t = int(time_axis.n_timesteps)
    if not np.isfinite(dt) or dt <= 0:
        return
    f_max = float(wps.max()) / (2 * np.pi)
    f_min = float(wps.min()) / (2 * np.pi)
    points_per_cycle = 1.0 / (f_max * dt) if f_max > 0 else np.inf
    if points_per_cycle < min_points_per_cycle:
        warnings.warn(
            f"load time step dt={dt:g} s resolves the highest mode ({f_max:.3g} Hz) with only "
            f"{points_per_cycle:.1f} points per cycle (< {min_points_per_cycle:g}). Check the "
            "time-axis scaling -- see cfdmod.dynamics.DimensionalData.",
            UserWarning,
            stacklevel=3,
        )
    cycles = n_t * dt * f_min
    if cycles < 10:
        warnings.warn(
            f"load record spans only {cycles:.1f} cycles of the fundamental ({f_min:.3g} Hz); "
            "peak statistics from it are not meaningful.",
            UserWarning,
            stacklevel=3,
        )


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
        check_sampling: Warn when the load time axis cannot carry the modal
            response (see :func:`check_modal_sampling`).
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
    check_sampling: bool = True
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

    if cfg.check_sampling:
        check_modal_sampling(load_source.time, wps)

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
    solver = sdof_exact_solver(natural_frequencies=wps, damping_ratio=cfg.damping_ratio)
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


class ComfortConfig(BaseModel):
    """Point-acceleration (comfort) recipe parameters.

    Evaluates the horizontal acceleration a building occupant feels at an
    off-center point ``point`` on each floor. The point translates with the
    floor and swings with its torsion, so the perceived displacement adds a
    rotational lever-arm term before differentiation:

        displ_angle = atan2(point - CM) + rot_z
        px = disp_x + cos(displ_angle) * r
        py = disp_y + sin(displ_angle) * r

    with ``r = |point - CM|``; accelerations are the second time-derivative
    of ``px`` / ``py``.

    Attributes:
        cm_positions: ``(n_floors, 2)`` CM offsets ``[XR, YR]`` per floor.
        point: ``(x, y)`` query point (same frame as ``cm_positions``).
        disp_x_field / disp_y_field / rot_z_field: Input field names.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cm_positions: Any
    point: Any = (0.0, 0.0)
    disp_x_field: str = "disp_x"
    disp_y_field: str = "disp_y"
    rot_z_field: str = "rot_z"


def build_point_accelerations(response: PointsDataSource, cfg: ComfortConfig) -> PointsDataSource:
    """Per-floor horizontal accelerations at an off-center point.

    Consumes a building-response :class:`PointsDataSource` (floor
    displacements ``disp_x`` / ``disp_y`` / ``rot_z``) and returns it
    augmented with time-resolved ``acc_x`` / ``acc_y`` / ``acc_mag``. Peak /
    comfort reduction is a separate step: apply
    :func:`cfdmod.core.ops.data_source_create.extreme_value.extreme_value`
    to ``acc_mag`` (or per axis).
    """
    dx = np.asarray(response.fields.read(cfg.disp_x_field), dtype=np.float64)
    dy = np.asarray(response.fields.read(cfg.disp_y_field), dtype=np.float64)
    rz = np.asarray(response.fields.read(cfg.rot_z_field), dtype=np.float64)

    cm = np.asarray(cfg.cm_positions, dtype=np.float64)  # (n_floors, 2)
    point = np.asarray(cfg.point, dtype=np.float64)  # (2,)
    rel = point[None, :] - cm  # (n_floors, 2)
    point_angle = np.arctan2(rel[:, 1], rel[:, 0])[:, None]  # (n_floors, 1)
    r = np.hypot(rel[:, 0], rel[:, 1])[:, None]  # (n_floors, 1)
    displ_angle = point_angle + rz  # (n_floors, n_t)

    px = dx + np.cos(displ_angle) * r
    py = dy + np.sin(displ_angle) * r

    work = response.with_field("_px", px).with_field("_py", py)
    work = derivative(work, DerivativeParams(order=2, field="_px", out="acc_x"))
    work = derivative(work, DerivativeParams(order=2, field="_py", out="acc_y"))
    acc_x = np.asarray(work.fields.read("acc_x"), dtype=np.float64)
    acc_y = np.asarray(work.fields.read("acc_y"), dtype=np.float64)
    acc_mag = np.hypot(acc_x, acc_y)

    return (
        response.with_field("acc_x", acc_x)
        .with_field("acc_y", acc_y)
        .with_field("acc_mag", acc_mag)
    )
