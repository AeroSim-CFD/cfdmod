"""Format-agnostic nodal modal model -> per-floor BuildingStructuralData.

The finite-element structural export (TQS PORTELS, and any other tool
that reports at the node level) gives modal data per *node*: nodal
coordinates, lumped nodal masses, and per-mode nodal displacements
``[DX, DY, RZ]``. The building dynamic-response recipe instead wants
per-*floor* rigid-diaphragm properties. This module bridges the two.

Aggregation (ports the legacy ``gen_nodes.py`` throwaway script, but
uses the tool's direct nodal ``RZ`` instead of reconstructing rotation
from two reference nodes): group nodes by slab elevation, then per floor

    M   = sum_node m
    XG  = sum_node m*x / M          (centre of mass)
    YG  = sum_node m*y / M
    I   = sum_node m*((x-XG)^2 + (y-YG)^2)     (polar inertia about CoM)
    R   = sqrt(I / M)               (radius of gyration)
    DX  = sum_node m*DX_node / M    (mass-weighted rigid-diaphragm shape)
    DY  = sum_node m*DY_node / M
    RZ  = sum_node m*RZ_node / M

Floors with zero total mass (non-slab levels: bare columns/beams) are
dropped by default.
"""

from __future__ import annotations

__all__ = ["NodalModel", "aggregate_to_building"]

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from cfdmod.dynamics.structural import BuildingStructuralData, mass_normalize_mode_shapes


class NodalModel(BaseModel):
    """Raw nodal modal model, as parsed from a structural export.

    Attributes:
        coords: ``(n_nodes, 3)`` nodal coordinates ``[X, Y, Z]``.
        mass: ``(n_nodes,)`` lumped translational nodal mass.
        periods: ``(n_modes,)`` modal natural periods (seconds).
        shapes: ``(n_nodes, n_modes, 3)`` nodal mode shapes ``[DX, DY, RZ]``.
        node_ids: optional ``(n_nodes,)`` original node identifiers (for reference).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    coords: Any
    mass: Any
    periods: Any
    shapes: Any
    node_ids: Any | None = None

    @property
    def n_nodes(self) -> int:
        return int(np.asarray(self.coords).shape[0])

    @property
    def n_modes(self) -> int:
        return int(np.asarray(self.periods).shape[0])


def aggregate_to_building(
    model: NodalModel,
    *,
    tol_z: float = 0.05,
    active_modes: list[int] | None = None,
    drop_massless: bool = True,
) -> BuildingStructuralData:
    """Aggregate a :class:`NodalModel` into a per-floor :class:`BuildingStructuralData`.

    Args:
        tol_z: Elevation clustering tolerance (m); nodes whose Z rounds to
            the same multiple of ``tol_z`` belong to one slab.
        active_modes: 1-based mode numbers to keep (``None`` keeps all).
        drop_massless: Drop clustered elevations with zero total mass
            (non-slab levels). When ``False`` they raise instead.

    Returns:
        A :class:`BuildingStructuralData` with floors ordered by ascending
        elevation, mass-normalized mode shapes, and ``cm_positions`` set to
        the per-floor centre of mass.
    """
    coords = np.asarray(model.coords, dtype=np.float64)
    mass = np.asarray(model.mass, dtype=np.float64)
    periods = np.asarray(model.periods, dtype=np.float64)
    shapes = np.asarray(model.shapes, dtype=np.float64)
    n_modes = periods.shape[0]

    if shapes.shape != (coords.shape[0], n_modes, 3):
        raise ValueError(
            f"shapes must be (n_nodes, n_modes, 3)=({coords.shape[0]}, {n_modes}, 3); "
            f"got {shapes.shape}"
        )

    keep = list(range(n_modes)) if active_modes is None else [m - 1 for m in active_modes]

    # Cluster nodes by elevation (round to tol_z), ascending.
    z_key = np.round(coords[:, 2] / tol_z).astype(np.int64)
    floor_props: list[tuple[float, float, float, float, float, float]] = []
    floor_shapes: list[np.ndarray] = []  # each (n_kept_modes, 3)

    for key in sorted(np.unique(z_key)):
        sel = z_key == key
        m = mass[sel]
        total = float(m.sum())
        elev = float(coords[sel, 2].mean())
        if total <= 0.0:
            if drop_massless:
                continue
            raise ValueError(f"slab at z={elev:.3f} has zero total mass")

        x, y = coords[sel, 0], coords[sel, 1]
        xg = float((m * x).sum() / total)
        yg = float((m * y).sum() / total)
        inertia = float((m * ((x - xg) ** 2 + (y - yg) ** 2)).sum())
        radius = float((inertia / total) ** 0.5)
        floor_props.append((elev, total, inertia, radius, xg, yg))

        # Mass-weighted rigid-diaphragm shape per kept mode.
        node_shapes = shapes[sel][:, keep, :]  # (n_sel, n_kept, 3)
        weighted = (m[:, None, None] * node_shapes).sum(axis=0) / total  # (n_kept, 3)
        floor_shapes.append(weighted)

    if not floor_props:
        raise ValueError("no slabs with positive mass were found in the nodal model")

    props = np.asarray(floor_props, dtype=np.float64)  # (n_floors, 6)
    elevations, floors_mass, _, floors_radius, xg, yg = props.T
    phi = np.stack(floor_shapes, axis=0)  # (n_floors, n_kept_modes, 3)

    floors_mass = np.asarray(floors_mass, dtype=np.float64)
    floors_radius = np.asarray(floors_radius, dtype=np.float64)
    phi = mass_normalize_mode_shapes(phi, floors_mass, floors_radius)

    freqs = 1.0 / periods[keep]
    wp = 2.0 * np.pi * freqs

    return BuildingStructuralData(
        mode_shapes=phi,
        natural_frequencies=wp,
        floor_points=np.column_stack([xg, yg, elevations]),
        cm_positions=np.column_stack([xg, yg]),
        floors_mass=floors_mass,
        floors_radius=floors_radius,
    )
