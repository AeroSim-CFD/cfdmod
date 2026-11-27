"""Projected (shadow) area of a surface mesh seen along a horizontal direction.

The frontal area of a building for one wind direction: every triangle is
projected on the vertical plane normal to the wind and the projections are
united, so surfaces that hide one another (the front and back facades, a
balcony in front of its wall, a leaf wall in front of the tower) count once.
The union is a 2D polygon -- the shadow. Its area is the reference area of a
drag coefficient; its intersection with horizontal bands gives the area per
floor.

``shapely`` (GEOS) does the union. It is part of the optional ``geometry``
extra and is imported only when :func:`projected_area` is called, so importing
this module stays cheap.
"""

from __future__ import annotations

__all__ = ["ProjectedArea", "projected_area"]

from dataclasses import dataclass
from typing import Any

import numpy as np

from cfdmod.geometry.matching import as_triangle_vertices

#: Default precision grid of the union, relative to the largest extent of the
#: projected geometry (0.1 mm on a 100 m building).
RELATIVE_GRID_SIZE = 1e-6


@dataclass(frozen=True)
class ProjectedArea:
    """Result of :func:`projected_area`.

    Attributes:
        shadow: Union of the projected triangles, a shapely ``Polygon`` or
            ``MultiPolygon`` in plane coordinates ``(u, z)``. Openings the wind
            sees through are interior rings.
        area: Area of ``shadow`` [length^2].
        u_axis: ``(3,)`` unit vector of the plane's horizontal axis in the
            model frame, ``direction x e_z``: the right-hand side of an
            observer upstream facing downstream, so ``(u, z)`` is the picture
            that observer sees. ``u = p . u_axis`` for a model point ``p``.
        grid_size: Precision grid the union was snapped to [length].
        z_intervals: ``(n + 1,)`` band edges along z, or None.
        band_areas: ``(n,)`` area of the shadow inside each band, or None.
    """

    shadow: Any
    area: float
    u_axis: np.ndarray
    grid_size: float
    z_intervals: np.ndarray | None = None
    band_areas: np.ndarray | None = None

    @property
    def width(self) -> float:
        """Extent of the shadow along ``u`` (the across-wind width)."""
        if self.shadow.is_empty:
            return 0.0
        u0, _, u1, _ = self.shadow.bounds
        return float(u1 - u0)


def _load_shapely():
    """Import shapely on demand, naming the extra when it is absent."""
    try:
        import shapely
    except ImportError as exc:
        raise ImportError(
            "cfdmod.geometry.projected_area requires the optional 'geometry' extras. "
            "Install with: pip install aerosim-cfdmod[geometry]"
        ) from exc
    return shapely


def projected_area(
    geometry,
    direction_deg: float,
    *,
    z_intervals=None,
    grid_size: float | None = None,
) -> ProjectedArea:
    """Projected (shadow) area of a surface seen along a horizontal wind direction.

    Args:
        geometry: The surface. An ``(n, 3, 3)`` array of triangle vertices, an
            lnas geometry / format, or a path to a ``.stl`` / ``.lnas`` file
            (see :func:`cfdmod.geometry.as_triangle_vertices`). Several
            surfaces (a tower and its leaf walls) go in one array, so their
            overlap is counted once.
        direction_deg: Direction the wind blows towards in the xy plane:
            ``0`` along ``+x``, ``90`` along ``+y`` (the convention of
            :func:`cfdmod.dynamics.shedding.wind_unit_vector`).
        z_intervals: Band edges along z, monotone increasing; ``n`` edges give
            ``n - 1`` bands (the ``z_intervals`` convention of the face-cut
            ops). Each band area is the shadow clipped to the band, so the
            band areas add up to the shadow area between the first and the
            last edge.
        grid_size: Precision grid of the union [length]. Default
            ``RELATIVE_GRID_SIZE`` times the largest extent of the projected
            geometry. Vertices are snapped to it, which keeps the union robust
            against the almost-coincident edges of adjacent triangles; the
            areas carry an error of about ``perimeter x grid_size`` (1e-6
            relative at the default). Projected triangles with an area below
            ``grid_size^2`` are left out: those are the faces parallel to the
            wind (edge-on, zero projected area).

    Returns:
        :class:`ProjectedArea` with the shadow polygon, its area and, when
        ``z_intervals`` is given, the area per band.

    Raises:
        ImportError: shapely is not installed (``geometry`` extra).
        ValueError: Empty or flat geometry, ``grid_size <= 0``, or
            ``z_intervals`` not a 1D increasing array of at least 2 edges.
    """
    shapely = _load_shapely()
    tris = as_triangle_vertices(geometry)
    if len(tris) == 0:
        raise ValueError("geometry has no triangles")

    a = np.deg2rad(float(direction_deg))
    u_axis = np.array([np.sin(a), -np.cos(a), 0.0])
    uz = np.stack([tris @ u_axis, tris[..., 2]], axis=-1)  # (n, 3, 2)

    lo, hi = uz.reshape(-1, 2).min(0), uz.reshape(-1, 2).max(0)
    extent = float((hi - lo).max())
    if grid_size is None:
        if extent == 0.0:
            raise ValueError("geometry projects to a single point")
        grid_size = RELATIVE_GRID_SIZE * extent
    grid_size = float(grid_size)
    if grid_size <= 0.0:
        raise ValueError(f"grid_size must be > 0; got {grid_size}")

    e1, e2 = uz[:, 1] - uz[:, 0], uz[:, 2] - uz[:, 0]
    tri_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    uz = uz[tri_area > grid_size**2]
    rings = np.concatenate([uz, uz[:, :1]], axis=1)
    shadow = shapely.union_all(shapely.polygons(rings), grid_size=grid_size)

    edges = band_areas = None
    if z_intervals is not None:
        edges = np.asarray(z_intervals, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError(
                f"z_intervals must be a 1D increasing array of at least 2 edges; got {edges}"
            )
        bands = shapely.box(lo[0] - extent, edges[:-1], hi[0] + extent, edges[1:])
        band_areas = shapely.area(shapely.intersection(shadow, bands, grid_size=grid_size))

    return ProjectedArea(
        shadow=shadow,
        area=float(shadow.area),
        u_axis=u_axis,
        grid_size=grid_size,
        z_intervals=edges,
        band_areas=band_areas,
    )
