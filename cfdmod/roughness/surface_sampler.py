import pathlib
from collections.abc import Sequence
from typing import Union

import numpy as np
from lnas import LnasFormat, LnasGeometry
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, QhullError

__all__ = [
    "SurfaceInput",
    "SurfaceSampler",
    "build_surface_sampler",
    "load_surface_vertices",
    "DEFAULT_MAX_SAMPLE_POINTS",
]

SurfaceInput = Union[pathlib.Path, str, LnasFormat, LnasGeometry, np.ndarray]
"""A surface accepted by the sampler: a path to an LNAS/STL file, an in-memory
``LnasFormat`` or ``LnasGeometry``, or a raw (N, 3) array of vertices."""

DEFAULT_MAX_SAMPLE_POINTS = 100_000
"""Default cap on the number of points fed to the triangulation.

The Delaunay build behind ``LinearNDInterpolator`` grows faster than linearly
with the point count, and real terrain surfaces carry far more vertices than
the drape needs. Above this many unique vertices the sample is thinned by XY
grid binning, with the boundary of the surface kept at full density so neither
the valid domain nor the accuracy of the drape near the edge changes.
"""


def _vertices_from_surface(surface: SurfaceInput) -> np.ndarray:
    """Extract the (N, 3) vertex array of a single surface input.

    Args:
        surface (SurfaceInput): Path to an LNAS/STL file, an in-memory
            ``LnasFormat`` / ``LnasGeometry``, or a raw (N, 3) vertex array.

    Returns:
        np.ndarray: Vertices as float64, shape (N, 3).
    """
    if isinstance(surface, LnasFormat):
        vertices = surface.geometry.vertices
    elif isinstance(surface, LnasGeometry):
        vertices = surface.vertices
    elif isinstance(surface, (str, pathlib.Path)):
        vertices = LnasFormat.from_file(pathlib.Path(surface)).geometry.vertices
    elif isinstance(surface, np.ndarray):
        vertices = surface
    else:
        raise TypeError(
            f"Unsupported surface input of type {type(surface).__name__}. "
            "Expected a path, an LnasFormat, an LnasGeometry or an (N, 3) array."
        )

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Surface vertices must have shape (N, 3), got {vertices.shape}")
    return vertices


def load_surface_vertices(surfaces: Sequence[SurfaceInput]) -> list[np.ndarray]:
    """Read every surface once and return their vertex arrays.

    Callers that need both the surfaces' bounding box and a sampler should use
    this and pass the resulting arrays to :func:`build_surface_sampler`, so a
    file is never read twice.

    Args:
        surfaces (Sequence[SurfaceInput]): Surfaces to read.

    Returns:
        list[np.ndarray]: One (N, 3) float64 vertex array per surface.
    """
    if len(surfaces) == 0:
        raise ValueError("At least one surface is required")
    return [_vertices_from_surface(s) for s in surfaces]


def _cell_grid(xy: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign every position to a cell of a square XY grid.

    Args:
        xy (np.ndarray): (N, 2) positions.
        n_cells (int): Target number of grid cells over the populated area.

    Returns:
        tuple[np.ndarray, np.ndarray]: The (N,) flat cell id of each position and
            the (2,) number of cells per axis.
    """
    lo = xy.min(axis=0)
    span = xy.max(axis=0) - lo
    # A degenerate span in one axis collapses that axis to a single row of cells.
    positive = span[span > 0]
    if positive.size == 0:
        return np.zeros(len(xy), dtype=np.int64), np.ones(2, dtype=np.int64)

    # Square cells, so the thinning does not favour one direction.
    cell = float(np.prod(positive) / n_cells) ** (1.0 / positive.size)
    n_per_axis = np.maximum(1, np.ceil(np.where(span > 0, span / cell, 0.0)).astype(np.int64))

    idx = np.minimum((np.where(span > 0, (xy - lo) / cell, 0.0)).astype(np.int64), n_per_axis - 1)
    return idx[:, 0] * n_per_axis[1] + idx[:, 1], n_per_axis


def _select_indices(xy: np.ndarray, n_cells: int) -> np.ndarray:
    """Pick the points to keep for a given grid resolution.

    Cells on the rim of the occupied region keep every one of their points, so
    the boundary of the surface stays at full resolution. Without that, the
    triangulation spans the gap between the sparse boundary and the thinned
    interior with very wide triangles, and the interpolated Z there is metres
    off. Interior cells keep a single representative each.

    Args:
        xy (np.ndarray): (N, 2) positions.
        n_cells (int): Target number of grid cells over the populated area.

    Returns:
        np.ndarray: Sorted indices into ``xy`` of the points to keep.
    """
    cell_id, n_per_axis = _cell_grid(xy, n_cells)
    occupied = np.unique(cell_id)

    def occupied_at(ids: np.ndarray, in_range: np.ndarray) -> np.ndarray:
        position = np.clip(np.searchsorted(occupied, ids), 0, len(occupied) - 1)
        return in_range & (occupied[position] == ids)

    n_y = n_per_axis[1]
    cell_x, cell_y = occupied // n_y, occupied % n_y
    interior_cell = (
        occupied_at(occupied - n_y, cell_x > 0)
        & occupied_at(occupied + n_y, cell_x < n_per_axis[0] - 1)
        & occupied_at(occupied - 1, cell_y > 0)
        & occupied_at(occupied + 1, cell_y < n_y - 1)
    )

    on_rim = np.isin(cell_id, occupied[~interior_cell])
    rim_idx = np.flatnonzero(on_rim)

    interior_idx = np.flatnonzero(~on_rim)
    _, first = np.unique(cell_id[interior_idx], return_index=True)

    return np.sort(np.concatenate([rim_idx, interior_idx[first]]))


def _hull_and_stride(xy: np.ndarray, indices: np.ndarray, budget: int) -> np.ndarray:
    """Cut a selection down to ``budget`` without shrinking the valid domain.

    Fallback for a footprint so jagged relative to the cap that no grid
    resolution fits. The XY convex hull is kept whole, because it is what bounds
    the domain the interpolator returns a value over; the rest is strided.

    Args:
        xy (np.ndarray): (N, 2) positions.
        indices (np.ndarray): Currently selected indices, over budget.
        budget (int): Number of indices to return.

    Returns:
        np.ndarray: Sorted indices, at most ``budget`` of them.
    """
    try:
        hull_idx = ConvexHull(xy).vertices.astype(np.intp)
    except QhullError:
        # Degenerate XY footprint (collinear or coincident); nothing to preserve.
        hull_idx = np.empty(0, dtype=np.intp)

    remaining = budget - len(hull_idx)
    if remaining <= 0:
        return np.sort(hull_idx)

    rest = indices[~np.isin(indices, hull_idx)]
    if len(rest) > remaining:
        rest = rest[np.linspace(0, len(rest) - 1, remaining).astype(np.intp)]
    return np.sort(np.concatenate([hull_idx, rest]))


def _thin_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Reduce a point cloud to at most ``max_points`` while keeping its boundary.

    Args:
        points (np.ndarray): (N, 3) deduplicated points.
        max_points (int): Upper bound on the returned point count.

    Returns:
        np.ndarray: (M, 3) points with M <= max_points (M == N if no thinning
            was needed).
    """
    if len(points) <= max_points:
        return points

    xy = points[:, :2]

    # Ceil-ing the cell count per axis, plus the rim kept at full density, means
    # a grid sized at the budget always lands a little over it. Rescale a few
    # times to land under the cap without wasting most of it.
    n_cells = max_points
    best: np.ndarray | None = None
    for _ in range(5):
        kept = _select_indices(xy, n_cells)
        if len(kept) <= max_points:
            if best is None or len(kept) > len(best):
                best = kept
            if len(kept) >= max_points * 0.7:
                break
        n_cells = max(1, int(n_cells * max_points / max(1, len(kept))))

    if best is None:
        best = _hull_and_stride(xy, kept, max_points)

    return points[best]


class SurfaceSampler:
    """Vectorized Z sampler over the union of one or more surfaces.

    Wraps a linear interpolation over the surfaces' vertices. Positions outside
    the sampled region -- the XY convex hull of the pooled vertices, which is
    what the underlying triangulation spans -- sample as NaN.
    """

    def __init__(self, points: np.ndarray):
        """
        Args:
            points (np.ndarray): (M, 3) points to interpolate over.
        """
        if len(points) < 3:
            raise ValueError(
                f"Need at least 3 surface points to interpolate, got {len(points)}. "
                "Check that the given surfaces are non-empty."
            )
        # `points` is handed out by the property; the interpolator is built
        # from it and would silently disagree with a mutated copy. A read-only
        # view guards that without freezing the caller's own array.
        read_only = np.asarray(points).view()
        read_only.flags.writeable = False
        self._points = read_only
        self._interpolator = LinearNDInterpolator(points[:, :2], points[:, 2])

    @property
    def n_points(self) -> int:
        """Number of points the interpolator was built over."""
        return len(self._points)

    @property
    def points(self) -> np.ndarray:
        """The (M, 3) points the interpolator was built over."""
        return self._points

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Sample surface Z at XY positions, in one vectorized call.

        Args:
            xy (np.ndarray): (N, 2) query positions.

        Returns:
            np.ndarray: (N,) Z values, NaN where the position falls outside the
                sampled region.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Query positions must have shape (N, 2), got {xy.shape}")
        if len(xy) == 0:
            return np.empty(0, dtype=np.float64)
        return np.asarray(self._interpolator(xy[:, 0], xy[:, 1]), dtype=np.float64)


def build_surface_sampler(
    surfaces: Sequence[SurfaceInput],
    max_points: int | None = DEFAULT_MAX_SAMPLE_POINTS,
) -> SurfaceSampler:
    """Build a bounded, vectorized Z sampler over a set of surfaces.

    Surfaces may be given as file paths or as in-memory geometry, so a caller
    that already holds the mesh does not pay an LNAS write plus read to use it.

    The vertices of all surfaces are pooled and deduplicated. If more than
    ``max_points`` unique vertices remain, the sample is thinned by XY grid
    binning: interior cells keep one representative each, while cells on the rim
    of the surface keep every point. That leaves the domain the sampler returns
    a value over unchanged, and keeps the drape near the surface edge as
    accurate as it is in the middle.

    Args:
        surfaces (Sequence[SurfaceInput]): Paths to LNAS/STL files, in-memory
            ``LnasFormat`` / ``LnasGeometry`` objects, or (N, 3) vertex arrays.
        max_points (int | None, optional): Cap on the number of interpolation
            points. None disables thinning. Defaults to
            ``DEFAULT_MAX_SAMPLE_POINTS``.

    Returns:
        SurfaceSampler: Sampler over the pooled surfaces.
    """
    if max_points is not None and max_points < 3:
        raise ValueError(f"max_points must be at least 3, got {max_points}")

    vertices = load_surface_vertices(surfaces)
    points = np.unique(np.concatenate(vertices), axis=0)
    if max_points is not None:
        points = _thin_points(points, max_points)
    return SurfaceSampler(points)
