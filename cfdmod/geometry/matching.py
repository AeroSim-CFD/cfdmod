"""Match the triangles of one geometry against a reference geometry.

The canonical use case: an ``.stl`` exported for a region of interest (a
facade, a roof panel, a floor strip) is a *sub-mesh* of the reference mesh
the simulation wrote its time series against. To read the reference data on
that region you need, for each triangle of the sub-mesh, the index of the
same triangle in the reference mesh.

:func:`match_triangles` returns exactly that: an index array in the target
(sub-mesh) order, with :data:`MISSING` (``-1``) wherever no reference
triangle corresponds. :meth:`TriangleMatch.as_float` gives the same array
as ``float64`` with ``NaN`` in the missing slots, for callers that prefer a
NaN sentinel.

Matching is by triangle centroid within a tolerance, cross-checked by
triangle area. Centroids are invariant to the vertex rotation and the
winding flip that STL round-trips introduce, and the tolerance absorbs the
``float32`` quantisation of the STL format itself (an absolute error that
scales with the coordinate magnitude, not with the model size -- hence the
default tolerance scale below).

Pure numpy plus a lazy ``scipy.spatial`` KD-tree; ``lnas`` and the mesh
readers are imported only when a path or a mesh object is passed in, so
importing this module stays cheap.
"""

from __future__ import annotations

__all__ = ["MISSING", "TriangleMatch", "as_triangle_vertices", "match_triangles"]

import pathlib
import warnings
from dataclasses import dataclass

import numpy as np

#: Sentinel stored in :attr:`TriangleMatch.indices` for a target triangle
#: with no counterpart in the reference geometry.
MISSING = -1


@dataclass(frozen=True)
class TriangleMatch:
    """Result of :func:`match_triangles`.

    Attributes:
        indices: ``(n_target,)`` ``int64``. ``indices[i]`` is the index in
            the reference geometry of target triangle ``i``, or
            :data:`MISSING` (``-1``) when it was not found. Ordered by the
            target geometry, so ``reference_field[m.indices]`` lines up with
            the target triangles (only valid where ``m.found``).
        distances: ``(n_target,)`` ``float64`` centroid distance to the
            matched reference triangle; ``inf`` where missing.
        tolerance: Absolute centroid tolerance actually used.
        ambiguous: ``(n_target,)`` ``bool``, True where more than one
            reference triangle sat within ``tolerance`` of the target
            centroid (coincident / duplicated reference triangles). The
            nearest one was taken.
    """

    indices: np.ndarray
    distances: np.ndarray
    tolerance: float
    ambiguous: np.ndarray

    @property
    def found(self) -> np.ndarray:
        """``(n_target,)`` bool mask of the target triangles that matched."""
        return self.indices != MISSING

    @property
    def missing(self) -> np.ndarray:
        """``(n_target,)`` bool mask of the target triangles with no match."""
        return self.indices == MISSING

    @property
    def n_found(self) -> int:
        return int(np.count_nonzero(self.found))

    @property
    def n_missing(self) -> int:
        return int(np.count_nonzero(self.missing))

    @property
    def is_complete(self) -> bool:
        """True when every target triangle found a reference triangle."""
        return self.n_missing == 0

    @property
    def is_injective(self) -> bool:
        """True when no two target triangles matched the same reference one."""
        found = self.indices[self.found]
        return int(np.unique(found).size) == int(found.size)

    def as_float(self) -> np.ndarray:
        """The indices as ``float64`` with ``NaN`` in place of :data:`MISSING`."""
        out = self.indices.astype(np.float64)
        out[self.missing] = np.nan
        return out

    def require_complete(self) -> np.ndarray:
        """Return the indices, raising if any target triangle is unmatched.

        Raises:
            ValueError: If at least one target triangle has no match.
        """
        if not self.is_complete:
            first = int(np.flatnonzero(self.missing)[0])
            raise ValueError(
                f"{self.n_missing} of {self.indices.size} target triangles have no "
                f"match in the reference geometry (first unmatched: index {first}); "
                f"centroid tolerance was {self.tolerance:.3e}"
            )
        return self.indices


def _tri_areas(tris: np.ndarray) -> np.ndarray:
    """Per-triangle area of a ``(n, 3, 3)`` vertex array."""
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def as_triangle_vertices(geometry, name: str = "geometry") -> np.ndarray:
    """Coerce a geometry-ish input to a ``(n, 3, 3)`` ``float64`` array.

    Accepts an ``(n, 3, 3)`` array of triangle vertices, an ``LnasGeometry``
    or ``LnasFormat`` (anything exposing ``triangle_vertices`` or
    ``geometry.triangle_vertices``), or a path to an ``.stl`` / ``.lnas``
    file.
    """
    if isinstance(geometry, (str, pathlib.Path)):
        path = pathlib.Path(geometry)
        suffix = path.suffix.lower()
        if suffix == ".stl":
            from cfdmod.io.geometry import read_stl

            tris, _ = read_stl(path)
            return np.asarray(tris, dtype=np.float64)
        if suffix == ".lnas":
            from lnas import LnasFormat

            fmt = LnasFormat.from_file(path)
            return np.asarray(fmt.geometry.triangle_vertices, dtype=np.float64)
        raise ValueError(
            f"{name}: unsupported geometry file {path.name!r} (expected .stl or .lnas)"
        )

    tris = getattr(geometry, "triangle_vertices", None)
    if tris is None:
        inner = getattr(geometry, "geometry", None)
        tris = getattr(inner, "triangle_vertices", None) if inner is not None else None
    if tris is None:
        tris = geometry

    arr = np.asarray(tris, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(
            f"{name}: expected triangle vertices of shape (n, 3, 3), got {arr.shape}; "
            "pass an array, an lnas geometry/format, or a path to a .stl / .lnas file"
        )
    return arr


def _tolerance_scale(tris: np.ndarray) -> float:
    """Length scale a relative tolerance is taken against.

    ``max(bounding-box diagonal, largest absolute coordinate)``: an ``.stl``
    stores ``float32``, so its absolute round-trip error grows with the
    distance from the origin, not with the size of the model. A mesh sitting
    at ``z = 900`` quantises ~1e-4 even if its bounding box is 1 unit wide.
    """
    if tris.size == 0:
        return 0.0
    flat = tris.reshape(-1, 3)
    diag = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    return max(diag, float(np.abs(flat).max()))


def match_triangles(
    reference,
    target,
    *,
    rtol: float = 1e-6,
    atol: float | None = None,
    area_rtol: float | None = 1e-2,
    warn_on_coarse_tolerance: bool = True,
) -> TriangleMatch:
    """Match the triangles of ``target`` against ``reference`` by centroid.

    Args:
        reference: The reference geometry -- the mesh whose triangle indices
            the caller wants. An ``(n, 3, 3)`` vertex array, an lnas
            geometry / format, or a path to a ``.stl`` / ``.lnas`` file.
        target: The geometry to locate inside ``reference``, typically a
            sub-mesh of it. Same accepted forms.
        rtol: Relative centroid tolerance, taken against the reference
            length scale (see :func:`_tolerance_scale`). Ignored if ``atol``
            is given.
        atol: Absolute centroid tolerance, in model units. Overrides
            ``rtol`` when set.
        area_rtol: Relative area agreement required of a candidate match; a
            nearest centroid whose area differs by more than this fraction
            is rejected as :data:`MISSING`. Set to ``None`` to skip the
            check (centroid proximity alone decides).
        warn_on_coarse_tolerance: Emit a ``RuntimeWarning`` when the
            tolerance is not clearly smaller than the closest pair of
            reference centroids, i.e. when a match could be ambiguous by
            construction.

    Returns:
        TriangleMatch: Indices in target order, ``-1`` where not found.

    Raises:
        ValueError: If either geometry cannot be read as ``(n, 3, 3)``
            vertices, or ``reference`` has no triangles while ``target``
            does.
    """
    ref_tris = as_triangle_vertices(reference, "reference")
    tgt_tris = as_triangle_vertices(target, "target")

    n_target = tgt_tris.shape[0]
    if n_target == 0:
        return TriangleMatch(
            indices=np.empty(0, dtype=np.int64),
            distances=np.empty(0, dtype=np.float64),
            tolerance=0.0,
            ambiguous=np.empty(0, dtype=bool),
        )
    if ref_tris.shape[0] == 0:
        raise ValueError(f"reference geometry has no triangles but target has {n_target}")

    tol = float(atol) if atol is not None else float(rtol) * _tolerance_scale(ref_tris)
    if tol <= 0.0:
        raise ValueError(f"centroid tolerance must be positive, got {tol!r}")

    from scipy.spatial import cKDTree

    ref_centroids = ref_tris.mean(axis=1)
    tgt_centroids = tgt_tris.mean(axis=1)
    tree = cKDTree(ref_centroids)

    if warn_on_coarse_tolerance and ref_centroids.shape[0] > 1:
        self_dist, _ = tree.query(ref_centroids, k=2)
        min_spacing = float(self_dist[:, 1].min())
        if min_spacing <= 2.0 * tol:
            warnings.warn(
                f"match_triangles: centroid tolerance {tol:.3e} is not clearly below the "
                f"closest reference centroid pair ({min_spacing:.3e}); matches may be "
                "ambiguous. Lower rtol / atol, or check the reference mesh for "
                "duplicated triangles.",
                RuntimeWarning,
                stacklevel=2,
            )

    # k=2 so a second reference triangle inside the tolerance is reported as
    # ambiguous instead of silently shadowed.
    k = 2 if ref_centroids.shape[0] > 1 else 1
    dist, idx = tree.query(tgt_centroids, k=k, distance_upper_bound=tol)
    dist = np.reshape(dist, (n_target, k))
    idx = np.reshape(idx, (n_target, k))

    # cKDTree marks "no neighbour within the bound" as dist=inf, idx=n_ref.
    found = np.isfinite(dist[:, 0])
    ambiguous = np.isfinite(dist[:, 1]) if k == 2 else np.zeros(n_target, dtype=bool)

    indices = np.where(found, idx[:, 0], MISSING).astype(np.int64)
    distances = np.where(found, dist[:, 0], np.inf).astype(np.float64)

    if area_rtol is not None:
        ref_areas = _tri_areas(ref_tris)
        tgt_areas = _tri_areas(tgt_tris)
        cand = indices != MISSING
        if np.any(cand):
            ref_a = ref_areas[indices[cand]]
            tgt_a = tgt_areas[cand]
            scale = np.maximum(np.maximum(np.abs(ref_a), np.abs(tgt_a)), np.finfo(np.float64).tiny)
            bad = np.abs(ref_a - tgt_a) / scale > float(area_rtol)
            if np.any(bad):
                reject = np.flatnonzero(cand)[bad]
                indices[reject] = MISSING
                distances[reject] = np.inf

    ambiguous = ambiguous & (indices != MISSING)
    return TriangleMatch(indices=indices, distances=distances, tolerance=tol, ambiguous=ambiguous)
