"""Mesh-field snapshots for the high-rise facade + structure-print stages.

Two rendering paths, so the stages run everywhere yet still produce
publication-quality images where the optional renderer is installed:

- **matplotlib (default, always available).** A pure-matplotlib 3-D triangle
  renderer (``mpl_toolkits.mplot3d``) colours each triangle by a per-triangle
  field (e.g. mean / peak Cp) and writes a PNG. Needs only the core deps, so
  it runs headless under nbconvert / the validators.
- **PyVista (optional, ``[vtk]`` extra).** :func:`write_field_vtp` dumps the
  per-triangle field to a ``.vtp`` and :func:`render_vtp_snapshot` drives
  :func:`cfdmod.snapshot.take_snapshot` for a contoured, colour-barred render.
  Both no-op with a clear message when PyVista/VTK is absent.

Facade selection reuses the library's normal-based grouping
(:func:`cfdmod.geometry.grouping.kinds.by_normal.apply_by_normal`): a
building's triangles bucket into ``+x`` / ``-x`` / ``+y`` / ``-y`` side faces
and ``+z`` roof by the cardinal direction of their outward normal.
"""

from __future__ import annotations

import pathlib

import numpy as np

# View presets for the matplotlib 3-D camera: (elevation, azimuth) in degrees.
STANDARD_VIEWS: dict[str, tuple[float, float]] = {
    "iso": (22.0, -60.0),
    "front": (8.0, -90.0),
    "back": (8.0, 90.0),
    "left": (8.0, 180.0),
    "right": (8.0, 0.0),
    "top": (89.0, -90.0),
}

# Cardinal-axis token -> human label for the side / roof faces.
FACADE_LABELS: dict[str, str] = {
    "n_+x": "face_+x",
    "n_-x": "face_-x",
    "n_+y": "face_+y",
    "n_-y": "face_-y",
    "n_+z": "roof",
}


def load_geometry(mesh_path: str | pathlib.Path):
    """Load an ``.lnas`` and return its ``LnasGeometry`` (vertices + triangles)."""
    from lnas import LnasFormat

    return LnasFormat.from_file(pathlib.Path(mesh_path)).geometry


def facade_groups(
    mesh_path: str | pathlib.Path,
    *,
    axes: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z"),
    tolerance_deg: float = 45.0,
) -> dict[str, np.ndarray]:
    """Triangle-index buckets per facade, by outward-normal direction.

    Returns a dict keyed ``n_+x`` ... ``n_+z`` (see :data:`FACADE_LABELS`) to
    the triangle indices whose normal best aligns with that cardinal axis.
    """
    from lnas import LnasFormat

    from cfdmod.geometry.grouping.kinds.by_normal import ByNormalGrouping, apply_by_normal

    mesh = LnasFormat.from_file(pathlib.Path(mesh_path))
    spec = ByNormalGrouping(axes=list(axes), tolerance_deg=tolerance_deg)
    return apply_by_normal(spec, mesh, allowed=None)


def triangle_field_figure(
    geometry,
    values: np.ndarray | None,
    *,
    view: tuple[float, float] = STANDARD_VIEWS["iso"],
    cmap: str = "turbo",
    clim: tuple[float, float] | None = None,
    subset: np.ndarray | None = None,
    title: str = "",
    cbar_label: str = "",
    face_color: str = "#c8d0d8",
):
    """Render a per-triangle field on a mesh to a matplotlib 3-D figure.

    Args:
        geometry: An ``LnasGeometry`` (uses ``triangle_vertices``).
        values: Per-triangle scalar array ``(n_tri,)``, or ``None`` to draw
            plain geometry in ``face_color``.
        view: ``(elevation, azimuth)`` camera angles, e.g. one of
            :data:`STANDARD_VIEWS`.
        clim: Colour limits; defaults to the finite min/max of ``values``.
        subset: Triangle indices to draw (e.g. one facade); ``None`` = all.
        cbar_label: Colourbar label; only shown when ``values`` is given.

    Returns a ``(fig, ax)`` matplotlib pair.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = np.asarray(geometry.triangle_vertices, dtype=np.float64)  # (n_tri, 3, 3)
    if subset is not None:
        subset = np.asarray(subset, dtype=np.int64)
        tris = tris[subset]

    fig = plt.figure(figsize=(6.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    coll = Poly3DCollection(tris, linewidths=0.1, edgecolors=(0, 0, 0, 0.15))
    if values is None:
        coll.set_facecolor(face_color)
    else:
        vals = np.asarray(values, dtype=np.float64)
        if subset is not None:
            vals = vals[subset]
        if clim is None:
            finite = vals[np.isfinite(vals)]
            clim = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        coll.set_cmap(cmap)
        coll.set_array(vals)
        coll.set_clim(*clim)
    ax.add_collection3d(coll)

    _equalize_3d(ax, tris)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if title:
        ax.set_title(title)
    if values is not None:
        cbar = fig.colorbar(coll, ax=ax, shrink=0.6, pad=0.1)
        if cbar_label:
            cbar.set_label(cbar_label)
    return fig, ax


def facade_index_per_triangle(groups: dict[str, np.ndarray], n_tri: int) -> np.ndarray:
    """Map each triangle to its facade-group index (NaN if unassigned).

    Handy as the ``values`` argument to :func:`triangle_field_figure` to show
    the facade partition as a categorical colouring.
    """
    out = np.full(n_tri, np.nan, dtype=np.float64)
    for idx, tri_ids in enumerate(sorted(groups)):
        out[groups[tri_ids]] = float(idx)
    return out


# -- field sampling along a line + time-window stats -----------------------


def triangle_centroids(geometry) -> np.ndarray:
    """Per-triangle centroids ``(n_tri, 3)`` from an ``LnasGeometry``."""
    tris = np.asarray(geometry.triangle_vertices, dtype=np.float64)  # (n_tri, 3, 3)
    return tris.mean(axis=1)


def sample_field_along_line(
    geometry,
    field: np.ndarray,
    p1,
    p2,
    *,
    n: int = 100,
):
    """Sample a per-triangle ``field`` along the segment ``p1 -> p2``.

    Each of ``n`` evenly spaced points takes the value of the nearest triangle
    (by centroid, via a KD-tree). Returns a DataFrame with the arc length ``s``,
    the sample coordinates ``x/y/z`` and ``value`` -- e.g. facade pressure vs
    height by sampling a vertical line down a facade.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    field = np.asarray(field, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    pts = p1[None, :] + t[:, None] * (p2 - p1)[None, :]  # (n, 3)

    tree = cKDTree(triangle_centroids(geometry))
    _, idx = tree.query(pts)
    s = t * float(np.linalg.norm(p2 - p1))
    return pd.DataFrame(
        {"s": s, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "value": field[idx]}
    )


def moving_average_stats(series: np.ndarray, dt: float, window_s: float) -> dict:
    """Rolling-window mean of a 1-D signal plus its peak stats.

    Reuses the library's canonical moving average
    (:func:`cfdmod.core.ops.field.moving_average`): ``window_s`` (in the
    series' time units) is rounded to the nearest odd sample count via
    :func:`~cfdmod.core.ops.field.moving_average.window_in_samples`, and the
    signal is edge-padded so the smoothed series ``ma`` stays the same length
    and aligned with the input. Returns ``mean`` (of the raw series), ``ma``,
    and ``ma_max`` / ``ma_min`` -- the "peak of the N-second moving average"
    used for facade design pressures.
    """
    from cfdmod.core.ops.field.moving_average import window_in_samples

    series = np.asarray(series, dtype=np.float64)
    if series.size == 0:
        return {
            "mean": float("nan"),
            "window": 0,
            "ma": series,
            "ma_max": float("nan"),
            "ma_min": float("nan"),
        }
    n = window_in_samples(window_s, dt)
    if n <= 1:
        ma = series
    else:
        pad = n // 2
        padded = np.pad(series, (pad, pad), mode="edge")
        ma = np.convolve(padded, np.ones(n) / n, mode="valid")
    return {
        "mean": float(series.mean()),
        "window": n,
        "ma": ma,
        "ma_max": float(ma.max()),
        "ma_min": float(ma.min()),
    }


def has_pyvista() -> bool:
    """True when the optional ``[vtk]`` extra (pyvista + vtk) is importable."""
    try:
        import pyvista  # noqa: F401

        return True
    except Exception:
        return False


def write_field_vtp(geometry, fields: dict[str, np.ndarray], path: str | pathlib.Path) -> bool:
    """Write per-triangle fields to a ``.vtp`` polydata (requires ``[vtk]``).

    ``fields`` maps a scalar name to a ``(n_tri,)`` array. Returns ``True`` on
    success, ``False`` (no file written) when VTK is unavailable, so callers
    can degrade to the matplotlib path.
    """
    try:
        import pandas as pd

        from cfdmod.io.vtk import create_polydata_for_cell_data, write_polydata
    except Exception:
        return False

    n_tri = int(np.asarray(geometry.triangle_vertices).shape[0])
    data = pd.DataFrame({"point_idx": np.arange(n_tri)})
    for name, arr in fields.items():
        data[name] = np.asarray(arr, dtype=np.float64)
    poly = create_polydata_for_cell_data(data, geometry)
    write_polydata(pathlib.Path(path), poly)
    return True


def render_vtp_snapshot(
    vtp_path: str | pathlib.Path,
    image_path: str | pathlib.Path,
    *,
    scalar: str,
    label: str,
    clim: tuple[float, float],
    n_divs: int = 10,
    view_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    zoom: float = 1.0,
    window_size: tuple[int, int] = (900, 900),
) -> bool:
    """Render a ``.vtp`` scalar to a PNG via PyVista (requires ``[vtk]``).

    A best-effort wrapper over :func:`cfdmod.snapshot.take_snapshot` with a
    single projection and a turbo colourbar. Returns ``True`` on success,
    ``False`` when VTK is unavailable.
    """
    try:
        from cfdmod.snapshot.config import (
            CameraConfig,
            LegendConfig,
            ProjectionConfig,
            SnapshotConfig,
        )
        from cfdmod.snapshot.snapshot import take_snapshot
    except Exception:
        return False

    cfg = SnapshotConfig(
        projections={
            "body": ProjectionConfig(file_path=pathlib.Path(vtp_path), scalar=scalar),
        },
        legend_config=LegendConfig(label=label, range=clim, n_divs=n_divs),
        camera=CameraConfig(zoom=zoom, view_up=view_up, window_size=window_size),
    )
    take_snapshot(pathlib.Path(image_path), cfg, off_screen=True)
    return True


def _equalize_3d(ax, tris: np.ndarray) -> None:
    """Give the 3-D axes an equal-aspect box around the drawn triangles."""
    pts = tris.reshape(-1, 3)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    ax.set_box_aspect(tuple(span))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
