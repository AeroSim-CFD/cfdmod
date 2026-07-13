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
  :func:`slice_field_on_plane` cuts the field mesh with a plane and returns the
  cross-section points / values / outline segments as a :class:`PlaneSlice`
  (rendered to a 2-D PNG by the pure-matplotlib :func:`render_plane_slice`).
  The VTK-backed paths (``write_field_vtp``, ``render_vtp_snapshot`` and
  ``slice_field_on_plane``) all no-op with a clear message when PyVista/VTK is
  absent.

Facade selection reuses the library's normal-based grouping
(:func:`cfdmod.geometry.grouping.kinds.by_normal.apply_by_normal`): a
building's triangles bucket into ``+x`` / ``-x`` / ``+y`` / ``-y`` side faces
and ``+z`` roof by the cardinal direction of their outward normal.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

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


# -- plane-slice field render (optional PyVista) ---------------------------

_AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


@dataclass
class PlaneSlice:
    """A field mesh cut by a plane: cross-section points, values and outline.

    Attributes:
        points: ``(n_pts, 3)`` slice-point coordinates.
        values: ``(n_pts,)`` scalar sampled onto ``points`` (aligned 1:1).
        segments: ``(n_seg, 2)`` index pairs into ``points`` -> outline lines
            tracing the building cross-section.
        origin: ``(3,)`` plane origin echoed back.
        normal: ``(3,)`` plane normal echoed back.
    """

    points: np.ndarray
    values: np.ndarray
    segments: np.ndarray
    origin: np.ndarray
    normal: np.ndarray


def slice_field_on_plane(
    geometry,
    field: np.ndarray,
    *,
    origin,
    normal=(0.0, 1.0, 0.0),
    association: str = "cell",
    field_name: str = "value",
) -> PlaneSlice | None:
    """Cut a per-triangle ``field`` on a mesh with a plane (requires ``[vtk]``).

    Builds a PyVista surface from the ``LnasGeometry`` vertices + triangles,
    attaches ``field`` as cell (default) or point data, and slices it with the
    plane ``(origin, normal)``. Returns a :class:`PlaneSlice` with the
    cross-section ``points``, per-point ``values`` and outline ``segments``.

    No-ops with a clear message and returns ``None`` when PyVista/VTK is absent,
    mirroring :func:`write_field_vtp` / :func:`render_vtp_snapshot`. When the
    plane misses the mesh, returns a :class:`PlaneSlice` with empty arrays.

    Args:
        geometry: An ``LnasGeometry`` (uses ``vertices`` + ``triangles``).
        field: Per-triangle (``association="cell"``) or per-vertex
            (``association="point"``) scalar array.
        origin: ``(3,)`` point on the slicing plane.
        normal: ``(3,)`` plane normal (default ``+y``).
        association: ``"cell"`` (per-triangle, primary) or ``"point"``.
        field_name: Name to store the scalar under on the PyVista mesh.
    """
    if not has_pyvista():
        print("[mesh_field] PyVista not installed ([vtk] extra); slice_field_on_plane is a no-op.")
        return None
    try:
        import pyvista as pv
    except Exception:
        return None

    origin = np.asarray(origin, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)

    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    triangles = np.asarray(geometry.triangles, dtype=np.int64)  # (n_tri, 3)
    n_tri = triangles.shape[0]
    faces = np.hstack(
        [np.full((n_tri, 1), 3, dtype=np.int64), triangles]
    ).ravel()  # [3, i0, i1, i2, 3, ...]
    mesh = pv.PolyData(vertices, faces)

    values_in = np.asarray(field, dtype=np.float64)
    if association == "point":
        mesh.point_data[field_name] = values_in
    else:
        mesh.cell_data[field_name] = values_in

    sliced = mesh.slice(normal=normal, origin=origin)

    empty = np.empty((0, 3), dtype=np.float64)
    if sliced.n_points == 0:
        return PlaneSlice(
            points=empty,
            values=np.empty((0,), dtype=np.float64),
            segments=np.empty((0, 2), dtype=np.int64),
            origin=origin,
            normal=normal,
        )

    points = np.asarray(sliced.points, dtype=np.float64)  # (n_pts, 3)

    if association == "point":
        values = np.asarray(sliced[field_name], dtype=np.float64)
    else:
        per_point = sliced.cell_data_to_point_data()
        values = np.asarray(per_point[field_name], dtype=np.float64)

    lines = np.asarray(sliced.lines, dtype=np.int64)
    if lines.size:
        segments = lines.reshape(-1, 3)[:, 1:]  # [2, a, b, ...] -> (n_seg, 2)
    else:
        segments = np.empty((0, 2), dtype=np.int64)

    return PlaneSlice(
        points=points,
        values=values,
        segments=segments,
        origin=origin,
        normal=normal,
    )


def render_plane_slice(
    slc: PlaneSlice | None,
    image_path: str | pathlib.Path,
    *,
    plane_axes: tuple[str, str] = ("x", "z"),
    mode: str = "tricontourf",
    cmap: str = "turbo",
    clim: tuple[float, float] | None = None,
    clim_percentile: tuple[float, float] = (2.0, 98.0),
    levels: int = 24,
    label: str = "p",
    title: str = "",
    draw_outline: bool = True,
    outline_kwargs: dict | None = None,
) -> bool:
    """Render a :class:`PlaneSlice` cross-section to a 2-D PNG (matplotlib only).

    Maps the two in-plane coordinates (``plane_axes``, default ``x`` / ``z`` for
    the ``p(x, z)`` view on a ``y=const`` plane), draws the field as filled
    contours (``mode="tricontourf"``, falling back to a scatter when the
    triangulation degenerates) or a ``"scatter"``, adds a colourbar and overlays
    the slice's own line cells as the building cross-section outline.

    Colour limits default to the ``clim_percentile`` percentiles of the values
    (robust to outliers). No-ops (returns ``False``) on a ``None`` or empty
    slice, so it degrades cleanly when the plane misses the mesh.

    Returns ``True`` on success, ``False`` when nothing was rendered.
    """
    if slc is None or slc.points.size == 0:
        print("[mesh_field] empty slice; nothing to render.")
        return False

    from matplotlib.collections import LineCollection

    from cfdmod import plot_config

    h = slc.points[:, _AXIS_INDEX[plane_axes[0]]]
    v = slc.points[:, _AXIS_INDEX[plane_axes[1]]]
    values = slc.values

    if clim is None:
        finite = values[np.isfinite(values)]
        if finite.size:
            clim = tuple(np.nanpercentile(finite, clim_percentile))
        else:
            clim = (0.0, 1.0)
    vmin, vmax = float(clim[0]), float(clim[1])

    plot_config.apply_style()
    fig, ax = plot_config.new_axes(
        xlabel=f"{plane_axes[0]} [m]",
        ylabel=f"{plane_axes[1]} [m]",
        title=title,
    )

    mappable = None
    if mode == "tricontourf":
        try:
            mappable = ax.tricontourf(h, v, values, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        except Exception:
            mappable = None
    if mappable is None:
        mappable = ax.scatter(h, v, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=6)

    fig.colorbar(mappable, ax=ax, label=label)

    if draw_outline and slc.segments.size:
        opts = {"colors": "black", "linewidths": 0.8, "zorder": 3}
        if outline_kwargs:
            opts.update(outline_kwargs)
        coords = np.column_stack([h, v])  # (n_pts, 2)
        lines = coords[slc.segments]  # (n_seg, 2, 2)
        ax.add_collection(LineCollection(lines, **opts))

    ax.set_aspect("equal")

    image_path = pathlib.Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    plot_config.close(fig)
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
