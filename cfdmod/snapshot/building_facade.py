"""Building-facade snapshot layout for cfdmod.snapshot.

Unfolds a rectangular building's four walls (N / E / S / W) side by side as
vertical strips, with the roof on top -- the layout wind engineers use for
facade pressure-coefficient deliverables. Returns a base :class:`SnapshotConfig`
whose projections are left blank; point it at a field with
:meth:`SnapshotConfig.retarget`. Pass ``z_band`` to clip every wall to a height
band (e.g. per 10 floors) so a tall tower reads at each level; use a shared
``value_range`` across directions / bands so the colours are comparable.

The per-wall rotations bring each outward-facing wall to face the (parallel)
camera as a vertical strip; wall widths follow the footprint (N/S span x, E/W
span y). Geometry only -- no mesh/VTK dependency here, so it is unit-testable
without the render extras.
"""

from __future__ import annotations

import numpy as np

from cfdmod.snapshot.config import (
    CameraConfig,
    LegendConfig,
    OverlayTextConfig,
    ProjectionConfig,
    SnapshotConfig,
    TransformationConfig,
)

# (label, rotate-to-upright-vertical-strip-facing-camera, footprint axis index for width).
# Each rotation brings that outward wall to face the parallel camera as an upright
# strip (building +z -> image +y); N/S span x, E/W span y.
_WALLS: tuple[tuple[str, tuple[int, int, int], int], ...] = (
    ("N", (90, 0, 180), 0),  # +y wall, width along x
    ("E", (0, -90, -90), 1),  # +x wall, width along y
    ("S", (-90, 0, 0), 0),  # -y wall, width along x
    ("W", (0, 90, 90), 1),  # -x wall, width along y
)


def building_facade_config(
    bbox_lo,
    bbox_hi,
    *,
    legend_label: str = "Cp",
    value_range: tuple[float, float] = (-1.5, 1.0),
    n_divs: int = 10,
    z_band: tuple[float, float] | None = None,
    with_roof: bool = True,
    gap: float | None = None,
    zoom: float = 1.1,
    window_size: tuple[int, int] = (1900, 1100),
    file_path: str = "",
    scalar: str = "",
) -> SnapshotConfig:
    """Base :class:`SnapshotConfig` for a building facade (walls side by side + roof).

    Args:
        bbox_lo / bbox_hi: building mesh bounding box ``(x, y, z)`` min / max.
        value_range / n_divs: colour scale; pass the SAME ``value_range`` for
            every direction and band of a statistic so the colours compare.
        z_band: ``(z_lo, z_hi)`` to clip every wall to a height band (per-floor
            bands); ``None`` renders the full height. Bands drop the roof.
        with_roof: place the roof projection above the wall row (full height only).
        gap: spacing between unfolded faces (default ``0.1 * height``).
        file_path / scalar: usually left blank -- set per field with
            :meth:`SnapshotConfig.retarget`.
    """
    lo = np.asarray(bbox_lo, dtype=np.float64)
    hi = np.asarray(bbox_hi, dtype=np.float64)
    cx, cy, _cz = (lo + hi) / 2
    lx, ly, lz = hi - lo
    ext = (lx, ly, lz)
    if gap is None:
        gap = 0.1 * lz

    clip = None
    if z_band is not None:
        z_lo, z_hi = z_band
        clip = TransformationConfig(
            translate=[float(cx), float(cy), float((z_lo + z_hi) / 2)],
            rotate=[0, 0, 0],
            scale=[3 * lx, 3 * ly, float(z_hi - z_lo)],
        )
    strip_h = (z_band[1] - z_band[0]) if z_band is not None else lz

    def proj(rotate, tx, ty):
        return ProjectionConfig(
            file_path=str(file_path),
            scalar=scalar,
            cell_data_to_point_data=False,
            clip_box=clip,
            transformation=TransformationConfig(
                translate=[float(tx - cx), float(ty - cy), 0.0],
                rotate=list(rotate),
                scale=[1, 1, 1],
            ),
        )

    widths = {name: ext[axis] for name, _rot, axis in _WALLS}
    total = sum(widths.values()) + gap * (len(_WALLS) - 1)
    projections: dict[str, ProjectionConfig] = {}
    texts: list[OverlayTextConfig] = []
    cursor = -total / 2
    for name, rotate, _axis in _WALLS:
        w = widths[name]
        px = cursor + w / 2
        cursor += w + gap
        projections[name] = proj(rotate, px, 0.0)
        texts.append(
            OverlayTextConfig(
                text=name, position=[float(px), float(-strip_h / 2 - gap)], font_size=11
            )
        )

    if with_roof and z_band is None:
        # the tower-top slab only (clip off the podium footprint), snug above the
        # wall row, no label -- it reads as the roof by position.
        top = float(hi[2])
        roof_depth = 6.0
        roof_clip = TransformationConfig(
            translate=[float(cx), float(cy), top - roof_depth / 2],
            rotate=[0, 0, 0],
            scale=[3 * lx, 3 * ly, roof_depth],
        )
        roof = ProjectionConfig(
            file_path=str(file_path),
            scalar=scalar,
            cell_data_to_point_data=False,
            clip_box=roof_clip,
            transformation=TransformationConfig(
                translate=[float(-cx), float(lz / 2 + ly / 2 - cy), 0.0],
                rotate=[0, 0, 0],
                scale=[1, 1, 1],
            ),
        )
        projections["roof"] = roof

    return SnapshotConfig(
        projections=projections,
        legend_config=LegendConfig(label=legend_label, range=tuple(value_range), n_divs=n_divs),
        camera=CameraConfig(zoom=zoom, view_up=[0, 1, 0], window_size=list(window_size)),
        text_overlay=texts,
    )
