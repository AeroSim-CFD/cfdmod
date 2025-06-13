import pathlib

import numpy as np
import pyvista as pv

from cfdmod.use_cases.snapshot.colormap import ColormapFactory
from cfdmod.use_cases.snapshot.config import (
    Projections,
    SnapshotConfig,
)


def get_mesh_center(mesh_bounds: list[float]) -> tuple[float, float, float]:
    """Calculates mesh center

    Args:
        mesh_bounds (list[float]): Mesh bounds (x_min, x_max, y_min, y_max, z_min, z_max)

    Returns:
        tuple[float, float, float]: Mesh center (x, y, z)
    """
    centerX = (mesh_bounds[1] + mesh_bounds[0]) / 2
    centerY = (mesh_bounds[3] + mesh_bounds[2]) / 2
    centerZ = (mesh_bounds[5] + mesh_bounds[4]) / 2

    return (centerX, centerY, centerZ)


def get_translation(
    bounds: list[float], for_projection: Projections, offset_val: float
) -> tuple[float, float, float]:
    """Calculates projection translation

    Args:
        bounds (list[float]): Mesh bounds (x_min, x_max, y_min, y_max, z_min, z_max)
        for_projection (Projections): Which projection to calculate translation
        offset_val (float): Value for offsetting projection from the center projection

    Returns:
        tuple[float, float, float]: Translation value (x, y, z)
    """
    x_translate = (bounds[1] - bounds[0]) / 2 + (bounds[5] - bounds[4]) / 2
    y_translate = (bounds[3] - bounds[2]) / 2 + (bounds[5] - bounds[4]) / 2

    if for_projection == Projections.x_plus:
        return (x_translate + offset_val, 0, x_translate)
    elif for_projection == Projections.x_minus:
        return (-(x_translate + offset_val), 0, x_translate)
    elif for_projection == Projections.y_plus:
        return (0, -(y_translate + offset_val), y_translate)
    elif for_projection == Projections.y_minus:
        return (0, y_translate + offset_val, y_translate)
    else:
        raise ValueError(f"Projection {for_projection} is not supported")


def take_snapshot(
    snapshot_config: SnapshotConfig,
):
    """Use pyvista renderer to take a snapshot

    Args:
        output_path (pathlib.Path): Output path for saving images
        snapshot_config (SnapshotConfig): Parameters for snapshot
    """

    plotter = pv.Plotter(window_size=snapshot_config.camera.window_size)
    plotter.enable_parallel_projection()

    sargs = dict(
        title=f"{snapshot_config.legend_config.label}\n",
        title_font_size=24,
        label_font_size=20,
        n_labels=snapshot_config.legend_config.n_divs + 1,
        italic=False,
        fmt="%.2f",
        font_family="arial",
        position_x=0.2,
        position_y=0.0,
        width=0.6,
    )

    lut = pv.LookupTable(cmap="turbo")
    lut.scalar_range = (
        snapshot_config.legend_config.range[0],
        snapshot_config.legend_config.range[1],
    )
    lut.n_values = snapshot_config.legend_config.n_divs
    lut.SetNanColor(1.0, 1.0, 1.0, 1.0)

    for projection in snapshot_config.projections:
        projection = snapshot_config.projections[projection]
        mesh = pv.read(projection.file_path)
        mesh.set_active_scalars(projection.scalar)

        feature_edges = mesh.extract_feature_edges(
            feature_angle=5,  # degrees, controls sensitivity
            boundary_edges=True,
            feature_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        plotter.add_mesh(feature_edges, color="black", line_width=2)

        mesh.rotate_x(projection.transformation.rotate[0], point=mesh.center, inplace=True)
        mesh.rotate_y(projection.transformation.rotate[1], point=mesh.center, inplace=True)
        mesh.rotate_z(projection.transformation.rotate[2], point=mesh.center, inplace=True)

        mesh.translate(projection.transformation.translate, inplace=True)

        mesh.cell_data_to_point_data()
        plotter.add_mesh(mesh, lighting=False, cmap=lut, scalar_bar_args=sargs, nan_color="white")
        feature_edges = mesh.extract_feature_edges(
            feature_angle=5,  # degrees, controls sensitivity
            boundary_edges=True,
            feature_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        plotter.add_mesh(feature_edges, color="black", line_width=2)

        if snapshot_config.colormap.style == "contour":
            mesh = mesh.cell_data_to_point_data()
            plotter.add_mesh(mesh, lighting=False, cmap=lut, scalar_bar_args=sargs)
            contours = mesh.contour(
                np.linspace(
                    snapshot_config.legend_config.range[0],
                    snapshot_config.legend_config.range[1],
                    snapshot_config.legend_config.n_divs + 1,
                ),
                scalars=projection.scalar,  # optional, if you want to contour along z-axis
            )
            plotter.add_mesh(contours, color="grey", line_width=2)
        elif snapshot_config.colormap.style == "flat":
            plotter.add_mesh(mesh, lighting=False, cmap=lut, scalar_bar_args=sargs)

    plotter.camera_position = "xy"
    plotter.camera.SetParallelProjection(True)

    camera = plotter.camera
    camera.SetFocalPoint(camera.GetFocalPoint() + np.array(snapshot_config.camera.offset_position))
    camera.SetPosition(camera.GetPosition() + np.array(snapshot_config.camera.offset_position))

    plotter.camera.up = snapshot_config.camera.view_up
    plotter.camera.zoom(snapshot_config.camera.zoom)

    plotter.show(jupyter_backend="static")
    plotter.screenshot(snapshot_config.name)
    plotter.close()
