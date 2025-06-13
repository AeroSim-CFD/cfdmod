import pathlib

import numpy as np
import pyvista as pv
from cfdmod.logger import logger

from cfdmod.use_cases.snapshot.colormap import ColormapFactory
from cfdmod.use_cases.snapshot.config import (
    LegendConfig,
    Projections,
    SnapshotConfig,
    TransformationConfig,
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
        projection_config = snapshot_config.projections[projection]

        mesh = pv.read(projection_config.file_path)
        mesh.set_active_scalars(projection_config.scalar)

        clip_box = projection_config.clip_box
        if clip_box.scale[0] and clip_box.scale[1] and clip_box.scale[2] != 0:
            mesh = clip(mesh, clip_box)
            if mesh.n_cells == 0:
                logger.warning(
                    f"The clip box in projection '{projection}' is cropping the model completely."
                )
                return

        transform(mesh, projection_config.transformation)

        mesh = mesh.cell_data_to_point_data()
        plotter.add_mesh(mesh, lighting=False, cmap=lut, scalar_bar_args=sargs, nan_color="white")

        contours = create_contours(mesh, projection_config.scalar, snapshot_config.legend_config)
        plotter.add_mesh(contours, color="grey", line_width=2)

        feature_edge = create_feature_edges(mesh)
        plotter.add_mesh(feature_edge, color="black", line_width=1)

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


def clip(mesh, clip_box: TransformationConfig):
    clip_cube = pv.Cube(
        center=mesh.center,
        x_length=clip_box.scale[0],
        y_length=clip_box.scale[1],
        z_length=clip_box.scale[2],
    )
    transform(clip_cube, clip_box)
    return mesh.clip_box(clip_cube, invert=False)


def transform(mesh, transformation: TransformationConfig):
    mesh.rotate_x(transformation.rotate[0], point=mesh.center, inplace=True)
    mesh.rotate_y(transformation.rotate[1], point=mesh.center, inplace=True)
    mesh.rotate_z(transformation.rotate[2], point=mesh.center, inplace=True)
    mesh.translate(transformation.translate, inplace=True)


def create_contours(mesh, scalar: str, legend_config: LegendConfig):
    return mesh.contour(
        np.linspace(
            legend_config.range[0],
            legend_config.range[1],
            legend_config.n_divs + 1,
        ),
        scalars=scalar,
    )


def create_feature_edges(mesh):
    return mesh.extract_feature_edges(
        feature_angle=30,
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        non_manifold_edges=False,
    )
