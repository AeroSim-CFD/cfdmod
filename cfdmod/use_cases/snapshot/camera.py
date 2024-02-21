import pathlib

import numpy as np
import pyvista as pv

from cfdmod.use_cases.snapshot.colormap import ColormapFactory
from cfdmod.use_cases.snapshot.config import (
    CameraConfig,
    ColormapConfig,
    ProjectionConfig,
    Projections,
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
    scalar_name: str,
    file_path: pathlib.Path,
    output_path: pathlib.Path,
    colormap_params: ColormapConfig,
    projection_params: ProjectionConfig,
    camera_params: CameraConfig,
):
    """Use pyvista renderer to take a snapshot

    Args:
        scalar_name (str): Variable name
        file_path (pathlib.Path): Input polydata file path
        output_path (pathlib.Path): Output path for saving images
        colormap_params (ColormapConfig): Parameters for colormap
        projection_params (ProjectionConfig): Parameters for projection
        camera_params (CameraConfig): Parameters for camera
    """
    original_mesh = pv.read(file_path)
    original_mesh.set_active_scalars(scalar_name)

    scalar_arr = original_mesh.active_scalars[~np.isnan(original_mesh.active_scalars)]
    scalar_range = np.array([scalar_arr.min(), scalar_arr.max()])
    colormap_divs = colormap_params.get_colormap_divs(scalar_range)

    sargs = dict(
        title=f"{scalar_name}\n",
        title_font_size=24,
        label_font_size=20,
        n_labels=colormap_divs + 1,
        italic=False,
        fmt="%.2f",
        font_family="arial",
        position_x=0.2,
        position_y=0.0,
        width=0.6,
    )
    plotter = pv.Plotter(window_size=camera_params.window_size)
    plotter.enable_parallel_projection()

    original_bounds = original_mesh.bounds
    original_center = get_mesh_center(original_bounds)

    original_mesh.rotate_x(projection_params.rotation[0], point=original_center, inplace=True)
    original_mesh.rotate_y(projection_params.rotation[1], point=original_center, inplace=True)
    original_mesh.rotate_z(projection_params.rotation[2], point=original_center, inplace=True)

    plotting_cmap = ColormapFactory(
        scalar_range=scalar_range, n_divs=colormap_divs
    ).build_default_colormap()

    plotter.add_mesh(original_mesh, lighting=False, cmap=plotting_cmap, scalar_bar_args=sargs)

    for projection in [p for p in Projections if p.value[0] in projection_params.axis]:
        axes = pv.Axes()
        duplicated_mesh = original_mesh.copy()

        axes.origin = original_center
        duplicated_mesh.rotate_x(projection.value[1][0], point=axes.origin, inplace=True)
        duplicated_mesh.rotate_y(projection.value[1][1], point=axes.origin, inplace=True)

        translation = get_translation(
            bounds=original_bounds, for_projection=projection, offset_val=projection_params.offset
        )

        duplicated_mesh = duplicated_mesh.translate(translation, inplace=True)
        plotter.add_mesh(
            duplicated_mesh, lighting=False, cmap=plotting_cmap, scalar_bar_args=sargs
        )

    plotter.camera_position = "xy"
    plotter.camera.SetParallelProjection(True)

    camera = plotter.camera
    camera.SetFocalPoint(camera.GetFocalPoint() + camera_params.offset_position)
    camera.SetPosition(camera.GetPosition() + camera_params.offset_position)

    plotter.camera.up = camera_params.view_up
    plotter.camera.zoom(camera_params.zoom)

    plotter.show()
    plotter.screenshot(output_path)
    plotter.close()
