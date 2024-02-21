import pathlib

import numpy as np
import pyvista as pv

from cfdmod.use_cases.pressure.snapshot.colormap import ColormapFactory
from cfdmod.use_cases.pressure.snapshot.config import (
    CameraConfig,
    ColormapConfig,
    ProjectionConfig,
)


def take_snapshot(
    scalar_name: str,
    file_path: pathlib.Path,
    output_path: pathlib.Path,
    colormap_params: ColormapConfig,
    project_params: ProjectionConfig,
    camera_params: CameraConfig,
):
    def get_mesh_center(mesh_bounds: list[float]) -> tuple[float, float, float]:
        centerX = (mesh_bounds[1] + mesh_bounds[0]) / 2
        centerY = (mesh_bounds[3] + mesh_bounds[2]) / 2
        centerZ = (mesh_bounds[5] + mesh_bounds[4]) / 2

        return (centerX, centerY, centerZ)
    
    original_mesh = pv.read(file_path)
    original_mesh.set_active_scalars(scalar_name)
    colormap_divs = 

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

    original_mesh.rotate_x(project_params.rotation[0], point=original_center, inplace=True)
    original_mesh.rotate_y(project_params.rotation[1], point=original_center, inplace=True)
    original_mesh.rotate_z(project_params.rotation[2], point=original_center, inplace=True)

    scalar_arr = original_mesh.active_scalars[~np.isnan(original_mesh.active_scalars)]
    scalar_range = (scalar_arr.min(), scalar_arr.max())

    plotting_cmap = ColormapFactory(
        scalar_range=scalar_range, n_divs=colormap_divs
    ).build_default_colormap()

    plotter.add_mesh(original_mesh, lighting=False, cmap=plotting_cmap, scalar_bar_args=sargs)

    for projection in Projections:
        axes = pv.Axes()
        duplicated_mesh = original_mesh.copy()

        axes.origin = original_center
        duplicated_mesh.rotate_x(projection.value[0], point=axes.origin, inplace=True)
        duplicated_mesh.rotate_y(projection.value[1], point=axes.origin, inplace=True)

        translation = [0, 0, 0]
        if projection == Projections.x_plus:
            translation[0] += (
                (original_bounds[1] - original_bounds[0]) / 2
                + (original_bounds[5] - original_bounds[4]) / 2
                + offset_value
            )
            translation[2] += (
                -(original_bounds[1] - original_bounds[0]) / 2
                + (original_bounds[5] - original_bounds[4]) / 2
            )
        elif projection == Projections.x_minus:
            translation[0] -= (
                (original_bounds[1] - original_bounds[0]) / 2
                + (original_bounds[5] - original_bounds[4]) / 2
                + offset_value
            )
            translation[2] += (
                -(original_bounds[1] - original_bounds[0]) / 2
                + (original_bounds[5] - original_bounds[4]) / 2
            )
        elif projection == Projections.y_plus:
            translation[1] -= (
                (original_bounds[3] - original_bounds[2]) / 2
                + (original_bounds[5] - original_bounds[4]) / 2
                + offset_value
            )
            translation[2] += (
                -(original_bounds[3] - original_bounds[2]) / 2
                + (original_bounds[5] - original_bounds[4]) / 2
            )
        elif projection == Projections.y_minus:
            translation[1] += (
                (original_bounds[3] - original_bounds[2]) / 2
                + (original_bounds[5] - original_bounds[4]) / 2
                + offset_value
            )
            translation[2] += (
                -(original_bounds[3] - original_bounds[2]) / 2
                + (original_bounds[5] - original_bounds[4]) / 2
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
