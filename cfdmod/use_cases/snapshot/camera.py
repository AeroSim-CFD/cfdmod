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
    data_source_paths: list[dict],
    output_path: pathlib.Path,
    colormap_params: ColormapConfig,
    projection_params: list[ProjectionConfig],
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

    meshes = {key: pv.read(path) for item in data_source_paths for key, path in item.items()}

    plotter = pv.Plotter(window_size=camera_params.window_size)
    plotter.enable_parallel_projection()

    # plotting_cmap = ColormapFactory(
    #     scalar_range=scalar_range, n_divs=colormap_divs
    # ).build_default_colormap()

    # lut = pv.LookupTable(cmap="turbo")
    # lut.scalar_range = (scalar_range[0], scalar_range[1])
    # lut.n_values = colormap_divs
    # lut.SetNanColor(1.0, 1.0, 1.0, 1.0)

    # feature_edges = original_mesh.extract_feature_edges(
    #     feature_angle=5,  # degrees, controls sensitivity
    #     boundary_edges=True,
    #     feature_edges=True,
    #     manifold_edges=False,
    #     non_manifold_edges=False,
    # )
    # plotter.add_mesh(feature_edges, color="black", line_width=2)

    # if colormap_params.style == "contour":
    #     original_mesh = original_mesh.cell_data_to_point_data()
    #     plotter.add_mesh(original_mesh, lighting=False, cmap=lut, scalar_bar_args=sargs)
    #     contours = original_mesh.contour(
    #         np.linspace(scalar_range[0], scalar_range[1], colormap_divs + 1),
    #         scalars=scalar_name,  # optional, if you want to contour along z-axis
    #     )
    #     plotter.add_mesh(contours, color="grey", line_width=2)
    # elif colormap_params.style == "flat":
    #     plotter.add_mesh(original_mesh, lighting=False, cmap=lut, scalar_bar_args=sargs)

    for projection in projection_params:
        axes = pv.Axes()
        current_mesh = meshes.get(projection.data_source_key).copy()
        current_mesh.set_active_scalars(scalar_name)

        scalar_arr = current_mesh.active_scalars[~np.isnan(current_mesh.active_scalars)]
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

        plotting_cmap = ColormapFactory(
            scalar_range=scalar_range, n_divs=colormap_divs
        ).build_default_colormap()

        lut = pv.LookupTable(cmap="turbo")
        lut.scalar_range = (scalar_range[0], scalar_range[1])
        lut.n_values = colormap_divs
        lut.SetNanColor(1.0, 1.0, 1.0, 1.0)

        feature_edges = current_mesh.extract_feature_edges(
            feature_angle=5,  # degrees, controls sensitivity
            boundary_edges=True,
            feature_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        plotter.add_mesh(feature_edges, color="black", line_width=2)

        # duplicated_mesh = original_mesh.copy()
        # transformed_box = reference_box.copy()

        # axes.origin = original_center

        current_mesh.rotate_x(projection.transformation.rotate[0], point=axes.origin, inplace=True)
        current_mesh.rotate_y(projection.transformation.rotate[1], point=axes.origin, inplace=True)
        current_mesh.rotate_z(projection.transformation.rotate[2], point=axes.origin, inplace=True)
        # transformed_box.rotate_x(projection.value[1][0], point=axes.origin, inplace=True)
        # transformed_box.rotate_y(projection.value[1][1], point=axes.origin, inplace=True)

        current_mesh.translate(projection.transformation.translate, inplace=True)
        # transformed_box = transformed_box.translate(translation, inplace=True)
        current_mesh.cell_data_to_point_data()
        plotter.add_mesh(
            current_mesh, lighting=False, cmap=lut, scalar_bar_args=sargs, nan_color="white"
        )
        feature_edges = current_mesh.extract_feature_edges(
            feature_angle=5,  # degrees, controls sensitivity
            boundary_edges=True,
            feature_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        plotter.add_mesh(feature_edges, color="black", line_width=2)

        if colormap_params.style == "contour":
            current_mesh = current_mesh.cell_data_to_point_data()
            plotter.add_mesh(current_mesh, lighting=False, cmap=lut, scalar_bar_args=sargs)
            contours = current_mesh.contour(
                np.linspace(scalar_range[0], scalar_range[1], colormap_divs + 1),
                scalars=scalar_name,  # optional, if you want to contour along z-axis
            )
            plotter.add_mesh(contours, color="grey", line_width=2)
        elif colormap_params.style == "flat":
            plotter.add_mesh(current_mesh, lighting=False, cmap=lut, scalar_bar_args=sargs)

    plotter.camera_position = "xy"
    plotter.camera.SetParallelProjection(True)

    camera = plotter.camera
    camera.SetFocalPoint(camera.GetFocalPoint() + np.array(camera_params.offset_position))
    camera.SetPosition(camera.GetPosition() + np.array(camera_params.offset_position))

    plotter.camera.up = camera_params.view_up
    plotter.camera.zoom(camera_params.zoom)

    plotter.show(jupyter_backend="static")
    plotter.screenshot(output_path)
    plotter.close()
