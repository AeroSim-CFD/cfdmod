import pathlib

import numpy as np
import pyvista as pv
from cfdmod.logger import logger

from cfdmod.use_cases.snapshot.config import (
    LabelsConfig,
    LegendConfig,
    ProjectionConfig,
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

        clip_box = projection_config.clip_box
        if all(dimension > 0 for dimension in clip_box.scale):
            mesh = clip(mesh, clip_box)
            if mesh.n_cells == 0:
                logger.warning(
                    f"The clip box in projection '{projection}' is cropping the model completely."
                )
                return

        transform(mesh, projection_config.transformation)

        if projection_config.scalar:
            mesh.set_active_scalars(projection_config.scalar)

            mesh = mesh.cell_data_to_point_data()
            plotter.add_mesh(
                mesh, lighting=False, cmap=lut, scalar_bar_args=sargs, nan_color=lut.nan_color
            )

            contours = create_contours(
                mesh, projection_config.scalar, snapshot_config.legend_config
            )
            plotter.add_mesh(contours, color="grey", line_width=2)

            if projection_config.labels_config:
                points, labels = create_labels(
                    mesh, projection_config, projection_config.labels_config
                )
                plotter.add_point_labels(
                    points=points,
                    labels=labels,
                    font_size=12,
                    text_color="black",
                    point_color="black",
                    point_size=5,
                    render_points_as_spheres=True,
                    shape_opacity=0,
                    always_visible=True,
                )
        else:
            plotter.add_mesh(mesh, lighting=False, color="white")

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
        center=[0, 0, 0],
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


def create_labels(mesh, projection_config: ProjectionConfig, labels_config: LabelsConfig):
    bounds = list(mesh.bounds)

    spacing_x, spacing_y = labels_config.spacing
    padding_left, padding_right, padding_bottom, padding_top = labels_config.padding

    x_min = bounds[0] + padding_left
    x_max = bounds[1] - padding_right
    y_min = bounds[2] + padding_bottom
    y_max = bounds[3] - padding_top

    size_x = x_max - x_min
    size_y = y_max - y_min

    if size_x < spacing_x:
        x_targets = [bounds[0] + (bounds[1] - bounds[0]) / 2]
    else:
        num_divisions_x = int(size_x // spacing_x)
        num_points_x = num_divisions_x + 1
        total_spacing_x = spacing_x * num_divisions_x
        center_x = (x_min + x_max) / 2
        x_start = center_x - total_spacing_x / 2
        x_end = center_x + total_spacing_x / 2
        x_targets = np.linspace(x_start, x_end, num_points_x)

    if size_y < spacing_y:
        y_targets = [bounds[2] + (bounds[3] - bounds[2]) / 2]
    else:
        num_divisions_y = int(size_y // spacing_y)
        num_points_y = num_divisions_y + 1
        total_spacing_y = spacing_y * num_divisions_y
        center_y = (y_min + y_max) / 2
        y_start = center_y - total_spacing_y / 2
        y_end = center_y + total_spacing_y / 2
        y_targets = np.linspace(y_start, y_end, num_points_y)

    z_level = bounds[5]

    X, Y, Z = np.meshgrid(x_targets, y_targets, [z_level], indexing="ij")
    target_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    closest_point_ids = []
    for pt in target_points:
        closest_point_id = mesh.find_closest_point(pt)
        closest_pt = mesh.points[closest_point_id]
        if np.linalg.norm(pt - closest_pt) < 10:
            closest_point_ids.append(closest_point_id)

    points = mesh.points[closest_point_ids]
    values = mesh.point_data[projection_config.scalar][closest_point_ids]
    labels = [f"{v:.2f}" for v in values]

    return points, labels