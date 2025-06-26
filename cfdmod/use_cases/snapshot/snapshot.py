import pathlib

import numpy as np
import pyvista as pv

from cfdmod.logger import logger
from cfdmod.use_cases.snapshot.config import (  # Projections,
    LegendConfig,
    OverlayTextConfig,
    ProjectionConfig,
    SnapshotConfig,
    TransformationConfig,
    ValueTagsConfig,
)
from cfdmod.use_cases.snapshot.image_processing import (
    crop_image,
    display_image,
    paste_overlay_image,
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


def take_snapshot(
    image_path: pathlib.Path | str,
    snapshot_config: SnapshotConfig,
):
    """Use pyvista renderer to take a snapshot

    Args:
        output_path (pathlib.Path): Output path for saving images
        snapshot_config (SnapshotConfig): Parameters for snapshot
    """
    if isinstance(image_path, str):
        image_path = pathlib.Path(image_path)

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
        add_mesh_projection_to_screenshot(
            plotter=plotter,
            projection_config=projection_config,
            colomap_lookup_table=lut,
            scalar_bar_args=sargs,
            legend_config=snapshot_config.legend_config,
        )

    combined_bounding_box = get_combined_bounding_box(plotter)
    for text_overlay_config in snapshot_config.text_overlay:
        add_text_overlay_to_screenshot(
            plotter, text_overlay_config, scene_borders=combined_bounding_box
        )

    plotter.camera_position = "xy"
    plotter.camera.SetParallelProjection(True)
    plotter.camera.up = snapshot_config.camera.view_up
    plotter.reset_camera()
    plotter.camera.zoom(snapshot_config.camera.zoom)
    camera = plotter.camera
    camera_offset = np.array(snapshot_config.camera.offset_position)

    focal_point = list(camera.GetFocalPoint())
    focal_point[0] -= camera_offset[0]
    focal_point[1] -= camera_offset[1]
    camera.SetFocalPoint(focal_point)
    camera_position = list(camera.GetPosition())
    camera_position[0] -= camera_offset[0]
    camera_position[1] -= camera_offset[1]
    camera.SetPosition(camera_position)

    camera.reset_clipping_range()

    plotter.screenshot(image_path)
    plotter.close()

    for image_overlay_config in snapshot_config.images_overlay:
        paste_overlay_image(
            main_image_path=image_path, image_to_overlay_config=image_overlay_config
        )

    if snapshot_config.image_crop is not None:
        crop_image(image_path=image_path, crop_cfg=snapshot_config.image_crop)

    display_image(image_path)


def add_mesh_projection_to_screenshot(
    plotter: pv.Plotter,
    projection_config: ProjectionConfig,
    colomap_lookup_table: pv.LookupTable,
    scalar_bar_args: dict,
    legend_config: LegendConfig,
):
    mesh = pv.read(projection_config.file_path)
    if projection_config.clip_box is not None:
        clip_box = projection_config.clip_box
        if all(dimension > 0 for dimension in clip_box.scale):
            mesh = clip_mesh(mesh, clip_box)
            if mesh.n_cells == 0:
                logger.warning("The clip box in projection is cropping the model completely.")
                return
    transform_mesh(mesh, projection_config.transformation)

    # move mesh to z=0 for better control of image
    center = mesh.center
    mesh = mesh.translate([0, 0, -center[2]])

    if projection_config.scalar is not None:
        mesh.set_active_scalars(projection_config.scalar)
        mesh = mesh.cell_data_to_point_data()
        plotter.add_mesh(
            mesh,
            lighting=False,
            cmap=colomap_lookup_table,
            scalar_bar_args=scalar_bar_args,
            nan_color=colomap_lookup_table.nan_color,
        )

        contours = create_contours(mesh, projection_config.scalar, legend_config)
        plotter.add_mesh(contours, color="grey", line_width=1)
        if projection_config.values_tag_config is not None:
            points, labels = create_value_tags(
                mesh, projection_config, projection_config.values_tag_config
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


def get_combined_bounding_box(plotter: pv.Plotter) -> np.ndarray:
    b = plotter.bounds
    return np.array([[b[0], b[1]], [b[2], b[3]], [b[4], b[5]]])


def add_text_overlay_to_screenshot(
    plotter: pv.Plotter, text_config: OverlayTextConfig, scene_borders: np.ndarray[3, 2]
):
    """Adds a text to the screenshot

    Args:
        plotter (pv.Plotter): Plotter handler
        watermark_image (Image): Watermark image
    """
    text = text_config.text
    position = text_config.position
    font_size = text_config.font_size
    angle = text_config.angle

    center = scene_borders.mean(axis=1)

    text_obj = pv.Text3D(text, depth=0.1)
    text_obj = text_obj.scale(font_size)
    text_obj = text_obj.rotate_z(angle)
    text_obj = text_obj.translate(
        (center[0] + position[0], center[1] + position[1], scene_borders[2, 1] + 1)
    )  # text 100m over geometries at z=0
    plotter.add_mesh(text_obj, color="black")


def clip_mesh(mesh: pv.DataSet, clip_box: TransformationConfig) -> pv.UnstructuredGrid:
    clip_cube = pv.Cube(
        center=[0, 0, 0],
        x_length=clip_box.scale[0],
        y_length=clip_box.scale[1],
        z_length=clip_box.scale[2],
    )
    transform_mesh(clip_cube, clip_box)
    return mesh.clip_box(clip_cube, invert=False)


def transform_mesh(mesh: pv.DataSet, transformation: TransformationConfig):
    mesh.rotate_x(transformation.rotate[0], point=mesh.center, inplace=True)
    mesh.rotate_y(transformation.rotate[1], point=mesh.center, inplace=True)
    mesh.rotate_z(transformation.rotate[2], point=mesh.center, inplace=True)
    mesh.translate(transformation.translate, inplace=True)


def create_contours(mesh: pv.DataSet, scalar: str, legend_config: LegendConfig) -> pv.PolyData:
    return mesh.contour(
        np.linspace(
            legend_config.range[0],
            legend_config.range[1],
            legend_config.n_divs + 1,
        ),
        scalars=scalar,
    )


def create_feature_edges(mesh: pv.DataSet) -> pv.DataSet:
    return mesh.extract_feature_edges(
        feature_angle=30,
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        non_manifold_edges=False,
    )


def create_value_tags(
    mesh: pv.DataSet, projection_config: ProjectionConfig, value_tags_config: ValueTagsConfig
) -> tuple[np.ndarray, pv.DataSetAttributes]:
    bounds = list(mesh.bounds)

    spacing_x, spacing_y = value_tags_config.spacing
    padding_left, padding_right, padding_bottom, padding_top = value_tags_config.padding

    x_min = bounds[0] + padding_left
    x_max = bounds[1] - padding_right
    y_min = bounds[2] + padding_bottom
    y_max = bounds[3] - padding_top

    size_x = x_max - x_min
    size_y = y_max - y_min

    if size_x < spacing_x:
        x_targets = [(bounds[0] + bounds[1]) / 2]
    else:
        num_divisions_x = int(size_x // spacing_x)
        num_points_x = num_divisions_x + 1
        total_spacing_x = spacing_x * num_divisions_x
        center_x = (x_min + x_max) / 2
        x_start = center_x - total_spacing_x / 2
        x_end = center_x + total_spacing_x / 2
        x_targets = np.linspace(x_start, x_end, num_points_x)

    if size_y < spacing_y:
        y_targets = [(bounds[2] + bounds[3]) / 2]
    else:
        num_divisions_y = int(size_y // spacing_y)
        num_points_y = num_divisions_y + 1
        total_spacing_y = spacing_y * num_divisions_y
        center_y = (y_min + y_max) / 2
        y_start = center_y - total_spacing_y / 2
        y_end = center_y + total_spacing_y / 2
        y_targets = np.linspace(y_start, y_end, num_points_y)

    z_level = bounds[5] - value_tags_config.z_offset

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
    dp = value_tags_config.decimal_places
    labels = [f"{v:.{dp}f}" for v in values]

    return points, labels
