__all__ = [
    "ImageConfig",
    "CropConfig",
    "OverlayImageConfig",
    "OverlayTextConfig",
    "TransformationConfig",
    "LegendConfig",
    "ColormapConfig",
    "CameraConfig",
    "ValueTagsConfig",
    "ProjectionConfig",
    "SnapshotConfig",
    "building_facade_config",
    "take_snapshot",
    "add_mesh_projection_to_screenshot",
    "get_combined_bounding_box",
    "add_text_overlay_to_screenshot",
    "clip_mesh",
    "transform_mesh",
    "create_contours",
    "create_feature_edges",
    "create_value_tags",
    "get_mesh_center",
]

try:
    import pyvista as _pyvista  # noqa: F401
except ImportError as _exc:  # pragma: no cover - depends on the install
    raise ImportError(
        "cfdmod.snapshot requires the optional 'snapshot' extras. "
        "Install with: pip install aerosim-cfdmod[snapshot]"
    ) from _exc

from cfdmod.snapshot.config import (
    ImageConfig,
    CropConfig,
    OverlayImageConfig,
    OverlayTextConfig,
    TransformationConfig,
    LegendConfig,
    ColormapConfig,
    CameraConfig,
    ValueTagsConfig,
    ProjectionConfig,
    SnapshotConfig,
)
from cfdmod.snapshot.building_facade import building_facade_config
from cfdmod.snapshot.snapshot import (
    take_snapshot,
    add_mesh_projection_to_screenshot,
    get_combined_bounding_box,
    add_text_overlay_to_screenshot,
    clip_mesh,
    transform_mesh,
    create_contours,
    create_feature_edges,
    create_value_tags,
    get_mesh_center,
)
