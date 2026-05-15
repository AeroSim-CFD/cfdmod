"""Tests for moment coefficient (Cm) functions."""

import numpy as np
import pandas as pd
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.pressure.functions import add_lever_arm_to_geometry_df, transform_Cm
from cfdmod.pressure.geometry import build_geometry_data, tabulate_geometry_data
from cfdmod.pressure.parameters import MomentBodyConfig, ZoningModel
from cfdmod.utils import convert_dataframe_into_matrix

pytestmark = pytest.mark.unit


def _moment_body(name="body", lever_origin=(0.0, 0.0, 10.0), **kwargs) -> MomentBodyConfig:
    return MomentBodyConfig(name=name, lever_origin=lever_origin, **kwargs)


@pytest.fixture()
def body_data():
    data = pd.DataFrame(
        {
            "cp": [0.1, 0.2, 0.3, 0.4],
            "time_normalized": [0, 0, 1, 1],
            "point_idx": [0, 1, 0, 1],
        }
    )
    yield convert_dataframe_into_matrix(
        data, row_data_label="time_normalized", value_data_label="cp"
    )


@pytest.fixture()
def body_geom():
    vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    yield LnasGeometry(vertices, triangles)


@pytest.fixture()
def geom_data(body_geom):
    parent = LnasFormat(version="", geometry=body_geom, surfaces={"body": np.array([0, 1])})
    yield build_geometry_data(
        body_label="body",
        sfc_list=["body"],
        zoning=ZoningModel(),
        mesh=parent,
    )


@pytest.fixture()
def geometry_df(geom_data, body_geom):
    geometry_dict = {"body": geom_data}
    yield tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=body_geom.areas,
        mesh_normals=body_geom.normals,
        transformation=TransformationConfig(),
    )


def test_add_lever_arm_fixed(geom_data, geometry_df):
    result_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        body_cfg=_moment_body(lever_origin=(0.0, 0.0, 10.0)),
        geometry_df=geometry_df,
    )

    assert all(f"r{d}" in result_df.columns for d in ["x", "y", "z"])
    # Body fixture: triangles span (0,0,0)-(10,10,0). Lever origin (0,0,10).
    # First triangle centroid is roughly (10/3, 10/3, 0) -> rz = -10.
    np.testing.assert_allclose(result_df["rz"].to_numpy(), [-10.0, -10.0])


def test_add_lever_arm_region_base(geom_data, geometry_df):
    result_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        body_cfg=_moment_body(lever_strategy="region_base"),
        geometry_df=geometry_df,
    )
    # All vertices live on z=0, so the auto-derived base z is 0 -> rz == cz == 0.
    np.testing.assert_allclose(result_df["rz"].to_numpy(), [0.0, 0.0])


def test_add_lever_arm_explicit_override(geom_data, geometry_df):
    # Default zoning produces a single region per body labelled "0-body".
    result_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        body_cfg=_moment_body(
            lever_origin=(0.0, 0.0, 0.0),
            region_lever_origins={0: (0.0, 0.0, 100.0)},
        ),
        geometry_df=geometry_df,
    )
    # Override puts origin at z=100 -> rz = 0 - 100 = -100 for both tris.
    np.testing.assert_allclose(result_df["rz"].to_numpy(), [-100.0, -100.0])


def test_add_lever_arm_override_beats_strategy(geom_data, geometry_df):
    result_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        body_cfg=_moment_body(
            lever_strategy="region_base",
            region_lever_origins={0: (0.0, 0.0, 100.0)},
        ),
        geometry_df=geometry_df,
    )
    # Region 0 uses the explicit override even though strategy is region_base.
    np.testing.assert_allclose(result_df["rz"].to_numpy(), [-100.0, -100.0])


def test_add_lever_arm_handles_unassigned_sentinel(geom_data, geometry_df):
    """Regression: a triangle whose centroid did not fall into any zoning
    cell carries the ``-1`` sentinel that
    :func:`cfdmod.pressure.geometry.get_indexing_mask` writes. The
    resulting ``region_idx`` is the string ``"-1-<body>"``. Earlier
    versions of ``_resolve_region_origin`` did
    ``int(label.split("-", 1)[0])``, which on ``"-1-body"`` produces
    ``int("")`` and crashes with ``ValueError`` -- so Cm blew up while
    Cf (which only groups by the string label) silently stayed alive.

    The fix is ``rsplit("-", 1)`` so the body suffix is peeled from the
    right; this test pins that.
    """
    geometry_df = geometry_df.copy()
    # Force the first triangle to look "unassigned" the way
    # get_indexing_mask would have left it.
    geometry_df.loc[0, "region_idx"] = "-1-body"

    # The pre-fix call raised ValueError("invalid literal for int() with base 10: ''").
    result_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        body_cfg=_moment_body(lever_origin=(0.0, 0.0, 10.0)),
        geometry_df=geometry_df,
    )
    # Sentinel triangles fall through to the fixed-strategy branch, so
    # they pick up the body's lever_origin (z=10) -- rz = -10 for both
    # triangles regardless of their region label.
    np.testing.assert_allclose(result_df["rz"].to_numpy(), [-10.0, -10.0])


def test_add_lever_arm_handles_unassigned_sentinel_region_base(geom_data, geometry_df):
    """Same regression under ``lever_strategy="region_base"``: the
    sentinel rows compute their own ``(mean, mean, min)`` from
    ``local_idx_in_region`` instead of crashing on the int parse."""
    geometry_df = geometry_df.copy()
    geometry_df.loc[0, "region_idx"] = "-1-body"

    result_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        body_cfg=_moment_body(lever_strategy="region_base"),
        geometry_df=geometry_df,
    )
    # All triangles live on z=0, so the auto base z is 0 -> rz == 0.
    np.testing.assert_allclose(result_df["rz"].to_numpy(), [0.0, 0.0])


def test_add_lever_arm_explicit_override_on_negative_key(geom_data, geometry_df):
    """The override map keys on the parsed int. With the new rsplit
    parser, a key like ``-1`` is honoured for sentinel rows."""
    geometry_df = geometry_df.copy()
    geometry_df.loc[0, "region_idx"] = "-1-body"

    result_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        body_cfg=_moment_body(
            lever_origin=(0.0, 0.0, 0.0),
            region_lever_origins={-1: (0.0, 0.0, 100.0), 0: (0.0, 0.0, 0.0)},
        ),
        geometry_df=geometry_df,
    )
    # Only the sentinel row (first triangle) sits on the -1 override
    # (z=100 -> rz = -100); the second still has region "0-body" -> rz = 0.
    np.testing.assert_allclose(result_df["rz"].to_numpy(), [-100.0, 0.0])


def test_lever_origin_cases_round_trip():
    """A MomentBodyConfig with lever_origin_cases preserves the dict."""
    body = MomentBodyConfig(
        name="pack",
        lever_origin_cases={
            "xmin_ymin": {0: (0.0, 0.0, 0.0), 1: (10.0, 0.0, 0.0)},
            "xmax_ymin": {0: (5.0, 0.0, 0.0), 1: (15.0, 0.0, 0.0)},
        },
    )
    assert set(body.lever_origin_cases) == {"xmin_ymin", "xmax_ymin"}
    assert body.lever_origin_cases["xmax_ymin"][1] == (15.0, 0.0, 0.0)


def test_lever_strategy_accepts_region_bbox_corners_xy():
    body = MomentBodyConfig(name="pack", lever_strategy="region_bbox_corners_xy")
    assert body.lever_strategy == "region_bbox_corners_xy"


def test_lever_origin_cases_with_strategy_warns():
    """Combining lever_origin_cases with a non-fixed strategy is ambiguous --
    cases win, so the user gets a UserWarning to flag the dead-code field."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MomentBodyConfig(
            name="pack",
            lever_strategy="region_bbox_corners_xy",
            lever_origin_cases={"a": {0: (0.0, 0.0, 0.0)}},
        )
    assert any("lever_origin_cases is set" in str(w.message) for w in caught)


def test_lever_origin_cases_with_fixed_strategy_silent():
    """The validator only warns on conflicting strategies, not on the
    canonical (fixed + cases) combination."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MomentBodyConfig(
            name="pack",
            lever_origin_cases={"a": {0: (0.0, 0.0, 0.0)}},
        )
    assert not any("lever_origin_cases is set" in str(w.message) for w in caught)


def test_transform_Cm(geom_data, body_data, body_geom, geometry_df):
    geometry_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=TransformationConfig(),
        body_cfg=_moment_body(lever_origin=(0.0, 0.0, 10.0)),
        geometry_df=geometry_df,
    )
    cm_data = transform_Cm(
        raw_cp=body_data, geometry_df=geometry_df, geometry=body_geom, nominal_volume=10
    )

    assert cm_data.notna().all().all()
    assert all(f"Cm{d}" in cm_data.columns for d in ["x", "y", "z"])
