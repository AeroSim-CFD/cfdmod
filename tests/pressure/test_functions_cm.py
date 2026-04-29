"""Tests for moment coefficient (Cm) functions."""

import numpy as np
import pandas as pd
import pytest
from lnas import LnasGeometry

from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.pressure.functions import add_lever_arm_to_geometry_df, transform_Cm
from cfdmod.pressure.geometry import GeometryData, tabulate_geometry_data
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
    yield GeometryData(
        mesh=body_geom, zoning_to_use=ZoningModel(), triangles_idxs=np.array([0, 1])
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
