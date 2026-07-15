"""Tests for cfdmod.snapshot config helpers (SnapshotConfig.retarget)."""

from __future__ import annotations

import pytest

# The snapshot package imports pyvista + IPython at load; skip where absent.
snapshot = pytest.importorskip("cfdmod.snapshot")

pytestmark = pytest.mark.unit

SnapshotConfig = snapshot.SnapshotConfig
ProjectionConfig = snapshot.ProjectionConfig
LegendConfig = snapshot.LegendConfig
CameraConfig = snapshot.CameraConfig


def _base() -> SnapshotConfig:
    return SnapshotConfig(
        projections={
            "top": ProjectionConfig(file_path="placeholder.vtp", scalar=""),
            "front": ProjectionConfig(file_path="placeholder.vtp", scalar=""),
        },
        legend_config=LegendConfig(label="base", range=(0.0, 1.0), n_divs=5),
        camera=CameraConfig(),
    )


def test_retarget_points_all_projections_and_sets_legend():
    base = _base()
    cfg = base.retarget(
        "case_cp.vtp", "cp/base_cp/mean", label="mean Cp", value_range=(-1.2, 0.8), n_divs=12
    )

    for proj in cfg.projections.values():
        assert str(proj.file_path) == "case_cp.vtp"
        assert proj.scalar == "cp/base_cp/mean"
    assert cfg.legend_config.label == "mean Cp"
    assert cfg.legend_config.range == (-1.2, 0.8)
    assert cfg.legend_config.n_divs == 12


def test_retarget_returns_copy_leaving_base_untouched():
    base = _base()
    base.retarget("other.vtp", "s", label="x", value_range=(0.0, 2.0), n_divs=3)

    # base is unchanged -> it can be retargeted again for the next direction/stat
    for proj in base.projections.values():
        assert str(proj.file_path) == "placeholder.vtp"
        assert proj.scalar == ""
    assert base.legend_config.label == "base"
    assert base.legend_config.range == (0.0, 1.0)
    assert base.legend_config.n_divs == 5


def test_retarget_keeps_legend_fields_when_not_overridden():
    base = _base()
    cfg = base.retarget("case_cp.vtp", "s")
    assert cfg.legend_config.label == "base"
    assert cfg.legend_config.range == (0.0, 1.0)
    assert cfg.legend_config.n_divs == 5
    # projections still retargeted
    assert all(p.scalar == "s" for p in cfg.projections.values())


building_facade_config = snapshot.building_facade_config


def test_building_facade_config_four_walls_and_roof():
    cfg = building_facade_config(
        [-10, -5, 0], [10, 5, 100], legend_label="Cp", value_range=(-1.5, 1.0)
    )
    assert set(cfg.projections) == {"N", "E", "S", "W", "roof"}
    # compass labels present as overlays (roof intentionally unlabeled)
    assert {t.text for t in cfg.text_overlay} == {"N", "E", "S", "W"}
    assert cfg.legend_config.range == (-1.5, 1.0)


def test_building_facade_config_band_drops_roof_and_clips():
    cfg = building_facade_config([-10, -5, 0], [10, 5, 100], z_band=(20.0, 40.0))
    assert set(cfg.projections) == {"N", "E", "S", "W"}  # no roof in a band
    for p in cfg.projections.values():
        assert p.clip_box is not None
        assert p.clip_box.scale[2] == 20.0  # band height


def test_building_facade_config_retarget_points_all_walls():
    cfg = building_facade_config([-10, -5, 0], [10, 5, 100]).retarget("case.vtp", "cp_mean")
    for p in cfg.projections.values():
        assert str(p.file_path) == "case.vtp"
        assert p.scalar == "cp_mean"
