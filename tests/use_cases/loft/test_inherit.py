import pathlib

import pytest

from cfdmod.use_cases.loft.parameters import LoftCaseConfig


@pytest.fixture
def cfg():
    yield LoftCaseConfig.from_file(pathlib.Path("./fixtures/tests/loft/loft_params.yaml"))


def test_inherit_from_default(cfg):
    default_cfg = cfg.cases["default"]
    inherited_cfg = cfg.cases["inherit_loft"]
    inherited_elem_size_cfg = cfg.cases["inherit_element_size"]

    assert inherited_cfg.mesh_element_size == default_cfg.mesh_element_size
    assert inherited_cfg.upwind_elevation == default_cfg.upwind_elevation
    assert inherited_cfg.loft_radius != default_cfg.loft_radius

    assert inherited_elem_size_cfg.mesh_element_size == default_cfg.mesh_element_size
    assert inherited_elem_size_cfg.upwind_elevation != default_cfg.upwind_elevation
    assert inherited_elem_size_cfg.loft_radius != default_cfg.loft_radius
