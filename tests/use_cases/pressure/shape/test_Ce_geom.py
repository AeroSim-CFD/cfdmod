import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig, ZoningConfig
from cfdmod.use_cases.pressure.shape.Ce_data import get_surface_dict
from cfdmod.use_cases.pressure.shape.Ce_geom import generate_regions_mesh, get_geometry_data
from cfdmod.use_cases.pressure.statistics import BasicStatisticModel
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


@pytest.fixture()
def zoning():
    zoning = ZoningModel(x_intervals=[0, 5, 10], y_intervals=[0, 10], z_intervals=[0, 10])
    zoning.offset_limits(0.1)
    yield zoning


@pytest.fixture()
def mesh():
    geom = LnasGeometry(
        vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
        triangles=np.array([[0, 1, 2], [2, 1, 3]]),
    )
    yield LnasFormat(
        version="", geometry=geom, surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])}
    )


@pytest.fixture()
def cfg(zoning):
    yield CeConfig(
        statistics=[
            BasicStatisticModel(stats="mean"),
            BasicStatisticModel(stats="rms"),
            BasicStatisticModel(stats="skewness"),
            BasicStatisticModel(stats="kurtosis"),
        ],
        zoning=ZoningConfig(global_zoning=zoning),
        sets={},
        transformation=TransformationConfig(),
    )


def test_get_geometry_data(cfg, mesh):
    sfc_dict = get_surface_dict(cfg=cfg, mesh=mesh)
    geometry_dict = get_geometry_data(surface_dict=sfc_dict, cfg=cfg, mesh=mesh)

    assert len(geometry_dict) == len(sfc_dict) == len(mesh.surfaces.keys())


def test_get_surface_dict(cfg, mesh):
    sfc_dict = get_surface_dict(cfg=cfg, mesh=mesh)

    assert [k in sfc_dict.keys() for k in mesh.surfaces.keys()]
    assert [k in mesh.surfaces.values() for k in sfc_dict.values()]


def test_generate_regions_mesh(cfg, mesh):
    sfc_dict = get_surface_dict(cfg=cfg, mesh=mesh)
    geometry_dict = get_geometry_data(surface_dict=sfc_dict, cfg=cfg, mesh=mesh)
    for geometry_data in geometry_dict.values():
        regions_mesh, regions_mesh_triangles_indexing = generate_regions_mesh(
            geom_data=geometry_data, cfg=cfg
        )

        assert len(regions_mesh_triangles_indexing) == len(regions_mesh.triangles)
