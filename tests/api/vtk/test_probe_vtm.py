import pathlib

import pytest

from cfdmod.api.vtk.probe_vtm import create_line, get_array_from_filter, probe_over_line, read_vtm


@pytest.fixture()
def p1():
    yield [10, 20, 1]


@pytest.fixture()
def p2():
    yield [10, 20, 40]


@pytest.fixture()
def vtm_path():
    yield pathlib.Path("./fixtures/tests/s1/vtm/example.vtm")


def test_create_line(p1, p2):
    line = create_line(p1, p2, numPoints=10)
    assert 10 == line.GetResolution()


def test_probe_vtm(p1, p2, vtm_path):
    line = create_line(p1, p2, numPoints=10)
    reader = read_vtm(vtm_path)
    probe = probe_over_line(line, reader)
    data = get_array_from_filter(probe, array_lbl="ux")

    assert len(data) == 11
