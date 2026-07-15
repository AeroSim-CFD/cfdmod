"""Opt-in memory + performance regression test for the per-floor load path.

Runs with ``pytest -m perf`` (excluded from the default suite). It writes a
synthetic body (the galpao mesh with a long random pressure series) to disk,
then measures the **peak RSS** of ``building.per_floor_loads`` in isolated
subprocesses -- once time-chunked, once whole-series -- and asserts that:

- the chunked run stays under a memory budget, and
- the whole-series run uses materially more memory than the chunked run.

The subprocess isolation matters: ``ru_maxrss`` is a high-water mark, so the
fixture's own allocations (building + writing the body) would otherwise mask the
compute footprint we want to bound.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.integration]

REPO = pathlib.Path(__file__).resolve().parents[2]
MESH = str(REPO / "fixtures" / "tests" / "pressure" / "galpao" / "galpao.normalized.lnas")

# Long enough that the whole-series arrays dominate memory, short enough to keep
# the opt-in run to a few seconds. galpao ~ 2915 triangles -> ~0.28 GiB / array.
N_TIMESTEPS = 12000
CHUNK = 1000
# Measured on this fixture: chunked ~700 MiB vs whole-series ~2900 MiB (~4x).
# Budget leaves CI headroom while still catching a regression back to full
# materialization (which jumps to ~2.9 GiB).
CHUNKED_BUDGET_MB = 1000.0
MIN_WHOLE_TO_CHUNKED_RATIO = 1.8

_RUNNER = """
import pathlib, resource, sys
import numpy as np
from cfdmod.adapters.xdmf_h5 import XdmfH5Storage
from cfdmod import building

data_dir, mesh, chunk_arg = sys.argv[1], sys.argv[2], sys.argv[3]
chunk = None if chunk_arg == "none" else int(chunk_arg)
st = XdmfH5Storage(pathlib.Path(data_dir))
body = st.read_data_source(pathlib.Path("bodies.synthetic"))
p_ref = st.read_data_source(pathlib.Path("points.ref"))
case = building.example_building_case(mesh, n_floors=8)
cf, cm = building.per_floor_loads(body, p_ref, mesh, case, chunk_size=chunk)
# touch results so nothing is optimised away
_ = float(np.asarray(cf.fields.read("cf_x")).sum() + np.asarray(cm.fields.read("cm_z")).sum())
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # KiB on Linux
"""


@pytest.fixture(scope="module")
def big_store(tmp_path_factory):
    from lnas import LnasFormat

    from cfdmod.adapters.memory import MemoryFieldStore
    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage
    from cfdmod.core.data_source import PointsDataSource, SurfaceDataSource
    from cfdmod.core.field_meta import FieldMeta
    from cfdmod.core.time_axis import TimeAxis
    from cfdmod.core.topology import ElementMeta, Topology

    geom = LnasFormat.from_file(pathlib.Path(MESH)).geometry
    verts = np.asarray(geom.vertices, dtype=np.float64)
    tris = np.asarray(geom.triangles, dtype=np.int32)
    n_tri = tris.shape[0]

    rng = np.random.default_rng(0)
    time = TimeAxis(initial_time=0.0, timestep_size=0.01, n_timesteps=N_TIMESTEPS)
    body = SurfaceDataSource(
        time=time,
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore(
            {"pressure": rng.standard_normal((n_tri, N_TIMESTEPS)).astype(np.float64)}
        ),
        field_meta={"pressure": FieldMeta(name="pressure")},
    )
    p_ref = PointsDataSource(
        time=time,
        topology=Topology.points(np.zeros((1, 3), dtype=np.float64)),
        elements=ElementMeta(),
        fields=MemoryFieldStore(
            {"pressure": rng.standard_normal((1, N_TIMESTEPS)).astype(np.float64)}
        ),
        field_meta={"pressure": FieldMeta(name="pressure")},
    )

    data_dir = tmp_path_factory.mktemp("perf_store")
    storage = XdmfH5Storage(data_dir)
    storage.write_data_source(pathlib.Path("bodies.synthetic"), body)
    storage.write_data_source(pathlib.Path("points.ref"), p_ref)

    runner = data_dir / "_runner.py"
    runner.write_text(_RUNNER)
    # free the big in-memory arrays before the subprocesses run
    del body, p_ref
    return data_dir, runner


def _peak_mb(data_dir: pathlib.Path, runner: pathlib.Path, chunk: str) -> float:
    out = subprocess.run(
        [sys.executable, str(runner), str(data_dir), MESH, chunk],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip().splitlines()[-1]) / 1024.0  # KiB -> MiB


def test_per_floor_loads_memory_bounded(big_store):
    data_dir, runner = big_store
    chunked_mb = _peak_mb(data_dir, runner, str(CHUNK))
    whole_mb = _peak_mb(data_dir, runner, "none")

    assert chunked_mb < CHUNKED_BUDGET_MB, (
        f"chunked peak RSS {chunked_mb:.0f} MiB exceeded budget {CHUNKED_BUDGET_MB:.0f} MiB"
    )
    assert whole_mb > chunked_mb * MIN_WHOLE_TO_CHUNKED_RATIO, (
        f"expected whole-series ({whole_mb:.0f} MiB) to dwarf chunked "
        f"({chunked_mb:.0f} MiB); chunking not reducing memory as expected"
    )
