"""Dependency-light import surface (issue #147).

A service consumer that only needs the v3 template schema and op catalog
must be able to import them without dragging in the heavy scientific
stack (h5py / matplotlib / pandas / pyarrow / vtk / trimesh). Each check
runs in a fresh interpreter so an unrelated test importing pandas cannot
mask a regression.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit

_HEAVY = ("h5py", "matplotlib", "pandas", "pyarrow", "vtk", "trimesh")


def _heavy_after(import_line: str) -> list[str]:
    code = textwrap.dedent(
        f"""
        import sys
        {import_line}
        heavy = [m for m in {_HEAVY!r} if m in sys.modules]
        print(",".join(heavy))
        """
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return [m for m in out.stdout.strip().split(",") if m]


def test_import_cfdmod_is_dependency_light():
    assert _heavy_after("import cfdmod") == []


def test_schema_and_catalog_import_light():
    assert _heavy_after("import cfdmod.core.pipeline_yaml") == []


def test_op_catalog_usable_without_heavy_stack():
    """list_ops() must work in a light interpreter (no template run needed)."""
    code = (
        "import cfdmod; "
        "infos = cfdmod.list_ops(); "
        "assert len(infos) >= 20; "
        "import sys; "
        f"assert not [m for m in {_HEAVY!r} if m in sys.modules], "
        "[m for m in sys.modules if m in " + repr(_HEAVY) + "]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_heavy_symbol_still_loads_lazily():
    """Accessing a heavy symbol pulls its dep -- lazy loading still works."""
    heavy = _heavy_after("import cfdmod; _ = cfdmod.XdmfH5Storage")
    assert "h5py" in heavy
