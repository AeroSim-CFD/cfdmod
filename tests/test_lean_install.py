"""Guard that a base install -- no optional extras -- can still import and run.

`aerosim-cfdmod` declares trimesh, vtk/pyvista, pytables and fast-simplification
as extras precisely so the library installs lean. That promise is only real if
nothing on a base install's import path reaches one of them at module scope.

It broke once: the `cfdmod` console script imported the altimetry sub-app
eagerly, which imported trimesh, so `cfdmod run` -- which has nothing to do with
altimetry -- died on any install that had not opted into `[geometry]`. These
tests reproduce that condition on purpose (the extras are installed in CI, so
they are masked out here rather than merely absent).
"""

from __future__ import annotations

import contextlib
import importlib
import sys

import pytest

pytestmark = pytest.mark.unit

# Every optional extra declared in pyproject.toml.
OPTIONAL_EXTRAS = (
    "trimesh",
    "vtk",
    "vtkmodules",
    "pyvista",
    "tables",
    "fast_simplification",
)


class _BlockExtras:
    """meta_path finder that makes the named top-level packages look absent."""

    def __init__(self, blocked: frozenset[str]) -> None:
        self.blocked = blocked

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise ImportError(f"{fullname} is masked by test_lean_install")
        return None


@contextlib.contextmanager
def extras_absent():
    """Run the body as if none of the optional extras were installed.

    Purges cfdmod and the extras from ``sys.modules`` so imports inside the body
    genuinely re-execute, then restores the interpreter state so the rest of the
    session is unaffected.
    """
    saved = dict(sys.modules)
    blocked = frozenset(OPTIONAL_EXTRAS)
    for name in list(sys.modules):
        root = name.split(".", 1)[0]
        if root == "cfdmod" or root in blocked:
            del sys.modules[name]
    finder = _BlockExtras(blocked)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.clear()
        sys.modules.update(saved)


def test_cli_builds_without_optional_extras():
    """`cfdmod --help` must work on a base install.

    Asserts the whole command tree resolves, not just that the import survived:
    `run` and `status` are the v3 entry points, and `altimetry` must still be
    listed rather than silently disappearing when its extra is missing.
    """
    with extras_absent():
        main = importlib.import_module("cfdmod.__main__")
        commands = {c.name for c in main.app.registered_commands}
        groups = {g.name for g in main.app.registered_groups}

    assert {"run", "status"} <= commands
    assert {"altimetry", "loft", "roughness", "regroup", "dynamics"} <= groups


@pytest.mark.parametrize(
    "module",
    [
        "cfdmod",
        "cfdmod.altimetry",
        "cfdmod.altimetry.cli",
        "cfdmod.altimetry.plots",
        "cfdmod.core",
        "cfdmod.io",
        "cfdmod.recipes",
    ],
)
def test_module_imports_without_optional_extras(module):
    with extras_absent():
        importlib.import_module(module)


def test_altimetry_names_the_missing_extra():
    """A base install that actually runs altimetry gets an actionable error."""
    with extras_absent():
        cli = importlib.import_module("cfdmod.altimetry.cli")
        with pytest.raises(ImportError, match=r"aerosim-cfdmod\[geometry\]"):
            cli._load_trimesh()


@pytest.mark.parametrize(
    ("do_import", "extra"),
    [
        (lambda: importlib.import_module("cfdmod.snapshot"), "snapshot"),
        (lambda: importlib.import_module("cfdmod.io").read_vtm, "vtk"),
    ],
    ids=["snapshot", "vtk"],
)
def test_extras_only_features_name_their_extra(do_import, extra):
    """Reaching an extras-only feature must say which extra to install.

    These two are *allowed* to need an extra -- unlike the CLI above, they are
    the feature the extra exists for. What is not allowed is a bare
    ``No module named 'pyvista'`` that leaves the caller guessing.
    """
    with extras_absent():
        with pytest.raises(ImportError, match=rf"aerosim-cfdmod\[{extra}\]"):
            do_import()
