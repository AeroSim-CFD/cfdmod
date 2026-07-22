"""cfdmod CI pipeline, defined as code with Dagger.

The exact same pipeline runs on a laptop and on any remote runner - there is no
platform-specific YAML. Everything executes inside containers, so a green run
locally means a green run everywhere.

Quick start (see ``.dagger/README.md`` for details)::

    dagger call --source=. check          # all checks, concurrently
    dagger call --source=. lint stdout    # a single check

The checks mirror the tooling cfdmod actually uses: ruff lint + format check,
the pytest suite (unit + integration; the opt-in ``perf`` benchmarks are
excluded, matching the project's default ``-m 'not perf'``), and the Sphinx
docs build. Everything runs inside a single content-addressed base image, so
each check reuses the same ``uv sync``.
"""

import asyncio
from typing import Annotated

import dagger
from dagger import Doc, dag, function, object_type

PYTHON_IMAGE = "python:3.12-slim-bookworm"

# System packages for the base image:
#   git/curl/ca-certificates - uv sync + source metadata
#   pandoc                    - nbsphinx converts the docs notebooks
#   libgl1/libglx-mesa0/...   - vtk + pyvista need OpenGL to import/render
#   xvfb                      - off-screen virtual display for the pyvista
#                               mesh-field render tests (headless host)
BASE_APT = [
    "git",
    "curl",
    "ca-certificates",
    "make",
    "pandoc",
    "libgl1",
    "libglx-mesa0",
    "libxrender1",
    "libxext6",
    "libglib2.0-0",
    "xvfb",
    "xauth",
]

# name -> the coroutine attribute on ``Cfdmod`` that runs it, for ``check``.
CHECKS = ["lint", "test", "docs"]


@object_type
class Cfdmod:
    """CI pipeline for the aerosim-cfdmod library."""

    source: Annotated[dagger.Directory, Doc("The cfdmod repository root")]

    def base(self) -> dagger.Container:
        """Python container with the project and all extras synced via uv.

        Layers are content-addressed, so every check reuses this build. All
        optional extras are installed so the full test suite (vtk, geometry,
        remesh) collects and runs, and the docs extra is present for the
        Sphinx build.
        """
        uv_cache = dag.cache_volume("cfdmod-uv-cache")
        apt = "apt-get update && apt-get install -y --no-install-recommends " + " ".join(BASE_APT)
        return (
            dag.container()
            .from_(PYTHON_IMAGE)
            .with_exec(["sh", "-c", apt])
            .with_exec(["pip", "install", "--no-cache-dir", "uv"])
            .with_env_variable("UV_LINK_MODE", "copy")
            .with_mounted_cache("/root/.cache/uv", uv_cache)
            .with_directory("/src", self.source)
            .with_workdir("/src")
            .with_exec(["uv", "sync", "--all-extras"])
        )

    @function
    async def lint(self) -> str:
        """ruff static analysis and format check (line length 99)."""
        return await (
            self.base()
            .with_exec(["uv", "run", "ruff", "check", "."])
            .with_exec(["uv", "run", "ruff", "format", "--check", "."])
            .stdout()
        )

    @function
    async def test(self) -> str:
        """Run the pytest suite (unit + integration; perf excluded by default).

        Wrapped in ``xvfb-run`` so the pyvista off-screen renders have a
        virtual display on the headless CI host.
        """
        return await self.base().with_exec(["xvfb-run", "-a", "uv", "run", "pytest"]).stdout()

    @function
    async def docs(self) -> str:
        """Build the Sphinx HTML docs (nbsphinx renders the notebook pages)."""
        return await self.base().with_exec(["sh", "-c", "cd docs && uv run make html"]).stdout()

    @function
    async def check(self) -> str:
        """Run every check concurrently and report a pass/fail summary.

        Raises if any check fails, so this is the single command a runner calls.
        """
        results = await asyncio.gather(
            *(getattr(self, name)() for name in CHECKS),
            return_exceptions=True,
        )
        lines: list[str] = []
        failed = False
        for name, result in zip(CHECKS, results):
            if isinstance(result, BaseException):
                failed = True
                lines.append(f"FAIL  {name}\n{result}")
            else:
                lines.append(f"ok    {name}")
        report = "\n".join(lines)
        if failed:
            raise RuntimeError("CI checks failed:\n" + report)
        return "All checks passed:\n" + report
