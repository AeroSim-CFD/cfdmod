"""DebugWriter -- versioned output roots for exploratory and deliverable artifacts.

The old notebooks stored results inline, so you had to open a notebook to see
its output. The new suite instead sprays images and tables into two roots:

    <base>/debug/<version>/<stage>/...        <- free-to-compare exploratory output
    <base>/deliverables/<version>/<stage>/... <- engineer-facing output

``version`` is a per-run label so multiple runs coexist (compare freely) and
re-runs of the *same* version overwrite in place (minimize, not eliminate,
overwriting). Pass an explicit version (e.g. a batch name or a date tag);
if omitted, a UTC timestamp is used.
"""

from __future__ import annotations

import datetime
import pathlib


class DebugWriter:
    """Resolve and create versioned paths under debug/ and deliverables/."""

    def __init__(
        self,
        base_dir: str | pathlib.Path,
        stage: str,
        version: str | None = None,
    ) -> None:
        self.base_dir = pathlib.Path(base_dir)
        self.stage = stage
        self.version = version or datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d_%H%M%S"
        )

    @property
    def debug_dir(self) -> pathlib.Path:
        d = self.base_dir / "debug" / self.version / self.stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def deliverables_dir(self) -> pathlib.Path:
        d = self.base_dir / "deliverables" / self.version / self.stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def debug_path(self, name: str) -> pathlib.Path:
        """Path for a debug artifact, creating any subdirectories in ``name``."""
        return self._resolve(self.debug_dir, name)

    def deliverable_path(self, name: str) -> pathlib.Path:
        """Path for a deliverable artifact, creating any subdirectories in ``name``."""
        return self._resolve(self.deliverables_dir, name)

    def savefig(self, fig, name: str, *, deliverable: bool = False, **savefig_kwargs) -> pathlib.Path:
        """Save a matplotlib figure to debug (default) or deliverables, return the path."""
        path = self.deliverable_path(name) if deliverable else self.debug_path(name)
        savefig_kwargs.setdefault("bbox_inches", "tight")
        savefig_kwargs.setdefault("dpi", 150)
        fig.savefig(path, **savefig_kwargs)
        return path

    @staticmethod
    def _resolve(root: pathlib.Path, name: str) -> pathlib.Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
