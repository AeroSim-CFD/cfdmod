"""``cfdmod run`` entry point: load a YAML template and execute it.

The v3 CLI surface is one command: ``cfdmod run <template.yaml>``. Every
recipe -- Cp, Cf, Cm, Ce, S1, dynamic, pedestrian comfort, or a custom
chain -- is expressed as a pipeline template. The runner doesn't need
to know which recipe it is; it walks the steps.

Storage adapter selection follows the YAML's input/output ``kind``:

- ``surface`` / ``points`` / ``volume``: an
  :class:`XdmfH5Storage` rooted at the template's directory. Files
  resolve to ``<root>/<path>.h5`` (+ optional ``.xdmf`` sidecar).
- ``memory``: an in-process :class:`MemoryStorage` -- useful for tests
  and for templates that chain other templates.

For now there is one default storage instance; future extensions may
let the YAML pick per-input adapters explicitly.
"""

from __future__ import annotations

__all__ = ["run_yaml"]

import pathlib

from cfdmod.adapters.xdmf_h5 import XdmfH5Storage
from cfdmod.core.data_source import DataSource
from cfdmod.core.pipeline_yaml import load_template, run_template


def run_yaml(
    template_path: pathlib.Path | str,
    *,
    output_root: pathlib.Path | str | None = None,
) -> dict[str, DataSource]:
    """Load and run a v3 pipeline template.

    Args:
        template_path: YAML file path. Relative paths inside the YAML
            resolve against the file's parent directory.
        output_root: Optional storage root override. By default the
            storage is rooted at the filesystem root (``/``) so the
            YAML's absolute / template-relative paths resolve directly;
            pass a directory here to redirect outputs.

    Returns:
        The dict of every named binding the runner produced (inputs +
        step outputs). Outputs declared in the YAML are written through
        the storage as a side effect.
    """
    template_path = pathlib.Path(template_path)
    template = load_template(template_path)

    root = pathlib.Path(output_root) if output_root is not None else pathlib.Path("/")
    storage = XdmfH5Storage(root)

    return run_template(template, storage=storage)
