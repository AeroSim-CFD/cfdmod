"""Execute the high-rise stage notebooks headless, in order, on fixtures.

Run: uv run python notebooks/examples/high_rise/_validate_notebooks.py
Runs 01 -> ... -> 06 sharing one temp OUTPUT_BASE, asserts each executes without
error and that the expected debug/deliverable/artifact files are produced. Does
NOT write outputs back into the committed notebooks.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import nbformat
from nbclient import NotebookClient

HERE = pathlib.Path(__file__).resolve().parent


def run_notebook(path: pathlib.Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
    )
    client.execute()
    print(f"  [ok ] executed {path.name}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        os.environ["CFDMOD_HR_OUTPUT_BASE"] = str(base)
        os.environ["CFDMOD_HR_VERSION"] = "smoke"
        print(f"OUTPUT_BASE = {base}")
        for name in (
            "01_inflow.ipynb",
            "02_cp.ipynb",
            "03_cf.ipynb",
            "04_dynamic.ipynb",
            "05_facade.ipynb",
            "06_structure.ipynb",
        ):
            run_notebook(HERE / name)

        expect = [
            base / "deliverables" / "smoke" / "inflow" / "reference_velocity.json",
            base / "artifacts" / "smoke" / "cp.time_series.h5",
            base / "deliverables" / "smoke" / "cp" / "cp_summary.csv",
            base / "deliverables" / "smoke" / "cf" / "per_floor_loads.csv",
            base / "debug" / "smoke" / "cf" / "per_floor_coefficients.png",
            base / "deliverables" / "smoke" / "dynamic" / "dynamic_response.csv",
            base / "debug" / "smoke" / "dynamic" / "peak_acceleration.png",
            base / "deliverables" / "smoke" / "facade" / "cp_mean_iso.png",
            base / "debug" / "smoke" / "facade" / "facade_n_+z.png",
            base / "deliverables" / "smoke" / "structure" / "geometry_iso.png",
            base / "debug" / "smoke" / "structure" / "floor_partition.png",
        ]
        missing = [p for p in expect if not (p.exists() and p.stat().st_size > 0)]
        debug_imgs = list((base / "debug" / "smoke").rglob("*.png"))
        print(f"  debug images produced: {len(debug_imgs)}")
        if missing:
            raise SystemExit(
                "missing expected outputs:\n  " + "\n  ".join(str(p) for p in missing)
            )
    print("\nAll high-rise notebooks executed and produced expected outputs.")


if __name__ == "__main__":
    main()
