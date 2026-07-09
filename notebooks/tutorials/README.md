# cfdmod tutorials

Four short notebooks introducing the v3 paradigm. Run in order; each builds on concepts from the previous.

| Notebook | What you learn |
|---|---|
| [`01_data_sources.ipynb`](01_data_sources.ipynb) | Build a `DataSource`, inspect time / topology / fields, persist via Memory and XDMF+H5 storage. |
| [`02_recipes.ipynb`](02_recipes.ipynb) | `build_cp`, `cf_pipeline`, compose your own ops with `compose(...)`. |
| [`03_pipelines.ipynb`](03_pipelines.ipynb) | Load a pipeline from a YAML template, register custom ops. |
| [`04_containers.ipynb`](04_containers.ipynb) | Multi-case containers: `join_by`, `filter_by`, parallel `map_values`. |

Each notebook runs end-to-end on synthetic data with no fixture files. They use only the public API on `cfdmod` (no internal imports). After you finish them, the production notebooks (`envelope`, `global_forces`, `psd_profile_velocity`, `hfpi_*`) demonstrate the same primitives on real cases.

## Reference

- Architecture overview: `docs/source/architecture/data_sources.md`
- Migration guide (legacy v2 -> v3): `docs/source/architecture/v3_migration.md`
- Recipe / op reference: `cfdmod.recipes.__all__`, `cfdmod.ops`
