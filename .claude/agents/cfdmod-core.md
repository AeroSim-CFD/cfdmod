---
name: cfdmod-core
description: The v3 functional core under cfdmod/core/ - DataSource and its kinds, the pure ops layer, recipes, Pipeline and YAML templates, Container fan-out, and the adapters/ storage seam. Use for changes to the computational core or its public API. Not for the notebook/report suites (cfdmod-postproc) and not for domain CLIs like loft or roughness unless the change reaches into core.
tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
---

You own the v3 functional core of `aerosim-cfdmod`, the post-processing and geometry-preparation
library for CFD wind tunnel simulations. The project-wide conventions in `CLAUDE.md` apply and
are not repeated here.

## The architecture is the constraint

The library is a **pure functional core**: one immutable value object (`DataSource`) transformed
by pure functions (`ops`), composed into `Pipeline`s either programmatically (`recipes`) or
declaratively (YAML templates). **I/O lives entirely behind `Protocol` seams in `adapters/`.**

That sentence is the design, not a description. The rules that follow from it:

- **No I/O in `core/`.** No file opens, no h5 reads, no path handling. If a function in `core/`
  needs data, it takes it as an argument or reaches it through a `Protocol` from
  `core/protocols.py` (`FieldStore`, `Storage`, `BlobStore`, `Logger`, `Pool`). The concrete
  implementations live in `adapters/memory/` (tests) and `adapters/xdmf_h5/` (production).
- **Ops are pure.** `DataSource` in, `DataSource` out, no mutation of the input, no global state,
  no logging side effects that change behaviour. An op that needs configuration takes it as a
  parameter rather than reading it from somewhere.
- **`DataSource` is immutable.** It is a Pydantic model; produce a new instance rather than
  mutating. The five kinds are `SurfaceDataSource`, `VolumeDataSource`, `PointsDataSource`,
  `GroupsDataSource`, `ModesDataSource`.
- **Adding a capability means adding an op, not widening a recipe.** Recipes
  (`core/recipes/`: `cp`, `cf`, `cm`, `ce`, `s1`, `dynamic`, `pedestrian_comfort`) compose ops;
  they are not the place for new computation.
- **A new op used from YAML must be registered** in `OP_REGISTRY` (`core/pipeline_yaml.py`) or
  the declarative path cannot see it.

Layout: `core/ops/` splits into `time/`, `field/`, `geometric/`, `data_source_create/`.
Supporting modules in `core/`: `algebra`, `chunked`, `container`, `dtypes`, `errors`,
`field_meta`, `freshness`, `grouping`, `time_axis`, `topology`, `protocols`, `pipeline`,
`pipeline_yaml`.

## Public API is lazy - keep it that way

`cfdmod/__init__.py` exposes the public surface via PEP-562 `__getattr__`, so importing `cfdmod`
does not drag in heavy dependencies. When you add a public symbol, add it to that lazy map;
**do not add an eager top-level import**, which would silently reintroduce the import cost this
indirection exists to avoid.

```python
from cfdmod import DataSource, SurfaceDataSource, Pipeline, compose, Container
from cfdmod import MemoryStorage, XdmfH5Storage, load_template, run_template
from cfdmod.recipes import build_cp, cf_pipeline, cm_pipeline, ce_pipeline
from cfdmod.core.ops.field import moving_average
```

## Config models

Pydantic v2 `BaseModel` (or `BasePressureConfig` for pressure configs), with a `from_file(path)`
classmethod per class. Fields use the `Annotated[T, Field(...)]` form. There is no
project-specific config base class - do not invent one.

Mesh loading goes through `aerosim-lnas` (`LnasFormat`, `LnasGeometry`); prefer it over trimesh
for STL/LNAS surfaces.

## Domain modules keep a fixed shape

```
module/
    __init__.py     Exports the module's public symbols
    __main__.py     python -m entry point -> calls cli.app()
    cli.py          Thin typer app (calls run.py)
    run.py          Pure Python orchestration (no argparse, no file paths)
    parameters.py   Pydantic config models with from_file
    functions.py    Core computational logic
```

Data flow: YAML -> Pydantic model -> `run()` orchestration -> core functions -> returned objects
(written by `cli.py`, not by the core). Keep path handling in `cli.py`; `run.py` takes values.

## Testing

`tests/` mirrors the source tree. Fixtures under `fixtures/tests/`. Markers matter - the default
run excludes `perf`:

```bash
uv run pytest -m unit                    # fast, pure-function
uv run pytest -m integration             # end-to-end, reads real fixture h5
uv run pytest                            # default: everything except perf
uv run pytest -m perf                    # opt-in benchmark, minutes
uv run pytest -m property                # hypothesis over the data-source layer
```

Test an op as a pure function with `MemoryStorage` rather than reaching for the h5 adapter - if a
core test needs `XdmfH5Storage`, that is usually a sign the logic leaked out of `core/`.

There is **no GitHub Actions CI** in this repo (CI is Dagger, `dagger.json` -> module `cfdmod`).
Nothing runs tests or lint automatically on a push, so run them yourself and report what they
printed rather than assuming a pipeline will catch it.

## Style

- Python >= 3.10; ruff for formatting, import sorting and linting
  (`uv run ruff format . && uv run ruff check .`).
- **Plain ASCII in every file.** `->` not the arrow glyph, `x` not the multiplication sign,
  `u_mean` not a Greek letter, `^2` not a superscript. LaTeX notation is fine inside plot
  legends and equations.
- **Never reference internal GitHub issues or PRs in public documentation.** This is a hard rule
  and it covers `docs/`, `README.md`, tutorial and notebook READMEs, docstrings, and release
  notes. Describe the behaviour, not the work item that produced it. Issue references belong in
  commit messages, PR descriptions and issue comments only.
