# CLAUDE.md - cfdmod project guide

## What this project is

`aerosim-cfdmod` is a Python library for post-processing and geometry preparation of CFD wind tunnel simulations. It covers:

- Terrain loft surface generation (circular projection to a fixed radius)
- Roughness element placement (linear grid and radial ring patterns)
- Pressure coefficient analysis (Cp, Cf, Cm, Ce)
- Inflow profile analysis
- Climate/wind statistics (Weibull, Gumbel, Lawson comfort)
- Altimetry section analysis
- ParaView/VTK snapshot automation

---

## Conventions

- **No Unicode or special characters** in any file (notebooks, scripts, YAML, markdown).
  Use plain ASCII only: `->` not arrows, `x` not multiplication sign, `u_mean` not Greek letters, `^2` not superscripts.
  For equations in legends, feel free to use Latex notation for equations and special symbols in Latex representation.
- **No inline comments in python3 -c "..." terminal commands** (shell escaping issues).
- Python >= 3.10; formatting with black + isort; linting with ruff.
- Configuration via YAML + Pydantic v2 `BaseModel` with `from_file(path)` classmethods.
- External mesh format: `aerosim-lnas` (`LnasFormat`, `LnasGeometry`). Prefer lnas over trimesh for loading STL/LNAS surfaces.
- Testing: pytest, fixture files under `fixtures/tests/`, tests mirror source structure under `tests/`.

---

## Current architecture (v2.0)

The library is API-first. All public symbols are importable from the top-level package:

```python
from cfdmod import LoftParams, generate_loft_surface
from cfdmod import RadialParams, radial_pattern
from cfdmod import mesh_summary, show_config, load_lnas
```

### Directory layout

```
cfdmod/
    cfdmod/
        __init__.py         Public API (~69 exported symbols)
        __main__.py         Global typer CLI (python -m cfdmod loft|roughness)
        notebook_utils.py   Notebook helpers: mesh_summary, show_config, load_lnas
        io/
            geometry/       STL I/O (export_stl, read_stl)
            vtk/            VTK/ParaView probe and write utilities
        config/
            hashable.py     HashableConfig base (Pydantic + SHA256, to_dict/to_yaml)
        analysis/
            inflow/         InflowData class and functions
        loft/               Terrain loft surface generation
        roughness/          Roughness element generation (linear + radial)
        pressure/           Cp/Cf/Cm/Ce post-processing
        altimetry/          Surface section analysis
        climate/            Wind rose, Gumbel, Weibull, Lawson
        analytical/         Analytical wind/aero models
        s1/                 S1 profile analysis
        snapshot/           ParaView snapshot automation
        logger.py
        utils.py            read_yaml, HDF5 helpers, dataframe utilities
    tests/                  Mirror of cfdmod/ structure
    fixtures/               YAML configs + STL/LNAS mesh files for tests
    notebooks/              Jupyter notebooks for analysis and generation
    docs/                   Sphinx documentation
```

### Layout inside each domain module

```
module/
    __init__.py     Exports all public symbols for the module
    __main__.py     python -m entry point -> calls cli.app()
    cli.py          Thin typer app (uses run.py functions)
    run.py          Pure Python orchestration (no argparse, no file paths)
    parameters.py   Pydantic config models with from_file/from_dict/to_dict/to_yaml
    functions.py    Core computational logic
    [helpers].py    Supporting modules
```

Data flow:
```
YAML file -> Pydantic model (parameters.py)
          -> run() orchestration (run.py)
          -> core functions (functions.py)
          -> output objects (returned to caller or written by cli.py)
```

### CLI entry points

Each module is runnable as a module and as a subcommand of the global CLI:

```bash
python -m cfdmod.loft --config ... --surface ... --output ...
python -m cfdmod.roughness --config ... --output ... --mode radial

python -m cfdmod loft --config ... --surface ... --output ...
python -m cfdmod roughness --config ... --output ...
```

After install, the `cfdmod` shell command is also available (via `[project.scripts]`).

### Config models

All config classes inherit from `HashableConfig` and support:

```python
params = LoftParams.from_file("config.yaml")   # from YAML
params = LoftParams.from_dict({...})            # from dict
params = LoftParams(field=value, ...)           # from kwargs

params.to_dict()   # -> dict
params.to_yaml()   # -> YAML string
```

### Backward-compatibility shims

`cfdmod/api/` and `cfdmod/use_cases/` are retained as thin re-export shims so that
existing scripts using old import paths continue to work. Do not add new code there.

---

## Notebook utilities

`cfdmod.notebook_utils` provides lightweight helpers for exploratory work:

- `mesh_summary(path)` - print triangle/vertex counts and bounding box for .lnas or .stl
- `show_config(config)` - pretty-print any HashableConfig as a dict
- `load_lnas(path)` - load an .lnas file and return the LnasFormat object

All three are also exported from the top-level `cfdmod` package.
[
---

