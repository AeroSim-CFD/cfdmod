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
- **No inline comments in python3 -c "..." terminal commands** (shell escaping issues).
- Python >= 3.10; formatting with black + isort; linting with ruff.
- Configuration via YAML + Pydantic v2 `BaseModel` with `from_file(path)` classmethods.
- External mesh format: `aerosim-lnas` (`LnasFormat`, `LnasGeometry`). Prefer lnas over trimesh for loading STL/LNAS surfaces.
- Testing: pytest, fixture files under `fixtures/tests/`, tests mirror source structure under `tests/`.

---

## Current architecture

```
cfdmod/
    cfdmod/
        api/
            geometry/       STL I/O (export_stl, read_stl)
            configs/        HashableConfig base (Pydantic + SHA256)
            vtk/            VTK/ParaView probe and write utilities
        analysis/
            inflow/         InflowData class and functions
        use_cases/
            altimetry/      Surface section analysis
            analytical/     Analytical wind/aero models
            climate/        Wind rose, Gumbel, Weibull, Lawson
            loft/           Terrain loft surface generation
            pressure/       Cp/Cf/Cm/Ce post-processing
            roughness_gen/  Roughness element generation (linear + radial)
            s1/             S1 profile analysis
            snapshot/       ParaView snapshot automation
        logger.py
        utils.py            read_yaml, HDF5 helpers, dataframe utilities
    tests/                  Mirror of cfdmod/ structure
    fixtures/               YAML configs + STL/LNAS mesh files for tests
    notebooks/              Jupyter notebooks for analysis and generation
```

### Recurring patterns inside each use_case

Every CLI use case follows the same layout:

```
use_case/
    __main__.py     python -m entry point
    main.py         main(*args) with argparse; orchestrates the pipeline
    parameters.py   Pydantic models; from_file(path) reads YAML
    functions.py    Core computational logic
    [helpers].py    Supporting modules
    __init__.py     Exports the public symbols
```

Data flow:
```
YAML file -> Pydantic model (parameters.py)
          -> main() orchestration (main.py)
          -> core functions (functions.py)
          -> output files (STL, CSV, VTK, ...)
```

---

## Current limitations and dissatisfaction

The structure above was designed around CLI execution and YAML-driven batch jobs.
This makes it awkward to use as a Python library from notebooks or other programs:

1. **No top-level public API.** `cfdmod/__init__.py` is empty. To use loft generation
   you must know to import `from cfdmod.use_cases.loft.functions import generate_loft_surface`.

2. **main() functions mix CLI parsing with orchestration.** There is no clean
   `run(config, data)` function callable from Python without going through argparse.

3. **Configuration objects are YAML-first.** `LoftParams`, `RadialParams`, etc. can be
   constructed from Python dicts but the pattern is not documented or enforced.
   Notebooks end up duplicating parameter logic instead of reusing config models.

4. **The api/ folder is not the API.** It contains low-level I/O helpers (STL, VTK)
   rather than being the programmatic interface to domain functionality.

5. **Deep import paths for common operations.** Users write long, fragile imports
   instead of `from cfdmod import generate_loft_surface`.

6. **Functions and parameters are split across files with no enforced contract.**
   There is no single place that describes what inputs and outputs a use_case exposes.

---

## Proposed refactoring towards an API-first structure

The goal is to make cfdmod usable as:
```python
from cfdmod.loft import LoftParams, generate_loft_surface
from cfdmod.roughness import RadialParams, radial_pattern
```

without needing to know internal file layout. The CLI entry points should be thin
wrappers around this library API, not the other way around.

### Proposed changes (for review)

#### 1. Flatten use_cases into top-level domain modules

Rename `use_cases/` to domain modules directly under `cfdmod/`:

```
cfdmod/
    loft/
    roughness/
    pressure/
    altimetry/
    climate/
    s1/
    snapshot/
    analysis/
```

Each module keeps its internal structure but its `__init__.py` exports everything
a user needs: params classes, core functions, and any plot helpers.

#### 2. Separate library from CLI in each module

Split `main.py` into two parts:

- `run.py` (or a `run()` function) - pure Python orchestration, takes config objects and
  returns results. No argparse, no file I/O assumptions.
- `cli.py` - thin argparse wrapper that calls `run()`.

Example (loft):
```python
# loft/run.py
def generate_loft(params: LoftParams, geom: LnasGeometry) -> LnasGeometry:
    ...

# loft/cli.py
def main(*args):
    args = get_args_process(*args)
    params = LoftParams.from_file(args.config)
    geom = LnasFormat.from_file(args.surface).geometry
    result = generate_loft(params, geom)
    ...
```

#### 3. Expose a clean top-level __init__.py

```python
# cfdmod/__init__.py
from cfdmod.loft import LoftParams, generate_loft_surface
from cfdmod.roughness import RadialParams, radial_pattern, LinearParams, linear_pattern
from cfdmod.pressure import ...
```

Users and notebooks can then do `import cfdmod; cfdmod.generate_loft_surface(...)`.

#### 4. Make config models notebook-friendly

Params classes should be constructible from keyword arguments as naturally as from YAML:

```python
# Currently works but undocumented:
params = RadialParams(element_params=..., r_start=300, ...)

# Goal: this should be the primary documented interface, with from_file as a convenience:
params = RadialParams.from_dict({...})
params = RadialParams.from_file("config.yaml")
```

Add `to_dict()` / `to_yaml()` methods so notebook users can persist their configs.

#### 5. Reorganize api/ to infrastructure/

Rename `api/` to `infrastructure/` (or `io/`) to better reflect its role as low-level
I/O and format helpers, distinct from the domain API:

```
cfdmod/
    io/
        stl.py
        vtk.py
        lnas.py       thin wrappers around aerosim-lnas
        yaml.py
    config/
        base.py       HashableConfig and other base classes
```

#### 6. Standardize function signatures across modules

Adopt consistent conventions:
- Core functions take typed config objects + data objects, return data objects.
- No functions take file paths (that belongs in CLI/IO layer).
- No functions write files (return arrays/geometries; let the caller decide format).

#### 7. Move notebook-specific helpers to a notebooks/ support module

Create `cfdmod/notebook_utils.py` (or `cfdmod/viz.py`) with:
- Quick-plot helpers for common outputs (loft surface, roughness layout, Cp maps)
- Convenience loaders that return ready-to-use objects from a file path

---

## Priority order for refactoring

Suggested sequence (each step is independently mergeable):

1. Expose a documented `__init__.py` with current symbols (no restructuring, low risk).
2. Split `main.py` -> `run.py` + `cli.py` for loft and roughness_gen as a pilot.
3. Add `to_dict()` / `to_yaml()` to all Pydantic param classes.
4. Rename `api/` -> `io/` + `config/`.
5. Flatten `use_cases/` to top-level domain modules.
6. Update all notebooks and tests to use the new import paths.
