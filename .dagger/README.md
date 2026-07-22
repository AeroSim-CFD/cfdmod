# cfdmod CI (Dagger)

cfdmod's CI is defined as code with [Dagger](https://dagger.io) (Apache-2.0).
The same pipeline runs on your laptop and on any remote runner - there is no
platform-specific YAML, and every step runs inside a container, so a green run
locally is a green run everywhere.

## One-time setup

Install the Dagger CLI (Docker must be running):

```bash
curl -fsSL https://dl.dagger.io/dagger/install.sh | sh
# then generate the SDK bindings for this module:
dagger develop
```

`dagger develop` populates `.dagger/sdk/` (git-ignored) and reconciles the
`engineVersion` in `dagger.json` with your installed CLI.

## Running checks

From the repository root, `--source=.` mounts the working tree:

```bash
dagger call --source=. check            # all checks, concurrently, with a summary
dagger call --source=. lint             # a single check (also: test, docs)
dagger call --source=. test
```

`check` runs the full pipeline:

| Function | What it does                                                        |
| -------- | ------------------------------------------------------------------- |
| `lint`   | `ruff check` + `ruff format --check` (line length 99)               |
| `test`   | `pytest` (unit + integration; the opt-in `perf` benchmarks excluded) |
| `docs`   | Sphinx HTML build (nbsphinx renders the notebook pages)             |

Everything runs inside one content-addressed base image: `python:3.12-slim`
with the project synced via `uv sync --all-extras` (so the vtk / geometry /
remesh extras are present for the full test suite and the docs extra for the
Sphinx build). The `test` stage runs under `xvfb-run` to give the pyvista
off-screen renders a virtual display.

## How the CI server runs this

This repo owns only its CI *definition* (the pipeline above). Scheduling,
checkout, and running many repos' pipelines are the CI server's job (shared
across repos), not this repo's. From the server's point of view the entrypoint
is a single command run against a checkout:

```bash
dagger call --source=. check
```
