# cfdmod

[![Testing Pipeline](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/testing.yaml/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/testing.yaml)
[![Docs Deploy](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/pages/pages-build-deployment)
[![Linting Workflow](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/linting.yaml/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/linting.yaml)
[![Release artifacts](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/build_and_deploy_artifacts.yaml/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/build_and_deploy_artifacts.yaml)

Package to provide analysis and processing tools for CFD cases

## Tests

This codebase uses Pytest framework and it features tests for loft, cp, s1profile, config_models and altimetry modules. To run the tests via CLI:

```bash
poetry run pytest <path/to/tests>
```

The tests can also be automated to run in different environments, and include dist build commands using <a href="https://tox.wiki/en/stable/" target="_blank">tox</a>:

```bash
poetry run tox
```

## Memory usage profiling

In order to check memory usage, _memory-profiler_ library is used.
First, install memory-profiler:

```bash
pip install -U memory-profiler
```

And activate the poetry virtual environment:

```bash
poetry shell
```

Then, run:

```bash
mprof run -C -M python path_to_script.py
mprof plot
```

That will plot the latest profiling data.

## Generating schemas

Schema files serve as a guide to fill config files.
To generate a schema file for every config model, use the following command:

```bash
poetry run python -m scripts.generate_schemas
```

In order to setup the schema in VSCode, edit `settings.json`, or the workspace file, to include:

```bash
"yaml.schemas": {
    "file:///path/to/cfdmod/output/schema-cfdmod.json": "**/cfdmod/**/\*.yaml"
}
```
