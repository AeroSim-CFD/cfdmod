# cfdmod

[![tests](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/tox_pipeline.yaml/badge.svg?branch=main)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/tox_pipeline.yaml)

Package to provide analysis and processing tools for CFD cases

## Tests

There are two ways of running the repository tests. Both use Unit test python framework and it features tests for loft, cp, s1profile, config_models and altimetry modules. To run the tests via CLI:

```bash
poetry run python -m unittest discover -v -s tests/ -p 'test_*.py'
```

Or you can run via <a href="https://code.visualstudio.com/docs/python/testing" target="_blank">Visual Studio Code</a>

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
