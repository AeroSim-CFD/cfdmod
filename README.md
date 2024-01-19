# cfdmod
Package to provide analysis and processing tools for CFD cases

## Tests

There are two ways of running the repository tests. Both use Unit test python framework and it features tests for loft, cp, s1profile, config_models and altimetry modules. To run the tests via CLI:

```bash
poetry run python -m unittest discover -v -s tests/ -p 'test_*.py'
```

Or you can run via <a href="https://code.visualstudio.com/docs/python/testing" target="_blank">Visual Studio Code</a>

## Memory usage profiling

In order to check memory usage, *memory_profiler* library is used.

To profile a function the python decorator @profile must be added to it.

Then, to run the profiling use the following command:

```bash
poetry run mprof run file_with_function.py
```

This librar also provides a command to plot the memory usage:

```bash
poetry run mprof plot
```

That will plot the latest profiling data.