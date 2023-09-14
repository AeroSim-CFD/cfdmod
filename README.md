# cfdmod
Package to provide analysis and processing tools for CFD cases

## Tests

There are two ways of running the repository tests. Both use Unit test python framework and it features tests for loft, cp, s1profile, config_models and altimetry modules. To run the tests via CLI:

```bash
poetry run python -m unittest discover -v -s tests/ -p 'test_*.py'
```

Or you can run via <a href="https://code.visualstudio.com/docs/python/testing" target="_blank">Visual Studio Code</a>

## Fixtures

In order to run tests or the examples of the use cases on the notebooks, you should download the needed files from
AWS S3. To download via command line:

```bash
cd cfdmod/
aws s3 cp s3://dev01-aerosim-eng-data/cfdmod-fixtures ./fixtures --recursive
```