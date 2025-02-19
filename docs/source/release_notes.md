# Release Notes

## v1.1.2

- Automated CI/CD workflow

## v1.1.1

Coefficient time series are now in normalized time scales.
Time values from the solver are normalized by the CST value in the solver time scale.

- Input body pressure data is normalized by the CST value
- Derived coefficients also use a normalized time scale
- Parameters for statistical model are in full scale and need to be normalized using full scale CST.

## v1.1.0

It features the refactor for pressure use case module.

- Updated time series format to matrix form
- Changed how direction logic is applied for Cf and Cm
- Updated statistics functions and models

## v1.0.0

First production stable release.
It features all consulting use cases:

- Loft, Altimetry, Pressure, Roughness Generation, S1 and Snapshots modules

It also includes an API for handling geometry and vtk objects (Probe from VTM).
All use cases have good testing code coverage, all passing.

## v0.1.0

First version of CFDMod. It is being refactored from the
original codebase [CFD-Scripting](https://github.com/AeroSim-CFD/cfd-scripting)

The API module currently supports STL reading and writing.
It also includes tools for extracting data from multiblock datasets.

The available use cases are

- Use cases

  - Altimetry:

    - Sections
    - Plots

  - Block Generation:

    - STL file with blocks

  - S1

    - From csv
    - From vtm
