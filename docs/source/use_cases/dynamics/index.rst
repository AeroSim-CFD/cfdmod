**********************************
Structural Model Import (Dynamics)
**********************************

The building dynamic-response pipeline needs the structure's modal model:
per-floor mass, polar inertia and radius of gyration, the centre of mass,
the natural periods, and the per-floor mode shapes (``DX``, ``DY`` and the
torsional rotation ``RZ``). The structural engineer produces that model in
their design software; these converters turn those exports into the internal
format that :class:`~cfdmod.dynamics.structural.BuildingStructuralData`
consumes, removing a manual, error-prone transcription step.

Supported sources
^^^^^^^^^^^^^^^^^

**TQS (Portico Espacial / PORTELS).** TQS exports the spatial-frame modal
analysis as a set of text files (Latin-1 encoded, ``//`` comment lines, comma
decimal separators, TAB-separated):

- ``PORTELS_MODOS.TXT`` -- one row per mode: number, period, angular frequency, frequency.
- ``PORTELS_NOS.TXT`` -- nodal coordinates (``No; X; Y; Z``).
- ``PORTELS_MASSAS.TXT`` -- lumped nodal masses.
- ``PORTELS_FORMAS2.TXT`` -- per-mode nodal mode shapes carrying the rotation (``No; DX; DY; RZ``).

This is **nodal** data. The reader groups nodes by slab elevation and reduces
each slab to its lumped floor properties (mass, centre of mass, polar inertia,
radius of gyration) and a mass-weighted rigid-diaphragm mode shape.

**Eberick.** Eberick models each storey as a rigid diaphragm, so its results
are already **per-floor**. The reader consumes a workbook with a floors table,
a modes table and a long-form shapes table. Column and sheet names default to a
documented Portuguese convention and are fully overridable.

Usage
^^^^^

From Python::

    from cfdmod.dynamics import read_tqs_portels, read_eberick

    structure = read_tqs_portels("path/to/portels_export/")
    # or: structure = read_eberick("path/to/modal.xlsx")

    cfg = structure.to_config(damping_ratio=0.015)
    # -> build_building_dynamic_response(load_source, cfg)

From the command line, writing the internal ``modes.csv`` / ``floors.csv`` /
``phi{m}.csv`` (round-trippable with
:meth:`~cfdmod.dynamics.structural.BuildingStructuralData.from_csvs`)::

    cfdmod dynamics <portels_export_dir> --out out_dir --format tqs
    cfdmod dynamics modal.xlsx --out out_dir --format eberick

Both readers return mass-normalized mode shapes (unit generalized mass), the
precondition the single-degree-of-freedom modal solver assumes.
