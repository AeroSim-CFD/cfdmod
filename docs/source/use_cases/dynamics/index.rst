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

**TQS (Portico Espacial).** TQS exports the spatial-frame modal analysis as a
set of text files (Latin-1 encoded, ``//`` comment lines, comma decimal
separators, TAB-separated). The file-name prefix is ``PORTELS_`` (older) or
``PORTELSSE_`` (newer); both are accepted:

- ``*_MODOS.TXT`` -- one row per mode: number, period, angular frequency, frequency.
- ``*_NOS.TXT`` -- nodal coordinates (``No; X; Y; Z``).
- ``*_MASSAS.TXT`` -- lumped nodal masses.
- ``*_FORMAS2.TXT`` -- per-mode nodal mode shapes carrying the rotation (``No; DX; DY; RZ``).
- ``*_PISOS.TXT`` -- optional floor table (``Piso; Nome; Nivel``), in newer exports.

This is **nodal** data. The reader groups nodes by slab elevation and reduces
each slab to its lumped floor properties (mass, centre of mass, polar inertia,
radius of gyration) and a mass-weighted rigid-diaphragm mode shape. When a
``PISOS`` floor table is present it defines the real slab elevations, so the
many intermediate FE node levels (beams, landings) collapse onto actual floors;
otherwise elevations are discovered by clustering the node ``Z`` values.

**Eberick (AltoQi).** Eberick models each storey as a rigid diaphragm, so its
results are already **per-floor** (no nodal aggregation). It delivers a pair of
spreadsheets in an export directory:

- ``DISTRIBUICAO_DAS_MASSAS_DOS_PAVIMENTOS.xlsx`` -- per-floor ``Pavimento;
  Altura; Elevacao (cm); Massa; Momento de inercia; Xcg (cm); Ycg (cm)``.
- ``FORMAS_MODAIS_DOS_PAVIMENTOS.xlsx`` -- one block per mode with its frequency
  (Hz) and a per-floor ``Pavimento; Dx (cm); Dy (cm); Rz (rad)`` table.

The reader skips the project-identifying header block, matches the files
case/accent-insensitively, and converts Eberick's centimetre / technical-mass
units to metres and kilograms (overridable via :class:`EberickUnits`). The
structural damping ratio lives in the companion "sistema de referencia" sheet;
pass it explicitly to ``to_config(damping_ratio=...)``.

Usage
^^^^^

From Python::

    from cfdmod.dynamics import read_tqs_portels, read_eberick

    structure = read_tqs_portels("path/to/portels_export/")
    # or: structure = read_eberick("path/to/eberick_export/")

    cfg = structure.to_config(damping_ratio=0.015)
    # -> build_building_dynamic_response(load_source, cfg)

From the command line, writing the internal ``modes.csv`` / ``floors.csv`` /
``phi{m}.csv`` (round-trippable with
:meth:`~cfdmod.dynamics.structural.BuildingStructuralData.from_csvs`)::

    cfdmod dynamics <portels_export_dir> --out out_dir --format tqs
    cfdmod dynamics <eberick_export_dir> --out out_dir --format eberick

Both readers return mass-normalized mode shapes (unit generalized mass), the
precondition the single-degree-of-freedom modal solver assumes.
