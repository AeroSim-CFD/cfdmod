**********************************
Structural Model Import (Dynamics)
**********************************

The building dynamic-response pipeline needs the structure's modal model:
per-floor mass, polar inertia and radius of gyration, the centre of mass,
the natural periods, and the per-floor mode shapes (``DX``, ``DY`` and the
torsional rotation ``RZ``). The structural engineer produces that model in
their design software; these converters turn those exports into the internal
:class:`~cfdmod.dynamics.structural.BuildingStructuralData` the recipe
consumes, removing a manual, error-prone transcription step.

Supported sources
^^^^^^^^^^^^^^^^^

**TQS (Portico Espacial), nodal set -- `read_tqs_portels`.** Latin-1 text
files (``//`` comment lines, comma decimals, TAB-separated). The file-name
prefix is ``PORTELS_`` (older) or ``PORTELSSE_`` (newer); both are accepted:

- ``*_MODOS.TXT`` -- one row per mode: number, period, angular frequency, frequency.
- ``*_NOS.TXT`` -- nodal coordinates (``No; X; Y; Z``).
- ``*_MASSAS.TXT`` -- lumped nodal masses.
- ``*_FORMAS2.TXT`` -- per-mode nodal mode shapes carrying rotation (``No; DX; DY; RZ``).
- ``*_PISOS.TXT`` -- optional floor table (``Piso; Nome; Nivel``), in newer exports.

This is **nodal** data and needs aggregation to per-floor (see below).

**TQS (Portico), per-floor set -- `read_tqs_portico`.** Some deliveries ship a
per-floor summary instead (TAB-separated, decimal point):

- ``PORTICO_MASSAS_PAVIMENTO.TXT`` -- per floor: ``Pavimento; Elevacao (cm);
  Massa X/Y/Z; Momento de inercia; Xcg (cm); Ycg (cm)``.
- ``PORTICO_MODOS_PAVIMENTO.TXT`` -- per mode: per-floor ``Pavimento; DX; DY; RZ``.
- ``modes.csv`` -- ``mode,period[,wp,freq]`` (the natural periods).

Already per-floor -- no nodal aggregation.

**Eberick (AltoQi) -- `read_eberick`.** Eberick models each storey as a rigid
diaphragm, so its results are already **per-floor**. A pair of spreadsheets:

- ``DISTRIBUICAO_DAS_MASSAS_DOS_PAVIMENTOS.xlsx`` -- per floor ``Pavimento;
  Altura; Elevacao (cm); Massa; Momento de inercia; Xcg (cm); Ycg (cm)``.
- ``FORMAS_MODAIS_DOS_PAVIMENTOS.xlsx`` -- one block per mode with its frequency
  (Hz) and a per-floor ``Pavimento; Dx (cm); Dy (cm); Rz (rad)`` table.

The reader skips the project-identifying header block and matches the files
case/accent-insensitively. The damping ratio lives in the companion "sistema de
referencia" sheet; pass it explicitly to ``to_config(damping_ratio=...)``.

How the conversion works
^^^^^^^^^^^^^^^^^^^^^^^^^

All three readers converge on the same internal model. The transformation:

1. **Nodal -> per-floor (TQS PORTELS only).** Nodes are grouped by slab
   elevation, and each slab is reduced to lumped floor properties
   (:func:`~cfdmod.dynamics.imports.aggregate_to_building`):

   - ``M   = sum_node m``
   - ``XG  = sum_node m*x / M``, ``YG = sum_node m*y / M`` (centre of mass)
   - ``I   = sum_node m*((x-XG)^2 + (y-YG)^2)`` (polar inertia about the CoM)
   - ``R   = sqrt(I / M)`` (radius of gyration)
   - ``DX  = sum_node m*DX_node / M`` (mass-weighted rigid-diaphragm shape; same for ``DY``, ``RZ``)

   When a ``PISOS`` table is present its levels define the real floors, so the
   many intermediate FE node elevations (beams, landings) collapse onto actual
   slabs; otherwise elevations are found by clustering node ``Z``. The Portico
   and Eberick sets already give these floor quantities directly.

2. **Units -> SI.** Portico and Eberick report lengths in centimetres and mass
   in ``tf.s^2/cm``; these are converted to metres and kilograms
   (:class:`~cfdmod.dynamics.imports.EberickUnits`, overridable). TQS PORTELS
   coordinates are already in metres.

3. **Mass-normalization.** Mode shapes are scaled to unit generalized mass
   (``sum_floor M*(DX^2 + DY^2 + (R*RZ)^2) = 1`` per mode) -- the precondition
   the single-degree-of-freedom modal solver assumes.

4. **Metadata.** The storey names are carried through in
   ``BuildingStructuralData.floor_labels`` and any extra per-floor columns (e.g.
   Eberick's storey height) in ``floor_metadata`` -- ignored by the recipe but
   preserved for reporting and written as extra columns in ``floors.csv``.

The result is a per-floor model: floors ascending by elevation, ``floor_points``
/ ``cm_positions`` from the centre of mass, ``floors_mass`` / ``floors_radius``,
``natural_frequencies`` (angular, from the periods), and mass-normalized
``mode_shapes``.

Usage
^^^^^

From Python::

    from cfdmod.dynamics import read_tqs_portels, read_tqs_portico, read_eberick

    structure = read_tqs_portels("path/to/portels_export/")
    # or: read_tqs_portico("path/to/portico_export/")
    # or: read_eberick("path/to/eberick_export/")

    cfg = structure.to_config(damping_ratio=0.015)
    # -> build_building_dynamic_response(load_source, cfg)

Each reader also accepts explicit file paths (for renamed files), e.g.
``read_eberick(dir, masses_file=..., formas_file=...)``.

From the command line, writing the internal ``modes.csv`` / ``floors.csv`` /
``phi{m}.csv`` (round-trippable with
:meth:`~cfdmod.dynamics.structural.BuildingStructuralData.from_csvs`; the
``floors.csv`` also carries a ``name`` column and any metadata)::

    cfdmod dynamics <export_dir> --out out_dir --format tqs
    cfdmod dynamics <export_dir> --out out_dir --format portico
    cfdmod dynamics <export_dir> --out out_dir --format eberick
