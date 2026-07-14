# Consulting post-processing template (cfdmod v3)

A copy-per-case set of **self-contained notebooks** for building post-processing
of a wind-tunnel / CFD consulting case. Copy this folder next to a case, edit one
line, and run the notebooks top to bottom.

The notebooks are the whole thing: config is read inline, the per-direction loop
is visible in a cell, and all heavy computation is called straight from the
installed `cfdmod` library. There is no per-case Python module, no CLI, and
nothing to keep in sync across cases -- reusable logic lives in `cfdmod`.

## Notebooks

| Notebook | Produces |
|---|---|
| `01_inflow.ipynb` | Directional design speed (NBR 6123 / EN 1991-1-4); ABL mean/TI vs code, spectrum, integral length per terrain category. Writes `_shared.json`. |
| `02_static_loads.ipynb` | Per-direction floor Fx/Fy/Mz profiles, directional global envelope, Eberick per-direction peak-load tables. |
| `03_facade.ipynb` | Per-triangle mean-Cp renders per facade. |

Run them in order: `01` writes a local `_shared.json` (reference/design speeds)
that `02` and `03` read, so you never copy-paste numbers between notebooks. Each
notebook still runs standalone (it falls back to computing those values inline if
`_shared.json` is absent).

## Use it on a new case

1. Copy this folder into the case, e.g. `<case>/post_processing_v3/`.
2. In the first config cell of each notebook, set the case root:

   ```python
   project = pathlib.Path("/data/eng/consulting/NNN_CaseName")
   ```

3. In `01_inflow`, set `REP` -- the representative ABL direction and the
   EU/NBR terrain-category labels per terrain category, e.g.

   ```python
   REP = {"0": ("000.0", "0", "I"), "3": ("022.5", "III", "III")}
   ```

4. Pick the `body` in `02` / `03` (defaults to the first body).
5. Run all cells. Figures render inline; save/versioned output is up to you.

## Expected case layout

The notebooks read the standard consulting `case_data` and results tree:

```
<case>/post_processing/pp_config/case_data/
    global_data.json          # analysis.batch_name, body_names, directions_cat*, H, L, V0
    params_cat<c>.yaml         # pressure_coefficient.simul_U_H, force/moment nominal_area/volume, floor z_intervals
    alturas_<body>.csv         # storey table: z_min, z_max, Pavimento
    wind_analysis_{NBR,EU}.csv
<case>/run/artifacts/lnas/<body>.lnas
<case>/results/<batch>/<Case>_<dir>/000/probes/hist_series/cp_analysis_<body>/
    bodies.<body>.h5           # body pressure timeseries
    points.point0.h5           # reference-pressure probe
<case>/results/<batch>/<Case>_noBody_<dir>/000/probes/hist_series/abl_profile/
    line.<lineN>.ux.h5 / .points.csv    # inflow ABL probe lines (01 only)
```

## Environment

Any kernel with **cfdmod v3** plus `numpy`, `pandas`, `scipy`, `matplotlib`,
`h5py`, `aerosim-lnas`, `ruamel.yaml`, and `tables` (PyTables, for reading the
ABL probe h5 in `01`). The notebooks do **not** import `nassu` -- the ABL probe
files are read directly with pandas.

## Notes

- The tiny glue kept visible in a cell (the reference-pressure point reader and
  the static-load dimensionalization) is a candidate to move into `cfdmod`; until
  then it stays in the notebook so it is easy to read and change.
- Wind-direction labels from `cfdmod.dynamics.plotting` are reformatted to a
  degree symbol in the notebook (the library default is being fixed upstream).
