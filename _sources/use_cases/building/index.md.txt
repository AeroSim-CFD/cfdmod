# Building Wind-Load Post-Processing

The **building** use case turns a pressure time series on a building surface
into the engineering deliverables of a wind study: per-floor force and moment
coefficients, the modal **dynamic response**, occupant-comfort accelerations,
design load cases, and a multi-direction fan-out over a parametric study.

It composes the v3 recipes and ops (see {doc}`/architecture/data_sources`);
nothing here is high-rise-specific -- the same helpers serve low-rise studies.
The Python surface lives under {mod}`cfdmod.building`; the modal model it
consumes is produced by the importers in {doc}`/use_cases/dynamics/index`.

## The pipeline

A case runs through a fixed sequence of stages, each a thin composition of
library recipes:

```text
pressure time series (surface)
  -> Cp                              (cp_from_pressure)
  -> per-floor Cf / Cm               (cf_per_floor, cm_per_floor)
  -> floor-load points               (floor_load_source)
  -> modal dynamic response          (solve_building_response)
  -> floor accelerations + comfort   (floor_accelerations, comfort_limit)
  -> load-case tables                (generate_load_cases)
```

A single case is aggregated into a {class}`~cfdmod.building.BuildingCase`,
which carries the pressure time series, the dynamic pressure and the geometry
so the downstream stages read a single value object rather than a pile of
loose arrays.

## Per-floor coefficients

The building surface is partitioned into floors (a height binning of the
triangles), and the pressure coefficient time series is integrated over each
floor's triangles into a per-floor force and moment coefficient. The floor
force coefficient is

$$
C_{f,i} = \frac{1}{A_{ref}} \sum_{t \in \text{floor } i} C_{p,t}\, \hat{n}_t\, A_t
$$

with the outward normal $\hat{n}_t$ and triangle area $A_t$, and
the moment coefficient likewise about the floor reference point. Normalisation
uses an **explicit reference area / volume** (`nominal_area` /
`nominal_volume`), not a per-region bounding-box area, so the coefficients
convert back to real-scale forces unambiguously.

## Peak estimation

Each deliverable reduces a fluctuating response (acceleration, floor force,
displacement) to a single design peak. Three methods are selectable per
deliverable ({func}`~cfdmod.building.peak_value`):

- `"max"` -- the observed extreme (optionally of the absolute value).
- `"peak-factor"` -- Davenport {footcite:t}`davenport1964gust`:
  $\hat{x} = \bar{x} + g\,\sigma_x$, with the **gust peak factor**

  $$
  g = \sqrt{2 \ln(\nu T)} + \frac{0.5772}{\sqrt{2 \ln(\nu T)}}
  $$

  where $\nu$ is the mean up-crossing rate (taken as the response
  frequency $f_0$, in Hz) and $T$ the averaging duration
  (default 600 s). The constant 0.5772 is the Euler-Mascheroni constant.
- `"gumbel"` -- fit a Gumbel distribution to block maxima and read the design
  fractile off it (more stable than the raw max for short records). See the
  extreme-value discussion in {doc}`/use_cases/pressure/statistics`.

## Dynamic response

Tall, flexible buildings resonate with the fluctuating wind load, so the
response is not simply the static application of the peak load. The building
is solved as a **modal single-degree-of-freedom (SDOF) system** per mode
({func}`~cfdmod.building.solve_building_response`):

1. The per-floor load coefficients are dimensionalised into a floor-load
   `PointsDataSource` (per-floor $f_x$, $f_y$, $m_z$) by
   the case dynamic pressure and reference area
   ({func}`~cfdmod.building.floor_load_source`).
2. Each mode shape is **mass-normalized to unit generalized mass**,
   $\sum_i M_i\,(\phi_{x,i}^2 + \phi_{y,i}^2 + (R_i\,\phi_{rz,i})^2) = 1$,
   the precondition the modal solver assumes.
3. The floor loads are projected onto each mode to a generalized modal load,
   the modal SDOF equation

   $$
   \ddot{q}_j + 2\,\xi_j\,\omega_j\,\dot{q}_j + \omega_j^2\,q_j = p_j(t)
   $$

   is integrated in time (RK45) with damping ratio $\xi_j$ and natural
   frequency $\omega_j$, and the modal coordinates are recomposed into
   per-floor displacements ($d_x$, $d_y$) and torsional rotation
   ($r_z$).
4. The response is reported as per-floor displacements plus
   **static-equivalent** floor forces / moments ($f_{eq,x}$,
   $f_{eq,y}$, $m_{eq,z}$) that the structural engineer applies
   statically, and as per-floor horizontal **accelerations** at an off-centre
   occupant point ({func}`~cfdmod.building.floor_accelerations`).

The natural periods, per-floor mass, radius of gyration, centre of mass and
mode shapes come from the structural model
({class}`~cfdmod.dynamics.structural.BuildingStructuralData`); see
{doc}`/use_cases/dynamics/index` for how that model is imported from TQS or
Eberick.

## Occupant comfort

Occupant comfort is judged by comparing the **peak horizontal acceleration**
at the top floor against a standard's acceptance limit. Three criteria are
supported ({func}`~cfdmod.building.comfort_limit`); all return the limit in
SI $m/s^2$ (convert to milli-g with
{func}`~cfdmod.building.mps2_to_milli_g`).

**NBR 6123** (serviceability limit, sec. 9.6.2), a decreasing power law in the
fundamental sway frequency $f_0$ {footcite:t}`nbr19886123`:

$$
a_{lim} = 0.01\, k_c\, f_0^{-0.445} \quad [m/s^2]
$$

with $k_c = 4.08$ (residential) or $6.12$ (commercial / office),
valid over $0.06 \le f_0 \le 1.00$ Hz at a 1-year return period.

**Melbourne and Palmer (1992)** serviceable-acceleration curve, a function of
$f_0$ and the return period $R$ in years
{footcite:t}`melbourne1992accelerations`:

$$
a = \sqrt{2 \ln(f_0 T)}\,\left(0.68 + \frac{\ln R}{5}\right)\,
    e^{-3.65 - 0.41 \ln f_0} \quad [m/s^2]
$$

with $T = 600$ s, valid over $0.06 < f_0 < 1.0$ Hz and
$0.5 < R < 10$ years.

**NBCC** occupant-comfort criterion, a flat (frequency-independent) limit at a
10-year return period: **15 milli-g** residential, **25 milli-g** office /
commercial.

## Design load cases

For the structural hand-off, the multi-direction response is reduced to
per-floor load-case tables ({func}`~cfdmod.building.generate_load_cases`,
written with {func}`~cfdmod.building.save_load_case_tables`):

- **Effective-load envelopes** -- per-floor $F_x$ / $F_y$ /
  $M_z$, one column per wind direction, for `min` / `max` (the signed
  per-floor envelopes) and `peak` (the governing envelope, whichever of
  min / max has the larger mean magnitude per direction and axis).
- **Companion load cases** -- for each principal axis the critical direction is
  selected and the accompanying loads on the other axes are carried along, in
  the form the structural software (e.g. Eberick) ingests.

## Multi-direction fan-out

Real consulting cases process many wind directions (and bodies, and Cp
configurations). The v3 building path solves one case at a time; the fan-out
driver ({func}`~cfdmod.building.run_fanout` with a
{class}`~cfdmod.building.FanoutPlan`) maps every `(direction, body, config)`
key through the whole Cp -> per-floor Cf/Cm -> dynamic-response chain and
collects the results in a {class}`~cfdmod.Container` -- the exact shape the
multi-direction reducers ({mod}`cfdmod.dynamics.cases`) and the load-case
tables above consume.

:::{note}
The `examples/high_rise/` suite in the repository is the reference
layout: thin notebooks, one per stage, orchestrating the
{mod}`cfdmod.building` helpers end-to-end and writing versioned debug /
deliverable output roots.
:::

```{footbibliography}
```
