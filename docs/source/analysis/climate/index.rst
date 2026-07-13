******************************
Climate and Wind Statistics
******************************

A CFD wind study produces **dimensionless** results -- pressure, force and
comfort coefficients that are independent of the actual wind speed. To turn
those into design values (design pressures, return-period accelerations,
pedestrian-comfort verdicts) they must be combined with the **local wind
climate**: how often the wind blows, how hard, and from which direction.

The :mod:`cfdmod.climate` module provides that statistical layer -- long-term
speed distributions, extreme-value analysis, directional statistics and the
Lawson pedestrian-comfort criterion. Climate ingestion is deliberately kept
*outside* the pipeline (it is small tabular data), and combined with the
pipeline's dimensionless output in a downstream step.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parent distribution (Weibull)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The long-term distribution of mean wind speed at a site is well described by a
**Weibull distribution**, whose probability density is

.. math::
   f(U) = \frac{k}{c}\left(\frac{U}{c}\right)^{k-1}
          \exp\!\left[-\left(\frac{U}{c}\right)^{k}\right]

with shape parameter :math:`k` and scale parameter :math:`c` (a
characteristic speed). ``fit_weibull`` fits :math:`(k, c)` to a speed record
and ``directional_weibull_fit`` fits one Weibull per wind-direction sector.
The helpers ``weibull_shape_from_mean_and_std`` /
``weibull_scale_from_mean_and_shape`` recover parameters from summary
statistics, and ``get_weibull_quantile`` /
``get_weibull_probability_between_velocities`` read probabilities off the
fitted distribution -- the exceedance frequencies a comfort or fatigue
assessment integrates over.

^^^^^^^^^^^^^^^^^^^^^^^^^
Extreme values (Gumbel)
^^^^^^^^^^^^^^^^^^^^^^^^^

Design wind loads are governed not by the typical speed but by the **extreme**
speed for a chosen return period. The maxima (annual, or per independent storm)
follow a **Type I / Gumbel** extreme-value distribution,

.. math::
   F(U) = \exp\!\left[-e^{-(U - \mu)/\beta}\right]

with location :math:`\mu` and scale :math:`\beta`. The design speed for a
return period :math:`R` (years) follows from the **reduced variate**
:math:`y_R = -\ln\!\left[-\ln\!\left(1 - 1/R\right)\right]`
(:func:`~cfdmod.climate.get_reduced_variate`), giving the return level

.. math::
   U_R = \mu + \beta\, y_R

(:func:`~cfdmod.climate.type_I_return_level`). ``fit_gumbel`` fits the
distribution; two estimators are provided -- ``fit_gumbel_BR_MIS`` (the
Method of Independent Storms after Vallis, 2019) and ``fit_gumbel_MLE_MIS``
(maximum-likelihood) -- and ``directional_gumbel_fit`` fits per sector.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Storm declustering and direction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extreme-value fitting assumes the maxima are **independent**. A raw time series
is not: one storm contributes many correlated hourly peaks. The **Method of
Independent Storms** extracts one peak per storm event
(:func:`~cfdmod.climate.get_storm_peaks`) and removes an event's samples before
picking the next (:func:`~cfdmod.climate.remove_storm_from_series`), so the
fit sees independent maxima.

Directional statistics are handled by the wind-rose helpers
(:mod:`cfdmod.climate.wind_rose`), which count wind occurrences per direction
sector -- the weights that combine per-sector fits into an omnidirectional
result, and that drive the pedestrian-comfort assessment below. Analytical
mean-velocity profiles (code-based :math:`U(z)` laws) live alongside in
:class:`~cfdmod.climate.WindProfile`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pedestrian comfort (Lawson)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pedestrian wind comfort combines the CFD velocity field at pedestrian level
with the site climate. The pipeline extracts per-probe velocity statistics
(mean / rms / peak) at pedestrian-level probes (the
:class:`~cfdmod.recipes.PedestrianComfortConfig` recipe); those per-direction
amplification ratios are then weighted by the local directional Weibull
climate into an **effective velocity** per probe
(``fit_average_velocity`` in :mod:`cfdmod.climate.lawson`). That effective
velocity is what the **Lawson comfort criterion** classifies against its
activity-category speed thresholds (sitting, standing, strolling, walking) to
produce the pedestrian-comfort verdict at each location.

.. seealso::
   :doc:`/analysis/inflow/index` for validating the simulated inflow profile
   against the target ABL, and :doc:`/use_cases/pressure/statistics` for the
   extreme-value correction applied to pressure coefficients.
