**********
Statistics
**********

For every coefficient signal, some statistical operations are performed in order to extract value from the data.

Minimum and maximum values
==========================

Finding the minimum and maximum values is pretty straightforward.
For example, the minimum and maximum values of a generic signal (:math:`x`) is:

.. math::
    \check{x} = min(x)

    \widehat{x} = max(x)

Sometimes it may be necessary to divide the time series in order to make it **memory efficient**.

Thus the minimum and maximum values are computed **for each subdivision** (:math:`n`), and the global minimum and maximum values are defined as:

.. math::
    \check{x} = min(\check{x_n})

    \widehat{x} = max(\widehat{x_n})

Average value
=============

The mean value is obtained by a sum over the signal, divided by how many samples there are.
For example, the mean value of a generic signal (:math:`x`) is:

.. math::
    \bar{x} = \mu = \frac{\int x(t) dt}{T} = \frac{1}{N} \sum{x_i}

Where :math:`t` is temporal variable, and :math:`T` is the sample total time.
For discrete signal :math:`N` is the size of the sample.

As mentioned before, the sample can be subdivided in :math:`n` subsamples.
Thus to compose the global average (:math:`\mu`) from each subdivision average, a weighted average must be performed based on each subdivision size :math:`N_n`:

.. math::
    \mu = \frac{\sum{\bar{x_n} N_n}}{\sum{N_n}} 

Ensemble Average
================

The ensemble average of signal fluctuations is a statistical measure that involves **averaging the values of a signal** across multiple instances or repetitions of an experiment or observation.

The ensemble average :math:`\langle x'(t) \rangle` of the **fluctuation of a signal** :math:`x(t)` is calculated by averaging the corresponding values across different trials or instances, for a m-th order moment:

.. math::
    \langle x'(t) ^ m \rangle = \langle (x(t) - \mu) ^ m \rangle = \frac{\sum{[x(t) - \mu] ^ m}}{N}

The following statistical operations use higher order moments.

Root Mean Square value
======================

The next statistic value is the Root Mean Square (RMS) value of a coefficient signal.
It measures the magnitude of a varying quantity. 
The RMS value is a way to represent the **"effective" or "equivalent" value of a varying quantity**.
For example, the RMS value of a generic signal (:math:`x`) is:

.. math::
    \tilde{x} = \sqrt{\frac{\sum{(x_i - \mu)^2}}{N}}

Where :math:`\mu` is the **global average** and :math:`N` is the number of time step samples.

It can also be defined based on the **ensemble average** of the signal fluctuation (:math:`x'`):

.. math::
    \tilde{x} = \sqrt{\langle x' ^ 2 \rangle} = \sqrt{\frac{\sum{(x_i - \mu)^2}}{N}}

The global RMS value for subdivided samples is computed **cumulating the second-order moment**.

For each subdivision, the second-order moment is calculated as:

.. math::
    \mu_2 = \sum_{i=1}^{N_n}{(x_i - \mu)^2}

Then the global RMS value is defined as:

.. math::
    \tilde{x} = \sqrt{\frac{\sum_{n} \mu_2}{N}}

Skewness
========

Skewness, a third-order statistical moment, characterizes the **asymmetry of the signal's probability distribution**.
A positive skewness indicates a longer tail on the right side of the distribution, while negative skewness suggests a longer left tail.

For example, the skewness value of a generic signal (:math:`x`) is:

.. math::
    Skew[x] = \frac{\langle x' ^ 3 \rangle}{\langle x' ^ 2 \rangle ^ {3/2}} = \frac{\sum{(x_i - \mu)^3}}{(\sum{(x_i - \mu)^2}) ^ {3 / 2}} \sqrt {N}

Where :math:`\mu` is the **global average**. 

The global skewness value for subdivided samples is computed **cumulating the second and third-order moment**.

.. math::
    \mu_2 = \sum_{i=1}^{N_n}{(x_i - \mu)^2}

    \mu_3 = \sum_{i=1}^{N_n}{(x_i - \mu)^3}

Then the global skewness value is defined as:

.. math::
    Skew[x] = \frac{\sum_{n} \mu_3}{(\sum_{n} \mu_2) ^ {3 / 2}} \sqrt {N}

Kurtosis
========

Kurtosis, a fourth-order moment, measures the **"tailedness" of the signal's distribution**.
A high kurtosis indicates heavy tails and a more peaked distribution, suggesting the presence of outliers or extreme values.
In the other hand, low kurtosis indicates lighter tails and a flatter distribution.

For example, the kurtosis value of a generic signal (:math:`x`) is:

.. math::
    Kurt[x] = \frac{\langle x' ^ 4 \rangle}{\langle x' ^ 2 \rangle ^ 2} = \frac{\sum{(x_i - \mu)^4}}{(\sum{(x_i - \mu)^2}) ^ {2}} N

Where :math:`\mu` is the **global average**. 

The global kurtosis value for subdivided samples is computed **cumulating the second and fourth-order moment**.

.. math::
    \mu_2 = \sum_{i=1}^{N_n}{(x_i - \mu)^2}

    \mu_4 = \sum_{i=1}^{N_n}{(x_i - \mu)^4}

Then the global kurtosis value is defined as:

.. math::
    Kurt[x] = \frac{\sum_{n} \mu_4}{(\sum_{n} \mu_2) ^ {2}} N

Extreme values
==============

Extreme value analysis is a statistical approach used to analyze the **behavior of extreme events in a dataset**.
In the context of CFD simulations, particularly for pressure coefficient signals, understanding extreme events is crucial for designing structures and systems that can **withstand extreme conditions**.

Extreme events in pressure coefficient signals often represent **critical scenarios such as peak loads on structures or components**.
The analysis involves fitting extreme value distributions to the data and extrapolating to estimate the occurrence of extreme events beyond the observed range.

Currently, CFD modules support two models of extreme value calculation:

- **Moving average**: The coefficient signal is smoothed using a time window, then its min and max values are assigned to the extreme peaks
- **Gumbel**: The coefficient signal time window is extrapolated using Gumbel statistical model.
- **Peak factor**: The value of the peak is calculated using the average, RMS and peak factor values.

The following configuration illustrates how to select and configure the appropriate model:

.. code-block:: yaml

    statistics:
      - stats: "mean"
      - stats: "rms"
      - stats: "skewness"
      - stats: "kurtosis"
      - stats: "mean_eq"
        params:
          scale_factor: 0.61
      - stats: "min"
        params:
          method_type: "Absolute"
      - stats: "max"
        params:
          method_type: "Gumbel"
          peak_duration: 3 # in seconds
          event_duration: 600 # in seconds. Period of extreme event
          n_subdivisions: 10 # Number of subdivisions
          non_exceedance_probability: 0.78 # Confidence parameter in %
      - stats: "max"
        params:
          method_type: "Peak"
          peak_factor: 3 # xtr = avg +- factor * rms
      - stats: "max"
        params:
          method_type: "Moving Average"
          window_size_real_scale: 3 # s

Moving average
^^^^^^^^^^^^^^

In the context of wind engineering, sometimes it may be necessary to smooth a coefficient signal so that the **peak values (maximum and minimum) are in the same order as the event time scale**.
Normally, atmospheric wind gusts **peak pressures last for about 3 seconds**, according to the wind standard NBR-6123.
However, other wind standards may define a different base time scale.

Therefore, the pressure coefficient peaks can be obtained after smoothing the pressure signal based on the event time scale.
That way, a peak that happens in a shorter period can be neglected/smoothed.

Applying a **moving average to a signal has the effect of smoothing it**.
The image below illustrates the effect of a moving average:

.. image:: /_static/pressure/moving_avg.png
    :width: 60 %
    :align: center

After smoothing the pressure signal, based on the wind gust time scale, the peaks from the smoothed signal are captured and assigned as a extreme value (minimum/negative or maximum/positive).

Gumbel model
^^^^^^^^^^^^

The Gumbel model is a widely used statistical model in extreme value theory for predicting the probability distribution of extreme values.

To determine the extreme values of a coefficient time series, the sample obtained by the simulation is subdivided according to a **characteristic design interval**.
This interval is related to the duration of the events that are **relevant for the structure design**.

.. image:: /_static/pressure/samples.png
    :width: 60 %
    :align: center

Then the peak values of each subdivided sample are computed, and ordered progressively, for the positive peak values, and regressively, for the negative peak values.


The last step is to fit the **Gumbel PDF** to the ordered data, and compute the extreme value for the reduced variable related to a probability of exceeding the peak value.
Firstly the extreme values for the samples are tabulated as follows:

.. list-table:: Sample extremes
   :widths: 25 25 25 25
   :header-rows: 1

   * - Sample number (i)
     - min (cp)
     - max (cp)
     - Reduced variable (y)
   * - 1
     - -0.3
     - 0.4
     - y(1)
   * - 2
     - -0.4
     - 0.38
     - y(2)
   * - 3
     - -0.28
     - 0.41
     - y(3)
   * - 4
     - -0.31
     - 0.43
     - y(4)
   * - 5
     - -0.2
     - 0.45
     - y(5)

The reduced variable :math:`y` is defined as:

.. math::
    y(i) &= -ln(-ln(P_i))

    P_i &= \frac{i}{N + 1}

Where :math:`i` indicates the subdivided sample index, :math:`N` is the number of subdivided samples, and :math:`P(i)` is a weighting value for the sample.

Then the values are ordered, and the Gumbel model is fit by:

.. math::
    y = \frac{1}{\beta}(x - \mu)

Where :math:`\beta` and :math:`\mu` are parameters of the fit. 
The value for reduced variable :math:`y` commonly used is 1.4, resulting in 78% of non-exceeding extreme values.

The method consists of the following steps:

- Subdivide the coefficient time series into samples
- Compute the extreme values for each sample and order them
- Fit Gumbel PDF model to the data
- Calculate the extreme value of the time series with a probability of exceeding this value for positive and negative values

.. note:: 
    For more information about extreme values for structure design, check out Chapter 13 (:footcite:t:`wyatt1990designer`)

Peak Factor
^^^^^^^^^^^

Using this method, the extreme value is calculated provided a peak factor :math:`f`.
Peak value is then defined using the RMS :math:`\tilde{x}` and average values :math:`\bar{x}`.
The expression for calculation is as follows:

.. math::

    \hat{x} = \bar{x} + f \tilde{x}

Mean Equivalent
===============

There are two ways of composing the wind load from coefficient data.
The first one is to use mean pressure distribution, and the dymanic pressure, **which is based on the peak base wind velocity**.
The definition of the first mode of peak wind load is:

.. math:: 
    \hat{P} = \bar{c_p} . \hat{q} = \bar{c_p}  \frac{1}{2}  \rho \hat{V_0}^2

Where :math:`\hat{P}` is the design peak load, :math:`\hat{q}` is the peak dynamic pressure, :math:`\rho` is the fluid density and :math:`\hat{V_0}` is the peak wind velocity.

For structure design purposes, the mean value of the coefficient time series can be misleading.
Thus the peak wind load can be composed by the peak value for the coefficient and the dynamic pressure, **which is based on the mean base wind velocity**.
The definition of the second mode of peak wind load is:

.. math:: 
    \hat{P} = \hat{c_p} . \bar{q} = \hat{c_p}  \frac{1}{2}  \rho \bar{V_0}^2

Where :math:`\bar{q}` is the average dynamic pressure, :math:`\rho` is the fluid density and :math:`\bar{V_0}` is the average wind velocity.

However, the peak value for the coefficient needs to be scaled according to the characteristic event duration.
This correction is performed using the values for the statistical factors (:math:`S_2`) from the :footcite:t:`nbr19886123` 6123.
The correction factor is defined as:

.. math::
    f = \left(\frac{S_{2,600s}}{S_{2,3s}} \right) ^ 2

The mean equivalent value is the worst case between the mean value and the extreme values scaled by the statistical factors.
For example, the mean equivalent value of a pressure coefficient signal is defined as:

.. math::
    abs(cp_{mean-eq}) &= max(abs(cp_{mean}), abs(f * cp_{xtr-max}), abs(f * cp_{xtr-min}))

    cp_{mean-eq} &= abs(cp_{mean-eq}) * sign(max(abs(val)))


.. important:: 
    **The sign of the mean equivalent value is the same as the value correspondent to the maximum absolute value** between the mean, extreme maximum and extreme minimum values.

.. footbibliography::