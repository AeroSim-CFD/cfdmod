**********
Statistics
**********

For every coefficient signal, some statistical operations are performed in order to extract value from the data.

Minimum and maximum values
==========================

Finding the minimum and maximum values is pretty straightforward.
For example, the minimum and maximum values of a pressure coefficient signal is:

.. math::
    cp_{min} = min(cp(t))

    cp_{max} = max(cp(t))

Average value
=============

The mean value is obtained by a sum over the signal, divided by how many samples there are.
For example, the mean value of a pressure coefficient signal is:

.. math::
    cp_{mean} = \frac{\int cp(t) dt}{T}

Where :math:`t` is temporal variable, and :math:`T` is the sample total time.

Root Mean Square value
======================

The next statistic value is the Root Mean Square (RMS) value of a coefficient signal.
It measures the magnitude of a varying quantity. 
The RMS value is a way to represent the **"effective" or "equivalent" value of a varying quantity**.
For example, the RMS value of a pressure coefficient signal is:

.. math::
    cp_{rms} = \frac{\sqrt{\sum{(cp(t) - cp_{mean})^2}}}{N}

Where :math:`N` is the number of time step samples.

Extreme values
==============

Extreme value analysis is a statistical approach used to analyze the **behavior of extreme events in a dataset**.
In the context of CFD simulations, particularly for pressure coefficient signals, understanding extreme events is crucial for designing structures and systems that can **withstand extreme conditions**.
Extreme events in pressure coefficient signals often represent **critical scenarios such as peak loads on structures or components**.
The analysis involves fitting extreme value distributions to the data and extrapolating to estimate the occurrence of extreme events beyond the observed range.

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
- Calculate the extreme value of the time series with a probability of exceeding this value

.. note:: 
    For more information about extreme values for structure design, check out Chapter 13 (:footcite:t:`wyatt1990designer`)

Mean Quasi static
=================

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
This correction is performed using the values for the statistical factors (:math:`S`) from the :footcite:t:`nbr19886123` 6123.
The correction factor is defined as:

.. math::
    f = \left(\frac{S_{2,600s}}{S_{2,3s}} \right) ^ 2

The mean quasi static value is the worst case between the mean value and the extreme value scaled by the statistical factors.
For example, the mean quasi static value of a pressure coefficient signal is defined as:

.. math::
    cp_{mean-qs} &= max(cp_{mean}, f cp_{xtr-max})   \text{   if  } cp_{mean} > 0

    cp_{mean-qs} &= min(cp_{mean}, f cp_{xtr-min})   \text{   if  } cp_{mean} < 0


.. footbibliography::