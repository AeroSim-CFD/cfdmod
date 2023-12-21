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

Root Mean Square value
======================

The next statistic value is the Root Mean Square (RMS) value of a coefficient signal.
It measures the magnitude of a varying quantity. 
The RMS value is a way to represent the **"effective" or "equivalent" value of a varying quantity**.
For example, the RMS value of a pressure coefficient signal is:

.. math::
    cp_{rms} = \frac{\sqrt{\sum{(cp(t) - cp_{mean})^2}}}{N}


Extreme values
==============

Extreme value analysis is a statistical approach used to analyze the **behavior of extreme events in a dataset**.
In the context of CFD simulations, particularly for pressure coefficient signals, understanding extreme events is crucial for designing structures and systems that can **withstand extreme conditions**.
Extreme events in pressure coefficient signals often represent **critical scenarios such as peak loads on structures or components**.
The analysis involves fitting extreme value distributions to the data and extrapolating to estimate the occurrence of extreme events beyond the observed range.

The Gumbel model is a widely used statistical model in extreme value theory for predicting the probability distribution of extreme values.

To determine the extreme values of a coefficient time series, the sample obtained by the simulation is subdivided according to a **characteristic design interval**.
This interval is related to the duration of the events that are **relevant for the structure design**.

Then the peak values of each subdivided sample are computed, and ordered progressively, for the positive peak values, and regressively, for the negative peak values.
The last step is to fit the **Gumbel PDF** to the ordered data, and compute the extreme value for the reduced variable related to a probability of exceeding the peak value.

The method consists of the following steps:

- Subdivide the coefficient time series into samples
- Compute the extreme values for each sample and order them
- Fit Gumbel PDF model to the data
- Calculate the extreme value of the time series with a probability of exceeding this value

.. note:: 
    For more information about extreme values for structure design, check out Chapter 13 (:footcite:t:`wyatt1990designer`)

Mean Quasi static
=================

For composing Mean Quasi Static, it is taken into account the statistical factors from the NBR 6123, and the calculated extreme events peaks.

For example, the mean quasi static value of a pressure coefficient signal is defined as:

.. math::
    f &= (\frac{S_{2,600s}}{S_{2,3s}}) ^ 2

    cp_{mean-qs} &= max(cp_{mean}, f cp_{xtr-max})   \text{   if  } cp_{mean} > 0

    cp_{mean-qs} &= min(cp_{mean}, f cp_{xtr-min})   \text{   if  } cp_{mean} < 0


.. footbibliography::