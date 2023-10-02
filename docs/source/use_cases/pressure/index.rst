********
Pressure
********

The **Pressure** module handles the analysis and post processing of pressure time series data over a body.
Data comes from CFD transient simulations, and by definition is attached to a mesh.

Mesh describes a body geometry, and contains a set of **discrete vertices** and a **set of triangles**.
Triangles represent the link between 3 different vertices. 
Illustration of a mesh and a mesh triangle can be seen below:

.. image:: /_static/pressure/mesh.png
    :width: 65 %
.. image:: /_static/pressure/triangle.png
    :width: 30 %

Pressure data is extracted at each of the mesh vertices.
The frequency for exporting pressure is set during the simulation setup.
The resulting data has the form of a signal, or a time series.
An example of a pressure signal is shown below:

.. image:: /_static/pressure/pressure_signal.png
    :width: 65 %
    :align: center

The analysis of this signal is based on statistical operations, such as finding the **maximum, minimum or average** values for each vertex.

However, to correctly access the pressure effects over a structure, by definition, it needs to account for the **static pressure**.
For example, consider a wind from a **Atmospheric Boundary Flow** coming onto a building, as suggests the image below:

.. image:: /_static/pressure/domain.png
    :width: 85 %
    :align: center

In order to access the effects of pressure in :math:`p_1` and :math:`p_2`, data of a probe far away from the building must be obtained.
This data is a time series of the **static pressure**, :math:`p_{\infty}`.
Multiple probes can be set to access the domain static pressure.

Normally, the static pressure probe is positioned at the frontside of the building, far above to avoid flow perturbations.

If the fluctuation of the domain static pressure signal is not relevant, its effects can be neglected.
Thus, the pressure signals over the structure does not need the static pressure time series.
Pressure signals examples can be seen below:

.. figure:: /_static/pressure/rho_inf_significant.png
    :width: 85 %
    :align: center

    Pressure signal where domain static pressure should be considered

.. figure:: /_static/pressure/rho_inf_not_significant.png
    :width: 85 %
    :align: center

    Pressure signal where domain static pressure can be neglected

Pressure analysis are usually performed in an adimensionalized form. 
To do so, it needs to be divided by a **dynamic pressure** :math:`q`:

.. math::
   q = \frac{1}{2} \bar{\rho}_{\infty} U_H ^ 2

Normally the pressure signals obtained with Nassu solver are exported using LBM units such as :math:`\rho`.
The transformation of :math:`\rho` into pressure units :math:`[Pa]`, uses the speed of sound in the medium :math:`c_s`:

.. math::
   p(t) - p_{ref} = c_s ^ 2 (\rho(t) - \rho_{ref})

Artifacts
=========

In order to use the **pressure module**, the user has to provide a set of artifacts:

#. A lnas file: It contains the information about the mesh.
#. HDF time series: It contains the pressure signals indexed by each of the mesh vertices.
#. Domain static pressure time series: It contains the pressure signals for probes far away from the building.
#. Zoning information (Optional): Necessary for defining the bounding area for calculating shape and net force coefficients. It is not necessary for pressure coefficient use case only. 

The available use cases for determinating different coefficients using this module are listed below:

* `Pressure Coefficient <./pressure_coefficient.rst>`_
* `Shape Coefficient <./shape_coefficient.rst>`_
* `Force Coefficient <./force_coefficient.rst>`_

.. toctree::
   :maxdepth: 1
   :caption: Pressure use Cases
   :hidden:

   Pressure Coefficient <./pressure_coefficient.rst>
   Shape Coefficient <./shape_coefficient.rst>
   Force Coefficient <./force_coefficient.rst>