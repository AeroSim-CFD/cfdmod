*****************
Force Coefficient
*****************

This coefficient is defined as a net resulting force coefficient of a body.
A body is composed by a **set of surfaces**.
For example, consider a building's canopy, where the lower surface is marked on red, and the upper surface is marked on green:

.. image:: /_static/pressure/marquee.png
    :width: 90 %
    :align: center

The net resulting force coefficient is defined as:

.. math::
   C_{f} = \frac{\sum F_{res}}{q A_{rep}} = \frac{F_{green} + F_{red}}{q A_{rep}} = \frac{\sum{c_{pi} A_{i}} + \sum{c_{pj} A_{j}}}{A_{rep}}

.. important:: Note that the net force coefficient has a direction attached to its definition. Its direction is the same as the resulting force direction.

It can also be defined for each axis direction:

.. math::
   C_{fx} = \frac{\sum Fx_{res}}{q A_{x}} = \frac{Fx_{green} + Fx_{red}}{q A_{x}} = \frac{\sum{c_{pi} A_{ix}} - \sum{c_{pj} A_{jx}}}{A_{x}}

.. math::
   C_{fy} = \frac{\sum Fy_{res}}{q A_{y}} = \frac{Fy_{green} + Fy_{red}}{q A_{y}} = \frac{\sum{c_{pi} A_{iy}} - \sum{c_{pj} A_{jy}}}{A_{y}}

.. math::
   C_{fz} = \frac{\sum Fz_{res}}{q A_{z}} = \frac{Fz_{green} + Fz_{red}}{q A_{z}} = \frac{\sum{c_{pi} A_{iz}} - \sum{c_{pj} A_{jz}}}{A_{z}}

The representative area is defined as a **projection of the bounding box** for the body composed by the selected surfaces, for each direction.
For example, the following image shows a generic building, and its bounding box's dimensions:

.. image:: /_static/pressure/shed.png
    :width: 90 %
    :align: center

To define the representative areas for each direction of the resulting force:

.. math::
   A_x = b h

   A_y = h l

   A_z = b l

One can also define the representative area as a vector:

.. math::
   A_{rep} = [A_x, A_y, A_z]

A common application of the net force coefficient requires sectioning the body in different regions.
To do so, the same logic applied to the shape coefficient is used to **determine the respecting region** of each of the body's triangles.
If its center lies inside the region intervals, then it belongs to it.

The result is a sectionated body in different **sub-bodies for each region**.
When sectioning the body, the respecting representative area should be the same as the sub-body representative area.

.. note:: Check out the `definitions <./definitions.rst>`_ section for more information about **region, surface and body** definitions.

Like the other coefficients, we can apply statistical analysis to the net force coefficient.

By definition, the net force coefficient is a **property of a body**.

It is used for primary and secondary structures design.
It can be seen as the resulting effect of the wind induced stress over a body.

An example of the parameters file required for calculating the net force coefficient can be seen below:

.. literalinclude:: /_static/pressure/Cf_params.yaml
    :language: yaml

To invoke and run the calculation, the following command can be used:

.. code-block:: Bash

   poetry run python -m cfdmod.use_cases.pressure.Cf \
      --output {OUTPUT_PATH} \
      --cp     {CP_SERIES_PATH} \
      --mesh   {LNAS_PATH} \
      --config {CONFIG_PATH}

Or it can be generated together with the pressure data conversion:

.. code-block:: Bash

   poetry run python -m cfdmod.use_cases.pressure \
      --output {OUTPUT_PATH} \
      --cp     {CP_SERIES_PATH} \
      --mesh   {LNAS_PATH} \
      --config {CONFIG_PATH} \
      --Cf

# TODO: reference the notebooks


Use cases:
==========

* **Canopies**
* **Roof vents**
* **Buildings**
* **Building paviments**

Artifacts:
==========

#. **A lnas file**: It contains the information about the mesh.
#. **Parameters file**: It contains which surface inside the mesh is going to be used for evaluating net force coefficient as well as other configs parameters.
#. **HDF time series**: It contains the pressure coefficient signals indexed by each of the mesh triangles.

Outputs:
========

#. **Dimensionless time series**: force coefficient time series for each body.
#. **Statistical results**: maximum, minimum, RMS and average values for the force coefficient time series, for each body.
#. **VTK File**: contains the statistical values inside the original mesh (VTK).