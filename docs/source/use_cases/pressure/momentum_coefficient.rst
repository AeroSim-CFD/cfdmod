********************
Momentum Coefficient
********************

Similarly to the force coefficient, this coefficient is defined as a resulting momentum coefficient of a **body**.

It is defined as a sum of the resulting momentum for each triangle of each surface of the body:

.. math::
   \vec{C_{M}} = \frac{\sum \vec{M_{res}}}{q V_{rep}} = \frac{\sum \vec{r_o} \times \vec{f_{i}}}{q V_{rep}}

.. math::
   \vec{f_i} = c_{pi} q \vec{A_i}

.. math::
   \vec{C_{M}} = \frac{\sum (\vec{r_o} \times \vec{A_i}) c_{pi}}{V_{rep}}

The position vector :math:`r_o` is defined for each triangle, from a common arbitrary points :math:`o`. One can also define it for each axis direction:

.. math::
   C_{M_x} = \frac{\sum M_{res_x}}{q V_{rep}} = \frac{\sum (r_{ox} A_{ix}) c_{pi}}{V_{rep}}

.. math::
   C_{M_y} = \frac{\sum M_{res_y}}{q V_{rep}} = \frac{\sum (r_{oy} A_{iy}) c_{pi}}{V_{rep}}

.. math::
   C_{M_z} = \frac{\sum M_{res_z}}{q V_{rep}} = \frac{\sum (r_{oz} A_{iz}) c_{pi}}{V_{rep}}

The representative volume :math:`V_{rep}` is defined as the volume of the structure's bounding box.
For example, consider a tall building:

.. image:: /_static/pressure/building.png
    :width: 45 %
    :align: center

The representative volume can be calculated as:

.. math::
   V_{rep} = b h l

A common application of the momentum coefficient requires sectioning the body in **different sub-bodies**.
To do so, the same logic applied to the force coefficient is used to **determine the respective sub-body** of each of the body's triangles.
If its center lies inside the sub-body volume, then it belongs to it.

The result is a sectionated body in different **sub-bodies for each interval**.
When sectioning the body, the respective representative volume should be the same as the sub-body representative volume.

.. note:: Check out the `definitions <./definitions.rst>`_ section for more information about **surface, body and sub-body** definitions.

Like the other coefficients, we can apply statistical analysis to the momentum coefficient.

By definition, the momentum coefficient is a **property of a body**.

It is used for primary and secondary structures design.
It can be seen as the resulting torsion effect of the wind induced stress over a body.

An example of the parameters file required for calculating the momentum coefficient can be seen below:

.. literalinclude:: /_static/pressure/Cm_params.yaml
    :language: yaml

To invoke and run the calculation, the following command can be used:

.. code-block:: Bash

   poetry run python -m cfdmod.use_cases.pressure.Cm \
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
      --Cm

# TODO: reference the notebooks

Use cases:
==========

* **Canopies**
* **Buildings**
* **Building paviments**

Artifacts:
==========

#. **A lnas file**: It contains the information about the mesh.
#. **HDF time series**: It contains the pressure coefficient signals indexed by each of the mesh triangles.
#. **Parameters file**: It contains the coordinate for the arbitrary point when evaluating the force moment, as well as other configs parameters.

Outputs:
========

#. **Dimensionless time series**: momentum coefficient time series for each body.
#. **Statistical results**: maximum, minimum, RMS and average values for the momentum coefficient time series, for each body.
#. **VTK File**: contains the statistical values inside the original mesh (VTK).