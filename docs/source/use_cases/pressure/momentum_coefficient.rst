********************
Momentum Coefficient
********************

Similarly to the force coefficient, this coefficient is defined as a resulting momentum coefficient of a body.

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

Like the other coefficients, we can apply statistical analysis to the momentum coefficient.

By definition, the momentum coefficient is a **property of a body**.

It is used for primary and secondary structures design.
It can be seen as the resulting torsion effect of the wind induced stress over a body.

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