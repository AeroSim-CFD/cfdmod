******************
Moment Coefficient
******************

The **Moment Coefficient**, :math:`C_M`, is a dimensionless parameter that provides a generalized representation of the **resultant moment** experienced by an object within a fluid flow.
It offers a means to evaluate the **cumulative effect** of pressure coefficients, :math:`c_p` across different regions of an object's surface and how these pressures translate into aerodynamic moment forces.

:math:`C_M` is a fundamental tool for **torsional effects** for the design and analysis of aerodynamic components.

Definition
==========

Similarly to the force coefficient, this coefficient is defined as a resulting moment coefficient of a **body**.

It is defined as a sum of the resulting moment for each triangle of each surface of the body:

.. math::
   \vec{C_{M}} = \frac{\sum \vec{M_{res}}}{q V_{nom}} = \frac{\sum \vec{r_o} \times \vec{f_{i}}}{q V_{nom}}

.. math::
   \vec{f_i} = c_{pi} q \vec{A_i}

.. math::
   \vec{C_{M}} = \frac{\sum (\vec{r_o} \times \vec{A_i}) c_{pi}}{V_{nom}}

The position vector :math:`r_o` is defined for each triangle, from a common arbitrary points :math:`o`. One can also define it for each axis direction:

.. math::
   C_{M_x} = \frac{\sum M_{res_x}}{q V_{nom}} = \frac{\sum (r_{oy} A_{iz} - r_{oz} A_{iy}) c_{pi}}{V_{nom}}

.. math::
   C_{M_y} = \frac{\sum M_{res_y}}{q V_{nom}} = \frac{\sum (r_{oz} A_{ix} - r_{ox} A_{iz}) c_{pi}}{V_{nom}}

.. math::
   C_{M_z} = \frac{\sum M_{res_z}}{q V_{nom}} = \frac{\sum (r_{ox} A_{iy} - r_{oy} A_{ix}) c_{pi}}{V_{nom}}


We define the nominal volume ($V_{nom}$) as a **user input**.
This is done to let the user define how they want to calculate its value.
For example, considering a rectangular tall building:

.. image:: /_static/pressure/building.png
    :width: 45 %
    :align: center

The nominal volume could be calculated with:

.. math::
   V_{nom} = b h l

Use Case
========

A common application of the moment coefficient requires sectioning the body in **different sub-bodies**.
To do so, the same logic applied to the force coefficient is used to **determine the respective sub-body** of each of the body's triangles.
If its center lies inside the sub-body volume, then it belongs to it.

The result is a sectionated body in different **sub-bodies for each interval**.
When sectioning the body, the respective nominal volume should be the same as the sub-body nominal volume.

.. note:: Check out the `definitions <./definitions.rst>`_ section for more information about **surface, body and sub-body** definitions.

Like the other coefficients, we can apply statistical analysis to the moment coefficient.

By definition, the moment coefficient is a **property of a body**.

It is used for **primary and secondary structures design**, such as **canopies**.
It can also be used for evaluating the resultant wind torsional effect over a **building** or the **building paviments**.
It can be seen as the **resulting torsion effect** of the wind induced stress over a body.

Artifacts
=========

In order to use the moment coefficient module, the user has to provide a **set of artifacts**:

#. **A lnas file**: It contains the information about the mesh.
#. **HDF time series**: It contains the pressure coefficient signals indexed by each of the mesh triangles.
#. **Parameters file**: It contains the coordinate for the arbitrary point when evaluating the force moment, as well as other configs parameters.

Which outputs the following data:

#. **Dimensionless time series**: moment coefficient time series for each body.
#. **Statistical results**: maximum, minimum, RMS and average values for the moment coefficient time series, for each body.
#. **VTK File**: contains the statistical values inside the original mesh (VTK).

An illustration of the moment coefficient module pipeline can be seen below:

.. image:: /_static/pressure/Cm_pipeline.png
    :width: 90 %
    :align: center

Usage
=====

An example of the parameters file required for calculating the moment coefficient can be seen below:

.. literalinclude:: /_static/pressure/Cm_params.yaml
    :language: yaml

To invoke and run the calculation, the following command can be used:

.. code-block:: Bash

   uv run python -m cfdmod.use_cases.pressure.Cm \
      --output {OUTPUT_PATH} \
      --cp     {CP_SERIES_PATH} \
      --mesh   {LNAS_PATH} \
      --config {CONFIG_PATH}

Or it can be generated together with the pressure data conversion:

.. code-block:: Bash

   uv run python -m cfdmod.use_cases.pressure \
      --output {OUTPUT_PATH} \
      --cp     {CP_SERIES_PATH} \
      --mesh   {LNAS_PATH} \
      --config {CONFIG_PATH} \
      --Cm

Another way to run the moment coefficient calculation, is through the `notebook <calculate_Cm.ipynb>`_

Data format
===========

.. note:: The rule for determining the region_idx is based on the **region index and the body name**.
        Input mesh can have multiple bodies, and each of them can be applied a specific zoning/region rule.
        Because of that, region_idx has to be composed by the **zoning region index joined by "-" and the body name**.
        This also guarantee that even if different bodies lie on the same region, the interpreted region for each of them will be different

.. note::
    For more information about the normalized time scale (:math:`t^*`), check the `Normalization section <./normalization.rst>`_ 

.. list-table:: :math:`C_{mx}(t)`
   :widths: 15 15 15 15 15
   :header-rows: 1

   * - time_idx/region_idx
     - Normalized time (:math:`t^*`)
     - 0-Body1
     - 1-Body1
     - 0-Body2
   * - 0
     - 10000
     - 1.25
     - 1.15
     - -1.1
   * - 1
     - 11000
     - 1.5
     - 0.9
     - -1.15

.. list-table:: :math:`C_{my}(t)`
   :widths: 15 15 15 15 15
   :header-rows: 1

   * - time_idx/region_idx
     - Normalized time (:math:`t^*`)
     - 0-Body1
     - 1-Body1
     - 0-Body2
   * - 0
     - 10000
     - 1.25
     - 1.15
     - -1.1
   * - 1
     - 11000
     - 1.5
     - 0.9
     - -1.15

.. list-table:: :math:`C_{mz}(t)`
   :widths: 15 15 15 15 15
   :header-rows: 1

   * - time_idx/region_idx
     - Normalized time (:math:`t^*`)
     - 0-Body1
     - 1-Body1
     - 0-Body2
   * - 0
     - 10000
     - 1.25
     - 1.15
     - -1.1
   * - 1
     - 11000
     - 1.5
     - 0.9
     - -1.15

.. list-table:: :math:`C_{mx} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - max
     - min
     - mean
     - std
     - skewness
     - kurtosis
   * - 0-Body1
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1-Body1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_{my} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - max
     - min
     - mean
     - std
     - skewness
     - kurtosis
   * - 0-Body1
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1-Body1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_{mz} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - max
     - min
     - mean
     - std
     - skewness
     - kurtosis
   * - 0-Body1
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1-Body1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`Regions(indexing)`
   :widths: 50 50
   :header-rows: 1

   * - region_idx
     - point_idx
   * - 0-Body1
     - 0
   * - 1-Body1
     - 1

.. list-table:: :math:`Regions(definition)`
   :widths: 10 10 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - x_min
     - x_max
     - y_min
     - y_max
     - z_min
     - z_max
     - Lx
     - Ly
     - Lz
   * - 0-Body1
     - 0
     - 100
     - 0
     - 50
     - 0
     - 20
     - 0.5
     - 0.8
     - 0.1
   * - 1-Body1
     - 100
     - 200
     - 0
     - 50
     - 0
     - 20
     - 0.5
     - 0.8
     - 0.1

.. toctree::
   :maxdepth: -1
   :hidden:

   Calculate Cm <calculate_Cm.ipynb>
