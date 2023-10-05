************************
Coefficient data formats
************************

:math:`c_p`
==============

.. list-table:: :math:`c_p(t)`
   :widths: 33 33 33
   :header-rows: 1

   * - point_idx
     - timestep
     - cp
   * - 0
     - 10000
     - 1.25
   * - 1
     - 10000
     - 1.15

.. list-table:: :math:`c_p (stats)`
   :widths: 20 10 10 10 10 20 20
   :header-rows: 1

   * - point_idx
     - max
     - min
     - avg
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

:math:`C_e`
==============

.. list-table:: :math:`C_e(t)`
   :widths: 33 33 33
   :header-rows: 1

   * - region_idx
     - timestep
     - C_e
   * - 0
     - 10000
     - 1.25
   * - 1
     - 10000
     - 1.15

.. list-table:: :math:`C_e (stats)`
   :widths: 20 10 10 10 10 20 20
   :header-rows: 1

   * - region_idx
     - max
     - min
     - avg
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_e(regions)`
   :widths: 10 10 10 10 10 10 10
   :header-rows: 1

   * - region_idx
     - x_min
     - x_max
     - y_min
     - y_max
     - z_min
     - z_max
   * - 0
     - 0
     - 100
     - 0
     - 50
     - 0
     - 20
   * - 1
     - 100
     - 200
     - 0
     - 50
     - 0
     - 20

:math:`C_f`
==============

.. list-table:: :math:`C_f(t)`
   :widths: 15 15 15 15 15
   :header-rows: 1

   * - sub_body_idx
     - timestep
     - Cf_x
     - Cf_y
     - Cf_z
   * - 0
     - 10000
     - 1.25
     - 1.15
     - -1.1
   * - 1
     - 10000
     - 1.5
     - 0.9
     - -1.15

.. list-table:: :math:`C_{fx} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - sub_body_idx
     - max
     - min
     - avg
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_{fy} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - sub_body_idx
     - max
     - min
     - avg
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_{fz} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - sub_body_idx
     - max
     - min
     - avg
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_f` (sub-bodies)
   :widths: 50 50
   :header-rows: 1

   * - point_idx
     - sub_body_idx
   * - 0
     - 0
   * - 1
     - 0

:math:`C_m`
==============

.. list-table:: :math:`C_m(t)`
   :widths: 15 15 15 15 15
   :header-rows: 1

   * - sub_body_idx
     - timestep
     - Cm_x
     - Cm_y
     - Cm_z
   * - 0
     - 10000
     - 1.25
     - 1.15
     - -1.1
   * - 1
     - 10000
     - 1.5
     - 0.9
     - -1.15

.. list-table:: :math:`C_{mx} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - sub_body_idx
     - max
     - min
     - avg
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_{my} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - sub_body_idx
     - max
     - min
     - avg
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_{mz} (stats)`
   :widths: 20 10 10 10 10 10 10
   :header-rows: 1

   * - sub_body_idx
     - max
     - min
     - avg
     - std
     - skewness
     - kurtosis
   * - 0
     - 1.25
     - 0.9
     - 1.1
     - 0.2
     - 0.1
     - 0.15
   * - 1
     - 1.15
     - 0.95
     - 1.13
     - 0.19
     - 0.11
     - 0.13

.. list-table:: :math:`C_m` (sub-bodies)
   :widths: 50 50
   :header-rows: 1

   * - point_idx
     - sub_body_idx
   * - 0
     - 0
   * - 1
     - 0