*********************
Pressure Coefficients
*********************

.. note::
    Introduce here the coefficients and this module.
    This is a copy of the artifacts from the index.

    I moved the parameters here as well

==========
Parameters
==========

.. note::
    Move this

A full example of the parameters file can be seen below:

.. literalinclude:: /_static/pressure/pressure_params.yaml
    :language: yaml
    :caption: pressure_params.yaml

When calculating shape coefficient, zoning parameters must be defined as well:

.. literalinclude:: /_static/pressure/zoning_params.yaml
    :language: yaml
    :caption: zoning_params.yaml

Artifacts
=========

.. note::
    You may speak about the artifacts from Nassu here, because they should be the same for all modules.

    But the output of each coefficient is specific to it, so leave the docs there

    Introduce each module with a little text, if you wish to cite them here.

In order to use the **pressure module**, the user has to provide a set of artifacts:

#. **A lnas file**: It contains the information about the mesh.
#. **HDF time series**: It contains the pressure signals indexed by each of the mesh triangles.
#. **Static reference pressure time series**: It contains the pressure signals for probes far away from the building.
#. **Parameters file**: It contains the values for adimensionalization as well as other configs parameters, such as zoning information.

The available use cases for determinating different coefficients using this module are listed below:

* `Pressure Coefficient <./pressure_coefficient.rst>`_
* `Shape Coefficient <./shape_coefficient.rst>`_
* `Force Coefficient <./force_coefficient.rst>`_
* `Momentum Coefficient <./momentum_coefficient.rst>`_

.. note::
    Is this best here or in module specific?

An Illustration of the modules pipeline can be seen below:

.. figure:: /_static/pressure/pressure_pipeline.png
    :width: 100 %
    :align: center


.. toctree::
   :maxdepth: 1
   :caption: Pressure use Cases
   :hidden:

   Pressure Coefficient <./pressure_coefficient.rst>
   Shape Coefficient <./shape_coefficient.rst>
   Force Coefficient <./force_coefficient.rst>
   Momentum Coefficient <./momentum_coefficient.rst>