****
Loft
****

The **Loft** module is used to extend the geometry of the terrain, used in CFD Simulations.
The loft extension is essential to guarantee that the terrain starts from a straight line.
Otherwise it would affect the wind flow.

Use Case
========

The loft geometry is generated in the **pre-processing** step of the CFD simulations.
In order to use the loft module, the **terrain surface artifact**, from the landscaping project must be generated beforehand.

The terrain surface must be conformed in a **cirular shape**, in order to simplify the process of rotating the domain to simulate other **wind source directions**.
An example of the shape of the input terrain surface is shown below:

.. image:: /_static/loft/terrain.png
    :width: 90 %
    :align: center

.. important:: 
    The user must guarantee that the terrain surface **does not have a hole** in it. 
    Otherwise, the **border detection will fail!**

Artifacts
=========

The loft module also has to receive a configuration file containing the **loft parameters**.
The loft parameters define the **geometric properties** of the output surfaces.
An example of the configuration file is shown below:

.. literalinclude:: /_static/loft/loft_params.yaml
    :language: yaml

Calling the loft module will generate the following files:

#. **Upwind and downwind lofts**: Untreated generated loft surfaces.
#. **Upwind and downwind remeshed lofts**: Remeshed loft surfaces with target element size.
#. **Terrain remeshed**: Remeshed terrain surface with target element size.

Usage
=====

To invoke and generate loft surfaces, the following command can be used:

.. code-block:: Bash

   poetry run python -m cfdmod.use_cases.loft \
      --config  {CONFIG_PATH}
      --surface {TERRAIN_PATH} \
      --output  {OUTPUT_PATH} \

Another way to run the generation, is through the `notebook <generate_loft.ipynb>`_

.. toctree::
   :maxdepth: -1
   :hidden:

   Loft use case<generate_loft.ipynb>