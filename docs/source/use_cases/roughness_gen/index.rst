*****************************
Roughness Elements Generation
*****************************

The **Roughness Elements Generation** module is used to generate the geometry of the roughness elements, used in CFD Simulations.
These elements are used to represent the roughness of the terrain, just like how it is done in physical wind tunnels.

A standard configuration of the objects used for representing atmospheric flow can be seen in the following image:

.. figure:: /_static/roughness_gen/wind_tunnel.png

The roughness of the terrain affects the mean velocity profile in the **Atmospheric Boundary Layer** (ABL). 
The following image shows this effect:

.. figure:: /_static/roughness_gen/ABL.png

According to Brazilian and European wind standards, the ABL profile can me represented by a roughness factor (z_0).
The objective of this module is to serve as a tool for generating the geometry of the roughness elements, in order to achieve a corresponding ABL profile.
The validation of the ABL profile is based on the mean velocity profile.
This profile is then used to obtain a corresponding roughness factor and compared to the ones presented by the standards.

Usage
^^^^^

There are several ways to use **Roughness Elements generation** module. The main one is to run as a module, using:

.. code-block:: Bash

   poetry run python -m cfdmod.use_cases.roughness_gen --config {CONFIG_PATH} --output {OUTPUT_PATH}

It takes two arguments: the path for the **.yaml configuration file** with the generation parameters and the **output path** for saving the .STL file.
For standard use, the user must fullfill a configuration file with the parameters.
One example of the configuration is as it follows:

.. literalinclude:: /_static/roughness_gen/roughness_params.yaml
    :language: yaml

The parameters consist in defining the number of replication in each axis.
The logic follows the **linear pattern** thought.

First, it is generated only a single element.
Then, it is replicated in the main axis, defined by the offset direction.
Later, the rows generate are replicated as well, in a way that the even rows are offseted by the defined parameter.
The direction of the offset is also defined in the configuration parameters.

The value for offseting even rows can be defined by setting the absolute value of the offset.

Output
^^^^^^

The expected output is a STL file containing the information for creating the geometry.
This file can be inspected in CAD softwares, such as mesh lab.
An example of the output can be seen below:

.. figure:: /_static/roughness_gen/elements.png

.. toctree::
   :maxdepth: -1
   :hidden:

   Generating roughness elements <gen_roughness_elements.ipynb>
