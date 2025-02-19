**********
S1 Profile
**********

The **S1** module is used to evaluate the topographic factor from velocity profiles.

Definition
^^^^^^^^^^

The topographic factor is defined as the ratio of acceleration between a reference velocity profile, usually called *Pitot*, and a given profile.

A topographic factor, in the general context of wind engineering, typically refers to a factor that accounts for the influence of local terrain features (such as hills, valleys, ridges, etc.) on the wind pressure or wind speed experienced by a structure.
This factor is used to adjust the wind loads calculated for a structure based on its location and the surrounding topography.

An illustration of the topographic factor can be seen below:

.. figure:: /_static/s1/topographic_factor.png

For more information about the topographic factor, consider consulting the brazilian wind standard **NBR 6123**.

These profiles are extracted from the mean velocity field, thus are mean velocity profiles.
Another way to input the velocity profile information is to load from a csv file.
The user can check the examples for further information.

Usage
^^^^^

In order to use the S1 module, there are some **notebooks** to guide the user.
They are separated according to the use case, whether the profiles are in `csv format <profiles_from_csv.ipynb>`_, of if they will be extracted from a `mean velocity field <profiles_from_vtm.ipynb>`_ file (**.vtm**).

This module is **not supposed to be ran as a python module**, since the use case rules are still being defined.

.. toctree::
   :maxdepth: -1
   :hidden:

   Profiles from csv <profiles_from_csv.ipynb>
   Profiles from vtm <profiles_from_vtm.ipynb>
