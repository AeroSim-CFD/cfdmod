**********
S1 Profile
**********

The **S1** module is used to evaluate the topographic factor from velocity profiles.
The topographic factor is defined as the ratio of acceleration between a reference velocity profile, usually called *Pitot*, and a given profile.
An illustration of the topographic factor can be seen below:

.. figure:: /_static/s1/topographic_factor.png

For more information about the topographic factor, consider consulting the brazilian wind standard **NBR 6123**.

These profiles are extracted from the mean velocity field, thus are mean velocity profiles.
Another way to input the velocity profile information is to load from a csv file.
The user can check the examples for further information.

.. toctree::
   :maxdepth: -1
   :hidden:

   Profiles from csv <profiles_from_csv.ipynb>
   Profiles from vtm <profiles_from_vtm.ipynb>
