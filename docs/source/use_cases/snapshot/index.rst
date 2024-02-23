********
Snapshot
********

Snapshots are **photos taken from a rendering view** of a VTK polydata.

It can be generated from any .vtp files having scalar arrays in it.

Rendering Polydata
^^^^^^^^^^^^^^^^^^

VTK polydata is a common format for combining a **geometric mesh and a array of scalars** (data inside cell centers).

To render it, we use PyVista library, which provides methods and objects for rendering and taking a snapshot of the render view.
After taking the snapshot, the output image is treated using Pillow, for **cropping and adding watermarks**, if necessary.

Usage
^^^^^

Snapshot module does not support running it as a script.
Instead it should be used in a jupyter notebook environment, providing a configuration file.

This limitation is due to the fact that PyVista renders the polydata in another window, and it only supports jupyter backend for static rendering.

Here is an example of a configuration file:

.. literalinclude:: /_static/snapshot/snapshot_params.yaml
    :language: yaml

In the documentation, there is a `notebook <take_snapshot.ipynb>`_ with an example for consuming it as an API.

.. toctree::
   :maxdepth: -1
   :hidden:

   Snapshot from .vtp <take_snapshot.ipynb>