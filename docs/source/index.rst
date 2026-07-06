.. CFD Mod documentation master file

CFD Mod docs
============

`CFD Mod <https://github.com/AeroSim-CFD/cfdmod>`_ **is a Python library for post-processing
and geometry preparation of CFD wind-tunnel simulations.** It covers pressure (``Cp``),
force (``Cf``), moment (``Cm``) and shape (``Ce``) coefficients; terrain loft and roughness
elements; inflow / climate / Lawson statistics; and ParaView snapshot automation.

.. note::

   v3 reorganizes post-processing around a single data structure -- the
   :class:`~cfdmod.DataSource` -- and composable ops driven by YAML
   pipeline templates. The legacy per-coefficient functions
   (``run_cp`` / ``run_cf`` / ``run_cm`` / ``run_ce``) and their
   ``*CaseConfig`` models are removed; every flow now runs through
   ``cfdmod run <template>``. See
   :doc:`architecture/v3_migration` for the mapping and
   `Release Notes <release_notes.html>`_ for the full changeset.

.. _quickstart:

Quickstart
----------

Post-processing is a pipeline template: a YAML document declaring inputs,
a sequence of ops, and outputs. Run it from the command line:

.. code-block:: bash

   cfdmod run path/to/cp.yaml

or in Python, over any storage backend:

.. code-block:: python

   from cfdmod import load_template, run_template, XdmfH5Storage

   template = load_template("path/to/cp.yaml")
   bindings = run_template(template, storage=XdmfH5Storage(root="."))

Example Cp / Cf / Cm / Ce templates ship under
``fixtures/tests/pressure/templates/``. The full worked example lives at
``notebooks/process_container_pack.ipynb`` in the repository; the
:doc:`architecture/data_sources` page explains the paradigm end-to-end.

.. toctree::
   :maxdepth: 1
   :caption: Architecture
   :hidden:

   Data sources & pipelines <architecture/data_sources.md>
   v3 migration guide <architecture/v3_migration.md>

.. toctree::
   :maxdepth: 1
   :caption: Library API
   :hidden:

   API Reference <api_reference.rst>

.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :hidden:

   Altimetry <use_cases/altimetry/index.rst>
   Loft <use_cases/loft/index.rst>
   S1 <use_cases/s1/index.rst>
   Roughness Elements <use_cases/roughness_gen/index.rst>
   Pressure <use_cases/pressure/index.rst>
   Snapshot <use_cases/snapshot/index.rst>

.. toctree::
   :maxdepth: 1
   :caption: Analysis
   :hidden:

   Inflow <analysis/inflow/index.rst>


.. toctree::
   :maxdepth: 1
   :caption: Others
   :hidden:

   Release Notes <release_notes.md>
