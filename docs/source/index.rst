.. CFD Mod documentation master file

CFD Mod docs
============

`CFD Mod <https://github.com/AeroSim-CFD/cfdmod>`_ **is a Python library for post-processing
and geometry preparation of CFD wind-tunnel simulations.** It covers pressure (``Cp``),
force (``Cf``), moment (``Cm``) and shape (``Ce``) coefficients; terrain loft and roughness
elements; inflow / climate / Lawson statistics; and ParaView snapshot automation.

.. note::

   v2.0 redesigned the pressure pipeline around external consumers:
   XDMF+H5 end-to-end, no input mutation, embedded post-processing
   metadata, multi-format geometry (``.lnas`` / ``.stl`` / ``.h5`` /
   ``.xdmf``), and flat output by default. See
   `Release Notes <release_notes.html>`_ for the full v2 changeset.

.. _quickstart:

Quickstart
----------

.. code-block:: python

   from cfdmod import (
       BasicStatisticModel, BodyConfig, BodyDefinition,
       CpCaseConfig, CpConfig, CfCaseConfig, CfConfig,
       CmCaseConfig, CmConfig, MomentBodyConfig, ZoningModel,
       run_cp, run_cf, run_cm,
   )
   from cfdmod.io.geometry.transformation_config import TransformationConfig

   cp_cfg = CpCaseConfig(pressure_coefficient={
       "default": CpConfig(
           statistics=[BasicStatisticModel(stats="mean")],
           timestep_range=(150.0, 260.0),
           simul_U_H=1.0, simul_characteristic_length=10.0,
           macroscopic_type="rho", reference_pressure="average",
       )
   })
   run_cp(body_h5="body.h5", probe_h5="probe.h5",
          cfg_path=cp_cfg, output="output")

   # Cf and Cm read geometry from the cp.time_series.h5 by default.
   run_cf(cp_h5="output/cp.default.time_series.h5",
          cfg_path=cf_cfg, output="output")
   run_cm(cp_h5="output/cp.default.time_series.h5",
          cfg_path=cm_cfg, output="output")

The full worked example -- including container partition auto-detection
and a four-corner overturning-moment scan -- lives at
``notebooks/process_container_pack.ipynb`` in the repository.

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
