pyCurves documentation
======================

pyCurves analyzes DNA (and/or RNA) shape from static PDB/mmCIF structures and molecular-dynamics trajectories.

.. warning::

   pyCurves is under active development. Interfaces and output details may
   change before publication.

Choose an interface
-------------------

Use the :doc:`command-line reference <cli>` for shell scripts and batch jobs.
Use the :doc:`Python API <python-api>` for notebooks and applications. Both
interfaces call the same analysis engine and use the same conventions.

Installation
------------

pyCurves requires Python 3.10 or newer. Python 3.12 is recommended.

.. code-block:: console

   git clone https://github.com/JinsenLi/pyCurves/
   cd pyCurves
   pip install .

Install the optional dependencies for trajectories, plotting, and the batch
trajectory engine:

.. code-block:: console

   pip install ".[all]"

First analysis
--------------

.. code-block:: console

   pycurves test_data/1A1F_b_c.pdb
   pycurves test_data/1A6Y.cif --format json --output-file 1a6y.json

The first command prints a Curves-style report. The second writes structured
JSON. See :ref:`pycurves-cli` for every static-analysis option.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   cli
   analysis-options
   outputs

.. toctree::
   :maxdepth: 2
   :caption: Python reference

   python-api
