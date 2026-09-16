Command-line reference
======================

The package installs six commands. Run any command with ``--help`` for the
parser-generated synopsis.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Command
     - Purpose
   * - ``pycurves``
     - Static PDB/mmCIF or Curves-input analysis.
   * - ``pycurves-md``
     - General trajectory analysis, including the global axis and per-frame
       topology annotation.
   * - ``pycurves-md-batch``
     - Vectorized standard-frame, local-axis trajectory analysis.
   * - ``pycurves-md-plot``
     - Extract and plot per-frame trajectory results.
   * - ``pycurves-viewer``
     - Build a single-file interactive HTML structure viewer.
   * - ``pycurves-pymol``
     - Build a structure-free PyMOL overlay.

.. _pycurves-cli:

``pycurves``: static structures
-----------------------------------

.. program:: pycurves

Synopsis
~~~~~~~~

.. code-block:: text

   pycurves INPUT [OPTIONS]
   pycurves INPUT.inp --pdb STRUCTURE [OPTIONS]
   pycurves --generate-inp-only INPUT... [OPTIONS]

With a PDB/mmCIF input, pyCurves infers a Curves topology, writes a generated
``.inp`` file under ``--output-dir``, and analyzes the first inferred
unit. With an explicit ``.inp``, the topology and Curves flags come from
that file.

Input and output options
~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --pdb PATH

   Coordinate file used with an explicit ``.inp`` input. This is optional
   when the input's ``file=`` value resolves to the intended structure.

.. option:: --output-dir DIRECTORY

   Directory for inferred Curves inputs. The default is the current directory.
   This does not choose the result location; use ``--output-file`` for that.

.. option:: --generate-inp-only, --inp-only

   Infer and write Curves inputs, then stop before fitting, axis construction,
   and parameter calculation. In this mode positional inputs may be individual
   structure files, shell globs, or directories. Directories are scanned for
   supported PDB/mmCIF extensions. Duplicate resolved paths are processed once.

.. option:: --no-output

   Run the analysis without printing or writing the final result. Generated
   ``.inp`` files and calculation side effects still occur.

.. option:: --format FORMAT

   Select ``curves``, ``json``, or ``csv``. The default is
   ``curves``. CSV requires Pandas, which is a base pyCurves dependency.

.. option:: --output-file PATH

   Write Curves text or JSON to one file. For CSV, PATH is a prefix:
   ``result`` produces files such as ``result_backbone.csv`` and
   ``result_local_inter_base_pair.csv``. A trailing ``.csv`` is removed
   from the prefix.

.. option:: --visualization, --visualize

   Add viewer geometry to JSON output. Use this with ``--format json``
   before calling ``pycurves-viewer`` or ``pycurves-pymol``.

.. option:: --verbose-opt

   Echo base-fitting and axis-minimization progress during analysis. This is
   diagnostic output and can be substantial.

Topology options
~~~~~~~~~~~~~~~~

.. option:: --continuous-strands

   Treat connected split-chain helical components as one biological structure
   during inferred-topology generation.

.. option:: --duplex-only

   Skip triplex/quadruplex inference and use the one-to-one duplex path.

.. option:: --altloc CODE

   Select one alternate conformation, such as ``A`` or ``B``. See
   :ref:`alternate-conformations`.

.. option:: --dssr-json PATH

   Use a DSSR JSON report as the topology source while reading coordinates from
   INPUT. This cannot be combined with an explicit ``.inp`` or
   ``--continuous-strands``.

.. option:: --dssr-unit SELECTOR

   Select a unit when the DSSR report contains multiple candidates. Examples:
   ``multiplet:1``, ``stem:1``, ``helix:2``, and ``pairs:1``.

Analysis options
~~~~~~~~~~~~~~~~

.. option:: --fit, --no-fit

   Override least-squares base fitting. With neither flag, use the inferred or
   Curves-input value.

.. option:: --grooves, --no-grooves

   Override groove analysis. Disable grooves for explicit triplex and
   quadruplex Curves inputs.

.. option:: --mini, --no-mini

   Override global-axis minimization. Local-axis mode always disables it.

.. option:: --comb, --no-comb

   Override combined-strand analysis.

.. option:: --ends, --no-ends

   Override Curves terminal virtual end levels.

.. option:: --frame-convention NAME

   Use ``standard`` (default) or ``legacy`` base frames.

.. option:: --axis-convention NAME

   Use the ``global`` Curves axis (default) or ``local`` Curves+ axis.

.. option:: --axis-weighting, --no-axis-weighting

   Override fitted-pair weighting during axis construction. Weighting is
   disabled by default.

The precise behavior and interactions of these flags are described in
:doc:`analysis-options`.

Examples
~~~~~~~~

.. code-block:: console

   # Curves-style terminal report
   pycurves structure.cif

   # Structured output
   pycurves structure.cif --format json --output-file structure.json

   # Local Curves+ axis (also forces standard frames)
   pycurves structure.cif --axis-convention local --format csv --output-file local

   # Review inferred inputs without calculating geometry
   pycurves --generate-inp-only "structures/*.cif" --output-dir inferred

``pycurves-md``: general trajectories
------------------------------------------

.. program:: pycurves-md

Synopsis
~~~~~~~~

.. code-block:: text

   pycurves-md TOPOLOGY [TRAJECTORY] [OPTIONS]

``TOPOLOGY`` is the reference PDB/mmCIF structure. ``TRAJECTORY`` can be
any format supported by the installed MDAnalysis or MDTraj reader. Omit it when
``TOPOLOGY`` is a multi-model PDB.

Core options
~~~~~~~~~~~~

.. option:: --inp PATH

   Reuse an explicit Curves topology instead of inferring one from the reference
   coordinates.

.. option:: --output-dir DIRECTORY

   Directory for automatically generated Curves input files. The default is the
   current directory.

.. option:: --output-file PATH

   JSON path or CSV prefix. The default JSON path is
   ``pycurves_trajectory.json``.

.. option:: --format FORMAT

   Select ``json`` (default) or ``csv``. Set an explicit
   ``--output-file`` when using CSV so the prefix is clear.

.. option:: --mode MODE

   ``summary`` (default) stores aggregate tables,
   ``per-frame`` stores every selected frame, and ``both`` stores both.
   Plotting and arbitrary notebook slicing require ``per-frame`` or
   ``both``.

.. option:: --frames SPEC

   Select comma-separated indices and half-open ranges, for example
   ``0,10,20:100:5``. A missing start means zero and a missing stop means
   the end. This option overrides ``--start``, ``--stop``, and
   ``--step``.

.. option:: --start INDEX

   First frame index to include. Defaults to zero.

.. option:: --stop INDEX

   Stop before this frame index. Omit it to continue to the end.

.. option:: --step STRIDE

   Process every STRIDE-th frame. The default is one and the value must be
   positive.

.. option:: --topology-mode MODE

   ``reference`` keeps the fixed reference pair map. ``annotate``
   additionally detects pair presence and pairing mode independently in each
   selected frame, without changing calculated topology.

.. option:: --no-warm-start

   Do not seed a minimized global-axis optimizer with the preceding processed
   frame. Use this to diagnose branch flips or frame-order sensitivity.

.. option:: --no-axis-continuity

   Do not align global-axis direction signs to the first processed frame.

.. option:: --verbose

   Print pyCurves calculation logs for every selected frame.

The general trajectory command also accepts the shared frame, axis, fitting,
groove, input-generation, and DSSR options listed for
:ref:`pycurves-cli`. The same incompatibilities apply.

Example
~~~~~~~

.. code-block:: console

   pycurves-md topology.pdb trajectory.xtc \
     --frames 1000:5000:10 \
     --axis-convention global \
     --mode both \
     --output-file dynamics.json

``pycurves-md-batch``: vectorized trajectories
--------------------------------------------------

.. program:: pycurves-md-batch

Use this command only for canonical, combined-strand analyses using standard
frames and the local Curves+ axis. Use ``pycurves-md`` for the global axis,
non-canonical contact-geometry frames, uncombined strands, terminal virtual
levels, DSSR topology, or per-frame topology annotation.

.. code-block:: text

   pycurves-md-batch TOPOLOGY [TRAJECTORY] [OPTIONS]

The batch command shares ``--inp``, ``--output-dir``,
``--output-file``, ``--format``, ``--mode``, and all frame-selection
options with ``pycurves-md``. Its default output file is
``pycurves_trajectory_batch.json``.

.. option:: --batch-size COUNT

   Number of selected frames placed in one vectorized coordinate batch.
   Default: 256. Larger batches can improve throughput but use more memory.
   The value must be positive.

.. option:: --workers COUNT

   Number of process workers for independent batches. Default: one. Benchmark
   two to four workers on long trajectories; process startup, file access, and
   already-parallel numerical work can make extra workers slower.

.. option:: --continuous-strands

   Treat connected helical components as continuous during inferred Curves
   input generation.

.. option:: --duplex-only

   Skip triplex and quadruplex inference and generate two-strand inputs.

.. option:: --altloc CODE

   Select an alternate conformation in the reference structure.

.. option:: --fit, --no-fit

   Override base fitting. Batch mode requires fitting; ``--no-fit`` is
   rejected.

.. option:: --comb, --no-comb

   Override combined-strand analysis. Batch mode requires ``comb=true``.

.. option:: --ends, --no-ends

   Override terminal virtual levels. Batch mode requires ``ends=false``.

.. option:: --grooves, --no-grooves

   Override groove calculation. With neither flag, use the inferred or
   Curves-input ``grv`` value.

.. option:: --frame-convention NAME

   Must resolve to ``standard``.

.. option:: --axis-convention NAME

   Must resolve to ``local``.

.. option:: --axis-weighting, --no-axis-weighting

   Override fitted-pair weighting for smooth-axis construction.

.. option:: --curvesplus-axis-steps

   Include the Curves+ smooth-axis inter-base-pair step table.

.. option:: --fit-quality

   Include vectorized base-fitting RMSD diagnostics.

.. code-block:: console

   pycurves-md-batch topology.pdb trajectory.xtc \
     --frames 0:10000:10 \
     --batch-size 256 \
     --workers 4 \
     --mode summary \
     --output-file dynamics_batch.json

``pycurves-md-plot``: trajectory plots and extraction
---------------------------------------------------------

.. program:: pycurves-md-plot

This command consumes JSON produced with ``pycurves-md --mode per-frame`` or
``--mode both``. Summary-only JSON does not contain the rows needed for
plots.

.. code-block:: text

   pycurves-md-plot JSON_FILE [OPTIONS]

.. option:: --outdir DIRECTORY

   Destination for PNG plots and exported CSV tables. Default:
   ``pycurves_md_plots``.

.. option:: --block NAME

   Select an analysis block; repeat the option to select several. Canonical
   blocks are ``step``, ``local_step``, ``strand_step``,
   ``local_strand_step``, ``base_pair``, ``local_base_pair``,
   ``axis``, ``axis_curvature``, ``axis_bending``,
   ``axis_bending_summary``, ``backbone``, and ``groove``.

   Presets are ``default``, ``global``, ``local``,
   ``intra_pair``, ``curvature``, ``axis_analysis``,
   ``torsions``, and ``all``. Without ``--block``, the
   ``default`` preset is used.

.. option:: --parameter NAME

   Plot only a named parameter; repeat for several. Without it, each block uses
   its standard parameter list.

.. option:: --levels SPEC

   Keep selected Curves levels. Commas and inclusive ranges are supported,
   for example ``4,5,8:12``.

.. option:: --drop-terminal COUNT

   Drop COUNT distinct levels from both ends after other level filtering.

.. option:: --strands SPEC

   Keep selected strand numbers for tables that have a ``strand`` column.

.. option:: --duplex-contains TEXT

   Keep rows whose duplex label contains TEXT literally; regular expressions
   are not used.

.. option:: --time-scale FACTOR

   Multiply source time values by FACTOR for plots. Default: 0.001, which
   converts picoseconds to nanoseconds for typical MD readers.

.. option:: --time-label TEXT

   X-axis label after time scaling. Default: ``time (ns)``.

.. option:: --export-csv

   Write the filtered long-form table for every selected block.

.. option:: --no-plots

   Extract data without creating PNG files. Combine with ``--export-csv``
   for a conversion-only workflow.

.. option:: --stat STATISTIC

   Use ``mean`` (default) or ``median`` as the center of time-series
   plots. Mean uses mean ± standard deviation; median uses the 16th and 84th
   percentiles.

.. option:: --diagnose-outliers

   Write per-parameter outlier CSV files containing available frame, time,
   level, strand, duplex, and step identifiers.

.. option:: --outlier-abs LIMIT

   Also flag finite values whose absolute magnitude is at least LIMIT.

.. option:: --outlier-z THRESHOLD

   Robust modified-z threshold used by outlier diagnostics. Default: 8. A
   nonpositive value disables the robust-z test.

.. code-block:: console

   pycurves-md-plot dynamics.json \
     --block local \
     --levels 4:20 \
     --drop-terminal 1 \
     --export-csv \
     --outdir md_plots

``pycurves-viewer``: interactive HTML
-----------------------------------------

.. program:: pycurves-viewer

The input must be static JSON generated with ``--visualization``.

.. code-block:: console

   pycurves structure.pdb --format json --visualization --output-file viewer.json
   pycurves-viewer viewer.json --output viewer.html

.. option:: -o PATH, --output PATH

   Output HTML path. Without it, ``RESULT.json`` becomes
   ``RESULT.viewer.html``.

.. option:: --structure PATH

   Override the coordinate file named in the JSON. Relative structure paths in
   the JSON are resolved relative to the JSON file.

The generated result is one HTML file containing the structure and pyCurves
results. It loads 3Dmol.js from 3Dmol.org when opened.

``pycurves-pymol``: PyMOL overlay
-------------------------------------

.. program:: pycurves-pymol

The input must contain geometry generated with ``--visualization``.

.. code-block:: console

   pycurves-pymol viewer.json --output viewer.pml

.. option:: -o PATH, --output PATH

   Output PML path. Without it, ``RESULT.json`` becomes ``RESULT.pml``.

.. option:: --scene-prefix NAME

   Prefix for generated PyMOL objects and groups. Unsafe characters are
   converted to underscores.

The PML contains only the pyCurves helical axis, backbone splines, base-pair
blocks, and groove connectors. Load the matching coordinate structure
separately in PyMOL.
