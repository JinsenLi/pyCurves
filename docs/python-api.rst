Python API
==========

The public API is designed for notebooks and scripts that need results in
memory. The high-level entry points mirror the installed commands:

.. autosummary::
   :nosignatures:

   pycurves_lib.curves_wrapper.CurvesWrapper
   pycurves_md.analyze_trajectory
   pycurves_md_batch.analyze_trajectory_batch
   pycurves_viewer.write_viewer
   pycurves_pymol.write_pymol_scene

Static structure analysis
-------------------------

Use :class:`~pycurves_lib.curves_wrapper.CurvesWrapper` for one structure or
one explicit Curves input.

Basic lifecycle
~~~~~~~~~~~~~~~

.. code-block:: python

   from pycurves_lib.curves_wrapper import CurvesWrapper

   runner = CurvesWrapper.from_file("structure.cif")
   runner.analyze()

   json_text = runner.output(fmt="json")
   features = runner.getFeatures()

``from_file()`` accepts either a coordinate structure or a Curves
``.inp``. For coordinates it infers and writes an input immediately. For an
``.inp``, it reads the coordinate path from the input's ``file=`` value.

``analyze()`` performs the calculation and returns the same runner.
``output()`` requires a completed analysis, returns the rendered string,
and also prints it when ``file=None``. Use ``write_output()`` to write
directly.

For the shortest run-and-render workflow:

.. code-block:: python

   runner = CurvesWrapper.from_file("structure.pdb")
   runner.run(output=False)
   runner.write_output("structure.json", fmt="json")

Constructor arguments
~~~~~~~~~~~~~~~~~~~~~

Prefer keyword arguments when constructing the class directly:

.. code-block:: python

   runner = CurvesWrapper(
       pdbfile="structure.cif",
       output_dir="inferred_inputs",
       frame_convention="standard",
       axis_convention="global",
       axis_weighting=False,
   )

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Behavior
   * - ``pdbfile``
     - PDB/mmCIF coordinate path. Required unless ``inpfile`` is supplied
       and resolves its own ``file=`` entry.
   * - ``inpfile``
     - Existing Curves topology/configuration. Mutually exclusive with
       ``dssr_json``.
   * - ``output_dir``
     - Destination for inferred Curves inputs. Default: current directory.
   * - ``continuous_strands``
     - Join connected split-chain helical components during inference.
   * - ``duplex_only``
     - Disable triplex/quadruplex inference.
   * - ``altloc``
     - One-character alternate conformation, or ``None`` for the first
       conformer in file order.
   * - ``frame_convention``
     - ``"standard"`` (default) or ``"legacy"``.
   * - ``axis_convention``
     - ``"global"`` (default) or ``"local"``. Local forces standard
       frames.
   * - ``axis_weighting``
     - ``True`` or ``False`` overrides axis weighting; ``None``
       keeps the configuration default.
   * - ``fit_override``
     - Override the Curves ``fit`` flag.
   * - ``grv_override``
     - Override the Curves ``grv`` (groove) flag.
   * - ``mini_override``
     - Override the Curves ``mini`` flag.
   * - ``comb_override``
     - Override the Curves ``comb`` flag.
   * - ``ends_override``
     - Override terminal virtual end levels.
   * - ``auto_generate_inp``
     - Infer an input during construction when no ``inpfile`` is supplied.
       Set false when you intend to call ``generate_inp()`` explicitly.
   * - ``dssr_json``
     - Optional DSSR topology report.
   * - ``dssr_unit``
     - Unit selector such as ``"stem:1"`` when a DSSR report is ambiguous.

See :doc:`analysis-options` for the scientific effect of these arguments.

Generating Curves inputs
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   runner = CurvesWrapper(
       pdbfile="structure.cif",
       output_dir="inferred_inputs",
       auto_generate_inp=False,
   )
   paths = runner.generate_inp(prefix="structure_auto")

``generate_inp()`` returns a list of written paths because one coordinate
file can contain multiple inferred units.

DSSR topology
~~~~~~~~~~~~~

.. code-block:: python

   runner = CurvesWrapper.from_file(
       "structure.cif",
       dssr_json="structure-dssr.json",
       dssr_unit="stem:1",
   )
   runner.analyze()

When no unit selector is required, inspect the available units before analysis:

.. code-block:: python

   runner = CurvesWrapper(
       pdbfile="structure.cif",
       dssr_json="structure-dssr.json",
       auto_generate_inp=False,
   )
   units = runner.list_dssr_units()

Result access
~~~~~~~~~~~~~

``getFeatures()`` returns NumPy-backed internal feature arrays and selected
derived records. For stable serialization and labelled tables, prefer
:class:`~pycurves_lib.io.curves_output.CurvesOutputFormatter`:

.. code-block:: python

   from pycurves_lib.io.curves_output import CurvesOutputFormatter

   runner = CurvesWrapper.from_file("structure.pdb").analyze()
   formatter = CurvesOutputFormatter(runner)

   frames = formatter.get_dataframes()
   steps = frames["local_inter_base_pair"]
   json_text = formatter.render("json")

The dataframe keys and JSON layout are listed in :doc:`outputs`.

Class reference
~~~~~~~~~~~~~~~

.. autoclass:: pycurves_lib.curves_wrapper.CurvesWrapper

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.from_file

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.analyze

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.analyze_molecule

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.run

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.output

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.write_output

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.getFeatures

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.generate_inp

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.list_dssr_units

.. automethod:: pycurves_lib.curves_wrapper.CurvesWrapper.normalize_conventions

.. autoclass:: pycurves_lib.io.curves_output.CurvesOutputFormatter

.. automethod:: pycurves_lib.io.curves_output.CurvesOutputFormatter.get_dataframes

.. automethod:: pycurves_lib.io.curves_output.CurvesOutputFormatter.render

.. automethod:: pycurves_lib.io.curves_output.CurvesOutputFormatter.render_csv

.. automethod:: pycurves_lib.io.curves_output.CurvesOutputFormatter.render_curves_text

.. automethod:: pycurves_lib.io.curves_output.CurvesOutputFormatter.render_json

General trajectory API
----------------------

:func:`pycurves_md.analyze_trajectory` is the in-memory equivalent of
``pycurves-md``.

.. code-block:: python

   from pycurves_md import analyze_trajectory

   payload = analyze_trajectory(
       "topology.pdb",
       "trajectory.xtc",
       frames="1000:5000:10",
       mode="both",
       frame_convention="standard",
       axis_convention="local",
   )

Important arguments
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Behavior
   * - ``topology_file``
     - Reference PDB/mmCIF structure.
   * - ``trajectory_file``
     - Trajectory path, or ``None`` for a multi-model PDB.
   * - ``inpfile``
     - Optional fixed Curves topology.
   * - ``frames``
     - Comma/range selector such as ``"0,10,20:100:5"``; overrides
       ``start``, ``stop``, and ``step``.
   * - ``start, stop, step``
     - Half-open range selection when ``frames`` is absent.
   * - ``mode``
     - ``"per-frame"``, ``"summary"``, or ``"both"``.
   * - ``fit, grooves, mini, comb, ends``
     - Boolean Curves overrides. ``None`` keeps inferred/input values.
   * - ``topology_mode``
     - ``"reference"`` or per-frame ``"annotate"``.
   * - ``verbose``
     - Echo analysis logs for each selected frame.
   * - ``warm_start``
     - Seed minimized global-axis optimization from the preceding frame.
   * - ``axis_continuity``
     - Keep global-axis signs aligned to the first processed frame.
   * - ``dssr_json, dssr_unit``
     - Optional DSSR-supplied reference topology.

The return value is a JSON-compatible dictionary. ``mode="per-frame"``
adds ``payload["frames"]``, ``mode="summary"`` adds
``payload["summary"]``, and ``"both"`` adds both.

.. autofunction:: pycurves_md.analyze_trajectory

.. autoclass:: pycurves_md.MDTrajectoryAnalyzer

.. automethod:: pycurves_md.MDTrajectoryAnalyzer.run

.. automethod:: pycurves_md.MDTrajectoryAnalyzer.write_csv

Vectorized trajectory API
-------------------------

:func:`pycurves_md_batch.analyze_trajectory_batch` is the notebook-facing
batch engine:

.. code-block:: python

   from pycurves_md_batch import analyze_trajectory_batch

   payload = analyze_trajectory_batch(
       "topology.pdb",
       "trajectory.xtc",
       frames="0:10000:10",
       batch_size=256,
       workers=4,
       mode="summary",
       grooves=True,
   )

This function supports only standard frames, the local axis, fitted bases,
combined strands, and no terminal virtual levels. ``batch_size`` controls
memory/throughput. ``workers`` controls process-level batch concurrency.
``curvesplus_axis_steps=True`` includes the smooth-axis step table and
``fit_quality=True`` includes fitting RMSD diagnostics.

.. autofunction:: pycurves_md_batch.analyze_trajectory_batch

Trajectory dataframe helpers
----------------------------

Use ``mode="per-frame"`` or ``mode="both"`` before extracting
time-resolved data.

.. code-block:: python

   from pycurves_md_plot import (
       add_time_axis,
       extract_block,
       filter_rows,
       parameter_timeseries,
       pivot_parameter_matrix,
   )

   steps = extract_block(payload, "local_step")
   steps = filter_rows(steps, level=range(5, 16), drop_terminal=1)
   steps = add_time_axis(steps, time_scale=0.001, time_label="time (ns)")

   twist = parameter_timeseries(
       steps,
       "twist",
       time_column="plot_time",
       aggregate=True,
   )
   heatmap = pivot_parameter_matrix(
       steps,
       "twist",
       index_column="plot_time",
       column="level",
   )

.. autofunction:: pycurves_md_plot.load_trajectory_payload

.. autofunction:: pycurves_md_plot.iter_pycurves_frames

.. autofunction:: pycurves_md_plot.extract_table

.. autofunction:: pycurves_md_plot.extract_block

.. autofunction:: pycurves_md_plot.extract_summary_table

.. autofunction:: pycurves_md_plot.extract_summary_block

.. autofunction:: pycurves_md_plot.filter_rows

.. autofunction:: pycurves_md_plot.wrap_degrees

.. autofunction:: pycurves_md_plot.add_time_axis

.. autofunction:: pycurves_md_plot.parameter_timeseries

.. autofunction:: pycurves_md_plot.pivot_parameter_matrix

.. autofunction:: pycurves_md_plot.outlier_rows

Viewer and PyMOL writers
------------------------

These functions consume an already generated visualization payload; they do not
rerun the scientific analysis.

``write_viewer()`` reads a pyCurves JSON result, resolves the structure from
``structure_file`` or the JSON's ``inputs.pdbfile``, and writes one HTML file.
``render_viewer_html()`` is the lower-level renderer for callers that already
have a loaded result dictionary and coordinate text.

``write_pymol_scene()`` validates the same visualization payload and writes a
``.pml`` overlay script. ``render_pymol_script()`` returns that script as a
string instead; ``scene_prefix`` controls the generated PyMOL object names.

.. autofunction:: pycurves_viewer.write_viewer

.. autofunction:: pycurves_viewer.render_viewer_html

.. autofunction:: pycurves_pymol.write_pymol_scene

.. autofunction:: pycurves_pymol.render_pymol_script
