Output formats and schemas
==========================

pyCurves exposes the same calculated records as Curves-style text, JSON, Pandas
dataframes, and CSV files. JSON is the best interchange format; dataframes are
the most convenient in Python.

Static JSON
-----------

``CurvesWrapper.output(fmt="json")`` and
``CurvesOutputFormatter.render_json()`` return a
``pycurves-slim-v2`` document with these top-level keys:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Contents
   * - ``program``
     - Producer name.
   * - ``format``
     - Schema identifier, currently ``pycurves-slim-v2``.
   * - ``frame_convention``
     - Selected base-frame name, compatible tools, reference source, and axis
       convention.
   * - ``analysis_options``
     - Effective combined-strand, groove, terminal-level, minimization, axis,
       weighting, and alternate-conformation settings.
   * - ``inputs``
     - Coordinate path, Curves input path, and generated input paths. DSSR
       fields are added when DSSR supplied the topology.
   * - ``sequence``
     - Strand count, level count, one-letter sequences, and residue identities.
   * - ``dataframes``
     - Named lists of row dictionaries described below.
   * - ``rebuild_frame_mappings``
     - Present with visualization output; transforms needed to rebuild
       interaction frames from standard base frames.
   * - ``visualization``
     - Present only when visualization geometry was requested.

Static dataframe keys
---------------------

Not every analysis produces every table. Axis-specific tables appear only for
the relevant convention, and groove tables appear only when groove calculation
is enabled.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Key
     - Contents
   * - ``global_base_axis``
     - Base displacement, inclination, and tip relative to the global Curves
       axis.
   * - ``global_base_pair_axis``
     - Base-pair displacement and orientation relative to the global axis.
   * - ``curvesplus_base_pair_axis``
     - Base-pair displacement and orientation relative to the local Curves+
       smooth axis.
   * - ``global_base_base``
     - Helical intra-base-pair parameters.
   * - ``local_base_base``
     - Local intra-base-pair shear, stretch, stagger, buckle, propeller, and
       opening.
   * - ``global_inter_base``
     - Global strand-by-strand inter-base step parameters.
   * - ``local_inter_base``
     - Local strand-by-strand inter-base step parameters.
   * - ``global_inter_base_pair``
     - Global base-pair step parameters.
   * - ``local_inter_base_pair``
     - Local base-pair step shift, slide, rise, tilt, roll, and twist.
   * - ``backbone``
     - Backbone torsions, sugar-pucker descriptors, residue identity, validity,
       status, warnings, and missing parameters.
   * - ``global_axis_curvature``
     - Per-level axis curvature/path records.
   * - ``global_axis_bending``
     - Per-level offsets and local bending direction.
   * - ``global_axis_bending_summary``
     - Per-strand path length, end-to-end distance, shortening, and overall
       bend angles.
   * - ``groove``
     - Minor/major width and depth plus diameter, flattened by level and
       sub-level in Pandas/CSV output.
   * - ``base_pair_observations``
     - Pair presence, observed/reference LW family, named mode, classification
       state, helical context, and diagnostics.
   * - ``annotations``
     - Modified-base and backbone-connectivity events.
   * - ``dssr_source_pairs``
     - DSSR source pair records; present in dataframe output when DSSR
       provenance is available.

Missing values
--------------

Gapped or uncomputed backbone positions remain in sequence-indexed tables with
null values. This keeps row positions aligned with the requested sequence.
Inspect the row's ``valid``, ``status``, ``warnings``, and
``missing_parameters`` fields before treating a numeric null as an error.

CSV output
----------

For static CLI output, ``--output-file PREFIX`` produces one
``PREFIX_TABLE.csv`` file per dataframe.

For trajectory CSV, summary tables use
``PREFIX_TABLE_summary.csv`` and per-frame tables use
``PREFIX_TABLE_frames.csv``. Frame and time are flattened into columns.

General trajectory JSON
-----------------------

``pycurves-md`` and :func:`pycurves_md.analyze_trajectory` return
``pycurves-trajectory-slim-v2``. Common top-level keys are:

``inputs``
   Topology, reference topology, trajectory, Curves input, and generated input
   paths.

``analysis_options``
   Effective analysis settings.

``selection``
   Frame selector, processed-frame count, and continuity settings.

``frame_convention``
   Frame and axis conventions.

``frames``
   Present for ``per-frame`` and ``both``. Each item contains
   ``frame``, ``time``, and ``dataframes``.

``summary``
   Present for ``summary`` and ``both``.

Vectorized trajectory JSON
--------------------------

The batch engine uses ``pycurves-trajectory-batch-curvesplus-v1``. It has
the same high-level ``inputs``, ``analysis_options``, ``selection``,
``frame_convention``, ``frames``, and ``summary`` organization, but
records the fixed standard-frame/local-axis constraints and batch settings.

Summary statistics
------------------

Numeric summary columns are emitted as ``NAME_count``, ``NAME_mean``,
and ``NAME_stddev``. Angular columns use a circular mean and
resultant-length standard deviation in degrees; they are not averaged as linear
numbers.

``base_pair_observations`` is categorical. Summary rows count each
pair/status/mode/LW/helical-context combination using ``frame_count`` and
``frame_fraction``.

Base-pair observations
----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Meaning
   * - ``pair_id``
     - Stable identifier made from the two sorted molecular subunit IDs.
   * - ``reference_pair``
     - True for a pair from the reference map; false for a newly detected
       per-frame pair.
   * - ``level``
     - Curves reference level, or null when a newly detected pair has no
       reference level.
   * - ``residue_1``, ``residue_2``
     - Human-readable chain, residue name, and residue number.
   * - ``pair_status``
     - ``present``, ``absent``, or ``uncertain``.
   * - ``pairing_mode``
     - Definitive named mode, or an empty string when unassigned.
   * - ``observed_lw_family``
     - Confident current-frame LW family such as ``cWW`` or ``tWH``.
   * - ``reference_lw_family``
     - LW family supplied by the reference/input topology.
   * - ``candidate_mode``
     - Tentative named mode when evidence is insufficient for assignment.
   * - ``classification_status``
     - ``assigned``, ``possible``, ``unassigned``, or ``conflict``.
   * - ``helical_context``
     - ``left_handed_cww`` for a coordinate-confirmed left-handed cWW run;
       otherwise empty.
   * - ``diagnostic_flags``
     - Machine-readable reasons requiring review; normally empty.

Empty observed mode or LW-family strings mean *unclassified*, not
non-canonical.

Annotation rows
---------------

``annotations`` does not duplicate base-pair rows.

* A ``modified_base`` row contains annotation type, severity, level,
  location, code, message, residue, and parent base.
* A ``backbone_link`` row contains annotation type, severity, level,
  location, code, message, source/target subunits, bond source, and distance.

The table is empty when neither event type occurs.
