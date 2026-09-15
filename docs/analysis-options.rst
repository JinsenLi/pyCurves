Analysis options and conventions
================================

This page explains the options shared by the command-line and Python
interfaces. CLI spelling uses hyphens, while Python functions use underscores.
For example, ``--axis-convention local`` corresponds to
``axis_convention="local"``.

Analysis flow
-------------

For a coordinate input, pyCurves performs these stages:

#. Load coordinates and select alternate conformations.
#. Infer a strand/base-pair topology, or read it from Curves input or DSSR JSON.
#. Fit reference bases to the selected coordinates.
#. Construct local base and base-pair frames.
#. Build either a global Curves axis or a local Curves+ axis.
#. Calculate helical, backbone, groove, curvature, and annotation tables.
#. Render Curves-style text, JSON, or CSV.

An option can affect only one stage. For example, axis weighting changes axis
construction but never adds, removes, or reclassifies a requested base pair.

Topology sources
----------------

Inferred topology
~~~~~~~~~~~~~~~~~

Passing a PDB or mmCIF structure without ``--dssr-json`` or an explicit
``.inp`` file selects coordinate-based topology inference. pyCurves can infer
duplex, triplex, and quadruplex arrangements. If no qualifying duplex is found,
the inferrer can produce a single-strand topology.

``continuous_strands=False`` keeps separated helical components separate.
Set ``continuous_strands=True`` (CLI: ``--continuous-strands``) when
connected split-chain helices should be treated as one biological helix.

``duplex_only=True`` (CLI: ``--duplex-only``) disables triplex and
quadruplex inference and uses the one-to-one duplex path.

Explicit Curves input
~~~~~~~~~~~~~~~~~~~~~

A Curves ``.inp`` file supplies the strand count, strand directions, residue
maps, analysis flags, and optional initial helical values. Pass the coordinate
file separately if the input's ``file=`` entry does not resolve:

.. code-block:: console

   pycurves helix.inp --pdb structure.pdb

Topology rows accept inclusive range notation such as ``1:12``. Pair
geometry tags use Leontis--Westhof notation, for example ``[cWW]``,
``[tWH]``, and ``[cSS]``; ``[unresolved]`` preserves an explicitly
unresolved pair.

DSSR JSON topology
~~~~~~~~~~~~~~~~~~

``--dssr-json REPORT`` uses a DSSR report for residue pairing and unit
selection while reading coordinates from the PDB/mmCIF input. It cannot be
combined with an explicit Curves input or ``--continuous-strands``.

If the report contains one valid unit, it is selected automatically. Otherwise
use ``--dssr-unit KIND:N``, where common selectors include
``multiplet:1``, ``stem:1``, ``helix:2``, and ``pairs:1``.

Base fitting
------------

``fit`` controls least-squares fitting of packaged base references to the
observed base atoms.

* ``None`` (no CLI override) uses the value inferred or read from the
  Curves input.
* ``True`` / ``--fit`` enables fitting.
* ``False`` / ``--no-fit`` disables fitting.

The vectorized batch trajectory engine requires fitting and rejects
``--no-fit``.

Reference-frame convention
--------------------------

standard
   Tsukuba/Curves+/3DNA-style base reference frames. This is the default.

legacy
   Curves 5.3-compatible base reference frames. Use this when direct numerical
   compatibility with legacy Curves frames is required.

Select the convention with ``--frame-convention standard|legacy`` or the
``frame_convention`` Python argument.

The local Curves+ axis is defined on standard frames. Selecting
``axis_convention="local"`` therefore forces
``frame_convention="standard"``, even if legacy was also requested.

Axis convention
---------------

global
   The default. Uses the Curves curvilinear-axis path. When minimization is
   enabled, JAX optimizes the helical-axis variables.

local
   Uses the Curves+ local/smooth axis. This mode disables global-axis
   minimization and trajectory warm starts, and forces standard base frames.

For a Curves 5.3 comparison, use both legacy frames and the global axis:

.. code-block:: console

   pycurves structure.pdb \
     --frame-convention legacy \
     --axis-convention global

Axis minimization
-----------------

``mini`` affects the global-axis path.

* ``None`` reads ``mini`` from the Curves input.
* ``True`` / ``--mini`` runs global-axis minimization.
* ``False`` / ``--no-mini`` constructs the axis once from the input
  helical values and continues with downstream calculations.

Local-axis mode always disables minimization. Legacy Curves
``line=.t.`` and ``dinu=.t.`` minimization modes are currently rejected
because the JAX optimizer does not implement their alternate objectives.

Axis weighting
--------------

``axis_weighting=True`` (CLI: ``--axis-weighting``) smoothly downweights
requested pairs whose fitted Curves base origins separate by 4--8 Å. Weighting
applies to global optimization and local-axis smoothing when combined-strand
analysis is active.

Weighting does **not**:

* change the input topology;
* create or remove a pair;
* assign a pairing family; or
* change local base, base-pair, step, or backbone parameters.

Axis-dependent rows report ``axis_weight``. Geometrically unsupported
base-pair-axis values are emitted as null.

Grooves, strand combination, and terminal levels
------------------------------------------------

``grooves``
   Controls minor/major groove calculation. ``None`` uses the inferred or
   Curves-input ``grv`` value. Curves 5.3-style groove analysis is duplex
   only; disable it for explicit three- or four-strand inputs.

``comb``
   Controls combined-strand analysis. ``None`` uses the inferred or
   Curves-input ``comb`` value. The vectorized batch engine requires
   combined analysis.

``ends``
   Controls Curves terminal virtual end levels. ``None`` uses the inferred
   or Curves-input value. The vectorized batch engine requires this to be
   false.

The CLI uses paired boolean flags: ``--grooves/--no-grooves``,
``--comb/--no-comb``, and ``--ends/--no-ends``.

.. _alternate-conformations:

Alternate conformations
-----------------------

``--altloc A`` or ``altloc="A"`` retains blank/shared atoms plus the
requested one-character alternate-conformation code. Without an explicit
selection, Gemmi keeps the first conformer listed in the file. That selection
is based on file order, not highest occupancy.

Trajectory topology mode
------------------------

The regular trajectory engine has two annotation modes:

reference
   Keep the reference pair map for all frames. This is the default.

annotate
   After calculating a frame from the fixed reference topology, independently
   detect coordinate-supported pair presence and pairing mode. Missing
   reference pairs are reported as absent and new pairs use
   ``reference_pair=false``.

Per-frame annotation does not change the pair map used to calculate geometric
parameters.

Trajectory continuity controls
------------------------------

For a minimized global axis, the regular trajectory engine uses two continuity
features by default:

warm start
   Seed each frame's optimizer from the preceding processed frame. Disable with
   ``--no-warm-start`` or ``warm_start=False`` when diagnosing
   optimizer branch changes.

axis continuity
   Align global-axis direction signs with the first processed frame. Disable
   with ``--no-axis-continuity`` or ``axis_continuity=False``.

Neither feature is used by the local-axis path.
