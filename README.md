# pyCurves

pyCurves is a Python implementation and extension of Curves-style nucleic acid
helical analysis. It reads PDB/mmCIF structures, infers DNA/RNA topology, and
calculates global (Curves) or local (Curves+) helical axes, base/base-pair parameters, groove measurements,
backbone torsions, curvature summaries, and machine-readable tables.

This project is under active development. Interfaces and output details may
change before publication. If you use it before publication, please cite this
GitHub repository.

## Installation

Use Python 3.10 or newer. Python 3.12 is recommended.

```bash
conda create -n pycurves python=3.12
conda activate pycurves

git clone https://github.com/JinsenLi/pyCurves
cd pyCurves
pip install .
```

That is enough for static PDB/mmCIF analysis and `.inp` generation.

For trajectory readers, plotting, the batch MD path, and legacy
compatibility helpers, install the single full optional set:

```bash
pip install ".[all]"
```

CPU JAX is installed by default. On GPU clusters, install the matching JAX build
for your CUDA environment before installing pyCurves.

## Quick Start

Run a structure directly to get Curves-like text output:

```bash
pycurves test_data/1A1F_b_c.pdb
pycurves test_data/1A6Y.cif
```

Write JSON or CSV tables:

```bash
pycurves test_data/1A1F_b_c.pdb --format json --output-file 1a1f.json
pycurves test_data/1A1F_b_c.pdb --format csv --output-file 1a1f_tables
```

Generate inferred Curves `.inp` files without running analysis:

```bash
pycurves --generate-inp-only test_data/1OH6.cif test_data/1QNB.cif --output-dir inp
pycurves --inp-only "test_data/*.cif" --output-dir inp
```

Analyze an existing Curves `.inp` file:

```bash
pycurves your_input.inp --pdb your_structure.pdb
```

## Main Commands

```bash
pycurves --help
pycurves-md --help
pycurves-md-batch --help
pycurves-md-plot --help
pycurves-viewer --help
```

Most users start with `pycurves`. The MD, batch, plot, and viewer commands are
available when those workflows are needed.

## What pyCurves Adds

- Gemmi-backed PDB/mmCIF loading with automatic topology inference for DNA/RNA
  duplexes, triplexes, and quadruplexes. 
- Global Curves 5.3-style curvilinear-axis minimization in JAX autograd.
- Tsukuba/Standard reference frame by default.
- Curves-like editable `.inp` input files.
- Non-canonical base pairs use contact-geometry reference frame.
- Curves-like text output, or JSON, and CSV outputs.
- Regular MD trajectory analysis and a superfast vectorized Curves+ batch path.
- Optional HTML viewer/pyMOL payload generation.
- ...

## Important CLI Options

```bash
pycurves [input.pdb|input.cif|input.inp] [options]
pycurves --generate-inp-only [structure ...] [options]
```

Common options:

- `--format {curves,json,csv}`: choose Curves-style text, JSON, or CSV output.
- `--output-file PATH`: write output to a file or CSV prefix.
- `--altloc CODE`: retain blank/shared atoms plus the requested one-character
  alternate-conformation code. If omitted, retain the first listed conformer.
- `--frame-convention standard|legacy`: use Curves+/3DNA-style standard
  frames by default, or choose legacy Curves 5.3-compatible frames.
- `--axis-convention global|local`: choose the global Curves minimization axis
  (the default) or the local Curves+ smooth-axis path.
- `--axis-weighting` / `--no-axis-weighting`: opt in to smoothly downweighting
  requested pairs whose fitted Curves base origins separate by 4--8 A during
  axis construction. The default is the unweighted axis.
- `--generate-inp-only` / `--inp-only`: infer `.inp` files and exit before
  fitting, minimization, or parameter calculation.
- `--continuous-strands`: treat connected split-chain helices as one biological
  helix when possible.
- `--duplex-only`: skip triplex and quadruplex inference and use one-to-one
  duplex inference.
- `--visualization`: include geometry needed by `pycurves-viewer` in JSON.

## Non-Canonical Pairing

pyCurves reports base-pair identity, interacting edges, and cis/trans
orientation separately from the geometry used for shape calculations. Canonical
Watson-Crick pairs keep the selected canonical frame convention; by default this
is the standard reference frame. Explicit `.inp` LW tags and
authoritative reference annotations may select non-canonical calculation
frames.

Generated `.inp` files can carry editable geometry tags such as `[cWW]`,
`[tWW]`, `[cWH]`, `[tWH]`, `[cWS]`, `[tSS]` or `[unresolved]`. Pair geometry in
`.inp` files is represented exclusively by explicit three-character LW tags.

The annotation report is part of the Curves text output, and annotation records
are always included in JSON/CSV results. Pair presence is `present`, `absent`,
or `uncertain`. `pairing_mode` uses controlled values: `watson_crick`,
`reverse_watson_crick`, `hoogsteen`, `reverse_hoogsteen`, `wobble`, or
`other_noncanonical`. A confident `tWW` observation is named
`reverse_watson_crick`. Uncertain calls
do not enter that field: for example, Section M renders
`candidate_mode=hoogsteen` with `classification_status=possible` as “possible
Hoogsteen.” Records keep `observed_lw_family` separate from
`reference_lw_family`, and retain diagnostic flags.

Coordinate-confirmed left-handed cWW pairs use
`helical_context="left_handed_cww"`. Other pairs use an empty string. This is
structural context for the pair; the internal normal-frame branch and sign are
not exposed in the slim report.

## MD Trajectories

Install the full optional set first:

```bash
pip install ".[all]"
```

Run trajectory summaries:

```bash
pycurves-md topology.pdb trajectory.xtc --mode summary --frames 1000:5000:10 --output-file dynamics.json
```

To annotate transient pairing independently in every analyzed frame, use:

```bash
pycurves-md topology.pdb trajectory.xtc --topology-mode annotate --mode both --output-file pairing_dynamics.json
```

`--topology-mode reference` is the default and keeps the reference pair map.
`annotate` reruns coordinate-only pair detection after parameter calculation,
reports missing reference pairs as absent, and adds newly detected pairs with
`reference_pair=false`. The authoritative `base_pair_observations` table
contains pair presence, observed/reference LW family, named pairing mode,
classification status, helical context, and diagnostic flags. Calculation-frame, contact, and
glycosidic details remain internal rather than being repeated per pair. In
JSON, `frame` and `time` belong to the containing frame object; flattened CSV
rows include them as columns.

Axis weighting is independent of topology annotation: it never changes the
requested residue map or assigns a pairing family. It affects both global-axis optimization and Curves+ local smoothing; local base, base-pair,
step, and backbone parameters remain calculated from the fixed input topology.
Axis-dependent rows report `axis_weight`, and geometrically unsupported
base-pair-axis values are null.

Each per-frame `base_pair_observations` row contains exactly these keys:

| Key                      | What users should expect                                                                                                                                       |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pair_id`                | Stable pair identifier made from the two sorted molecular subunit IDs, for example `2:129`.                                                                    |
| `reference_pair`         | `true` for a pair from the reference map; `false` for a pair newly detected in this frame.                                                                     |
| `level`                  | Curves reference level, or `null` for a newly detected pair that has no reference level.                                                                       |
| `residue_1`, `residue_2` | Human-readable chain, residue name, and residue number, for example `B:DT2`.                                                                                   |
| `pair_status`            | `present`, `absent`, or `uncertain` from current-frame coordinate evidence.                                                                                    |
| `pairing_mode`           | Definitive named mode: `watson_crick`, `reverse_watson_crick`, `hoogsteen`, `reverse_hoogsteen`, `wobble`, or `other_noncanonical`; otherwise an empty string. |
| `observed_lw_family`     | Confident current-frame LW family such as `cWW`, `tWW`, `cHW`, or `tWH`; otherwise an empty string.                                                            |
| `reference_lw_family`    | LW family supplied by the reference/input topology; may be populated when `observed_lw_family` is empty.                                                       |
| `candidate_mode`         | Tentative named mode when the evidence supports a possibility but not a definitive assignment; otherwise an empty string.                                      |
| `classification_status`  | `assigned`, `possible`, `unassigned`, or `conflict`.                                                                                                           |
| `helical_context`        | `left_handed_cww` when the pair belongs to a coordinate-confirmed left-handed cWW run; otherwise an empty string.                                              |
| `diagnostic_flags`       | Machine-readable reasons requiring review; normally an empty list.                                                                                             |

Empty `pairing_mode` and `observed_lw_family` values mean **unclassified**, not
noncanonical. For example, `pair_status="present"`,
`reference_lw_family="cWW"`, and `observed_lw_family=""` means that the
reference pair is present but the current coordinates did not support a
definitive observed LW assignment.

The separate `annotations` table does not repeat base-pair rows. A
`modified_base` row contains `annotation_type`, `severity`, `level`,
`location`, `code`, `message`, `residue`, and `parent_base`. A `backbone_link`
row contains `annotation_type`, `severity`, `level`, `location`, `code`,
`message`, `source_subunit`, `target_subunit`, `bond_source`, and `distance`.
The table is empty when neither event type occurs.

In `summary` mode `base_pair_observations` becomes a categorical state profile:
each row reports one pair/status/mode/LW/helical-context combination with
`frame_count` and `frame_fraction`. Identifiers and diagnostic fields are not
numerically averaged. The separate `annotations` table is reserved for modified-base and
backbone-connectivity events. This mode is currently implemented only in
`pycurves-md`, not `pycurves-md-batch`.

Summary tables report numeric columns as `*_mean` and `*_stddev`. Angular columns
use a circular mean and the resultant-length standard deviation
`sqrt(-2 log(R))`, where `R` is the mean resultant length, reported in degrees.

Store both per-frame rows and summary statistics:

```bash
pycurves-md topology.pdb trajectory.dcd --mode both --format json --output-file dynamics_full.json
```

For canonical two-strand local-axis (Curves+)/standard-frame trajectories, the
vectorized batch path can be much faster:

```bash
pycurves-md-batch topology.pdb trajectory.xtc --axis-convention local --frame-convention standard --batch-size 256 --workers 4 --mode summary --output-file dynamics_batch.json
```

`--workers` processes independent coordinate batches concurrently while keeping
trajectory reading and output ordering in the parent process. The default is 1;
benchmark 2--4 workers for long trajectories because short runs, slow storage,
and already-parallel groove calculations may not benefit.

Use `pycurves-md` for global-axis minimization, non-canonical contact-geometry
frames, `--no-comb`, or `--ends`.

## MD Analysis In Notebooks

For exploratory MD work, use the Python helpers directly instead of writing JSON
and calling `pycurves-md-plot`. Use `mode="per-frame"` or `mode="both"` when
you want to slice levels, strands, time windows, or individual parameters in a
notebook. `mode="summary"` is compact, but it does not keep per-frame rows.

```python
from pycurves_md import analyze_trajectory
from pycurves_md_plot import (
    add_time_axis,
    extract_block,
    extract_summary_block,
    filter_rows,
    parameter_timeseries,
    pivot_parameter_matrix,
    wrap_degrees,
)

payload = analyze_trajectory(
    "topology.pdb",
    "trajectory.xtc",
    frames="1000:5000:10",
    mode="both",
    frame_convention="standard",
    axis_convention="local",
)

# Long-form DataFrames from the in-memory payload.
steps = extract_block(payload, "step")
base_pairs = extract_block(payload, "base_pair")
grooves = extract_block(payload, "groove")

# Work on a subsection, then build custom plots/tables.
mid_steps = filter_rows(steps, level=range(5, 16), drop_terminal=1)
mid_steps = add_time_axis(mid_steps, time_scale=0.001, time_label="time (ns)")
mid_steps["twist"] = wrap_degrees(mid_steps["twist"])

twist_series = parameter_timeseries(
    mid_steps,
    "twist",
    time_column="plot_time",
    aggregate=True,
)
twist_heatmap = pivot_parameter_matrix(
    mid_steps,
    "twist",
    index_column="plot_time",
    column="level",
)

# Summary tables are available when mode="summary" or mode="both".
bp_summary = extract_summary_block(payload, "base_pair")
```

For canonical two-strand standard-frame analyses, the vectorized batch path has
a matching notebook helper:

```python
from pycurves_md_batch import analyze_trajectory_batch
from pycurves_md_plot import extract_block

payload = analyze_trajectory_batch(
    "topology.pdb",
    "trajectory.xtc",
    frames="0:10000:10",
    batch_size=256,
    mode="per-frame",
)
steps = extract_block(payload, "step")
```

Use `analyze_trajectory` rather than the batch helper when you need global-axis
minimization, non-canonical contact-geometry frames, `comb=False`, or terminal
end-level handling.

## Plotting And Viewer

After `pip install ".[all]"`, plot MD JSON output:

```bash
pycurves-md-plot dynamics.json --outdir md_plots
pycurves-md-plot dynamics.json --export-csv --no-plots --outdir raw_csvs
```

Generate a self-contained HTML viewer from visualization JSON:

```bash
pycurves structure.pdb --format json --visualization --output-file viewer.json
pycurves-viewer viewer.json --output viewer.html
```

Generate a PyMOL inspection scene from the same visualization JSON:

```bash
pycurves-pymol viewer.json --output viewer.pml
```

Open the scene in PyMOL with `@viewer.pml`. The PML is a structure-free
overlay containing only the helical axis, backbone splines, base-pair color
blocks, and groove width connector lines. The PyMOL object panel exposes
individual axis points, backbone strands, base-pair blocks, and groove lines
under grouped dropdowns, so they can be toggled one by one. Load the source
PDB/mmCIF separately if you want to inspect coordinates underneath the
pyCurves geometry.

## Python API

```python
from pycurves_lib.curves_wrapper import CurvesWrapper

runner = CurvesWrapper.from_file("test_data/1A1F_b_c.pdb")
runner.analyze()
json_text = runner.output(fmt="json")
```

Generate `.inp` files programmatically without analysis:

```python
from pycurves_lib.curves_wrapper import CurvesWrapper

runner = CurvesWrapper(
    pdbfile="test_data/1QNB.cif",
    output_dir="inferred_inputs",
    auto_generate_inp=False,
)
inp_files = runner.generate_inp(prefix="1QNB_auto")
```

## Output

JSON output uses the `pycurves-slim-v2` schema. It contains metadata plus flat
`dataframes` records for local/global parameters, backbone, groove, curvature,
and annotations. Gapped or uncomputed backbone positions are kept with `null`
values so sequence-indexed tables stay aligned; each row also reports `valid`,
`status`, `warnings`, and `missing_parameters`. Residue names and chain IDs are
stored separately. CSV output writes one file per dataframe using the
`--output-file` value as the prefix.

## Code Guides

For a file-by-file map of the top-level commands, see
[`SCRIPT_REFERENCE.md`](SCRIPT_REFERENCE.md). Each Python package directory has
the same two-level documentation pattern: a short `README.md` and a detailed
`SCRIPT_REFERENCE.md`.
