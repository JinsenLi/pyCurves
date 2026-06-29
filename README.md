# pyCurves

pyCurves is a Python implementation and extension of Curves-style nucleic acid
helical analysis. It reads PDB/mmCIF structures, infers DNA/RNA topology, and
calculates helical axes, base/base-pair parameters, groove measurements,
backbone torsions, curvature summaries, and machine-readable tables.

This project is under active development. Interfaces and output details may
change before publication. If you use it before publication, please cite this
GitHub repository.

## Install

Use Python 3.10 or newer. Python 3.12 is recommended.

```bash
conda create -n pycurves python=3.12
conda activate pycurves

git clone https://github.com/JinsenLi/pyCurves
cd pyCurves
pip install .
```

That is enough for static PDB/mmCIF analysis and `.inp` generation.

For trajectory readers, plotting, the experimental batch MD path, and legacy
compatibility helpers, install the single full optional set:

```bash
pip install ".[all]"
```

CPU JAX is installed by default. On GPU clusters, install the matching JAX build
for your CUDA environment before installing pyCurves.

## Quick Examples

Run a structure directly:

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

Use Curves+/3DNA-style local frames:

```bash
pycurves test_data/1A1F_b_c.pdb --frame-convention standard --format json --output-file 1a1f_standard.json
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

- PDB/mmCIF loading with automatic topology inference for DNA/RNA structures.
- Legacy Curves 5.3-style curvilinear-axis minimization in Python/JAX.
- Standard Curves+/3DNA-compatible local frame mode.
- Non-canonical-aware frame selection for mismatches, Hoogsteen/reverse
  Hoogsteen contacts, and other edge-pair geometries.
- Editable geometry markers in generated `.inp` files, for example `[cWW]`,
  `[tWH]`, and `[tSS]`.
- Text, JSON, and CSV outputs for local/global helical parameters, grooves,
  backbone torsions, curvature, and annotations.
- MD trajectory analysis and an experimental vectorized Curves+ batch path.
- Optional HTML viewer payload generation.

## Important CLI Options

```bash
pycurves [input.pdb|input.cif|input.inp] [options]
pycurves --generate-inp-only [structure ...] [options]
```

Common options:

- `--format {curves,json,csv}`: choose Curves-style text, JSON, or CSV output.
- `--output-file PATH`: write output to a file or CSV prefix.
- `--frame-convention legacy|standard`: choose legacy Curves frames or
  Curves+/3DNA-style standard frames.
- `--axis-convention legacy|curvesplus`: choose the legacy pyCurves/JAX axis or
  the Curves+ smooth-axis path.
- `--generate-inp-only` / `--inp-only`: infer `.inp` files and exit before
  fitting, minimization, or parameter calculation.
- `--continuous-strands`: treat connected split-chain helices as one biological
  helix when possible.
- `--fit`, `--grooves`, `--mini`, `--comb`, and `--ends`: override inferred
  analysis flags. Each also accepts the `--no-*` form.
- `--no-annotations`: suppress the pyCurves annotation report.
- `--visualization`: include geometry needed by `pycurves-viewer` in JSON.

## Non-Canonical Pairing

pyCurves detects base-pair identity, interacting edges, and cis/trans orientation
so that the right local frames and strand-direction signs can be used in shape
calculations. Canonical Watson-Crick pairs keep the canonical legacy or standard
frames. Non-canonical pairs use contact-geometry frames only when the observed
edge/contact evidence is strong enough.

Generated `.inp` files can carry editable geometry tags such as `[cWW]`,
`[tWW]`, `[cWH]`, `[tWH]`, `[cWS]`, or `[tSS]`. Mismatches are still reported as
mismatches even when they have a clear edge-contact geometry.

The annotation report and JSON/CSV tables include the detected pair class,
observed edge/orientation tag, source mmCIF pair records when available, and
warnings for source pairs that do not belong to the current generated `.inp`
topology.

## MD Trajectories

Install the full optional set first:

```bash
pip install ".[all]"
```

Run trajectory summaries:

```bash
pycurves-md topology.pdb trajectory.xtc --mode summary --frames 1000:5000:10 --output-file dynamics.json
```

Store both per-frame rows and summary statistics:

```bash
pycurves-md topology.pdb trajectory.dcd --mode both --format json --output-file dynamics_full.json
```

For canonical two-strand Curves+/standard-frame trajectories, the experimental
batch path can be 100x faster:

```bash
pycurves-md-batch topology.pdb trajectory.xtc --axis-convention curvesplus --frame-convention standard --batch-size 256 --mode summary --output-file dynamics_batch.json
```

Use `pycurves-md` for legacy-axis minimization, non-canonical contact-geometry
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
    axis_convention="curvesplus",
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

Use `analyze_trajectory` rather than the batch helper when you need legacy-axis
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

runner = CurvesWrapper.from_file("test_data/1A1F_b_c.pdb", frame_convention="standard")
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

JSON output uses the `pycurves-slim-v1` schema. It contains metadata plus flat
`dataframes` records for local/global parameters, backbone, groove, curvature,
and annotations. Gapped or uncomputed positions are kept with `null` values so
sequence-indexed tables stay aligned. CSV output writes one file per dataframe
using the `--output-file` value as the prefix.
