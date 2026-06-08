# pyCurves

pyCurves is a Python implementation and extension of Curves-style nucleic acid
helical analysis. It reads PDB/mmCIF structures, infers DNA/RNA topology,
calculates helical axes, base/base-pair parameters, groove widths, backbone
torsions, curvature summaries, and exports the results as Curves-style text,
slim JSON, or CSV tables.

The default mode keeps the legacy Curves 5.3 curvilinear-axis minimization, but
pyCurves also supports standard Curves+/3DNA-compatible local reference frames
for reproducibility and downstream comparison.

## Quick Start

(Optional) It is highly recommended to use a standalone environment to install pyCurves.

```
conda create -n pycurves python=3.12
conda activate pycurves
```

Install from the source tree:

```bash
git clone https://github.com/JinsenLi/pyCurves
pip install .
```

Run a structure directly from PDB or mmCIF:

```bash
pycurves test_data/1A1F_b_c.pdb
pycurves test_data/1A6Y.cif
```

Save a machine-readable result:

```bash
pycurves test_data/1A1F_b_c.pdb --format json --output-file 1a1f.json
```

Generate CSV tables:

```bash
pycurves test_data/1A1F_b_c.pdb --format csv --output-file 1a1f_tables
```

Use standard Curves+/3DNA-style local frames:

```bash
pycurves test_data/1A1F_b_c.pdb \
  --frame-convention standard \
  --format json \
  --output-file 1a1f_standard.json
```

Create an interactive HTML viewer:

```bash
pycurves test_data/1BNK_b_c.pdb \
  --format json \
  --visualization \
  --output-file 1bnk_viewer_payload.json

pycurves-viewer 1bnk_viewer_payload.json --output 1bnk_viewer.html
```

If you already have a legacy Curves `.inp` file, pass it directly. Add `--pdb`
when the coordinate file is not specified inside the input file:

```bash
pycurves your_input.inp --pdb your_structure.pdb
```

## What pyCurves Adds

* Automatic topology inference for PDB/mmCIF files, including split chains,
  modified bases, mismatches, gaps, and non-canonical pairing annotations.
* Legacy Curves 5.3 global curvilinear-axis minimization in Python/JAX.
* Standard Curves+/3DNA-compatible local base and base-step parameter mode.
* Local and global shape tables for base-base and inter-base-pair parameters.
* Groove widths, backbone torsions, sugar pucker, curvature, and bending
  summaries in text, JSON, or CSV output.
* Batch trajectory analysis for MD simulations.
* Optional HTML/WebGL visualization of the molecular structure, helical axis,
  backbone splines, base blocks, local frames, and parameter locations.

## Installation

Basic install:

```bash
pip install .
```

The package installs these command-line entry points:

```bash
pycurves --help
pycurves-md --help
pycurves-md-plot --help
pycurves-viewer --help
```

Optional dependency groups:

```bash
pip install ".[md]"              # MDAnalysis and MDTraj trajectory readers
pip install ".[plot]"            # matplotlib/ijson plotting tools
pip install ".[legacy-process]"  # Biopython process_dna compatibility helper
pip install ".[all]"             # everything above
```

JAX is required for the legacy Curves-style minimization. The default
dependency installs CPU-compatible JAX. For GPU clusters, install the matching
JAX build before installing pyCurves, for example:

```bash
pip install -U "jax[cuda12]"
```

The repository also includes a small install helper:

```bash
python install_pycurves.py install --editable
python install_pycurves.py uninstall --yes
```

## Main CLI

```bash
pycurves [input.pdb|input.cif|input.inp] [options]
```

Common options:

* `--format {curves,json,csv}`: print a Curves-style report, slim JSON, or CSV
  tables.
* `--output-file PATH`: write output to a file or CSV prefix instead of
  `stdout`.
* `--frame-convention legacy|standard`: choose base reference frames.
  `legacy` is Curves 5.3-style; `standard` is Curves+/3DNA-compatible.
* `--axis-convention legacy|curvesplus`: choose the global-axis construction.
  `legacy` runs the pyCurves/JAX minimizer. `curvesplus` reproduces the Curves+
  smooth-axis path and skips pyCurves-only global-axis tables.
* `--continuous-strands`: treat connected multi-chain helices as one biological
  helix when topology inference would otherwise split them.
* `--fit` / `--no-fit`, `--grooves` / `--no-grooves`, `--mini` / `--no-mini`,
  `--comb` / `--no-comb`, `--ends` / `--no-ends`: override analysis flags.
* `--no-annotations`: suppress the non-canonical pairing/modification report.
* `--visualization`: add geometry needed by `pycurves-viewer` to JSON output.

Use `pycurves --help` for the full option list.

## Output

The standard JSON schema is `pycurves-slim-v1`. It is intentionally compact:

* `frame_convention`, `analysis_options`, `inputs`, and `sequence` metadata.
* A flat `dataframes` dictionary with table-like records.
* Local tables such as `local_base_base`, `local_inter_base`,
  `local_inter_base_pair`, `backbone`, and `groove`.
* Global tables from legacy-axis runs, including `global_base_axis`,
  `global_base_base`, `global_inter_base`, `global_inter_base_pair`,
  `global_axis_curvature`, `global_axis_bending`, and
  `global_axis_bending_summary`.

Gapped or uncomputed paired positions are retained with `null` parameter values
so sequence-indexed tables stay aligned. Debug-heavy internals such as raw
gradients, derivative arrays, LSFit atom diagnostics, and topology arrays are
not included in the standard JSON.

CSV output writes one file per dataframe using the `--output-file` value as the
filename prefix.

## MD Trajectory Analysis

Install MD support first:

```bash
pip install ".[md]"
```

Run summary statistics over a trajectory:

```bash
pycurves-md topology.pdb trajectory.xtc \
  --mode summary \
  --frames 1000:5000:10 \
  --output-file dynamics.json
```

Store both per-frame data and summary statistics:

```bash
pycurves-md topology.pdb trajectory.dcd \
  --mode both \
  --format json \
  --output-file dynamics_full.json
```

Useful MD-only options:

* `--frames SPEC`: comma-separated frame indices/ranges, for example
  `0,10,100:500:5`.
* `--start`, `--stop`, `--step`: regular frame window selection.
* `--mode {per-frame,summary,both}`: choose frame-level output, aggregate
  summary statistics, or both.
* `--no-warm-start`: do not initialize each frame from the previous frame.
* `--no-axis-continuity`: do not force global-axis signs to stay aligned with
  the first processed frame.

Static and MD runs share the same core options, including `--frame-convention`,
`--axis-convention`, `--continuous-strands`, `--fit`, `--grooves`, and `--comb`.

## Plotting MD Results

Install plotting support:

```bash
pip install ".[plot]"
```

Generate the default plot set:

```bash
pycurves-md-plot dynamics.json --outdir md_plots
```

Plot selected feature families:

```bash
pycurves-md-plot dynamics.json --block global --block local --outdir shape_plots
pycurves-md-plot dynamics.json --block curvature --block axis_bending --outdir axis_plots
pycurves-md-plot dynamics.json --block torsions --block groove --outdir diagnostic_plots
```

Export extracted tables without plotting:

```bash
pycurves-md-plot dynamics.json --export-csv --no-plots --outdir raw_csvs
```

## HTML Viewer

The viewer is built from JSON generated with `--visualization`:

```bash
pycurves structure.pdb --format json --visualization --output-file viewer.json
pycurves-viewer viewer.json --output viewer.html
```

The generated HTML file is self-contained except for the browser-side 3Dmol.js
dependency.

## Python API

The high-level API is `CurvesWrapper`:

```python
from pycurves_lib.curves_wrapper import CurvesWrapper

runner = CurvesWrapper.from_file(
    "test_data/1A1F_b_c.pdb",
    frame_convention="standard",
)
runner.analyze()
json_text = runner.output(fmt="json")
```


## Project Layout

Root scripts are command-line entry points. Reusable code lives in
`pycurves_lib/`:

* `pycurves.py`: single-structure CLI.
* `pycurves_md.py`: trajectory CLI.
* `pycurves_md_plot.py`: MD JSON plotting helper.
* `pycurves_viewer.py`: HTML viewer generator.
* `process_dna.py`: compatibility wrapper for external pipelines.
* `pycurves_lib/core/`: base fitting, backbone analysis, helical-axis
  minimization, local/global parameter calculations, groove analysis, and
  convention-specific math.
* `pycurves_lib/io/`: molecular loading, input parsing, output formatting,
  reference-base libraries, and viewer payload generation.
* `pycurves_lib/topology/`: automatic topology inference and pairing
  annotations.
* `pycurves_lib/md/`: trajectory-reader adapters.
* `pycurves_lib/data/`: modified-base mapping data.
