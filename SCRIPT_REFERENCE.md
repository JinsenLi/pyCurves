# Top-Level Script Reference

The repository root contains the installed command entry points, notebook-facing
helpers, visualization generators, and a few focused regression tests. The
scientific implementation lives under [`pycurves_lib`](pycurves_lib/README.md).

## Installed commands

| File                   | Installed command   | Responsibility                                                                                                                                                                             |
| ---------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pycurves.py`          | `pycurves`          | Thin static-analysis entry point. It delegates argument parsing and execution to `pycurves_lib.cli.pycurves_main` and turns expected user-facing exceptions into clean process exits.      |
| `pycurves_md.py`       | `pycurves-md`       | Public trajectory API and CLI facade. It extends the core trajectory analyzer with optional DSSR-selected reference topology, then reuses that topology for every frame.                   |
| `pycurves_md_batch.py` | `pycurves-md-batch` | Experimental vectorized trajectory runner for standard base frames and the Curves+ axis. It buffers coordinates, processes NumPy batches, and emits per-frame or streaming summary output. |
| `pycurves_md_plot.py`  | `pycurves-md-plot`  | Reads trajectory JSON, extracts named result tables into Pandas, filters or reshapes them, finds outliers, and writes overview plots or CSV files.                                         |
| `pycurves_viewer.py`   | `pycurves-viewer`   | Combines visualization JSON and its source PDB/mmCIF into a self-contained HTML viewer built around 3Dmol.js.                                                                              |
| `pycurves_pymol.py`    | `pycurves-pymol`    | Converts the same visualization payload into a structure-free PyMOL `.pml` overlay with grouped axis, backbone, base-pair, and groove objects.                                             |

The command names are declared in `pyproject.toml`. All modules also expose a
`main()` function and can be run directly with Python.

## Execution paths

### Static structures

1. `pycurves.py` calls `pycurves_lib.cli.pycurves_main.main()`.
2. The CLI builds a `pycurves_lib.curves_wrapper.CurvesWrapper`.
3. The wrapper loads or generates a Curves `.inp`, runs the analysis pipeline,
   and selects Curves text, JSON, or CSV output.

### Trajectories

`pycurves_md.py` is the general path. It analyzes selected frames one at a time,
supports both legacy and Curves+ axes, and can re-annotate base pairing per
frame. `pycurves_md_batch.py` is the constrained high-throughput path: it
requires standard frames, a Curves+ axis, combined strands, base fitting, and no
terminal virtual levels.

### Visualization and plotting

`pycurves_md_plot.py` consumes trajectory payloads. `pycurves_viewer.py` and
`pycurves_pymol.py` consume static JSON produced with `--visualization`; neither
reruns the scientific analysis.

## Public Python helpers

- `pycurves_md.analyze_trajectory(...)` is the notebook-friendly scalar
  trajectory API.
- `pycurves_md_batch.analyze_trajectory_batch(...)` is the notebook-friendly
  vectorized API.
- `pycurves_md_plot.extract_block(...)`, `filter_rows(...)`,
  `parameter_timeseries(...)`, and `pivot_parameter_matrix(...)` support custom
  analysis without invoking the plotting CLI.
- `pycurves_viewer.render_viewer_html(...)` and
  `pycurves_pymol.render_pymol_script(...)` render already-loaded result
  dictionaries.
