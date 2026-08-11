# Molecular-Dynamics Script Reference

## `trajectory_loader.py`

Defines `TrajectoryFrame` and `TrajectoryLoader`. The loader yields coordinates,
frame indices, and optional time values through this preference order:

1. MDAnalysis for ordinary topology/trajectory pairs;
2. MDTraj when MDAnalysis is unavailable, including optimized explicit-index or
   sequential range handling;
3. a built-in multi-model PDB reader when no trajectory file is supplied.

MDTraj nanometer coordinates are converted to angstroms before analysis.

## `trajectory_cli.py`

Owns the general scalar trajectory engine and its CLI implementation.
`MDTrajectoryAnalyzer` creates one reference topology and template molecule,
copies each frame's coordinates into that model, and calls
`CurvesWrapper.analyze_molecule()`. It supports warm-starting the legacy
optimizer, consistent axis signs across frames, optional per-frame pair
re-annotation, JSON/CSV output, and `per-frame`, `summary`, or `both` modes.

The module also parses frame specifications, reports available indices, flattens
groove records, summarizes numeric/categorical tables, and writes table CSVs.
The root `pycurves_md.py` facade adds DSSR topology support.

## `trajectory_statistics.py`

Provides shared summary math:

- mergeable Chan-Welford accumulators for linear values;
- compensated circular resultant sums for angular values;
- a grouped `BatchSummaryAccumulator` that avoids materializing every frame;
- sugar-pucker, BI/BII, and alpha/gamma population tables;
- scalar helpers used by both trajectory engines.

Angular standard deviations use `sqrt(-2 log(R))`, with `R` the mean resultant
length.

## `batch_curvesplus.py`

Implements `BatchCurvesPlusMDAnalyzer`, the vectorized standard-frame/Curves+
engine. It precomputes base-fit and backbone index templates, fits many frames
at once, calculates local frames and rigid-body values with NumPy/SciPy arrays,
constructs the smooth Curves+ axis, and either returns per-frame records or
feeds a streaming summary accumulator.

This path intentionally rejects unsupported analysis combinations. Use the
scalar analyzer for legacy minimization, contact-geometry/noncanonical frames,
non-combined analysis, or terminal virtual levels.

## `batch_groove.py`

Vectorized companion to the batch analyzer. It batches axis and backbone spline
interpolation, then applies a faithful groove-window scanner per frame. Numba is
used as an optional accelerator when installed; the NumPy/Python path remains
the functional fallback.

## `__init__.py`

Package marker only. User-facing APIs are exposed from `pycurves_md.py` and
`pycurves_md_batch.py` at the repository root.
