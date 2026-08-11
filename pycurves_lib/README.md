# pycurves_lib

This package is the reusable pyCurves engine. The root-level commands are thin
front ends around `CurvesWrapper`, which coordinates topology, I/O, numerical
analysis, annotations, and output formatting.

- `cli`: command-line parsing and dispatch
- `core`: frames, axis optimization, helical parameters, and grooves
- `data`: modified-base mappings and packaged reference data
- `io`: coordinate/configuration loading and result serialization
- `md`: trajectory iteration, batching, and statistics
- `topology`: inferred and DSSR-supplied nucleic-acid topology

See [`SCRIPT_REFERENCE.md`](SCRIPT_REFERENCE.md) for the wrapper layers and the
full analysis call sequence.
