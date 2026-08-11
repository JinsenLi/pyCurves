# pycurves_lib Script Reference

## Wrapper modules

### `curves_wrapper_core.py`

Defines the base `CurvesWrapper` and owns the end-to-end static analysis
pipeline. Its main responsibilities are:

- accept either a coordinate file or an existing Curves `.inp`;
- infer and write `.inp` files through `RobustTopologyInferrer` when needed;
- cache parsed configuration and base-reference libraries while creating fresh
  runtime state for each analysis;
- load coordinates, fit base frames, annotate pair geometry, calculate
  backbone values, choose an axis implementation, and calculate final results;
- expose text/JSON/CSV output plus the compatibility-oriented `getFeatures()`
  dictionary;
- analyze an already populated `MolecularStructure`, which is how trajectory
  workflows avoid reloading atom metadata for every frame.

The analysis sequence inside `_analyze_loaded_context()` is:

1. `BaseLocator.locate_all()` fits per-base frames.
2. `annotate_context()` classifies bases and pair contacts.
3. `build_axis_reference_frames()` derives sign-continuous shape/axis frames.
4. `BackboneAnalyzer.analyze()` calculates backbone and sugar geometry.
5. `HelicalOptimizerJAX` minimizes the legacy axis, or the lightweight base
   optimizer supplies state for Curves+ or fixed-Z paths.
6. `HelicalCalculator.calculate_all()` produces the reported parameter tables.

### `curves_wrapper_dssr.py`

Extends the core wrapper with an optional DSSR JSON topology source. It rejects
conflicting explicit `.inp` or continuous-strand options, selects a DSSR
stem/helix/pair unit, writes a DSSR-derived `.inp`, records provenance, and
attaches source pair annotations to static or trajectory molecules.

### `curves_wrapper.py`

Stable public import for `CurvesWrapper`. Its small override makes requested
DSSR annotations authoritative when equivalent mmCIF annotations are already
present, while retaining unrelated source annotations.

### `__init__.py`

Marks the package. Public callers should import `CurvesWrapper` from
`pycurves_lib.curves_wrapper`; no API is re-exported from the package root.

## Package boundaries

| Package | Detailed guide |
| --- | --- |
| CLI | [`cli/SCRIPT_REFERENCE.md`](cli/SCRIPT_REFERENCE.md) |
| Numerical core | [`core/SCRIPT_REFERENCE.md`](core/SCRIPT_REFERENCE.md) |
| Packaged data | [`data/SCRIPT_REFERENCE.md`](data/SCRIPT_REFERENCE.md) |
| Input/output | [`io/SCRIPT_REFERENCE.md`](io/SCRIPT_REFERENCE.md) |
| Molecular dynamics | [`md/SCRIPT_REFERENCE.md`](md/SCRIPT_REFERENCE.md) |
| Topology and annotations | [`topology/SCRIPT_REFERENCE.md`](topology/SCRIPT_REFERENCE.md) |
