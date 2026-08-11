# Core Script Reference

## Module map

### `curves_dataclasses.py`

Defines the central data model and base-fitting setup:

- `MolecularStructure` stores atom metadata, coordinates, bonds, source pair
  annotations, alternate conformations, and crystallographic metadata;
- `HelicalConfig` mirrors Curves control flags and numeric settings;
- `BackboneTopology` and `HelicalParameters` hold the indexed numerical arrays;
- `CurvesContext` allocates one analysis state and retains readable aliases for
  translated Fortran names;
- `BaseLocator` maps topology levels to residues, fits legacy or standard base
  reference frames, records fit quality, and creates optional terminal frames.

### `curves_analyzer.py`

Contains two stages. `BackboneAnalyzer` locates sugar-phosphate neighbors,
records phosphodiester connectivity, and calculates torsions, valence angles,
and sugar-pucker values. `HelicalOptimizer` prepares Curves 5.3-style helical
axis state, active ranges, seed parameters, optional terminal extensions, and
reports. The JAX subclass supplies the normal minimization implementation.

### `curves_optimizer_jax.py`

Implements the legacy curvilinear-axis objective in JAX with 64-bit arrays. A
jitted objective returns both gradients and axis state; SciPy BFGS drives the
optimization. The class also evaluates configured XYTP parameters once when
`mini=.f.` and writes optimized arrays back into the shared context.

### `curves_calculator.py`

`HelicalCalculator` converts frames and optimized axis state into reportable
values. It calculates global strand/base-pair axes, local intra-pair and step
parameters, curvature/bending summaries, and groove results. `calculate_all()`
selects the legacy or Curves+ global-axis path while preserving a common output
surface.

### `parameter_conventions.py`

Owns convention-specific shape mathematics and derived interaction frames.

- `build_interaction_reference_frames()` creates contact-aware shape frames for
  resolved noncanonical pairs, provisional contacts, and left-handed cWW runs.
- `build_axis_reference_frames()` chooses sign-equivalent contact frames that
  stay continuous against neighboring fitted frames.
- `LegacyParameterConvention` reproduces Curves 5.3 local formulas, using the
  standard machinery where contact geometry requires it.
- `StandardParameterConvention` uses midpoint rigid-body decomposition for
  Curves+/3DNA-style base, base-pair, and step parameters.
- `convention_for_context()` selects the implementation from the configured
  frame convention.

### `curvesplus_axis.py`

Mixin implementing the Curves+ `axis`/`smooth` route. It forms mean base-pair
frames, derives and smooths screw axes, calculates base-pair/axis and inter-pair
tables, and exposes the smooth axis to the existing groove scanner.

### `curves_groove.py`

Mixin containing the scalar Curves groove algorithm. It interpolates the axis
and selected backbone atoms, scans opposing backbone splines, applies van der
Waals corrections, and returns minor/major width, depth, angle, and display
geometry where valid.

### `__init__.py`

Package marker with no exports.

## Normal stage order

```text
fit base frames
  -> classify/derive interaction frames
  -> analyze backbone
  -> construct or minimize the helical axis
  -> calculate local/global parameters
  -> scan grooves when enabled
```

Most code outside this package should use `CurvesWrapper` instead of creating
these classes directly; the wrapper establishes the required shared state and
ordering.
