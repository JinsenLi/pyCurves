# I/O Script Reference

## Input modules

### `curves_mol_loader.py`

`MolecularLoader` reads PDB/BRK and mmCIF, including gzip-compressed files. It
uses Gemmi for normal parsing, retains a fixed-column PDB fallback, applies
explicit alternate-conformation selection, and populates `MolecularStructure`.
Post-processing standardizes atom names, identifies residues, imports source
pair/bond annotations, builds connectivity, and can materialize relevant mmCIF
or detected crystallographic symmetry mates.

### `curves_config_loader.py`

Parses Curves `.inp` files into `HelicalConfig` plus strand/level maps. It
handles namelist flags and numbers, signed strand directions, compact mapping
tokens, gaps, initial XYTP rows, terminal rows, runtime overrides, and explicit
Leontis-Westhof markers such as `[cWW]` or `[tWH]`.

### `base_reference.py`

Loads legacy or standard base templates and fits them to residue coordinates.
It normalizes prime/star atom aliases, parses `standard_b.lib`, exposes template
origins/axes, performs least-squares frame fitting, and supplies relative fitted
geometry helpers used by topology and annotation code.

### `dssr_json.py`

Normalizes DSSR JSON into immutable residue, pair, unit, and document records.
The loader tolerates supported DSSR layout variations, validates identities,
parses DSSR nucleotide identifiers, exposes stems/helices/synthetic pair units,
and raises `DSSRJSONError` when unsafe normalization would otherwise be needed.

## Output modules

### `curves_output_core.py`

Builds the canonical result records and renders them as Curves-style text, JSON,
or Pandas DataFrames. It converts NumPy values safely, preserves missing values
as JSON nulls, assembles local/global parameter, backbone, groove, curvature,
annotation, and base-pair-observation tables, and emits metadata describing the
input and chosen conventions.

### `curves_visualization_payload.py`

`VisualizationPayloadMixin` derives display-ready geometry from completed
results: helical-axis points, backbone paths, base plates/pair blocks, groove
connectors, annotations, and parameter tables. This keeps viewers independent
of the scientific calculation code.

### `curves_output.py`

Stable public formatter. It re-exports the core output surface and extends JSON
and DataFrame output with DSSR topology provenance and DSSR source-pair rows
when present.

### `__init__.py`

Package marker only.

## Main data flow

```text
PDB/mmCIF -> MolecularLoader -> MolecularStructure
.inp      -> ConfigLoader    -> HelicalConfig + strand maps
DSSR JSON -> DSSRDocument    -> optional topology builder

completed CurvesWrapper -> CurvesOutputFormatter
                        -> text / JSON / DataFrames / visualization payload
```
