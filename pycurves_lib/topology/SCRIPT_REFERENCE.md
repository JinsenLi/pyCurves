# Topology Script Reference

## Coordinate-driven topology

### `topology_inferrer.py`

`RobustTopologyInferrer` turns a loaded `MolecularStructure` into one or more
`InferredTopology` objects and writable Curves `.inp` files. Its major stages
are:

1. collect nucleic-acid residues and trace covalently connected strands;
2. combine coordinate contacts, fitted standard-frame geometry, source
   annotations, and Leontis-Westhof exemplars into base-pair candidates;
3. choose one-to-one pairs and project them onto strand order;
4. repair plausible mismatches, gaps, and terminal register extensions without
   silently inventing unsupported pairs;
5. partition complexes or emit per-duplex/single-strand topologies;
6. attach explicit geometry markers only where downstream frame construction
   needs them.

`InferredTopology.to_inp_text()` serializes the final signed strand directions,
level map, analysis flags, and pair markers.

### `lw_exemplars.py`

Loads the packaged FR3D-derived envelopes and conservatively classifies fitted
standard-frame pairs into all 18 directed Leontis-Westhof families. It computes
relative descriptors, checks exact envelope membership, scores distance from
the envelope center, applies an ambiguity margin, and returns diagnostic
`LWClassification` records.

## Annotation modules

### `base_annotations.py`

Builds the authoritative static annotation tables. It combines pair identity,
source annotations, fitted frames, heavy-atom contacts, LW evidence,
glycosidic/strand orientation, modified-base information, and fit quality to
report pair presence, named pairing mode, observed/reference LW families,
classification status, contact-frame choice, and diagnostic warnings. It also
renders the human-readable annotation sections used in Curves text output.

### `frame_annotations.py`

Re-runs coordinate-only pair detection for one trajectory frame. It matches
current candidates to reference pairs, reports absent/uncertain pairs, adds new
transient pairs, and preserves reference topology fields separately from
observed frame classifications.

## DSSR-driven topology

### `dssr_topology_core.py`

Resolves normalized DSSR residues against loaded coordinates, selects a stem,
helix, or synthetic pair unit, validates representability, constructs Curves
strand maps, orients LW tags to the chosen strand order, and records source
rows plus topology provenance. Ambiguous selection and unsafe topology have
separate public error types.

### `dssr_topology.py`

Public linear-time orientation implementation. A small dynamic-programming pass
chooses per-pair swaps and rail directions, requiring every adjacent pair to
follow two continuous backbones before accepting a DSSR unit.

### `dssr_topology_linear.py`

Backward-compatible import alias for the public builder in
`dssr_topology.py`.

### `__init__.py`

Package marker only.

## Typical flow

```text
coordinates -> RobustTopologyInferrer -> InferredTopology -> .inp
       or
DSSR JSON + coordinates -> DSSRTopologyBuilder -> InferredTopology -> .inp

loaded context -> base_annotations -> parameter-frame selection and output
trajectory frame -> frame_annotations -> observed pair-state records
```
