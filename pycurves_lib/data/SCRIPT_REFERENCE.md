# Data Script Reference

## `modified_bases.py`

Provides the shared residue-normalization API:

- `parent_base_name()` maps canonical aliases and known modifications to A, C,
  G, T, or U while retaining supported special bases;
- `is_modified_base()` distinguishes nonstandard residue names;
- `is_known_modified_base()` checks whether a modified residue has a supported
  parent mapping;
- `nakb_modified_base_parent()` and `modified_base_parent()` load and cache the
  packaged map, with a small built-in fallback for common modifications.

## Data files

| File | Use |
| --- | --- |
| `modified_to_change_data.json` | NAKB-derived residue-to-standard-base mappings. |
| `reference/standard_b.lib` | Curves+/standard base templates used by `BaseReferenceLibrary`. |
| `reference/lw_exemplars.json` | Conservative FR3D-derived envelopes for all 18 directed Leontis-Westhof families. |
| `reference/lw_exemplars.LICENSE.txt` | Attribution and licensing for the exemplar data. |

## `__init__.py`

Package marker only. Reference files are resolved through package resources or
paths relative to the installed module.
