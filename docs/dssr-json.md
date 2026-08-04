# DSSR JSON topology input

pyCurves can use an existing DSSR JSON report as the source of base pairing and
strand topology while retaining pyCurves/Curves analysis conventions. The
coordinate structure is still required because DSSR JSON does not contain the
complete atomic model used by pyCurves.

```console
pycurves structure.cif --dssr-json structure.cif_full.json
pycurves structure.cif --dssr-json structure.cif_full.json --dssr-unit stem:2
pycurves structure.cif --dssr-json structure.cif_full.json --dssr-unit helix:1
```

The same options are accepted by `pycurves-md`; the selected reference topology
is resolved once and reused for every trajectory frame.

Python usage:

```python
from pycurves_lib.curves_wrapper import CurvesWrapper

runner = CurvesWrapper.from_file(
    "structure.cif",
    dssr_json="structure.cif_full.json",
    dssr_unit="stem:2",
)
runner.run()
```

## Selection rules

- A full report with exactly one DSSR stem selects that stem automatically.
- A full report with multiple stems requires an explicit `stem:N` selection.
- `helix:N` is accepted only when the DSSR helix forms two continuous backbone
  rails. Composite/coaxial helices must be analyzed through their constituent
  stems.
- A pair-only report is partitioned into continuous, unambiguous two-rail
  segments. One segment is selected automatically; multiple segments require
  `pairs:N`.
- A report with no analyzable pair segment produces a clear error rather than
  falling back silently to geometric inference.

DSSR chain identifiers are case-sensitive. Residues are resolved by chain,
residue number, insertion code, model when present, and residue identity.

## Output provenance

Structured JSON output includes `topology_provenance` with the DSSR program
version, report kind, selected unit, strand directions, source coordinate name,
and mapping warnings. With annotations enabled, normalized DSSR root pairs are
also included as `dssr_source_pairs`. CSV output exposes the same source pairs as
a separate table.

The generated `.inp` file is retained as an editable/reproducible intermediate;
users do not need to author it themselves.
