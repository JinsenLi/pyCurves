"""Stable public import for the DSSR-aware CurvesWrapper."""

from __future__ import annotations

from pycurves_lib.curves_wrapper_dssr import CurvesWrapper as _DSSRCurvesWrapper


class CurvesWrapper(_DSSRCurvesWrapper):
    def attach_dssr_annotations(self, molecule) -> None:
        source_rows = [dict(row) for row in self.dssr_source_base_pairs]
        if not source_rows:
            return

        def residue_key(row, side):
            return (
                str(row.get(f"{side}_chain_id", "")).strip(),
                int(row.get(f"{side}_residue_id") or 0),
                str(row.get(f"{side}_insertion_code", "")).strip(),
            )

        def pair_key(row):
            return frozenset((residue_key(row, "i"), residue_key(row, "j")))

        dssr_keys = {pair_key(row) for row in source_rows}
        existing = [
            row for row in list(getattr(molecule, "source_base_pairs", None) or [])
            if row.get("source") != "DSSR JSON" and pair_key(row) not in dssr_keys
        ]
        # DSSR is the requested topology authority, so it must win annotation
        # deduplication over equivalent mmCIF source rows.
        molecule.source_base_pairs = source_rows + existing


__all__ = ["CurvesWrapper"]
