"""Public output formatter with DSSR provenance extensions."""

from __future__ import annotations

import json

from pycurves_lib.io import curves_output_core as _core
from pycurves_lib.io.curves_output_core import *  # noqa: F401,F403


_to_jsonable = _core._to_jsonable


class CurvesOutputFormatter(_core.CurvesOutputFormatter):
    def render_json(self) -> str:
        payload = json.loads(super().render_json())
        provenance = getattr(self.runner, "dssr_provenance", None)
        if provenance:
            payload["inputs"]["dssr_json"] = getattr(self.runner, "dssr_json", None)
            payload["topology_provenance"] = provenance
            if self.include_annotations:
                payload["dssr_source_pairs"] = _to_jsonable(
                    list(getattr(self.runner, "dssr_source_base_pairs", []) or [])
                )
        return json.dumps(_to_jsonable(payload), indent=2, allow_nan=False) + "\n"

    def get_dataframes(self):
        frames = super().get_dataframes()
        provenance = getattr(self.runner, "dssr_provenance", None)
        source_rows = list(getattr(self.runner, "dssr_source_base_pairs", []) or [])
        if provenance and source_rows:
            import pandas as pd

            frames["dssr_source_pairs"] = pd.DataFrame(_to_jsonable(source_rows))
        return frames


__all__ = [
    name for name in dir(_core)
    if not name.startswith("_") and name != "CurvesOutputFormatter"
] + ["CurvesOutputFormatter", "_to_jsonable"]
