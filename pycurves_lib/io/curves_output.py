from __future__ import annotations

import contextlib
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pycurves_lib.data.modified_bases import parent_base_name
from pycurves_lib.topology.base_annotations import (
    annotate_context,
    base_pair_geometry_annotation,
    base_pair_geometry_tag,
    render_section_m,
)
from pycurves_lib.io.curves_visualization_payload import VisualizationPayloadMixin


def _to_jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _all_finite(values) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=float))))


STEP_PARAMETERS = ("shift", "slide", "rise", "tilt", "roll", "twist")
AXIS_PARAMETERS = ("xdisp", "ydisp", "inclin", "tip")
BASE_BASE_PARAMETERS = ("shear", "stretch", "stagger", "buckle", "propel", "opening")
SUGAR_PUCKERS = (
    "C3'-endo", "C4'-exo", "O1'-endo", "C1'-exo", "C2'-endo",
    "C3'-exo", "C4'-endo", "O1'-exo", "C1'-endo", "C2'-exo",
)


def _parameter_record(values, parameter_names, **metadata):
    """Build one structured parameter row, or None if the values are invalid."""
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if not _all_finite(values[:len(parameter_names)]):
        return None
    row = dict(metadata)
    for idx, name in enumerate(parameter_names):
        row[name] = values[idx]
    return row


def _nullable_parameter_record(values, parameter_names, **metadata):
    """Build a structured row, preserving invalid/missing values as JSON null."""
    row = dict(metadata)
    if values is None:
        values = [None] * len(parameter_names)
    else:
        values = np.asarray(values, dtype=float)
    for idx, name in enumerate(parameter_names):
        try:
            value = float(values[idx])
        except (TypeError, ValueError, IndexError):
            value = np.nan
        row[name] = value if np.isfinite(value) else None
    return row


def _json_number(value):
    """Return a finite JSON number, otherwise None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


class CurvesOutputFormatter(VisualizationPayloadMixin):
    """Render pyCurves results as Curves-style text or structured JSON/Pandas."""

    def __init__(self, runner, annotations: bool = True, visualization: bool = False):
        self.runner = runner
        self.include_annotations = annotations
        self.include_visualization = visualization
        self._annotation_cache = None

    def get_dataframes(self):
        """Return a dictionary of Pandas DataFrames for all computed parameters."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Pandas or one of its dependencies is not installed. "
                "Run 'pip install pandas python-dateutil' to use get_dataframes()."
            ) from exc
        
        self._require_results()
        dataframes = self._build_dataframes()
        
        dfs = {}
        for k, v in dataframes.items():
            if k == "groove" and isinstance(v, dict) and "data" in v:
                flat_list = []
                for lvl, l_data in v["data"].items():
                    for sub_lvl, sl_data in l_data["sub_levels"].items():
                        row = {
                            "atom_defining_backbone": v.get("atom_defining_backbone", ""),
                            "total_levels": v.get("levels", 0),
                            "total_sub_levels": v.get("sub_levels", 0),
                            "level": int(lvl),
                            "base_pair": l_data.get("base_pair", ""),
                            "sub_level": int(sub_lvl),
                        }
                        row.update(sl_data)
                        flat_list.append(row)
                dfs[k] = pd.DataFrame(flat_list)
            else:
                dfs[k] = pd.DataFrame(v)
        return dfs

    def render(self, fmt: str = "curves", header_name: str = "pyCurves") -> str:
        fmt = fmt.lower()
        if fmt in {"curves", "lis", "text", "stdout"}:
            return self.render_curves_text(header_name=header_name)
        if fmt == "json":
            return self.render_json()
        if fmt == "csv":
            return self.render_csv()
        raise ValueError(f"Unknown output format {fmt!r}. Use 'curves', 'json', or 'csv'.")

    def render_csv(self) -> str:
        """Returns a string describing the CSV data or writes it to a zip file in memory (used programmatically).
        For CLI, curves_wrapper.py or pycurves.py should handle saving the DataFrames directly.
        """
        dfs = self.get_dataframes()
        buf = io.StringIO()
        for k, df in dfs.items():
            buf.write(f"--- {k} ---\n")
            buf.write(df.to_csv(index=False) + "\n")
        return buf.getvalue()

    def render_curves_text(self, header_name: str = "pyCurves") -> str:
        self._require_results()
        annotations = self._annotations()
        outaxe_text = self._capture(self.runner.calc.outaxe).rstrip()
        records = self._build_dataframes()
        pieces = [
            self._header(header_name=header_name),
            self.runner.analysis_log.rstrip(),
            outaxe_text,
            self._render_section_l_local_base_base(records).rstrip(),
        ]
        if self.include_annotations:
            pieces.append(render_section_m(annotations).rstrip())
        return "\n\n".join(piece for piece in pieces if piece) + "\n"

    def render_json(self) -> str:
        self._require_results()
        dataframes = self._build_dataframes()
        payload = {
            "program": "pyCurves",
            "format": "pycurves-slim-v1",
            "frame_convention": self._frame_convention_payload(),
            "analysis_options": self._analysis_options_payload(),
            "inputs": {
                "inpfile": self.runner.inpfile,
                "pdbfile": self.runner.pdbfile,
                "generated_inpfiles": self.runner.generated_inpfiles,
            },
            "sequence": self._sequence_payload(),
            "dataframes": dataframes,
        }
        if self.include_visualization:
            payload["visualization"] = self._visualization_payload(dataframes)
        return json.dumps(_to_jsonable(payload), indent=2, allow_nan=False) + "\n"

    def _frame_convention_payload(self) -> Dict[str, Any]:
        cfg = self.runner.ctx.cfg
        name = getattr(cfg, "frame_convention", "legacy")
        library = getattr(self.runner.ctx, "base_reference_library", None)
        return {
            "name": name,
            "compatible_with": ["Curves+", "3DNA", "x3dna"] if name == "standard" else ["Curves 5.3"],
            "reference_source": getattr(library, "source", "setup.f hardcoded bref"),
            "axis_convention": getattr(cfg, "axis_convention", "legacy"),
        }

    def _analysis_options_payload(self) -> Dict[str, Any]:
        cfg = self.runner.ctx.cfg
        return {
            "comb": bool(getattr(cfg, "comb", False)),
            "groove": bool(getattr(cfg, "grv", False)),
            "ends": bool(getattr(cfg, "ends", False)),
            "mini": bool(getattr(cfg, "mini", False)),
            "axis_convention": getattr(cfg, "axis_convention", "legacy"),
        }

    def _sequence_payload(self) -> Dict[str, Any]:
        ctx = self.runner.ctx
        strands = []
        for strand in range(ctx.nst):
            residues = []
            sequence = []
            for level in range(1, ctx.nux + 1):
                if ctx.ni_map[strand, level - 1] <= 0:
                    continue
                subunit = int(ctx.ni_map[strand, level - 1])
                atom_idx = int(ctx.molecule.subunit_boundaries[subunit - 1])
                residue_name = str(ctx.molecule.residue_names[atom_idx]).strip()
                chain_id = str(ctx.molecule.chain_ids[atom_idx]).strip() if ctx.molecule.chain_ids is not None else ""
                parent_base = parent_base_name(residue_name)
                sequence.append(str(parent_base or "N")[:1])
                residues.append({
                    "level": level,
                    "chain_id": chain_id,
                    "residue_name": residue_name,
                    "parent_base": parent_base,
                    "residue_id": int(ctx.molecule.residue_ids[atom_idx]),
                })
            strands.append({
                "strand": strand + 1,
                "sequence": "".join(sequence),
                "residues": residues,
            })
        return {
            "n_strands": ctx.nst,
            "n_levels": ctx.nux,
            "strands": strands,
        }

    def _primary_sequence_levels(self) -> List[int]:
        """Return actual strand-1 residue levels, excluding Curves gap levels."""
        ctx = self.runner.ctx
        calc = self.runner.calc
        return [level for level in range(1, ctx.nux + 1) if calc._has_level(0, level)]

    def _step_label_between(self, strand: int, first_level: int, second_level: int) -> str:
        calc = self.runner.calc
        first = calc._residue_label(strand, first_level)
        second = calc._residue_label(strand, second_level)
        if first is None or second is None:
            return "-"
        first_base, _, first_id = first
        second_base, _, second_id = second
        return f"{first_base}{first_id:3d}/{second_base}{second_id:3d}"

    def _build_dataframes(self) -> Dict[str, Any]:
        """Build the slim dataframe dictionary used by JSON, CSV, and viewers."""
        ctx = self.runner.ctx
        calc = self.runner.calc
        annotations = self._annotations() if self.include_annotations else {}
        curvesplus_axis = self._uses_curvesplus_axis()
        
        records = {}

        # 1. Global Base-Axis
        if not curvesplus_axis:
            base_axis = []
            for strand in range(ctx.nst):
                for level in range(1, ctx.nux + 1):
                    if not calc._has_level(strand, level): continue
                    label = calc._residue_unit_label(strand, level)
                    if label is None: continue
                    res_name, res_id = label
                    
                    p = ctx.params.helical[strand, level]
                    if not _all_finite([p[0], p[1], p[3], p[4]]):
                        continue
                    base_axis.append({
                        "strand": strand + 1,
                        "level": level,
                        "residue_name": res_name,
                        "residue_id": res_id,
                        "xdisp": p[0],
                        "ydisp": p[1],
                        "inclin": p[3],
                        "tip": p[4]
                    })
            records["global_base_axis"] = base_axis

        primary_levels = self._primary_sequence_levels()

        # 2. Global/Curves+ Base Pair-Axis and global Base-Base
        bp_axis = []
        base_base = []
        if ctx.cfg.comb and ctx.nst > 1:
            for partner_strand in range(1, ctx.nst):
                for sequence_index, level in enumerate(primary_levels, start=1):
                    has_pair = calc._has_level(0, level) and calc._has_level(partner_strand, level)
                    duplex = calc._duplex_id(0, partner_strand, level)
                    axis_row = _nullable_parameter_record(
                        calc._global_base_pair_axis_values(partner_strand, level) if has_pair else None,
                        AXIS_PARAMETERS,
                        partner_strand=partner_strand + 1,
                        sequence_index=sequence_index,
                        level=level,
                        duplex=duplex,
                    )
                    bp_axis.append(axis_row)

                    if curvesplus_axis or partner_strand != 1:
                        continue
                
                    base_base.append(_nullable_parameter_record(
                        calc._global_base_base_values(partner_strand, level) if has_pair else None,
                        BASE_BASE_PARAMETERS,
                        partner_strand=partner_strand + 1,
                        sequence_index=sequence_index,
                        level=level,
                        duplex=duplex,
                    ))
            if curvesplus_axis:
                records["curvesplus_base_pair_axis"] = bp_axis
            else:
                records["global_base_pair_axis"] = bp_axis
                records["global_base_base"] = base_base

        if ctx.cfg.comb and ctx.nst > 1:
            local_base_base = []
            for partner_strand in range(1, ctx.nst):
                for sequence_index, level in enumerate(primary_levels, start=1):
                    has_pair = calc._has_level(0, level) and calc._has_level(partner_strand, level)
                    values = calc._local_base_base_values(partner_strand, level) if has_pair else None
                    local_base_base.append(_nullable_parameter_record(
                        values,
                        BASE_BASE_PARAMETERS,
                        partner_strand=partner_strand + 1,
                        sequence_index=sequence_index,
                        level=level,
                        duplex=calc._duplex_id(0, partner_strand, level),
                    ))
            records["local_base_base"] = local_base_base

        # 3. Global Inter-Base
        if not curvesplus_axis:
            inter_base = []
            for strand in range(ctx.nst):
                for level in range(1, ctx.nux):
                    if not (calc._has_level(strand, level) and calc._has_level(strand, level+1)): continue
                    step = calc._step_label(strand, level+1)
                    row = _parameter_record(
                        ctx.params.inter_base[strand, level+1],
                        STEP_PARAMETERS,
                        strand=strand + 1,
                        level=level,
                        step=step,
                    )
                    if row is not None:
                        inter_base.append(row)
            records["global_inter_base"] = inter_base
        
        # 4. Local Inter-Base
        local_inter_base = []
        for strand in range(ctx.nst):
            for level in range(1, ctx.nux):
                if not (calc._has_level(strand, level) and calc._has_level(strand, level+1)): continue
                step = calc._step_label(strand, level+1)
                row = _parameter_record(
                    calc.local_inter_base[level+1, :, strand],
                    STEP_PARAMETERS,
                    strand=strand + 1,
                    level=level,
                    step=step,
                )
                if row is not None:
                    local_inter_base.append(row)
        records["local_inter_base"] = local_inter_base

        # 5. Inter-Base Pair
        if ctx.cfg.comb and ctx.nst > 1:
            global_inter_bp = []
            local_inter_bp = []
            for partner_strand in range(1, ctx.nst):
                for sequence_index, (level, next_level) in enumerate(zip(primary_levels, primary_levels[1:]), start=1):
                    adjacent = next_level == level + 1
                    has_step = (
                        adjacent
                        and calc._has_level(0, level)
                        and calc._has_level(partner_strand, level)
                        and calc._has_level(0, next_level)
                        and calc._has_level(partner_strand, next_level)
                    )
                    duplex = f"{calc._duplex_id(0, partner_strand, level)}/{calc._duplex_id(0, partner_strand, next_level)}"
                    step = self._step_label_between(0, level, next_level)
                    if not curvesplus_axis:
                        global_row = _nullable_parameter_record(
                            calc._global_inter_base_pair_values(partner_strand, next_level) if has_step else None,
                            STEP_PARAMETERS,
                            partner_strand=partner_strand + 1,
                            sequence_index=sequence_index,
                            level=level,
                            next_level=next_level,
                            step=step,
                            duplex=duplex,
                        )
                        global_inter_bp.append(global_row)

                    local_row = _nullable_parameter_record(
                        calc.local_inter_base_pair[next_level, :, partner_strand] if has_step else None,
                        STEP_PARAMETERS,
                        partner_strand=partner_strand + 1,
                        sequence_index=sequence_index,
                        level=level,
                        next_level=next_level,
                        step=step,
                        duplex=duplex,
                    )
                    local_inter_bp.append(local_row)
            if not curvesplus_axis:
                records["global_inter_base_pair"] = global_inter_bp
            records["local_inter_base_pair"] = local_inter_bp

        # 6. Backbone Parameters
        backbone = []
        for strand in range(ctx.nst):
            for level in range(1, ctx.nux + 1):
                if not calc._has_level(strand, level): continue
                label = calc._residue_unit_label(strand, level)
                if label is None: continue
                res_name, res_id = label

                tor = ctx.backbone.torsions[strand, level]
                pucker = ctx.backbone.sugar_pucker[strand, level]
                phase = _json_number(pucker[1])
                pucker_index = 0
                if phase is not None:
                    pucker_index = int((phase % 360.0) / 36.0)
                    pucker_index = max(0, min(len(SUGAR_PUCKERS) - 1, pucker_index))
                backbone.append({
                    "strand": strand + 1,
                    "level": level,
                    "residue_name": res_name,
                    "residue_id": res_id,
                    "c1_c2": _json_number(tor[4]),
                    "c2_c3": _json_number(tor[5]),
                    "phase": phase,
                    "amplitude": _json_number(pucker[0]),
                    "pucker": SUGAR_PUCKERS[pucker_index],
                    "c1_prime": _json_number(tor[0]),
                    "c2_prime": _json_number(tor[1]),
                    "c3_prime": _json_number(tor[2]),
                    "chi": _json_number(tor[12]),
                    "gamma": _json_number(tor[8]),
                    "delta": _json_number(tor[9]),
                    "epsilon": _json_number(tor[6]),
                    "zeta": _json_number(tor[7]),
                    "alpha": _json_number(tor[10]),
                    "beta": _json_number(tor[11]),
                })
        records["backbone"] = backbone
        if not curvesplus_axis:
            records["global_axis_curvature"] = self._axis_curvature_records()
            bending_rows, bending_summary = self._axis_bending_records()
            records["global_axis_bending"] = bending_rows
            records["global_axis_bending_summary"] = bending_summary

        # 7. Groove Parameters
        if ctx.cfg.comb and ctx.nst > 1 and getattr(ctx.cfg, "grv", False):
            if not hasattr(calc, "groove_params"):
                self._capture(calc.groove)
            if hasattr(calc, "groove_params") and calc.groove_params:
                records["groove"] = self._groove_records(calc.groove_params)

        if self.include_annotations:
            records["annotations"] = self._slim_annotation_records(annotations)
            records["noncanonical_base_pairs"] = self._noncanonical_base_pair_records(annotations)

        return records

    def _uses_curvesplus_axis(self) -> bool:
        cfg = self.runner.ctx.cfg
        return (
            str(getattr(cfg, "frame_convention", "legacy")).lower() == "standard"
            and str(getattr(cfg, "axis_convention", "legacy")).lower() == "curvesplus"
        )

    def _render_section_l_local_base_base(self, records: Dict[str, Any]) -> str:
        rows = records.get("local_base_base", [])
        if not rows:
            return ""

        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row.get("partner_strand", 0)), []).append(row)

        lines = [
            "  --------------------------------",
            "  |L| Local Base-Base Parameters |",
            "  --------------------------------",
        ]
        for partner_strand in sorted(grouped):
            lines.extend([
                "",
                f"  Strand 1 with strand {partner_strand} ...",
                "",
                "    Duplex          Shear    Stretch  Stagger  Buckle   Propel  Opening",
                "                    (Sx)      (Sy)     (Sz)    (kappa)  (omega) (sigma)",
            ])
            for row in sorted(grouped[partner_strand], key=lambda item: int(item.get("level", 0))):
                if not _all_finite([row.get(name) for name in BASE_BASE_PARAMETERS]):
                    continue
                lines.append(
                    f"  {int(row.get('level', 0)):3d}) {str(row.get('duplex', '')):<9s} "
                    f"{float(row.get('shear', 0.0)):8.2f} {float(row.get('stretch', 0.0)):8.2f} "
                    f"{float(row.get('stagger', 0.0)):8.2f} {float(row.get('buckle', 0.0)):8.2f} "
                    f"{float(row.get('propel', 0.0)):8.2f} {float(row.get('opening', 0.0)):8.2f}"
                )
        return "\n".join(lines)

    def _axis_curvature_records(self) -> List[Dict[str, Any]]:
        ctx = self.runner.ctx
        calc = self.runner.calc
        rows = []
        section_strands = [0] if ctx.cfg.comb else list(range(ctx.nst))
        for strand in section_strands:
            _, _, start, stop = calc._axis_bounds(strand)
            for level in range(start + 1, stop + 1):
                values = calc.vkin[level, :, strand]
                if not _all_finite(values[:7]):
                    continue
                step = calc._step_label(strand, level) if calc._has_level(strand, level - 1) and calc._has_level(strand, level) else "-"
                rows.append({
                    "strand": strand + 1,
                    "level": level - 1,
                    "next_level": level,
                    "step": step,
                    "ax": values[0],
                    "ay": values[1],
                    "ainc": values[2],
                    "atip": values[3],
                    "adis": values[4],
                    "angle": values[5],
                    "path": values[6],
                    "dc": int(calc.dcod[level, strand]) if hasattr(calc, "dcod") else 0,
                })
        return rows

    def _axis_bending_records(self):
        ctx = self.runner.ctx
        calc = self.runner.calc
        rows = []
        summary = []
        section_strands = [0] if ctx.cfg.comb else list(range(ctx.nst))
        for strand in section_strands:
            _, _, start, stop = calc._axis_bounds(strand)
            axis_points = ctx.params.ox if ctx.cfg.comb else self.runner.opt.hho[:, :, strand]
            path_length = float(sum(calc.vkin[level, 6, strand] for level in range(start + 1, stop + 1)))
            end_to_end = float(np.linalg.norm(axis_points[stop] - axis_points[start]))
            shortening = 100.0 * (1.0 - end_to_end / path_length) if path_length > 0.0 else 0.0
            start_point = axis_points[start]
            end_to_end_vector = axis_points[stop] - start_point
            end_to_end_unit = end_to_end_vector / (np.linalg.norm(end_to_end_vector) + 1e-12)

            for level in range(start, stop + 1):
                rel = axis_points[level] - start_point
                projection = float(np.dot(rel, end_to_end_unit))
                offset = float(np.linalg.norm(rel - end_to_end_unit * projection))
                local_direction = 0.0 if level in {start, stop} else float(calc.bend[level, strand])
                label = calc._residue_label(strand, level)
                if label is None:
                    residue_name = "-"
                    residue_id = None
                else:
                    residue_name, _, residue_id = label
                rows.append({
                    "strand": strand + 1,
                    "level": level,
                    "residue_name": residue_name,
                    "residue_id": residue_id,
                    "offset": _json_number(offset),
                    "local_direction": _json_number(local_direction),
                })

            summary.append({
                "strand": strand + 1,
                "path_length": _json_number(path_length),
                "end_to_end": _json_number(end_to_end),
                "shortening_percent": _json_number(shortening),
                "overall_bend_uu": _json_number(calc.bend[start, strand]),
                "overall_bend_pp": _json_number(calc.bend[stop, strand]),
            })
        return rows, summary

    def _slim_annotation_records(self, annotations: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row in annotations.get("base_pair_annotations", []):
            if not self._is_reportable_base_pair_annotation(row):
                continue
            rows.append({
                "annotation_type": "base_pair",
                "severity": "warn" if row.get("is_mismatch") else "info",
                "level": row.get("level"),
                "location": f"level {row.get('level')}",
                "code": row.get("pair_family", ""),
                "message": row.get("pair_subtype", ""),
                "geometry_annotation": base_pair_geometry_annotation(row),
                "leontis_westhof": base_pair_geometry_tag(row),
                "residue_1": row.get("residue_1"),
                "residue_2": row.get("residue_2"),
                "base_1": row.get("base_1"),
                "base_2": row.get("base_2"),
                "edge_pair": row.get("edge_pair", ""),
                "glycosidic_orientation": row.get("glycosidic_orientation", ""),
                "strand_direction": row.get("strand_direction", ""),
                "frame_mode": row.get("frame_mode", ""),
                "contact_confidence": row.get("contact_confidence", ""),
                "contact_count": self._contact_count(row),
                "shape_parameters_supported": row.get("shape_parameters_supported", True),
                "shape_skip_reason": row.get("shape_skip_reason", ""),
            })
        for row in annotations.get("modified_base_annotations", []):
            rows.append({
                "annotation_type": "modified_base",
                "severity": "info",
                "level": row.get("level"),
                "location": f"strand {row.get('strand')} level {row.get('level')}",
                "code": "modified_base",
                "message": f"{row.get('residue_name', '')} fitted with {row.get('parent_base', '')} parent-base template.".strip(),
                "residue": row.get("residue_name"),
                "parent_base": row.get("parent_base"),
            })
        return rows

    def _noncanonical_base_pair_records(self, annotations: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row in annotations.get("base_pair_annotations", []):
            if not self._is_noncanonical_base_pair(row):
                continue
            rows.append({
                "level": row.get("level"),
                "strand_1": row.get("strand_1"),
                "strand_2": row.get("strand_2"),
                "residue_1": row.get("residue_1"),
                "residue_2": row.get("residue_2"),
                "base_1": row.get("base_1"),
                "base_2": row.get("base_2"),
                "pair_family": row.get("pair_family", ""),
                "pair_subtype": row.get("pair_subtype", ""),
                "geometry_annotation": base_pair_geometry_annotation(row),
                "leontis_westhof": base_pair_geometry_tag(row),
                "edge_pair": row.get("edge_pair", ""),
                "edge_1": row.get("edge_1", ""),
                "edge_2": row.get("edge_2", ""),
                "glycosidic_orientation": row.get("glycosidic_orientation", ""),
                "strand_direction": row.get("strand_direction", ""),
                "frame_mode": row.get("frame_mode", ""),
                "contact_confidence": row.get("contact_confidence", ""),
                "contact_count": self._contact_count(row),
                "source_pair_number": row.get("source_pair_number"),
                "geometry_flag": row.get("geometry_flag", ""),
                "manual_geometry_tag": row.get("manual_geometry_tag", ""),
                "shape_parameters_supported": row.get("shape_parameters_supported", True),
                "shape_skip_reason": row.get("shape_skip_reason", ""),
            })
        return rows

    @staticmethod
    def _is_reportable_base_pair_annotation(row: Dict[str, Any]) -> bool:
        return bool(
            row.get("is_hoogsteen")
            or row.get("is_mismatch")
            or row.get("has_modified_base")
            or row.get("pair_family") not in {"watson_crick", ""}
            or row.get("geometry_flag")
            or row.get("frame_mode") == "contact_geometry"
        )

    @staticmethod
    def _is_noncanonical_base_pair(row: Dict[str, Any]) -> bool:
        return bool(
            row.get("is_hoogsteen")
            or row.get("is_mismatch")
            or row.get("pair_family") not in {"watson_crick", ""}
            or row.get("geometry_flag")
            or row.get("frame_mode") == "contact_geometry"
        )

    @staticmethod
    def _contact_count(row: Dict[str, Any]) -> int:
        count = row.get("contact_count")
        if count is not None:
            try:
                return int(count)
            except (TypeError, ValueError):
                pass
        return len(row.get("contact_atom_pairs") or [])

    def _groove_records(self, groove_params: Dict[str, Any]) -> Dict[str, Any]:
        groove = _to_jsonable(groove_params)
        if self.include_visualization:
            return groove
        for level_data in groove.get("data", {}).values():
            for sub_level_data in level_data.get("sub_levels", {}).values():
                sub_level_data.pop("geometry", None)
        return groove

    def _annotations(self) -> Dict[str, List[Dict[str, Any]]]:
        if self._annotation_cache is None:
            self._annotation_cache = annotate_context(self.runner.ctx)
        return self._annotation_cache

    def _header(self, header_name: str) -> str:
        cfg = self.runner.ctx.cfg
        namelist = self._read_namelist()
        today = datetime.now().strftime("%d %b %y")

        file_name = self.runner.pdbfile or namelist.get("file", "")
        lis_name = namelist.get("lis", Path(self.runner.inpfile or "").stem)
        pdb_name = namelist.get("pdb", "")

        title = f"******  {header_name.upper():^17s}  *****"
        return "\n".join([
            "",
            "     ***********************************               **************",
            f"     {title:<35s}               *  {today:>9s} *",
            "     ***********************************               **************",
            "",
            "",
            f"  FILE : {str(file_name):<32s} LIS  : {str(lis_name):<32s}",
            f"  dna  : {namelist.get('dna', ''):<32s} axin : {namelist.get('axin', ''):<32s}",
            f"  axout: {namelist.get('axout', ''):<32s} daf  : {namelist.get('daf', ''):<32s}",
            f"  PDB  : {str(pdb_name):<32s}",
            "",
            f"  acc  : {cfg.acc:8.3f}  wid  : {cfg.wid:8.3f}",
            "",
            (
                f"  maxn : {cfg.maxn:5d}  ior  : {cfg.ior:5d}  ibond: {cfg.ibond:5d}  "
                f"splin: {cfg.spline:5d}  break: {cfg.break_lvl:5d}"
            ),
            f"  nleve: {getattr(cfg, 'nlevel', 3):5d}  nbac : {getattr(cfg, 'nbac', 7):5d}",
            "",
            (
                f"  ends : {self._tf(cfg.ends):>5s}  supp : {self._tf(cfg.supp):>5s}  "
                f"COMB : {self._tf(cfg.comb):>5s}  dinu : {self._tf(cfg.dinu):>5s}  mini : {self._tf(cfg.mini):>5s}"
            ),
            (
                f"  rest : {self._tf(cfg.rest):>5s}  line : {self._tf(cfg.line):>5s}  "
                f"zaxe : {self._tf(cfg.zaxe):>5s}  FIT  : {self._tf(cfg.fit):>5s}  test : {self._tf(cfg.test):>5s}"
            ),
            (
                f"  GRV  : {self._tf(cfg.grv):>5s}  old  : {self._tf(cfg.old):>5s}  "
                f"axonl: {self._tf(cfg.axonly):>5s}"
            ),
        ])

    def _read_namelist(self) -> Dict[str, str]:
        if not self.runner.inpfile:
            return {}
        text = Path(self.runner.inpfile).read_text(encoding="utf-8", errors="ignore")
        values = {}
        for key, value in re.findall(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^,\s&]+)", text):
            values[key.lower()] = value.strip().strip("'\"")
        return values

    @staticmethod
    def _capture(func) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            func()
        return buf.getvalue()

    @staticmethod
    def _tf(value: bool) -> str:
        return "T" if value else "F"

    def _require_results(self) -> None:
        if self.runner.ctx is None or self.runner.opt is None or self.runner.calc is None:
            raise RuntimeError("Call analyze() before rendering output.")

