from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pycurves_lib.io.base_reference import (
    BaseFrameFitter,
    BaseReferenceLibrary,
    is_fitted_cww_pose,
    relative_base_frame_geometry,
)
from pycurves_lib.data.modified_bases import is_modified_base, parent_base_name
from pycurves_lib.topology.lw_exemplars import (
    LWClassification,
    get_lw_exemplar_library,
)

WC_PAIRS = {("A", "T"), ("T", "A"), ("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE_PAIRS = {
    ("G", "T"), ("T", "G"), ("G", "U"), ("U", "G"),
    ("I", "C"), ("C", "I"), ("I", "U"), ("U", "I"),
    ("I", "A"), ("A", "I"),
}
HOOGSTEEN_BASE_SETS = {frozenset(("A", "T")), frozenset(("A", "U")), frozenset(("G", "C"))}
HOOGSTEEN_PURINES = {"A", "G"}
PAIRING_MODE_LABELS = {
    "watson_crick": "Watson-Crick",
    "reverse_watson_crick": "reverse Watson-Crick",
    "hoogsteen": "Hoogsteen",
    "reverse_hoogsteen": "reverse Hoogsteen",
    "wobble": "wobble",
    "other_noncanonical": "other noncanonical",
}
PAIRING_MODES = frozenset(PAIRING_MODE_LABELS)
CLASSIFICATION_STATUSES = frozenset({"assigned", "possible", "unassigned", "conflict"})
PAIR_STATUSES = frozenset({"present", "absent", "uncertain"})
HOOGSTEEN_CONTACT_CUTOFF = 3.7
WATSON_CONTACT_PRESENT_CUTOFF = 3.3
EDGE_CONTACT_CUTOFF = 3.8
MIN_CONTACT_FRAME_PAIRS = 2
GLYCOSIDIC_SIDE_EPSILON = 0.25

CANONICAL_WC_CONTACTS = {
    ("A", "T"): (("N1", "N3"), ("N6", "O4")),
    ("A", "U"): (("N1", "N3"), ("N6", "O4")),
    ("G", "C"): (("N1", "N3"), ("N2", "O2"), ("O6", "N4")),
}

# Edge buckets are the only chemistry vocabulary used by pyCurves frame
# construction.  Pyrimidine C-H is folded into H for internal use.
BASE_EDGE_ATOMS = {
    "A": {
        "W": {"N1", "N6"},
        "H": {"N6", "N7", "C8"},
        "S": {"N3", "C2", "O2'", "O2*"},
    },
    "G": {
        "W": {"N1", "N2", "O6"},
        "H": {"O6", "N7", "C8"},
        "S": {"N2", "N3", "C2", "O2'", "O2*"},
    },
    "I": {
        "W": {"N1", "O6"},
        "H": {"O6", "N7", "C8"},
        "S": {"N3", "C2", "O2'", "O2*"},
    },
    "C": {
        "W": {"O2", "N3", "N4"},
        "H": {"C5", "C6"},
        "S": {"O2", "C2", "O2'", "O2*"},
    },
    "T": {
        "W": {"O2", "N3", "O4"},
        "H": {"C5", "C6", "C7"},
        "S": {"O2", "C2", "O2'", "O2*"},
    },
    "U": {
        "W": {"O2", "N3", "O4"},
        "H": {"C5", "C6"},
        "S": {"O2", "C2", "O2'", "O2*"},
    },
}
EDGE_ORDER = {"W": 0, "H": 1, "S": 2}
GLYCOSIDIC_ATOMS = {
    "A": "N9",
    "G": "N9",
    "I": "N9",
    "P": "N9",
    "R": "N9",
    "C": "N1",
    "T": "N1",
    "U": "N1",
    "Y": "N1",
}
SUGAR_C1_ATOMS = ("C1'", "C1*")


def annotate_context(ctx) -> Dict[str, List[Dict[str, Any]]]:
    """Build pyCurves-native annotations for noncanonical and modified bases."""
    # Coordinates may change between trajectory frames even when the topology
    # context is reused, so never retain fitted annotation geometry here.
    setattr(ctx, "_annotation_standard_base_fit_cache", {})
    base_fit_quality = list(getattr(ctx, "annotations", {}).get("base_fit_quality", []))
    backbone_links = list(getattr(ctx, "annotations", {}).get("backbone_links", []))
    source_base_pairs = _source_base_pair_annotations(ctx)
    source_by_level = {
        int(row["mapped_level"]): row
        for row in source_base_pairs
        if row.get("mapped_level") is not None
    }
    base_pairs = _classify_base_pairs(ctx, source_by_level)
    ctx.pair_contact_geometries = _pair_contact_geometry_index(base_pairs)
    skipped = []
    warnings = _collect_warnings(ctx, base_pairs, base_fit_quality, source_base_pairs)
    modified = [
        row for row in base_fit_quality
        if row.get("is_modified") or row.get("missing_fit_atoms") or row.get("ignored_base_atoms")
    ]
    annotations = {
        "base_pair_annotations": base_pairs,
        "source_base_pair_annotations": source_base_pairs,
        "modified_base_annotations": modified,
        "base_fit_quality": base_fit_quality,
        "skipped_shape_parameters": skipped,
        "noncanonical_warnings": warnings,
        "backbone_links": backbone_links,
    }
    ctx.annotations.update(annotations)
    # Older output paths used these lists to suppress unsupported/noncanonical
    # rows. Keep the keys for compatibility; current shape calculations report
    # the available Hoogsteen/noncanonical values directly.
    ctx.annotations["unsupported_shape_levels"] = []
    ctx.annotations["unsupported_shape_steps"] = []
    return annotations


def render_section_m(annotations: Dict[str, List[Dict[str, Any]]]) -> str:
    """Render the human-readable |M| annotation report."""
    warnings = annotations.get("noncanonical_warnings", [])
    base_pairs = annotations.get("base_pair_annotations", [])
    source_base_pairs = annotations.get("source_base_pair_annotations", [])
    modified = annotations.get("modified_base_annotations", [])

    lines = [
        "  --------------------------------",
        "  |M| pyCurves Annotation Report |",
        "  --------------------------------",
        "",
    ]

    unusual_pairs = [
        row for row in base_pairs
        if (
            not row.get("is_canonical")
            or row.get("has_modified_base")
            or row.get("diagnostic_flags")
            or row.get("candidate_mode")
            or row.get("pair_status") != "present"
            or row.get("frame_mode") in {
                "contact_geometry",
                "provisional_contact_geometry",
            }
            or row.get("normal_branch_mode") == "left_handed_cww"
        )
    ]
    source_unusual = [
        row for row in source_base_pairs
        if _should_report_unmapped_source_pair(row)
    ]
    if not warnings and not unusual_pairs and not source_unusual and not modified:
        lines.extend([
            "  No unusual base-pair identity, modified-base, or base-fitting events were detected.",
            "",
        ])
        return "\n".join(lines)

    if unusual_pairs:
        lines.extend([
            "  Base pair classification",
            "",
            "   Lvl  Strands  Pair       Family              Notes",
            "  ---------------------------------------------------------------",
        ])
        for row in unusual_pairs:
            notes = []
            if row.get("has_modified_base"):
                notes.append("modified")
            pair_status = str(row.get("pair_status") or "present")
            if pair_status != "present":
                notes.append(f"status={pair_status}")
            pairing_mode = str(row.get("pairing_mode") or "").strip()
            if pairing_mode and pairing_mode != "watson_crick":
                notes.append(f"mode={pairing_mode_label(pairing_mode)}")
            candidate_mode = str(row.get("candidate_mode") or "").strip()
            if candidate_mode:
                notes.append(f"possible {pairing_mode_label(candidate_mode)}")
            observed_geometry = base_pair_observed_geometry_annotation(row)
            if observed_geometry:
                notes.append(observed_geometry)
            reference_lw = str(row.get("reference_lw_family") or "").strip()
            if reference_lw and reference_lw != row.get("observed_lw_family"):
                notes.append(f"reference=[{reference_lw}]")
            diagnostics = list(row.get("diagnostic_flags") or [])
            notes.extend(flag for flag in diagnostics if flag not in notes)
            if row.get("frame_mode") == "contact_geometry":
                notes.append("contact_geometry_frames")
            elif row.get("frame_mode") == "provisional_contact_geometry":
                notes.append("provisional_contact_geometry_frames")
            candidate_lw = str(row.get("candidate_lw_family") or "").strip()
            if candidate_lw:
                notes.append(f"possible [{candidate_lw}]")
            if row.get("normal_branch_mode") == "left_handed_cww":
                notes.append("normal_branch=left_handed_cww")
                notes.append(f"normal_sign={int(row.get('pair_normal_sign', -1)):+d}")
                states = [
                    str(row.get(key) or "")
                    for key in ("glycosidic_state_1", "glycosidic_state_2")
                ]
                states = [state for state in states if state]
                if states:
                    notes.append(f"chi={'/'.join(states)}")
            if row.get("contact_confidence"):
                notes.append(f"conf={row['contact_confidence']}")
            contact_count = row.get("contact_count")
            if contact_count:
                notes.append(f"contacts={contact_count}")
            if row.get("pair_subtype"):
                subtype = row["pair_subtype"]
                if subtype not in notes:
                    notes.append(subtype)
            if row.get("shape_skip_reason") and not row.get("is_hoogsteen"):
                notes.append(row["shape_skip_reason"])
            pair = f"{row.get('base_1', '?')}-{row.get('base_2', '?')}"
            strands = f"{row.get('strand_1', '?')}/{row.get('strand_2', '?')}"
            lines.append(
                f"  {row.get('level', 0):4d}  {strands:>7s}  {pair:<9s} "
                f"{row.get('pair_family', ''):<19s} {', '.join(notes)}"
            )
        lines.append("")

    if source_unusual:
        lines.extend([
            "  Source base-pair annotations not represented as Curves paired levels",
            "",
            "   Pair  Residues                  Pairing mode        Source",
            "  ------------------------------------------------------------------------",
        ])
        for row in source_unusual:
            residues = f"{row.get('residue_1', '?')} / {row.get('residue_2', '?')}"
            pair_number = row.get("pair_number") or 0
            pairing_mode = str(row.get("pairing_mode") or "")
            candidate_mode = str(row.get("candidate_mode") or "")
            display_mode = (
                pairing_mode_label(pairing_mode)
                if pairing_mode
                else f"possible {pairing_mode_label(candidate_mode)}"
                if candidate_mode
                else "unassigned"
            )
            lines.append(
                f"  {pair_number:5d}  {residues:<24s} "
                f"{display_mode:<19s} {row.get('source', '')}"
            )
        lines.append("")

    if modified:
        lines.extend([
            "  Modified/nonstandard base fitting",
            "",
            "   Str  Lvl  Residue       Parent  RMSD     Missing fit atoms      Ignored base atoms",
            "  --------------------------------------------------------------------------------",
        ])
        for row in modified:
            residue = _format_residue(row)
            missing = ",".join(row.get("missing_fit_atoms", [])) or "-"
            ignored = ",".join(row.get("ignored_base_atoms", [])) or "-"
            rmsd = row.get("rmsd")
            rmsd_text = f"{float(rmsd):7.3f}" if isinstance(rmsd, (int, float, np.floating)) else "      -"
            lines.append(
                f"  {row.get('strand', 0):4d} {row.get('level', 0):4d}  {residue:<13s} "
                f"{row.get('parent_base', '?'):<6s} {rmsd_text}  {missing:<22s} {ignored}"
            )
        lines.append("")

    if warnings:
        lines.extend([
            "  Warnings",
            "",
            "   Sev  Code                 Location        Message",
            "  -------------------------------------------------------------------------------",
        ])
        for warning in warnings:
            location = warning.get("location", "")
            lines.append(
                f"  {warning.get('severity', ''):<4s} {warning.get('code', ''):<20s} "
                f"{location:<15s} {warning.get('message', '')}"
            )
        lines.append("")
    return "\n".join(lines)


def render_section_l(annotations: Dict[str, List[Dict[str, Any]]]) -> str:
    """Backward-compatible alias for the annotation report renderer."""
    return render_section_m(annotations)


def base_pair_geometry_tag(row: Dict[str, Any]) -> str:
    """Return a compact cWW/tWH-style tag when edge and orientation are known."""
    manual_tag = str(row.get("manual_geometry_tag") or "").strip()
    if manual_tag:
        return manual_tag
    if "lw_family_confident" in row and not bool(row.get("lw_family_confident")):
        return ""
    orientation = str(row.get("glycosidic_orientation") or "").strip().lower()
    prefix = {"cis": "c", "trans": "t"}.get(orientation, "")

    edge_1 = str(row.get("edge_1") or "").strip().upper()
    edge_2 = str(row.get("edge_2") or "").strip().upper()
    if not (edge_1 and edge_2):
        edge_pair = str(row.get("edge_pair") or "").strip().upper()
        if "/" in edge_pair:
            parts = [part.strip() for part in edge_pair.split("/", 1)]
            edge_1, edge_2 = parts[0], parts[1]

    if prefix and edge_1 and edge_2:
        return f"{prefix}{edge_1}{edge_2}"
    return manual_tag


def base_pair_geometry_annotation(row: Dict[str, Any]) -> str:
    """Return the user-facing calculation/reference geometry label."""
    tag = base_pair_geometry_tag(row)
    if tag:
        return f"[{tag}]"
    edge_pair = str(row.get("edge_pair") or "").strip()
    return edge_pair


def base_pair_observed_geometry_tag(row: Dict[str, Any]) -> str:
    """Return only geometry assigned from the current coordinates."""
    if "observed_lw_family" in row:
        return str(row.get("observed_lw_family") or "").strip()
    return base_pair_geometry_tag(row)


def base_pair_observed_geometry_annotation(row: Dict[str, Any]) -> str:
    tag = base_pair_observed_geometry_tag(row)
    return f"[{tag}]" if tag else ""


def pairing_mode_label(mode: str) -> str:
    """Return the human-readable label for one controlled pairing mode."""
    return PAIRING_MODE_LABELS.get(str(mode or "").strip(), "")


def base_pair_pairing_classification(
    row: Dict[str, Any],
    source_assignment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify a present pair without mixing its mode with uncertainty."""
    stored_mode = str(row.get("pairing_mode") or "").strip()
    if stored_mode == "other":
        stored_mode = "other_noncanonical"
    if stored_mode and stored_mode not in PAIRING_MODES:
        stored_mode = ""

    base_1 = parent_base_name(str(row.get("parent_base_1") or row.get("base_1") or ""))
    base_2 = parent_base_name(str(row.get("parent_base_2") or row.get("base_2") or ""))
    pair_status = str(row.get("pair_status") or "present").strip().lower()
    if pair_status not in PAIR_STATUSES:
        pair_status = "uncertain"
    diagnostics = list(dict.fromkeys(
        str(flag) for flag in row.get("diagnostic_flags", []) if flag
    ))
    if pair_status != "present":
        candidate = str(row.get("candidate_mode") or "").strip()
        if candidate not in PAIRING_MODES:
            candidate = ""
        return {
            "pairing_mode": "",
            "candidate_mode": candidate,
            "classification_status": "possible" if candidate else "unassigned",
            "diagnostic_flags": diagnostics,
        }

    tag = str(row.get("observed_lw_family") or "").strip()
    if not tag and not row.get("reference_lw_family"):
        tag = base_pair_geometry_tag(row)
    geometry_mode = _pairing_mode_from_lw_tag(tag, base_1, base_2)
    if geometry_mode:
        diagnostics.extend(
            _hoogsteen_protonation_diagnostics(geometry_mode, base_1, base_2)
        )
        return {
            "pairing_mode": geometry_mode,
            "candidate_mode": "",
            "classification_status": "assigned",
            "diagnostic_flags": list(dict.fromkeys(diagnostics)),
        }

    if tag.lower() == "cww":
        if (base_1, base_2) in WC_PAIRS:
            mode = "watson_crick"
        elif (base_1, base_2) in WOBBLE_PAIRS:
            mode = "wobble"
        else:
            mode = "other_noncanonical"
        return {
            "pairing_mode": mode,
            "candidate_mode": "",
            "classification_status": "assigned",
            "diagnostic_flags": diagnostics,
        }
    if (
        len(tag) == 3
        and tag[0].lower() in {"c", "t"}
        and tag[1].upper() in {"W", "H", "S"}
        and tag[2].upper() in {"W", "H", "S"}
    ):
        return {
            "pairing_mode": "other_noncanonical",
            "candidate_mode": "",
            "classification_status": "assigned",
            "diagnostic_flags": diagnostics,
        }


    if stored_mode:
        return {
            "pairing_mode": stored_mode,
            "candidate_mode": "",
            "classification_status": "assigned",
            "diagnostic_flags": diagnostics,
        }

    source_assignment = dict(source_assignment or {})
    source_status = str(source_assignment.get("classification_status") or "").strip()
    source_mode = str(source_assignment.get("pairing_mode") or "").strip()
    source_candidate = str(source_assignment.get("candidate_mode") or "").strip()
    if source_status == "assigned" and source_mode in PAIRING_MODES:
        return {
            "pairing_mode": source_mode,
            "candidate_mode": "",
            "classification_status": "assigned",
            "diagnostic_flags": diagnostics,
        }
    if source_candidate in PAIRING_MODES:
        return {
            "pairing_mode": "",
            "candidate_mode": source_candidate,
            "classification_status": "possible",
            "diagnostic_flags": list(dict.fromkeys(diagnostics + ["source_assignment_uncertain"])),
        }

    if (base_1, base_2) in WOBBLE_PAIRS:
        return {
            "pairing_mode": "",
            "candidate_mode": "wobble",
            "classification_status": "possible",
            "diagnostic_flags": list(dict.fromkeys(diagnostics + ["wobble_identity_without_cww_geometry"])),
        }
    if "possible_hoogsteen" in diagnostics:
        return {
            "pairing_mode": "",
            "candidate_mode": "hoogsteen",
            "classification_status": "possible",
            "diagnostic_flags": diagnostics,
        }
    return {
        "pairing_mode": "",
        "candidate_mode": "",
        "classification_status": "unassigned",
        "diagnostic_flags": diagnostics,
    }


def base_pair_pairing_mode(row: Dict[str, Any], source_mode: str = "") -> str:
    """Backward-compatible convenience wrapper returning assigned modes only."""
    source_assignment = None
    if source_mode:
        normalized = str(source_mode).strip()
        reverse = {
            "Watson-Crick": "watson_crick",
            "Hoogsteen": "hoogsteen",
            "reverse Hoogsteen": "reverse_hoogsteen",
            "wobble": "wobble",
            "other": "other_noncanonical",
            **{mode: mode for mode in PAIRING_MODES},
        }
        normalized = reverse.get(normalized, "")
        if normalized:
            source_assignment = {
                "pairing_mode": normalized,
                "classification_status": "assigned",
            }
    return base_pair_pairing_classification(row, source_assignment)["pairing_mode"]


def _pairing_mode_from_lw_tag(tag: str, base_1: str, base_2: str) -> str:
    normalized = str(tag or "").strip()
    if len(normalized) != 3:
        return ""
    orientation = normalized[0].lower()
    edge_1 = normalized[1].upper()
    edge_2 = normalized[2].upper()
    if orientation == "t" and edge_1 == edge_2 == "W":
        return "reverse_watson_crick"
    if not _has_classic_hoogsteen_edges(base_1, edge_1, base_2, edge_2):
        return ""
    if orientation == "c":
        return "hoogsteen"
    if orientation == "t":
        return "reverse_hoogsteen"
    return ""
def _hoogsteen_protonation_diagnostics(
    mode: str,
    base_1: str,
    base_2: str,
) -> List[str]:
    if (
        mode in {"hoogsteen", "reverse_hoogsteen"}
        and frozenset((base_1, base_2)) == frozenset(("G", "C"))
    ):
        return ["cytosine_protonation_unresolved"]
    return []




def _classify_base_pairs(ctx, source_by_level: Optional[Dict[int, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    source_by_level = source_by_level or {}
    rows = []
    for level in range(1, ctx.nux + 1):
        active = [strand for strand in range(ctx.nst) if _residue_for(ctx, strand, level) is not None]
        if len(active) < 2:
            continue
        if len(active) > 2:
            rows.append({
                "level": level,
                "pair_id": ":".join(str(value) for value in sorted((
                    int(_residue_for(ctx, active[0], level)["subunit"]),
                    int(_residue_for(ctx, active[1], level)["subunit"]),
                ))),
                "reference_pair": True,
                "subunit_1": int(_residue_for(ctx, active[0], level)["subunit"]),
                "subunit_2": int(_residue_for(ctx, active[1], level)["subunit"]),
                "strand_1": active[0] + 1,
                "strand_2": active[1] + 1,
                "residue_1": _format_residue(_residue_for(ctx, active[0], level)),
                "residue_2": _format_residue(_residue_for(ctx, active[1], level)),
                "base_1": parent_base_name(_residue_for(ctx, active[0], level)["residue_name"]),
                "base_2": parent_base_name(_residue_for(ctx, active[1], level)["residue_name"]),
                "parent_base_1": parent_base_name(_residue_for(ctx, active[0], level)["residue_name"]),
                "parent_base_2": parent_base_name(_residue_for(ctx, active[1], level)["residue_name"]),
                "pair_family": "ambiguous_topology",
                "identity_class": "ambiguous_topology",
                "pair_subtype": f"{len(active)} active strands at this level",
                "pair_status": "uncertain",
                "observed_lw_family": "",
                "reference_lw_family": "",
                "pairing_mode": "",
                "candidate_mode": "",
                "classification_status": "unassigned",
                "diagnostic_flags": ["ambiguous_topology"],
                "is_canonical": False,
                "is_mismatch": False,
                "is_hoogsteen": False,
                "has_modified_base": any(is_modified_base(_residue_for(ctx, s, level)["residue_name"]) for s in active),
                "confidence": "topology_warning",
                "method": "identity_and_inp_topology",
                "evidence_source": "reference_topology",
                "shape_skip_reason": "",
            })
            continue

        s1, s2 = active
        r1 = _residue_for(ctx, s1, level)
        r2 = _residue_for(ctx, s2, level)
        b1 = parent_base_name(r1["residue_name"])
        b2 = parent_base_name(r2["residue_name"])
        family, subtype, canonical = _pair_family(b1, b2)
        source_pair = source_by_level.get(level)
        source_hoogsteen = bool(source_pair and source_pair.get("is_hoogsteen"))
        manual_geometry = _pair_geometry_marker(ctx, level, s1 + 1, s2 + 1)
        contact_geometry = _contact_geometry_for_pair(
            ctx,
            s1,
            s2,
            level,
            r1,
            r2,
            source_hoogsteen=source_hoogsteen,
            canonical_identity=canonical,
            manual_geometry=manual_geometry,
        )
        diagnostic_flags = _geometry_diagnostics(
            ctx, s1, s2, level, contact_geometry
        )
        pair_status = _pair_status_from_geometry(contact_geometry)
        observed_lw_family = str(
            contact_geometry.get("observed_lw_family") or ""
        )
        reference_lw_family = str(
            contact_geometry.get("reference_lw_family")
            or _source_lw_family_for_strand_order(
                source_pair, first_strand=s1 + 1
            )
        )
        reference_pairing_mode = (
            str(source_pair.get("pairing_mode") or "") if source_pair else ""
        ) or _pairing_mode_from_lw_tag(reference_lw_family, b1, b2)
        calculation_is_hoogsteen = reference_pairing_mode in {
            "hoogsteen", "reverse_hoogsteen"
        }
        if (
            pair_status == "present"
            and observed_lw_family
            and (
                family == "mismatch"
                or observed_lw_family.lower() == "tww"
            )
        ):
            pair_family = "hbonded_noncanonical"
            pair_subtype = observed_lw_family
            confidence = contact_geometry.get("confidence", "coordinate_geometry")
            method = "coordinate_geometry"
        else:
            pair_family = family
            pair_subtype = subtype
            confidence = contact_geometry.get("confidence", "identity")
            method = "identity_and_coordinate_geometry"
        frame_mode = contact_geometry.get(
            "frame_mode",
            "legacy_canonical" if canonical else "fitted_fallback",
        )
        frame_basis = contact_geometry.get("frame_basis", "")
        subunit_1 = int(r1["subunit"])
        subunit_2 = int(r2["subunit"])
        pair_id = f"{min(subunit_1, subunit_2)}:{max(subunit_1, subunit_2)}"
        pair_row = {
            "pair_id": pair_id,
            "reference_pair": True,
            "level": level,
            "strand_1": s1 + 1,
            "strand_2": s2 + 1,
            "subunit_1": subunit_1,
            "subunit_2": subunit_2,
            "residue_1": _format_residue(r1),
            "residue_2": _format_residue(r2),
            "base_1": b1,
            "base_2": b2,
            "parent_base_1": b1,
            "parent_base_2": b2,
            "identity_class": family,
            "pair_family": pair_family,
            "pair_subtype": pair_subtype,
            "pair_status": pair_status,
            "observed_lw_family": observed_lw_family,
            "reference_lw_family": reference_lw_family,
            "candidate_lw_family": contact_geometry.get(
                "candidate_lw_family", ""
            ),
            "pairing_mode": "",
            "candidate_mode": "",
            "classification_status": "unassigned",
            "diagnostic_flags": diagnostic_flags,
            "reference_pairing_mode": reference_pairing_mode,
            "reference_classification_status": (
                source_pair.get("classification_status", "") if source_pair else ""
            ),
            "calculation_is_hoogsteen": calculation_is_hoogsteen,
            "is_canonical": False,
            "is_mismatch": False,
            "is_hoogsteen": False,
            "has_modified_base": is_modified_base(r1["residue_name"]) or is_modified_base(r2["residue_name"]),
            "confidence": confidence,
            "method": method,
            "evidence_source": contact_geometry.get("observed_geometry_source", "coordinates"),
            "edge_1": contact_geometry.get("edge_1", ""),
            "edge_2": contact_geometry.get("edge_2", ""),
            "edge_pair": contact_geometry.get("edge_pair", ""),
            "glycosidic_orientation": contact_geometry.get("glycosidic_orientation", ""),
            "lw_strand_orientation": contact_geometry.get("lw_strand_orientation", ""),
            "strand_direction": contact_geometry.get("strand_direction", ""),
            "topology_strand_direction": contact_geometry.get("topology_strand_direction", ""),
            "frame_mode": frame_mode,
            "frame_basis": frame_basis,
            "input_geometry_tag": contact_geometry.get(
                "input_geometry_tag", ""
            ),
            "geometry_resolution_status": contact_geometry.get(
                "geometry_resolution_status", ""
            ),
            "contact_atom_pairs": contact_geometry.get("contact_atom_pairs", []),
            "contact_count": contact_geometry.get("contact_count", 0),
            "contact_confidence": contact_geometry.get("confidence", ""),
            "lw_family_confident": bool(
                contact_geometry.get("observed_lw_family_confident")
            ),
            "manual_geometry_tag": contact_geometry.get("manual_geometry_tag", ""),
            "contact_geometry": contact_geometry,
            "source_pair_number": source_pair.get("pair_number") if source_pair else None,
            "shape_parameters_supported": True,
            "shape_skip_reason": "",
        }
        classification = base_pair_pairing_classification(pair_row)
        pair_row.update(classification)
        mode = pair_row["pairing_mode"]
        if mode in {"hoogsteen", "reverse_hoogsteen"}:
            pair_row["pair_family"] = "hoogsteen"
        elif mode == "reverse_watson_crick":
            pair_row["pair_family"] = mode
        pair_row["is_canonical"] = (
            pair_status == "present" and mode == "watson_crick"
        )
        pair_row["is_mismatch"] = (
            pair_status == "present" and family == "mismatch"
        )
        pair_row["is_hoogsteen"] = mode in {
            "hoogsteen", "reverse_hoogsteen"
        }
        rows.append(pair_row)
    return rows


def _pair_family(base_1: str, base_2: str) -> Tuple[str, str, bool]:
    pair = (base_1, base_2)
    if pair in WC_PAIRS:
        return "watson_crick", "canonical_identity", True
    if pair in WOBBLE_PAIRS:
        return "wobble", "recognized_noncanonical_identity", False
    if "unknown" in pair:
        return "unknown", "unrecognized_base_identity", False
    return "mismatch", "noncanonical_identity", False


def _geometry_diagnostics(
    ctx,
    strand_1: int,
    strand_2: int,
    level: int,
    contact_geometry: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return diagnostic evidence without promoting it to a pair class."""
    diagnostics: List[str] = []
    try:
        residue_1 = _residue_for(ctx, strand_1, level)
        residue_2 = _residue_for(ctx, strand_2, level)
        if residue_1 is None or residue_2 is None:
            return ["missing_residue_geometry"]
        observed_lw = str(
            (contact_geometry or {}).get("observed_lw_family") or ""
        )
        resolution_status = str(
            (contact_geometry or {}).get("geometry_resolution_status") or ""
        )
        if resolution_status == "unresolved":
            diagnostics.append("unresolved_lw_geometry")
            if int((contact_geometry or {}).get("contact_count") or 0) == 1:
                diagnostics.append("single_independent_contact")
            candidate_lw = str(
                (contact_geometry or {}).get("candidate_lw_family") or ""
            )
            if candidate_lw:
                diagnostics.append(f"possible_lw_{candidate_lw.lower()}")
        if not observed_lw and _has_hoogsteen_heavy_atom_contacts(
            ctx, residue_1, residue_2
        ):
            diagnostics.append("possible_hoogsteen")
    except Exception:
        diagnostics.append("geometry_evaluation_failed")
    return diagnostics


def _pair_status_from_geometry(contact_geometry: Dict[str, Any]) -> str:
    """Classify pair presence from current-coordinate evidence only."""
    fitted = dict(contact_geometry.get("fitted_pair_geometry") or {})
    eligible = bool(fitted.get("eligible"))
    contact_count = int(contact_geometry.get("contact_count") or 0)
    confident_lw = bool(
        contact_geometry.get("observed_lw_family_confident")
    )
    unresolved_contact = (
        contact_geometry.get("input_geometry_tag") == "unresolved"
        and contact_geometry.get("frame_mode")
        == "provisional_contact_geometry"
    )
    # Multiple independent edge contacts can establish a real, highly
    # buckled pair even when the fitted normal-angle envelope is exceeded.
    if contact_count >= 2 and confident_lw:
        return "present"
    if eligible and contact_count >= 1 and unresolved_contact:
        return "present"
    if eligible and contact_count >= 1 and (confident_lw or contact_count >= 2):
        return "present"
    if eligible or contact_count >= 1:
        return "uncertain"
    return "absent"



def _standard_base_fit_for_residue(ctx, residue: Dict[str, Any]) -> Optional[Dict[str, object]]:
    """Fit one mapped residue for annotation, independent of calculation convention."""
    try:
        subunit = int(residue["subunit"])
    except (KeyError, TypeError, ValueError):
        return None

    cache = getattr(ctx, "_annotation_standard_base_fit_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_annotation_standard_base_fit_cache", cache)
    if subunit in cache:
        return cache[subunit]

    result = None
    try:
        mol = ctx.molecule
        boundaries = np.asarray(mol.subunit_boundaries, dtype=int)
        start = int(boundaries[subunit - 1])
        end = int(boundaries[subunit])
        residue_atoms: Dict[str, int] = {}
        for atom_idx in range(start, end):
            atom_name = str(mol.atom_names[atom_idx]).strip().upper()
            residue_atoms.setdefault(atom_name, atom_idx)

        library = getattr(ctx, "_annotation_standard_base_library", None)
        fitter = getattr(ctx, "_annotation_standard_base_fitter", None)
        if library is None or fitter is None:
            library = BaseReferenceLibrary.load("standard")
            fitter = BaseFrameFitter(library)
            setattr(ctx, "_annotation_standard_base_library", library)
            setattr(ctx, "_annotation_standard_base_fitter", fitter)

        base = parent_base_name(residue["residue_name"])
        template = library.template_for_base(base)
        if template is not None:
            result = fitter.fit(template, residue_atoms, mol.coordinates)
    except (AttributeError, IndexError, TypeError, ValueError):
        result = None

    cache[subunit] = result
    return result


def _standard_fitted_pair_geometry(
    ctx,
    residue_1: Dict[str, Any],
    residue_2: Dict[str, Any],
) -> Optional[Dict[str, object]]:
    fit_1 = _standard_base_fit_for_residue(ctx, residue_1)
    fit_2 = _standard_base_fit_for_residue(ctx, residue_2)
    if fit_1 is None or fit_2 is None:
        return None
    return relative_base_frame_geometry(fit_1, fit_2)


def _lw_exemplar_classification(
    ctx,
    residue_1: Dict[str, Any],
    residue_2: Dict[str, Any],
) -> Optional[LWClassification]:
    fit_1 = _standard_base_fit_for_residue(ctx, residue_1)
    fit_2 = _standard_base_fit_for_residue(ctx, residue_2)
    if fit_1 is None or fit_2 is None:
        return None
    library = getattr(ctx, "_annotation_lw_exemplar_library", None)
    if library is None:
        library = get_lw_exemplar_library()
        setattr(ctx, "_annotation_lw_exemplar_library", library)
    return library.classify_fits(
        parent_base_name(residue_1["residue_name"]),
        fit_1,
        parent_base_name(residue_2["residue_name"]),
        fit_2,
    )


def _contact_geometry_for_pair(
    ctx,
    strand_1: int,
    strand_2: int,
    level: int,
    residue_1: Dict[str, Any],
    residue_2: Dict[str, Any],
    source_hoogsteen: bool,
    canonical_identity: bool,
    manual_geometry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_1 = parent_base_name(residue_1["residue_name"])
    base_2 = parent_base_name(residue_2["residue_name"])
    atom_map_1 = _atom_map_for_residue(ctx, int(residue_1["subunit"]))
    atom_map_2 = _atom_map_for_residue(ctx, int(residue_2["subunit"]))
    contacts = _selected_edge_contacts(base_1, base_2, atom_map_1, atom_map_2)
    contact_atoms_1 = [item["atom_1"] for item in contacts]
    contact_atoms_2 = [item["atom_2"] for item in contacts]
    edge_1, edge_1_score, edge_1_ambiguous = _dominant_edge(base_1, contact_atoms_1)
    edge_2, edge_2_score, edge_2_ambiguous = _dominant_edge(base_2, contact_atoms_2)
    manual_geometry = dict(manual_geometry or {})
    unresolved_requested = bool(
        manual_geometry.get("kind") == "unresolved"
        or str(manual_geometry.get("tag") or "").lower() == "unresolved"
    )
    manual_lw_requested = bool(manual_geometry) and not unresolved_requested
    fitted_geometry = _standard_fitted_pair_geometry(ctx, residue_1, residue_2)
    fitted_cww = is_fitted_cww_pose(fitted_geometry)
    lw_classification = _lw_exemplar_classification(ctx, residue_1, residue_2)
    confident_exemplar = bool(
        lw_classification is not None and lw_classification.confident
    )
    canonical_contact_orientation = _canonical_watson_contact_orientation(
        base_1, base_2, atom_map_1, atom_map_2
    )
    coordinate_orientation = (
        (lw_classification.glycosidic_orientation if confident_exemplar else "")
        or ("cis" if fitted_cww else "")
        or canonical_contact_orientation
        or _glycosidic_orientation(base_1, base_2, atom_map_1, atom_map_2, contacts)
    )
    coordinate_edge_1 = edge_1
    coordinate_edge_2 = edge_2
    if confident_exemplar:
        coordinate_edge_1 = lw_classification.edge_1
        coordinate_edge_2 = lw_classification.edge_2
    elif fitted_cww:
        coordinate_edge_1 = coordinate_edge_2 = "W"
    has_reliable_coordinate_contacts = (
        len(contacts) >= MIN_CONTACT_FRAME_PAIRS
        and bool(coordinate_edge_1)
        and bool(coordinate_edge_2)
        and not edge_1_ambiguous
        and not edge_2_ambiguous
    )
    coordinate_trans_ww = bool(
        coordinate_edge_1 == coordinate_edge_2 == "W"
        and coordinate_orientation == "trans"
        and has_reliable_coordinate_contacts
    )
    observed_lw_family = (
        lw_classification.tag
        if confident_exemplar
        else "cWW"
        if fitted_cww
        else "tWW"
        if coordinate_trans_ww
        else ""
    )
    observed_lw_family_confident = bool(
        confident_exemplar or fitted_cww or coordinate_trans_ww
    )
    provisional_unresolved = bool(
        unresolved_requested and not observed_lw_family_confident
    )
    topology_strand_direction = _strand_direction(ctx, strand_1, strand_2)
    candidate_lw_family = ""
    orientation_prefix = {
        "cis": "c",
        "trans": "t",
    }.get(str(coordinate_orientation).lower(), "")
    if (
        provisional_unresolved
        and orientation_prefix
        and coordinate_edge_1
        and coordinate_edge_2
        and not edge_1_ambiguous
        and not edge_2_ambiguous
        and infer_lw_strand_orientation(
            coordinate_orientation,
            coordinate_edge_1,
            coordinate_edge_2,
        ) == topology_strand_direction
    ):
        candidate_lw_family = (
            f"{orientation_prefix}{coordinate_edge_1}{coordinate_edge_2}"
        )

    if manual_lw_requested:
        edge_1 = str(manual_geometry.get("edge_1", "")).upper()
        edge_2 = str(manual_geometry.get("edge_2", "")).upper()
        edge_1_ambiguous = False
        edge_2_ambiguous = False
    elif confident_exemplar:
        edge_1 = lw_classification.edge_1
        edge_2 = lw_classification.edge_2
        edge_1_ambiguous = False
        edge_2_ambiguous = False
    elif fitted_cww:
        edge_1 = edge_2 = "W"
        edge_1_ambiguous = False
        edge_2_ambiguous = False
    edge_pair = f"{edge_1}/{edge_2}" if edge_1 and edge_2 else ""
    manual_glycosidic_orientation = (
        manual_geometry.get("glycosidic_orientation", "")
        if manual_lw_requested
        else ""
    )
    glycosidic_orientation = (
        manual_glycosidic_orientation or coordinate_orientation
    )
    manual_lw_strand_orientation = (
        (
            manual_geometry.get("lw_strand_orientation")
            or manual_geometry.get("strand_direction")
        )
        if manual_lw_requested
        else ""
    )
    lw_strand_orientation = (
        ""
        if provisional_unresolved
        else manual_lw_strand_orientation or infer_lw_strand_orientation(
            glycosidic_orientation,
            edge_1,
            edge_2,
        )
    )

    has_reliable_contacts = (
        len(contacts) >= MIN_CONTACT_FRAME_PAIRS
        and bool(edge_1)
        and bool(edge_2)
        and not edge_1_ambiguous
        and not edge_2_ambiguous
    )
    # An explicit LW tag in the input is authoritative even if the raw
    # coordinate contact finder cannot recover two short atom pairs. The frame
    # builder can fall back to the complete atoms belonging to those edges.
    has_manual_frame_geometry = (
        manual_lw_requested and bool(edge_1) and bool(edge_2)
    )
    has_provisional_contact_geometry = (
        provisional_unresolved
        and bool(contacts)
    )
    forced_noncanonical = bool(source_hoogsteen)
    if has_manual_frame_geometry:
        # cWW is the canonical standard-frame family.  Other directed LW
        # families, including tWW, use their explicitly requested contact
        # geometry and its coordinate-derived normal branch.
        is_cww = (
            glycosidic_orientation == "cis"
            and edge_1 == edge_2 == "W"
        )
        frame_mode = "fitted_fallback" if is_cww else "contact_geometry"
    elif unresolved_requested and observed_lw_family_confident:
        is_cww = observed_lw_family == "cWW"
        frame_mode = "fitted_fallback" if is_cww else "contact_geometry"
    elif has_provisional_contact_geometry:
        frame_mode = "provisional_contact_geometry"
    elif unresolved_requested:
        frame_mode = "fitted_fallback"
    elif canonical_identity and not forced_noncanonical:
        frame_mode = "legacy_canonical"
    else:
        # Coordinate-only/source annotations remain descriptive. They must not
        # silently replace the standard fitted calculation frames.
        frame_mode = "fitted_fallback"
    frame_basis = (
        "atom_contact_axis"
        if frame_mode == "provisional_contact_geometry"
        else "observed_edge"
        if frame_mode == "contact_geometry"
        else ""
    )

    contact_supported_trans_ww = bool(
        edge_1 == edge_2 == "W"
        and glycosidic_orientation == "trans"
        and has_reliable_contacts
    )
    lw_family_confident = bool(
        manual_lw_requested
        or confident_exemplar
        or fitted_cww
        or contact_supported_trans_ww
    )
    if provisional_unresolved and has_provisional_contact_geometry:
        confidence = "provisional_atom_contact_geometry"
    elif provisional_unresolved:
        confidence = "unresolved_geometry"
    elif manual_lw_requested:
        confidence = "manual_inp_geometry"
    elif confident_exemplar:
        confidence = "fitted_lw_exemplar"
    elif fitted_cww:
        confidence = "fitted_standard_frames"
    elif has_reliable_contacts:
        confidence = "edge_contacts"
    elif contacts:
        confidence = "weak_or_ambiguous_contacts"
    elif forced_noncanonical:
        confidence = "source_without_contacts"
    else:
        confidence = "identity"

    return {
        "level": int(level),
        "strand_1": int(strand_1 + 1),
        "strand_2": int(strand_2 + 1),
        "base_1": base_1,
        "base_2": base_2,
        "edge_1": edge_1,
        "edge_2": edge_2,
        "edge_pair": edge_pair,
        "observed_lw_family": observed_lw_family,
        "reference_lw_family": (
            manual_geometry.get("tag", "") if manual_lw_requested else ""
        ),
        "candidate_lw_family": candidate_lw_family,
        "observed_lw_family_confident": observed_lw_family_confident,
        "observed_edge_1": coordinate_edge_1,
        "observed_edge_2": coordinate_edge_2,
        "observed_glycosidic_orientation": coordinate_orientation,
        "observed_geometry_source": (
            "fitted_lw_exemplar"
            if confident_exemplar
            else "fitted_standard_frames"
            if fitted_cww
            else "coordinates"
        ),
        "orientation": manual_geometry.get("orientation", ""),
        "glycosidic_orientation": glycosidic_orientation,
        "lw_strand_orientation": lw_strand_orientation,
        "strand_direction": lw_strand_orientation,
        "strand_direction_source": (
            manual_geometry.get("strand_direction_source", "")
            if manual_lw_strand_orientation
            else "coordinate_provisional"
            if provisional_unresolved
            else "inferred_from_contact_geometry" if lw_strand_orientation else ""
        ),
        "topology_strand_direction": topology_strand_direction,
        "frame_mode": frame_mode,
        "frame_basis": frame_basis,
        "contact_atom_pairs": contacts,
        "contact_count": len(contacts),
        "confidence": confidence,
        "geometry_source": (
            "inp"
            if manual_lw_requested
            else "inp_unresolved_coordinates"
            if unresolved_requested
            else "fitted_lw_exemplar"
            if confident_exemplar
            else "fitted_standard_frames"
            if fitted_cww
            else "coordinates"
        ),
        "lw_family_confident": lw_family_confident,
        "lw_exemplar": lw_classification.as_dict() if lw_classification else {},
        "fitted_pair_geometry": fitted_geometry or {},
        "edge_score_1": edge_1_score,
        "edge_score_2": edge_2_score,
        "edge_1_ambiguous": edge_1_ambiguous,
        "edge_2_ambiguous": edge_2_ambiguous,
        "manual_geometry_tag": (
            manual_geometry.get("tag", "") if manual_lw_requested else ""
        ),
        "input_geometry_tag": (
            str(manual_geometry.get("tag") or "") if manual_geometry else ""
        ),
        "geometry_resolution_status": (
            "unresolved"
            if provisional_unresolved
            else "resolved" if unresolved_requested else ""
        ),
        "manual_geometry_strand": manual_geometry.get("annotated_strand"),
        "source_hoogsteen": bool(source_hoogsteen),
    }


def _selected_edge_contacts(
    base_1: str,
    base_2: str,
    atom_map_1: Dict[str, np.ndarray],
    atom_map_2: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    atoms_1 = _edge_contact_atoms(base_1)
    atoms_2 = _edge_contact_atoms(base_2)
    candidates = []
    for atom_1 in atoms_1:
        coord_1 = atom_map_1.get(atom_1)
        if coord_1 is None:
            continue
        for atom_2 in atoms_2:
            coord_2 = atom_map_2.get(atom_2)
            if coord_2 is None:
                continue
            distance = float(np.linalg.norm(coord_1 - coord_2))
            if distance > EDGE_CONTACT_CUTOFF:
                continue
            candidates.append({
                "atom_1": atom_1,
                "atom_2": atom_2,
                "distance": distance,
                "edges_1": sorted(_edges_for_atom(base_1, atom_1), key=lambda item: EDGE_ORDER.get(item, 99)),
                "edges_2": sorted(_edges_for_atom(base_2, atom_2), key=lambda item: EDGE_ORDER.get(item, 99)),
                "weak_contact": atom_1.startswith("C") or atom_2.startswith("C"),
            })

    candidates.sort(key=lambda item: (bool(item["weak_contact"]), item["distance"]))
    used_1 = set()
    used_2 = set()
    selected = []
    for candidate in candidates:
        if candidate["atom_1"] in used_1 or candidate["atom_2"] in used_2:
            continue
        used_1.add(candidate["atom_1"])
        used_2.add(candidate["atom_2"])
        selected.append(candidate)
        if len(selected) >= 4:
            break
    return selected


def _edge_contact_atoms(base: str) -> List[str]:
    atoms = set()
    for edge_atoms in BASE_EDGE_ATOMS.get(base, {}).values():
        atoms.update(edge_atoms)
    return sorted(atoms)


def _edges_for_atom(base: str, atom_name: str) -> List[str]:
    return [
        edge
        for edge, atoms in BASE_EDGE_ATOMS.get(base, {}).items()
        if atom_name in atoms
    ]


def _dominant_edge(base: str, contact_atoms: List[str]) -> Tuple[str, float, bool]:
    if not contact_atoms:
        return "", 0.0, False
    scores = []
    for edge, atoms in BASE_EDGE_ATOMS.get(base, {}).items():
        matched = [atom for atom in contact_atoms if atom in atoms]
        if not matched:
            continue
        unique = [
            atom for atom in matched
            if len(_edges_for_atom(base, atom)) == 1
        ]
        score = float(len(matched)) + 0.35 * float(len(unique)) - 0.01 * EDGE_ORDER.get(edge, 99)
        scores.append((score, edge))
    if not scores:
        return "", 0.0, False
    scores.sort(reverse=True)
    best_score, best_edge = scores[0]
    ambiguous = len(scores) > 1 and abs(best_score - scores[1][0]) < 0.25
    return best_edge, best_score, ambiguous


def _strand_direction(ctx, strand_1: int, strand_2: int) -> str:
    try:
        return "parallel" if int(ctx.idr[strand_1]) == int(ctx.idr[strand_2]) else "antiparallel"
    except Exception:
        return "unknown"


def infer_lw_strand_orientation(
    glycosidic_orientation: str,
    edge_1: str,
    edge_2: str,
) -> str:
    orientation = str(glycosidic_orientation or "").strip().lower()
    orientation = {"cis": "c", "trans": "t"}.get(orientation, orientation)
    first_edge = str(edge_1 or "").strip().upper()
    second_edge = str(edge_2 or "").strip().upper()
    if orientation not in {"c", "t"} or first_edge not in EDGE_ORDER or second_edge not in EDGE_ORDER:
        return ""
    one_hoogsteen_edge = (first_edge == "H") ^ (second_edge == "H")
    cis_is_parallel = one_hoogsteen_edge
    is_parallel = cis_is_parallel if orientation == "c" else not cis_is_parallel
    return "parallel" if is_parallel else "antiparallel"


def _canonical_watson_contact_orientation(
    base_1: str,
    base_2: str,
    atom_map_1: Dict[str, np.ndarray],
    atom_map_2: Dict[str, np.ndarray],
) -> str:
    pattern = CANONICAL_WC_CONTACTS.get((base_1, base_2))
    reversed_pattern = False
    if pattern is None:
        pattern = CANONICAL_WC_CONTACTS.get((base_2, base_1))
        reversed_pattern = pattern is not None
    if not pattern:
        return ""
    matches = 0
    for atom_1, atom_2 in pattern:
        if reversed_pattern:
            atom_1, atom_2 = atom_2, atom_1
        distance = _atom_distance(atom_map_1, atom_1, atom_map_2, atom_2)
        if distance is not None and distance <= EDGE_CONTACT_CUTOFF:
            matches += 1
    return "cis" if matches >= 2 else ""


def _glycosidic_orientation(
    base_1: str,
    base_2: str,
    atom_map_1: Dict[str, np.ndarray],
    atom_map_2: Dict[str, np.ndarray],
    contacts: List[Dict[str, Any]],
) -> str:
    gly_1 = _glycosidic_atom_point(base_1, atom_map_1)
    gly_2 = _glycosidic_atom_point(base_2, atom_map_2)
    sugar_1 = _sugar_c1_point(atom_map_1)
    sugar_2 = _sugar_c1_point(atom_map_2)
    if gly_1 is None or gly_2 is None or sugar_1 is None or sugar_2 is None:
        return ""

    contact_points_1 = [
        atom_map_1[item["atom_1"]]
        for item in contacts
        if item.get("atom_1") in atom_map_1
    ]
    contact_points_2 = [
        atom_map_2[item["atom_2"]]
        for item in contacts
        if item.get("atom_2") in atom_map_2
    ]
    if contact_points_1 and contact_points_2:
        axis_start = np.mean(contact_points_1, axis=0)
        axis_stop = np.mean(contact_points_2, axis=0)
    else:
        axis_start = _mean_edge_point(base_1, atom_map_1)
        axis_stop = _mean_edge_point(base_2, atom_map_2)
    if axis_start is None or axis_stop is None:
        return ""

    contact_axis = _unit_vector(axis_stop - axis_start)
    pair_normal = _pair_base_normal(base_1, base_2, atom_map_1, atom_map_2)
    if contact_axis is None or pair_normal is None:
        return ""

    side_axis = _unit_vector(np.cross(pair_normal, contact_axis))
    if side_axis is None:
        return ""

    axis_midpoint = 0.5 * (axis_start + axis_stop)
    bond_midpoint_1 = 0.5 * (gly_1 + sugar_1)
    bond_midpoint_2 = 0.5 * (gly_2 + sugar_2)
    side_1 = float(np.dot(bond_midpoint_1 - axis_midpoint, side_axis))
    side_2 = float(np.dot(bond_midpoint_2 - axis_midpoint, side_axis))
    if abs(side_1) < GLYCOSIDIC_SIDE_EPSILON or abs(side_2) < GLYCOSIDIC_SIDE_EPSILON:
        return ""
    return "cis" if side_1 * side_2 > 0.0 else "trans"


def _glycosidic_atom_point(base: str, atom_map: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    atom_name = GLYCOSIDIC_ATOMS.get(base)
    return atom_map.get(atom_name) if atom_name else None


def _sugar_c1_point(atom_map: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    for atom_name in SUGAR_C1_ATOMS:
        point = atom_map.get(atom_name)
        if point is not None:
            return point
    return None


def _mean_edge_point(base: str, atom_map: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    points = [atom_map[name] for name in _edge_contact_atoms(base) if name in atom_map]
    if not points:
        return None
    return np.mean(points, axis=0)


def _pair_base_normal(
    base_1: str,
    base_2: str,
    atom_map_1: Dict[str, np.ndarray],
    atom_map_2: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    normal_1 = _base_normal(base_1, atom_map_1)
    normal_2 = _base_normal(base_2, atom_map_2)
    if normal_1 is None or normal_2 is None:
        return normal_1 if normal_2 is None else normal_2
    if np.dot(normal_1, normal_2) < 0.0:
        normal_2 = -normal_2
    combined = _unit_vector(normal_1 + normal_2)
    return combined if combined is not None else normal_1


def _base_normal(base: str, atom_map: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    points = [atom_map[name] for name in _edge_contact_atoms(base) if name in atom_map]
    if len(points) < 3:
        return None
    coords = np.asarray(points, dtype=float)
    centered = coords - np.mean(coords, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return _unit_vector(vh[-1])


def _unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-10:
        return None
    return np.asarray(vector, dtype=float) / norm


def _is_hoogsteen_edge_pair(contact_geometry: Dict[str, Any]) -> bool:
    if not contact_geometry:
        return False
    edge_1 = contact_geometry.get("edge_1")
    edge_2 = contact_geometry.get("edge_2")
    return _has_classic_hoogsteen_edges(
        contact_geometry.get("base_1"), edge_1,
        contact_geometry.get("base_2"), edge_2,
    )


def _has_classic_hoogsteen_edges(base_1: str, edge_1: str, base_2: str, edge_2: str) -> bool:
    if frozenset((base_1, base_2)) not in HOOGSTEEN_BASE_SETS:
        return False
    if base_1 in HOOGSTEEN_PURINES:
        return edge_1 == "H" and edge_2 == "W"
    if base_2 in HOOGSTEEN_PURINES:
        return edge_1 == "W" and edge_2 == "H"
    return False


def _pair_geometry_marker(ctx, level: int, strand_1: int, strand_2: int) -> Optional[Dict[str, Any]]:
    markers = getattr(ctx, "pair_geometry_markers", {}) or {}
    marker = markers.get((strand_1, level))
    if marker:
        return dict(marker)
    marker = markers.get((strand_2, level))
    if marker:
        reversed_marker = dict(marker)
        reversed_marker["edge_1"] = marker.get("edge_2", "")
        reversed_marker["edge_2"] = marker.get("edge_1", "")
        tag = str(marker.get("tag", ""))
        if len(tag) == 3:
            reversed_marker["tag"] = f"{tag[0]}{tag[2]}{tag[1]}"
        return reversed_marker
    marker = markers.get(level)
    if marker:
        return dict(marker)
    return None


def _pair_contact_geometry_index(base_pairs: List[Dict[str, Any]]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    index: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for row in base_pairs:
        geometry = row.get("contact_geometry") or {}
        if not geometry:
            continue
        strand_1 = int(row.get("strand_1", 0)) - 1
        strand_2 = int(row.get("strand_2", 0)) - 1
        level = int(row.get("level", 0))
        if strand_1 < 0 or strand_2 < 0 or level <= 0:
            continue
        index[(strand_1, strand_2, level)] = geometry
        index[(strand_2, strand_1, level)] = geometry
    return index


def _has_hoogsteen_heavy_atom_contacts(ctx, residue_1: Dict[str, Any], residue_2: Dict[str, Any]) -> bool:
    base_1 = parent_base_name(residue_1["residue_name"])
    base_2 = parent_base_name(residue_2["residue_name"])
    bases = {base_1, base_2}
    atom_map_1 = _atom_map_for_residue(ctx, int(residue_1["subunit"]))
    atom_map_2 = _atom_map_for_residue(ctx, int(residue_2["subunit"]))

    if bases in ({"A", "T"}, {"A", "U"}):
        adenine_atoms = atom_map_1 if base_1 == "A" else atom_map_2
        pyrimidine_atoms = atom_map_2 if base_1 == "A" else atom_map_1
        n7_n3 = _atom_distance(adenine_atoms, "N7", pyrimidine_atoms, "N3")
        n6_o4 = _atom_distance(adenine_atoms, "N6", pyrimidine_atoms, "O4")
        n1_n3 = _atom_distance(adenine_atoms, "N1", pyrimidine_atoms, "N3")
        return _is_hoogsteen_contact_pair(n7_n3, n6_o4, n1_n3)

    if bases == {"G", "C"}:
        guanine_atoms = atom_map_1 if base_1 == "G" else atom_map_2
        cytosine_atoms = atom_map_2 if base_1 == "G" else atom_map_1
        n7_n3 = _atom_distance(guanine_atoms, "N7", cytosine_atoms, "N3")
        o6_n4 = _atom_distance(guanine_atoms, "O6", cytosine_atoms, "N4")
        n1_n3 = _atom_distance(guanine_atoms, "N1", cytosine_atoms, "N3")
        return _is_hoogsteen_contact_pair(n7_n3, o6_n4, n1_n3)

    if _has_modified_guanine_pyrimidine_hoogsteen_contact(
        residue_1,
        residue_2,
        base_1,
        base_2,
        atom_map_1,
        atom_map_2,
    ):
        return True

    return False


def _has_modified_guanine_pyrimidine_hoogsteen_contact(
    residue_1: Dict[str, Any],
    residue_2: Dict[str, Any],
    base_1: str,
    base_2: str,
    atom_map_1: Dict[str, np.ndarray],
    atom_map_2: Dict[str, np.ndarray],
) -> bool:
    # Legacy Curves has no explicit Hoogsteen taxonomy. For modified G/T-like
    # contacts such as IGU/T, keep the identity classification noncanonical but
    # annotate the observed guanine N7 to pyrimidine N3 Hoogsteen-edge contact.
    if not (is_modified_base(residue_1["residue_name"]) or is_modified_base(residue_2["residue_name"])):
        return False
    if "G" not in {base_1, base_2}:
        return False
    if not ({base_1, base_2} & {"T", "U"}):
        return False

    guanine_atoms = atom_map_1 if base_1 == "G" else atom_map_2
    pyrimidine_atoms = atom_map_2 if base_1 == "G" else atom_map_1
    n7_n3 = _atom_distance(guanine_atoms, "N7", pyrimidine_atoms, "N3")
    return n7_n3 is not None and n7_n3 <= HOOGSTEEN_CONTACT_CUTOFF


def _is_hoogsteen_contact_pair(
    hoogsteen_edge_distance: Optional[float],
    second_contact_distance: Optional[float],
    watson_edge_distance: Optional[float],
) -> bool:
    if hoogsteen_edge_distance is None or second_contact_distance is None:
        return False
    if hoogsteen_edge_distance > HOOGSTEEN_CONTACT_CUTOFF:
        return False
    if second_contact_distance > HOOGSTEEN_CONTACT_CUTOFF:
        return False
    return (
        watson_edge_distance is None
        or watson_edge_distance >= WATSON_CONTACT_PRESENT_CUTOFF
        or hoogsteen_edge_distance + 0.6 < watson_edge_distance
    )


def _atom_map_for_residue(ctx, subunit: int) -> Dict[str, np.ndarray]:
    mol = ctx.molecule
    start = int(mol.subunit_boundaries[subunit - 1])
    end = int(mol.subunit_boundaries[subunit])
    atom_map: Dict[str, np.ndarray] = {}
    for atom_idx in range(start, end):
        atom_name = str(mol.atom_names[atom_idx]).strip().upper()
        atom_map.setdefault(atom_name, np.asarray(mol.coordinates[atom_idx], dtype=float))
    return atom_map


def _atom_distance(
    atom_map_1: Dict[str, np.ndarray],
    atom_1: str,
    atom_map_2: Dict[str, np.ndarray],
    atom_2: str,
) -> Optional[float]:
    coord_1 = atom_map_1.get(atom_1)
    coord_2 = atom_map_2.get(atom_2)
    if coord_1 is None or coord_2 is None:
        return None
    return float(np.linalg.norm(coord_1 - coord_2))


def _source_lw_family_for_strand_order(
    source_pair: Optional[Dict[str, Any]],
    *,
    first_strand: int,
) -> str:
    if not source_pair:
        return ""
    tag = str(source_pair.get("source_lw_family") or "").strip()
    if len(tag) != 3:
        return tag
    try:
        mapped_first = int(source_pair.get("mapped_strand_1") or 0)
        mapped_second = int(source_pair.get("mapped_strand_2") or 0)
    except (TypeError, ValueError):
        return tag
    if mapped_second == first_strand and mapped_first != first_strand:
        return f"{tag[0]}{tag[2]}{tag[1]}"
    return tag


def _source_pairing_assignment(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize source nomenclature while preserving uncertainty/conflicts."""
    base_1 = parent_base_name(str(row.get("i_residue_name") or ""))
    base_2 = parent_base_name(str(row.get("j_residue_name") or ""))
    name = str(row.get("dssr_name") or "").strip()
    approximate_name = name.startswith("~")
    compact_name = (
        name.lower().replace(" ", "").replace("_", "").replace("-", "").lstrip("~")
    )
    named_mode = ""
    if compact_name.startswith("rhoogsteen") or "reversehoogsteen" in compact_name:
        named_mode = "reverse_hoogsteen"
    elif "hoogsteen" in compact_name:
        named_mode = "hoogsteen"
    elif "wobble" in compact_name:
        named_mode = "wobble"

    tag = str(
        row.get("hbond_type_leontis_westhof")
        or row.get("dssr_lw")
        or ""
    ).strip()
    tag_mode = _pairing_mode_from_lw_tag(tag, base_1, base_2)
    if tag.lower() == "cww":
        if (base_1, base_2) in WC_PAIRS:
            tag_mode = "watson_crick"
        elif (base_1, base_2) in WOBBLE_PAIRS:
            tag_mode = "wobble"
        else:
            tag_mode = "other_noncanonical"

    diagnostics: List[str] = []
    diagnostics.extend(_hoogsteen_protonation_diagnostics(
        named_mode or tag_mode, base_1, base_2
    ))
    if named_mode and tag_mode and named_mode != tag_mode:
        return {
            "pairing_mode": "",
            "candidate_mode": named_mode,
            "classification_status": "conflict",
            "diagnostic_flags": diagnostics + ["source_name_lw_conflict"],
            "source_lw_family": tag,
        }
    if approximate_name and named_mode:
        return {
            "pairing_mode": "",
            "candidate_mode": named_mode,
            "classification_status": "possible",
            "diagnostic_flags": diagnostics + ["source_name_approximate"],
            "source_lw_family": tag,
        }
    if named_mode:
        if compact_name.startswith("swobble"):
            diagnostics.append("shifted_wobble_variant")
        return {
            "pairing_mode": named_mode,
            "candidate_mode": "",
            "classification_status": "assigned",
            "diagnostic_flags": diagnostics,
            "source_lw_family": tag,
        }
    if tag_mode:
        return {
            "pairing_mode": tag_mode,
            "candidate_mode": "",
            "classification_status": "assigned",
            "diagnostic_flags": diagnostics,
            "source_lw_family": tag,
        }

    hbond_type_28 = str(row.get("hbond_type_28") or "").strip()
    complementary_hoogsteen = frozenset((base_1, base_2)) in HOOGSTEEN_BASE_SETS
    if hbond_type_28 in {"23", "24"} and complementary_hoogsteen:
        saenger_mode = "hoogsteen" if hbond_type_28 == "23" else "reverse_hoogsteen"
        return {
            "pairing_mode": saenger_mode,
            "candidate_mode": "",
            "classification_status": "assigned",
            "diagnostic_flags": (
                _hoogsteen_protonation_diagnostics(saenger_mode, base_1, base_2)
                + ["source_saenger_classification"]
            ),
            "source_lw_family": tag,
        }
    if row.get("is_hoogsteen"):
        return {
            "pairing_mode": "",
            "candidate_mode": "hoogsteen",
            "classification_status": "possible",
            "diagnostic_flags": (
                _hoogsteen_protonation_diagnostics("hoogsteen", base_1, base_2)
                + ["source_hoogsteen_without_ordered_geometry"]
            ),
            "source_lw_family": tag,
        }
    if (base_1, base_2) in WOBBLE_PAIRS:
        return {
            "pairing_mode": "",
            "candidate_mode": "wobble",
            "classification_status": "possible",
            "diagnostic_flags": ["source_wobble_identity_without_cww_geometry"],
            "source_lw_family": tag,
        }
    return {
        "pairing_mode": "",
        "candidate_mode": "",
        "classification_status": "unassigned",
        "diagnostic_flags": diagnostics,
        "source_lw_family": tag,
    }


def _source_pairing_mode(row: Dict[str, Any]) -> str:
    """Return only a definitive controlled source mode."""
    return _source_pairing_assignment(row)["pairing_mode"]


def _source_base_pair_annotations(ctx) -> List[Dict[str, Any]]:
    source_rows = list(getattr(ctx.molecule, "source_base_pairs", None) or [])
    if not source_rows:
        return []

    residue_locations = _residue_locations(ctx)
    annotations = []
    seen_pairs = set()
    for row in source_rows:
        i_lookup_chain = str(row.get("i_generated_chain_id") or row.get("i_chain_id", "")).strip()
        j_lookup_chain = str(row.get("j_generated_chain_id") or row.get("j_chain_id", "")).strip()
        i_key = (i_lookup_chain, int(row.get("i_residue_id") or 0))
        j_key = (j_lookup_chain, int(row.get("j_residue_id") or 0))
        unordered_key = tuple(sorted((i_key, j_key)))
        if unordered_key in seen_pairs:
            continue
        seen_pairs.add(unordered_key)
        i_locations = residue_locations.get(i_key, [])
        j_locations = residue_locations.get(j_key, [])
        in_current_topology = bool(i_locations or j_locations)
        mapped_level = None
        mapped_strands = None
        for left in i_locations:
            for right in j_locations:
                if left["level"] == right["level"] and left["strand"] != right["strand"]:
                    mapped_level = left["level"]
                    mapped_strands = (left["strand"], right["strand"])
                    break
            if mapped_level is not None:
                break

        source_assignment = _source_pairing_assignment(row)
        pairing_mode = source_assignment["pairing_mode"]
        annotation = {
            "source": row.get("source", ""),
            "pair_number": row.get("pair_number"),
            "pair_name": row.get("pair_name", ""),
            "residue_1": _format_source_residue(row, "i"),
            "residue_2": _format_source_residue(row, "j"),
            "chain_1": row.get("i_chain_id", ""),
            "chain_2": row.get("j_chain_id", ""),
            "mapped_chain_1": i_lookup_chain,
            "mapped_chain_2": j_lookup_chain,
            "residue_id_1": row.get("i_residue_id"),
            "residue_id_2": row.get("j_residue_id"),
            "base_1": parent_base_name(row.get("i_residue_name", "")),
            "base_2": parent_base_name(row.get("j_residue_name", "")),
            "pair_family": (
                "hoogsteen"
                if pairing_mode in {"hoogsteen", "reverse_hoogsteen"}
                else "source_annotated"
            ),
            "pairing_mode": pairing_mode,
            "candidate_mode": source_assignment["candidate_mode"],
            "classification_status": source_assignment["classification_status"],
            "diagnostic_flags": source_assignment["diagnostic_flags"],
            "source_lw_family": source_assignment["source_lw_family"],
            "is_hoogsteen": pairing_mode in {
                "hoogsteen", "reverse_hoogsteen"
            },
            "hbond_type_28": row.get("hbond_type_28", ""),
            "hbond_type_12": row.get("hbond_type_12", ""),
            "hbond_type_leontis_westhof": row.get(
                "hbond_type_leontis_westhof", ""
            ),
            "opening": row.get("opening"),
            "shear": row.get("shear"),
            "stretch": row.get("stretch"),
            "stagger": row.get("stagger"),
            "buckle": row.get("buckle"),
            "propeller": row.get("propeller"),
            "mapped_level": mapped_level,
            "mapped_strand_1": mapped_strands[0] if mapped_strands else None,
            "mapped_strand_2": mapped_strands[1] if mapped_strands else None,
            "topology_status": _source_pair_topology_status(mapped_level, in_current_topology),
            "shape_parameters_supported": mapped_level is not None,
            "shape_skip_reason": "" if mapped_level is not None else _source_pair_topology_status(mapped_level, in_current_topology),
        }
        annotations.append(annotation)
    return annotations


def _collect_warnings(ctx, base_pairs, base_fit_quality, source_base_pairs) -> List[Dict[str, Any]]:
    warnings = []
    for row in base_pairs:
        location = f"level {row.get('level')}"
        mismatch_reported = False
        if row.get("pair_family") == "ambiguous_topology":
            warnings.append(_warning("warn", "ambiguous_topology", location, row.get("pair_subtype", "")))
        elif row.get("is_hoogsteen"):
            geometry = base_pair_observed_geometry_annotation(row)
            label = pairing_mode_label(row.get("pairing_mode", ""))
            suffix = f" {geometry}" if geometry else ""
            warnings.append(_warning(
                "info",
                "hoogsteen_pair",
                location,
                f"{row['residue_1']} paired with {row['residue_2']} is classified as {label}{suffix} from current-coordinate evidence.",
            ))
        elif (
            row.get("classification_status") == "possible"
            and row.get("candidate_mode") == "hoogsteen"
        ):
            warnings.append(_warning(
                "info",
                "possible_hoogsteen_pair",
                location,
                f"{row['residue_1']} paired with {row['residue_2']} has evidence consistent with possible Hoogsteen pairing, but no definitive ordered LW assignment.",
            ))
        elif row.get("frame_mode") == "provisional_contact_geometry":
            candidate = str(row.get("candidate_lw_family") or "").strip()
            suffix = f"; possible [{candidate}]" if candidate else ""
            warnings.append(_warning(
                "info",
                "unresolved_contact_geometry_pair",
                location,
                f"{row['residue_1']} paired with {row['residue_2']} uses a provisional contact frame from {row.get('contact_count', 0)} independent contact(s){suffix}; no LW family is assigned.",
            ))
        elif row.get("frame_mode") == "contact_geometry":
            geometry = base_pair_geometry_annotation(row) or row.get("edge_pair") or "unknown edges"
            warnings.append(_warning("info", "contact_geometry_pair", location, f"{row['residue_1']} paired with {row['residue_2']} uses {geometry} contact-geometry frames for local shape parameters."))
        elif row.get("is_mismatch"):
            warnings.append(_mismatch_warning(row, location))
            mismatch_reported = True
        elif row.get("pairing_mode") == "wobble":
            warnings.append(_warning("info", "wobble_pair", location, f"{row['residue_1']} paired with {row['residue_2']} is classified as wobble from current-coordinate evidence."))
        if row.get("is_mismatch") and not mismatch_reported:
            warnings.append(_mismatch_warning(row, location))
        if row.get("has_modified_base"):
            warnings.append(_warning("info", "modified_base_pair", location, f"{row['residue_1']} / {row['residue_2']} contains modified or nonstandard residue names."))

    for row in base_fit_quality:
        location = f"strand {row.get('strand')} level {row.get('level')}"
        residue = _format_residue(row)
        if row.get("missing_fit_atoms"):
            warnings.append(_warning("warn", "missing_fit_atoms", location, f"{residue} missing fit atoms: {', '.join(row['missing_fit_atoms'])}."))
        if row.get("ignored_base_atoms"):
            warnings.append(_warning("info", "ignored_modified_atoms", location, f"{residue} extra atoms were ignored by the parent-base template."))
        if row.get("is_modified"):
            warnings.append(_warning("info", "parent_template_fit", location, f"{residue} fitted with {row.get('parent_base', '?')} parent-base template."))
    for row in source_base_pairs:
        if _should_report_unmapped_source_pair(row):
            warnings.append(_warning(
                "warn",
                "hoogsteen_source_pair",
                f"source pair {row.get('pair_number')}",
                f"{row.get('residue_1')} paired with {row.get('residue_2')} is Hoogsteen in the mmCIF table but is not represented as a Curves paired level.",
            ))
    return warnings


def _mismatch_warning(row: Dict[str, Any], location: str) -> Dict[str, str]:
    geometry = base_pair_observed_geometry_annotation(row)
    suffix = f" Observed geometry: {geometry}." if geometry else ""
    return _warning(
        "warn",
        "mismatch_pair",
        location,
        f"{row['residue_1']} paired with {row['residue_2']} is not Watson-Crick/wobble by identity.{suffix}",
    )


def _source_pair_topology_status(mapped_level, in_current_topology: bool) -> str:
    if mapped_level is not None:
        return "mapped_to_curves_level"
    if in_current_topology:
        return "source_pair_not_in_current_inp_topology"
    return "source_pair_outside_current_inp_topology"


def _should_report_unmapped_source_pair(row: Dict[str, Any]) -> bool:
    return bool(
        row.get("is_hoogsteen")
        and row.get("mapped_level") is None
        and row.get("topology_status") != "source_pair_outside_current_inp_topology"
    )


def _warning(severity: str, code: str, location: str, message: str) -> Dict[str, str]:
    return {"severity": severity, "code": code, "location": location, "message": message}


def _residue_for(ctx, strand: int, level: int) -> Optional[Dict[str, Any]]:
    if level < 1 or level > ctx.nux:
        return None
    subunit = int(ctx.ni_map[strand, level - 1])
    if subunit <= 0:
        return None
    mol = ctx.molecule
    atom_idx = int(mol.subunit_boundaries[subunit - 1])
    return {
        "strand": strand + 1,
        "level": level,
        "subunit": subunit,
        "residue_name": str(mol.residue_names[atom_idx]).strip().upper(),
        "residue_id": int(mol.residue_ids[atom_idx]),
        "chain_id": str(mol.chain_ids[atom_idx]).strip() if mol.chain_ids is not None else "",
    }


def _residue_locations(ctx) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    locations: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for strand in range(ctx.nst):
        for level in range(1, ctx.nux + 1):
            residue = _residue_for(ctx, strand, level)
            if residue is None:
                continue
            key = (str(residue.get("chain_id", "")).strip(), int(residue.get("residue_id", 0)))
            locations.setdefault(key, []).append(residue)
    return locations


def _format_source_residue(row: Dict[str, Any], prefix: str) -> str:
    chain = str(row.get(f"{prefix}_chain_id", "")).strip()
    name = str(row.get(f"{prefix}_residue_name", "")).strip()
    resid = row.get(f"{prefix}_residue_id", "")
    return f"{chain}:{name}{resid}" if chain else f"{name}{resid}"


def _format_residue(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return "?"
    chain = str(row.get("chain_id", "")).strip()
    name = str(row.get("residue_name", "")).strip()
    resid = row.get("residue_id", "")
    return f"{chain}:{name}{resid}" if chain else f"{name}{resid}"
