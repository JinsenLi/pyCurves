from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pycurves_lib.data.modified_bases import is_modified_base, parent_base_name

WC_PAIRS = {("A", "T"), ("T", "A"), ("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE_PAIRS = {("G", "U"), ("U", "G"), ("I", "C"), ("C", "I"), ("I", "U"), ("U", "I"), ("I", "A"), ("A", "I")}
HOOGSTEEN_CONTACT_CUTOFF = 3.7
WATSON_CONTACT_PRESENT_CUTOFF = 3.3
EDGE_CONTACT_CUTOFF = 3.8
MIN_CONTACT_FRAME_PAIRS = 2
GLYCOSIDIC_SIDE_EPSILON = 0.25

# Edge buckets are the only chemistry vocabulary used by pyCurves frame
# construction.  Pyrimidine C-H is folded into H for internal use.
BASE_EDGE_ATOMS = {
    "A": {
        "W": {"N1", "N6"},
        "H": {"N6", "N7", "C8"},
        "S": {"N3", "C2"},
    },
    "G": {
        "W": {"N1", "N2", "O6"},
        "H": {"O6", "N7", "C8"},
        "S": {"N2", "N3", "C2"},
    },
    "I": {
        "W": {"N1", "O6"},
        "H": {"O6", "N7", "C8"},
        "S": {"N3", "C2"},
    },
    "C": {
        "W": {"O2", "N3", "N4"},
        "H": {"C5", "C6"},
        "S": {"O2", "C2"},
    },
    "T": {
        "W": {"O2", "N3", "O4"},
        "H": {"C5", "C6", "C7"},
        "S": {"O2", "C2"},
    },
    "U": {
        "W": {"O2", "N3", "O4"},
        "H": {"C5", "C6"},
        "S": {"O2", "C2"},
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
    base_fit_quality = list(getattr(ctx, "annotations", {}).get("base_fit_quality", []))
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

    if not warnings:
        lines.extend([
            "  No unusual base-pair identity, modified-base, or base-fitting events were detected.",
            "",
        ])
        return "\n".join(lines)

    unusual_pairs = [
        row for row in base_pairs
        if (
            not row.get("is_canonical")
            or row.get("has_modified_base")
            or row.get("geometry_flag")
            or row.get("frame_mode") == "contact_geometry"
        )
    ]
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
            if row.get("geometry_flag"):
                notes.append(row["geometry_flag"])
            geometry_label = base_pair_geometry_annotation(row)
            if geometry_label:
                notes.append(geometry_label)
            elif row.get("edge_pair"):
                notes.append(row["edge_pair"])
            if row.get("glycosidic_orientation") and not geometry_label:
                notes.append(f"gly={row['glycosidic_orientation']}")
            if row.get("strand_direction") and not geometry_label:
                notes.append(f"dir={row['strand_direction']}")
            if row.get("frame_mode") == "contact_geometry":
                notes.append("contact_geometry_frames")
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

    source_unusual = [
        row for row in source_base_pairs
        if row.get("is_hoogsteen") and row.get("mapped_level") is None
    ]
    if source_unusual:
        lines.extend([
            "  Source base-pair annotations not represented as Curves paired levels",
            "",
            "   Pair  Residues                  Family      Source",
            "  ----------------------------------------------------------------",
        ])
        for row in source_unusual:
            residues = f"{row.get('residue_1', '?')} / {row.get('residue_2', '?')}"
            pair_number = row.get("pair_number") or 0
            lines.append(
                f"  {pair_number:5d}  {residues:<24s} "
                f"{row.get('pair_family', ''):<11s} {row.get('source', '')}"
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
    """Return the user-facing geometry label, including strand direction suffix."""
    tag = base_pair_geometry_tag(row)
    direction = str(row.get("strand_direction") or "").strip().lower()
    suffix = {"parallel": "p", "antiparallel": "ap"}.get(direction, "")
    if tag and suffix:
        return f"[{tag}:{suffix}]"
    if tag:
        return f"[{tag}]"
    edge_pair = str(row.get("edge_pair") or "").strip()
    if edge_pair and suffix:
        return f"[{edge_pair}:{suffix}]"
    return edge_pair


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
                "strand_1": active[0] + 1,
                "strand_2": active[1] + 1,
                "residue_1": _format_residue(_residue_for(ctx, active[0], level)),
                "residue_2": _format_residue(_residue_for(ctx, active[1], level)),
                "base_1": parent_base_name(_residue_for(ctx, active[0], level)["residue_name"]),
                "base_2": parent_base_name(_residue_for(ctx, active[1], level)["residue_name"]),
                "parent_base_1": parent_base_name(_residue_for(ctx, active[0], level)["residue_name"]),
                "parent_base_2": parent_base_name(_residue_for(ctx, active[1], level)["residue_name"]),
                "pair_family": "ambiguous_topology",
                "pair_subtype": f"{len(active)} active strands at this level",
                "is_canonical": False,
                "is_mismatch": False,
                "is_hoogsteen": False,
                "has_modified_base": any(is_modified_base(_residue_for(ctx, s, level)["residue_name"]) for s in active),
                "confidence": "topology_warning",
                "method": "identity_and_inp_topology",
                "geometry_flag": "",
                "shape_parameters_supported": True,
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
        marked_hoogsteen = _hoogsteen_marker_matches(ctx, level, s1 + 1, s2 + 1)
        contact_geometry = _contact_geometry_for_pair(
            ctx,
            s1,
            s2,
            level,
            r1,
            r2,
            source_hoogsteen=source_hoogsteen,
            marked_hoogsteen=marked_hoogsteen,
            canonical_identity=canonical,
            manual_geometry=manual_geometry,
        )
        geometry_flag = _geometry_flag(ctx, s1, s2, level, contact_geometry)
        if source_hoogsteen:
            geometry_flag = "hoogsteen_from_source"
        elif marked_hoogsteen:
            geometry_flag = "hoogsteen_from_inp"
        is_hoogsteen = source_hoogsteen or marked_hoogsteen or geometry_flag == "possible_hoogsteen"
        if source_hoogsteen:
            pair_family = "hoogsteen"
            pair_subtype = "source_annotation"
            confidence = "source_mmcif"
            method = "mmcif_ndb_struct_na_base_pair"
        elif marked_hoogsteen:
            pair_family = "hoogsteen"
            pair_subtype = "inp_marker"
            confidence = "inp_topology"
            method = "inp_hoogsteen_marker"
        elif geometry_flag == "possible_hoogsteen":
            pair_family = "possible_hoogsteen"
            pair_subtype = contact_geometry.get("edge_pair") or subtype
            confidence = "heuristic_geometry"
            method = "identity_and_base_pair_geometry"
        elif contact_geometry.get("edge_pair") and family == "mismatch":
            pair_family = "hbonded_noncanonical"
            pair_subtype = contact_geometry.get("edge_pair") or subtype
            confidence = contact_geometry.get("confidence", "heuristic_geometry")
            method = "edge_contact_geometry"
        else:
            pair_family = family
            pair_subtype = contact_geometry.get("edge_pair") if contact_geometry.get("frame_mode") == "contact_geometry" else subtype
            confidence = "identity"
            method = "identity_and_base_pair_geometry"
        frame_mode = contact_geometry.get(
            "frame_mode",
            "legacy_canonical" if canonical and not geometry_flag else "fitted_fallback",
        )
        rows.append({
            "level": level,
            "strand_1": s1 + 1,
            "strand_2": s2 + 1,
            "residue_1": _format_residue(r1),
            "residue_2": _format_residue(r2),
            "base_1": b1,
            "base_2": b2,
            "parent_base_1": b1,
            "parent_base_2": b2,
            "pair_family": pair_family,
            "pair_subtype": pair_subtype,
            "is_canonical": canonical and frame_mode == "legacy_canonical" and not geometry_flag and not source_hoogsteen and not marked_hoogsteen,
            "is_mismatch": family == "mismatch" and not is_hoogsteen,
            "is_hoogsteen": is_hoogsteen,
            "has_modified_base": is_modified_base(r1["residue_name"]) or is_modified_base(r2["residue_name"]),
            "confidence": confidence,
            "method": method,
            "geometry_flag": geometry_flag,
            "edge_1": contact_geometry.get("edge_1", ""),
            "edge_2": contact_geometry.get("edge_2", ""),
            "edge_pair": contact_geometry.get("edge_pair", ""),
            "glycosidic_orientation": contact_geometry.get("glycosidic_orientation", ""),
            "strand_direction": contact_geometry.get("strand_direction", ""),
            "frame_mode": frame_mode,
            "contact_atom_pairs": contact_geometry.get("contact_atom_pairs", []),
            "contact_count": contact_geometry.get("contact_count", 0),
            "contact_confidence": contact_geometry.get("confidence", ""),
            "manual_geometry_tag": contact_geometry.get("manual_geometry_tag", ""),
            "contact_geometry": contact_geometry,
            "source_pair_number": source_pair.get("pair_number") if source_pair else None,
            "shape_parameters_supported": True,
            "shape_skip_reason": "",
        })
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


def _geometry_flag(ctx, strand_1: int, strand_2: int, level: int, contact_geometry: Optional[Dict[str, Any]] = None) -> str:
    try:
        residue_1 = _residue_for(ctx, strand_1, level)
        residue_2 = _residue_for(ctx, strand_2, level)
        if residue_1 is None or residue_2 is None:
            return ""
        if contact_geometry and _is_hoogsteen_edge_pair(contact_geometry):
            return "possible_hoogsteen"
        if _has_hoogsteen_heavy_atom_contacts(ctx, residue_1, residue_2):
            return "possible_hoogsteen"
    except Exception:
        return ""
    return ""


def _contact_geometry_for_pair(
    ctx,
    strand_1: int,
    strand_2: int,
    level: int,
    residue_1: Dict[str, Any],
    residue_2: Dict[str, Any],
    source_hoogsteen: bool,
    marked_hoogsteen: bool,
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
    manual_requested = bool(manual_geometry)
    if manual_requested:
        edge_1 = str(manual_geometry.get("edge_1", "")).upper()
        edge_2 = str(manual_geometry.get("edge_2", "")).upper()
        edge_1_ambiguous = False
        edge_2_ambiguous = False
    edge_pair = f"{edge_1}/{edge_2}" if edge_1 and edge_2 else ""
    strand_direction = manual_geometry.get("strand_direction") or _strand_direction(ctx, strand_1, strand_2)
    manual_glycosidic_orientation = (
        manual_geometry.get("glycosidic_orientation")
        if manual_geometry.get("strand_direction_source") == "explicit"
        else ""
    )
    glycosidic_orientation = manual_glycosidic_orientation or _glycosidic_orientation(
        base_1,
        base_2,
        atom_map_1,
        atom_map_2,
        contacts,
    )

    has_reliable_contacts = (
        len(contacts) >= MIN_CONTACT_FRAME_PAIRS
        and bool(edge_1)
        and bool(edge_2)
        and not edge_1_ambiguous
        and not edge_2_ambiguous
    )
    has_manual_frame_contacts = manual_requested and bool(edge_1) and bool(edge_2) and len(contacts) >= MIN_CONTACT_FRAME_PAIRS
    has_usable_edge_geometry = has_reliable_contacts or has_manual_frame_contacts
    observed_watson_crick = (
        has_usable_edge_geometry
        and edge_1 == "W"
        and edge_2 == "W"
        and strand_direction == "antiparallel"
    )
    watson_watson_geometry = has_usable_edge_geometry and edge_1 == "W" and edge_2 == "W"
    forced_noncanonical = bool(source_hoogsteen or marked_hoogsteen)
    if canonical_identity and observed_watson_crick and not forced_noncanonical:
        frame_mode = "legacy_canonical"
    elif canonical_identity and not has_usable_edge_geometry and not forced_noncanonical:
        frame_mode = "legacy_canonical"
    elif watson_watson_geometry:
        frame_mode = "fitted_fallback"
    elif has_usable_edge_geometry:
        frame_mode = "contact_geometry"
    else:
        frame_mode = "fitted_fallback"

    if manual_requested:
        confidence = "manual_inp_geometry"
    elif has_reliable_contacts:
        confidence = "edge_contacts"
    elif contacts:
        confidence = "weak_or_ambiguous_contacts"
    elif forced_noncanonical:
        confidence = "source_or_inp_without_contacts"
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
        "orientation": manual_geometry.get("orientation", ""),
        "glycosidic_orientation": glycosidic_orientation,
        "strand_direction": strand_direction,
        "strand_direction_source": manual_geometry.get("strand_direction_source", "topology"),
        "frame_mode": frame_mode,
        "contact_atom_pairs": contacts,
        "contact_count": len(contacts),
        "confidence": confidence,
        "edge_score_1": edge_1_score,
        "edge_score_2": edge_2_score,
        "edge_1_ambiguous": edge_1_ambiguous,
        "edge_2_ambiguous": edge_2_ambiguous,
        "manual_geometry_tag": manual_geometry.get("tag", ""),
        "manual_geometry_strand": manual_geometry.get("annotated_strand"),
        "source_hoogsteen": bool(source_hoogsteen),
        "marked_hoogsteen": bool(marked_hoogsteen),
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
    if {edge_1, edge_2} != {"H", "W"}:
        return False
    bases = {contact_geometry.get("base_1"), contact_geometry.get("base_2")}
    return (
        (bases <= {"A", "T", "U"} and "A" in bases)
        or bases == {"G", "C"}
    )


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
    atom_map_1 = _atom_map_for_residue(ctx, int(residue_1["subunit"]))
    atom_map_2 = _atom_map_for_residue(ctx, int(residue_2["subunit"]))

    if {base_1, base_2} <= {"A", "T", "U"} and "A" in {base_1, base_2}:
        adenine_atoms = atom_map_1 if base_1 == "A" else atom_map_2
        pyrimidine_atoms = atom_map_2 if base_1 == "A" else atom_map_1
        n7_n3 = _atom_distance(adenine_atoms, "N7", pyrimidine_atoms, "N3")
        n6_o4 = _atom_distance(adenine_atoms, "N6", pyrimidine_atoms, "O4")
        n1_n3 = _atom_distance(adenine_atoms, "N1", pyrimidine_atoms, "N3")
        return _is_hoogsteen_contact_pair(n7_n3, n6_o4, n1_n3)

    if {base_1, base_2} == {"G", "C"}:
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


def _hoogsteen_marker_matches(ctx, level: int, strand_1: int, strand_2: int) -> bool:
    markers = getattr(ctx, "hoogsteen_markers", set()) or set()
    if level in markers:
        return True
    return (
        (strand_1, level) in markers
        or (strand_2, level) in markers
        or (strand_1, strand_2, level) in markers
        or (strand_2, strand_1, level) in markers
    )


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
            "pair_family": "hoogsteen" if row.get("is_hoogsteen") else "source_annotated",
            "is_hoogsteen": bool(row.get("is_hoogsteen")),
            "hbond_type_28": row.get("hbond_type_28", ""),
            "hbond_type_12": row.get("hbond_type_12", ""),
            "opening": row.get("opening"),
            "shear": row.get("shear"),
            "stretch": row.get("stretch"),
            "stagger": row.get("stagger"),
            "buckle": row.get("buckle"),
            "propeller": row.get("propeller"),
            "mapped_level": mapped_level,
            "mapped_strand_1": mapped_strands[0] if mapped_strands else None,
            "mapped_strand_2": mapped_strands[1] if mapped_strands else None,
            "topology_status": "mapped_to_curves_level" if mapped_level is not None else "source_pair_not_in_current_inp_topology",
            "shape_parameters_supported": mapped_level is not None,
            "shape_skip_reason": "" if mapped_level is not None else "source_pair_not_in_current_inp_topology",
        }
        annotations.append(annotation)
    return annotations


def _collect_warnings(ctx, base_pairs, base_fit_quality, source_base_pairs) -> List[Dict[str, Any]]:
    warnings = []
    for row in base_pairs:
        location = f"level {row.get('level')}"
        if row.get("pair_family") == "ambiguous_topology":
            warnings.append(_warning("warn", "ambiguous_topology", location, row.get("pair_subtype", "")))
        elif row.get("is_hoogsteen"):
            geometry = base_pair_geometry_annotation(row) or row.get("edge_pair") or "Hoogsteen-like"
            warnings.append(_warning("info", "hoogsteen_pair", location, f"{row['residue_1']} paired with {row['residue_2']} is {geometry}; local shape parameters use contact-geometry frames when reliable contacts are available."))
        elif row.get("frame_mode") == "contact_geometry":
            geometry = base_pair_geometry_annotation(row) or row.get("edge_pair") or "unknown edges"
            warnings.append(_warning("info", "contact_geometry_pair", location, f"{row['residue_1']} paired with {row['residue_2']} uses {geometry} contact-geometry frames for local shape parameters."))
        elif row.get("is_mismatch"):
            warnings.append(_warning("warn", "mismatch_pair", location, f"{row['residue_1']} paired with {row['residue_2']} is not Watson-Crick/wobble by identity."))
        elif row.get("pair_family") == "wobble":
            warnings.append(_warning("info", "wobble_pair", location, f"{row['residue_1']} paired with {row['residue_2']} is recognized as wobble/noncanonical."))
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
        if row.get("is_hoogsteen") and row.get("mapped_level") is None:
            warnings.append(_warning(
                "warn",
                "hoogsteen_source_pair",
                f"source pair {row.get('pair_number')}",
                f"{row.get('residue_1')} paired with {row.get('residue_2')} is Hoogsteen in the mmCIF table but is not represented as a Curves paired level.",
            ))
    return warnings


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
