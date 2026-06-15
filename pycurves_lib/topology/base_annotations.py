from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pycurves_lib.data.modified_bases import is_modified_base, parent_base_name

WC_PAIRS = {("A", "T"), ("T", "A"), ("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE_PAIRS = {("G", "U"), ("U", "G"), ("I", "C"), ("C", "I")}
HOOGSTEEN_CONTACT_CUTOFF = 3.7
WATSON_CONTACT_PRESENT_CUTOFF = 3.3


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
    skipped = _skipped_shape_parameters(base_pairs, source_base_pairs)
    warnings = _collect_warnings(ctx, base_pairs, base_fit_quality, source_base_pairs, skipped)
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
        if not row.get("is_canonical") or row.get("has_modified_base") or row.get("geometry_flag")
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
            if row.get("pair_subtype"):
                notes.append(row["pair_subtype"])
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
        geometry_flag = _geometry_flag(ctx, s1, s2, level)
        source_pair = source_by_level.get(level)
        source_hoogsteen = bool(source_pair and source_pair.get("is_hoogsteen"))
        marked_hoogsteen = _hoogsteen_marker_matches(ctx, level, s1 + 1, s2 + 1)
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
            pair_subtype = subtype
            confidence = "heuristic_geometry"
            method = "identity_and_base_pair_geometry"
        else:
            pair_family = family
            pair_subtype = subtype
            confidence = "identity"
            method = "identity_and_base_pair_geometry"
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
            "is_canonical": canonical and not geometry_flag and not source_hoogsteen and not marked_hoogsteen,
            "is_mismatch": family == "mismatch" and not is_hoogsteen,
            "is_hoogsteen": is_hoogsteen,
            "has_modified_base": is_modified_base(r1["residue_name"]) or is_modified_base(r2["residue_name"]),
            "confidence": confidence,
            "method": method,
            "geometry_flag": geometry_flag,
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


def _geometry_flag(ctx, strand_1: int, strand_2: int, level: int) -> str:
    try:
        residue_1 = _residue_for(ctx, strand_1, level)
        residue_2 = _residue_for(ctx, strand_2, level)
        if residue_1 is None or residue_2 is None:
            return ""
        if _has_hoogsteen_heavy_atom_contacts(ctx, residue_1, residue_2):
            return "possible_hoogsteen"
    except Exception:
        return ""
    return ""


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


def _wrap_180(value: float) -> float:
    if abs(value) > 180.0:
        value -= np.sign(value) * 360.0
    return float(value)


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


def _skipped_shape_parameters(base_pairs, source_base_pairs) -> List[Dict[str, Any]]:
    return []


def _collect_warnings(ctx, base_pairs, base_fit_quality, source_base_pairs, skipped) -> List[Dict[str, Any]]:
    warnings = []
    for row in base_pairs:
        location = f"level {row.get('level')}"
        if row.get("pair_family") == "ambiguous_topology":
            warnings.append(_warning("warn", "ambiguous_topology", location, row.get("pair_subtype", "")))
        elif row.get("is_hoogsteen"):
            warnings.append(_warning("info", "hoogsteen_pair", location, f"{row['residue_1']} paired with {row['residue_2']} is Hoogsteen-like; local shape parameters use Hoogsteen-aware fitted frames."))
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
