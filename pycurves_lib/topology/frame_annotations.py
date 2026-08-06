from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from pycurves_lib.data.modified_bases import parent_base_name
from pycurves_lib.io.base_reference import is_fitted_cww_pose
from pycurves_lib.topology.base_annotations import (
    WC_PAIRS,
    WOBBLE_PAIRS,
    base_pair_pairing_classification,
)
from pycurves_lib.topology.topology_inferrer import (
    BasePairCandidate,
    RobustTopologyInferrer,
)


def infer_frame_pair_observations(
    molecule,
    reference_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return coordinate-only pair presence and mode for one trajectory frame."""
    inferrer = RobustTopologyInferrer(molecule)
    candidates = inferrer.coordinate_pair_candidates(one_to_one=True)
    remaining = {
        _pair_key(candidate.first, candidate.second): candidate
        for candidate in candidates
    }

    observations: List[Dict[str, Any]] = []
    for reference in reference_rows:
        row = dict(reference)
        key = _reference_pair_key(row)
        if key is None:
            row.update({
                "pair_status": "uncertain",
                "observed_lw_family": "",
                "pairing_mode": "",
                "candidate_mode": "",
                "classification_status": "unassigned",
                "diagnostic_flags": ["reference_pair_identity_unavailable"],
                "evidence_source": "frame_coordinate_topology",
                "reference_pair": True,
            })
            observations.append(row)
            continue

        candidate = remaining.pop(key, None)
        reference_fields = {
            name: row.get(name)
            for name in (
                "reference_lw_family",
                "reference_pairing_mode",
                "reference_classification_status",
                "calculation_is_hoogsteen",
            )
        }
        if candidate is None:
            row.update(_absent_reference_fields(key))
            observations.append(row)
            continue

        observed = _candidate_observation(
            candidate,
            inferrer,
            preferred_first=int(row["subunit_1"]),
        )
        row.update(observed)
        row.update(reference_fields)
        row["level"] = reference.get("level")
        row["strand_1"] = reference.get("strand_1")
        row["strand_2"] = reference.get("strand_2")
        row["reference_pair"] = True
        observations.append(row)

    for key in sorted(remaining):
        row = _candidate_observation(remaining[key], inferrer)
        row["reference_pair"] = False
        observations.append(row)
    return observations


def _candidate_observation(
    candidate: BasePairCandidate,
    inferrer: RobustTopologyInferrer,
    preferred_first: Optional[int] = None,
) -> Dict[str, Any]:
    reverse = (
        preferred_first is not None
        and candidate.second == preferred_first
        and candidate.first != preferred_first
    )
    first_subunit = candidate.second if reverse else candidate.first
    second_subunit = candidate.first if reverse else candidate.second
    first_strand = candidate.second_strand if reverse else candidate.first_strand
    second_strand = candidate.first_strand if reverse else candidate.second_strand
    residue_1 = inferrer.residues[first_subunit]
    residue_2 = inferrer.residues[second_subunit]

    observed_lw = ""
    lw = candidate.lw_classification
    if lw is not None and lw.confident:
        observed_lw = _reverse_lw_tag(lw.tag) if reverse else lw.tag
    elif is_fitted_cww_pose(candidate.fitted_geometry):
        observed_lw = "cWW"

    contacts = [
        {
            "atom_1": atom_2 if reverse else atom_1,
            "atom_2": atom_1 if reverse else atom_2,
            "distance": float(distance),
        }
        for atom_1, atom_2, distance in candidate.atom_pairs
    ]
    base_1 = parent_base_name(residue_1.res_name)
    base_2 = parent_base_name(residue_2.res_name)
    identity_class = _identity_class(base_1, base_2)
    diagnostics: List[str] = []
    if candidate.is_hoogsteen and not observed_lw:
        diagnostics.append("possible_hoogsteen")
    if candidate.pair_family == "hbonded_noncanonical":
        diagnostics.append("generic_donor_acceptor_contacts")

    row: Dict[str, Any] = {
        "pair_id": f"{min(first_subunit, second_subunit)}:{max(first_subunit, second_subunit)}",
        "level": None,
        "strand_1": first_strand + 1,
        "strand_2": second_strand + 1,
        "subunit_1": first_subunit,
        "subunit_2": second_subunit,
        "residue_1": _format_residue(residue_1),
        "residue_2": _format_residue(residue_2),
        "base_1": base_1,
        "base_2": base_2,
        "parent_base_1": base_1,
        "parent_base_2": base_2,
        "identity_class": identity_class,
        "pair_family": candidate.pair_family,
        "pair_subtype": observed_lw or candidate.pair_family,
        "pair_status": "present",
        "observed_lw_family": observed_lw,
        "reference_lw_family": "",
        "pairing_mode": "",
        "candidate_mode": "",
        "classification_status": "unassigned",
        "diagnostic_flags": diagnostics,
        "evidence_source": "frame_coordinate_topology",
        "contact_atom_pairs": contacts,
        "contact_count": candidate.hbond_count,
        "contact_confidence": (
            "fitted_lw_exemplar"
            if lw is not None and lw.confident
            else "fitted_standard_frames"
            if observed_lw == "cWW"
            else "donor_acceptor_contacts"
        ),
        "is_canonical": False,
        "is_mismatch": False,
        "is_hoogsteen": False,
        "has_modified_base": False,
    }
    row.update(base_pair_pairing_classification(row))
    mode = row["pairing_mode"]
    if mode in {"hoogsteen", "reverse_hoogsteen"}:
        row["pair_family"] = "hoogsteen"
    elif mode == "watson_crick":
        row["pair_family"] = "watson_crick"
    elif mode == "wobble":
        row["pair_family"] = "wobble"
    row["is_canonical"] = mode == "watson_crick"
    row["is_mismatch"] = identity_class == "mismatch"
    row["is_hoogsteen"] = mode in {"hoogsteen", "reverse_hoogsteen"}
    return row


def _absent_reference_fields(key: Tuple[int, int]) -> Dict[str, Any]:
    return {
        "pair_id": f"{key[0]}:{key[1]}",
        "pair_status": "absent",
        "observed_lw_family": "",
        "pairing_mode": "",
        "candidate_mode": "",
        "classification_status": "unassigned",
        "diagnostic_flags": ["reference_pair_not_detected"],
        "evidence_source": "frame_coordinate_topology",
        "contact_atom_pairs": [],
        "contact_count": 0,
        "contact_confidence": "",
        "is_canonical": False,
        "is_mismatch": False,
        "is_hoogsteen": False,
        "reference_pair": True,
    }


def _reference_pair_key(row: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    try:
        return _pair_key(int(row["subunit_1"]), int(row["subunit_2"]))
    except (KeyError, TypeError, ValueError):
        return None


def _pair_key(first: int, second: int) -> Tuple[int, int]:
    return tuple(sorted((int(first), int(second))))


def _reverse_lw_tag(tag: str) -> str:
    value = str(tag or "").strip()
    if len(value) != 3:
        return value
    return f"{value[0]}{value[2]}{value[1]}"


def _identity_class(base_1: str, base_2: str) -> str:
    pair = (base_1, base_2)
    if pair in WC_PAIRS:
        return "watson_crick"
    if pair in WOBBLE_PAIRS:
        return "wobble"
    if not base_1 or not base_2:
        return "unknown"
    return "mismatch"


def _format_residue(residue) -> str:
    label = f"{residue.res_name}{residue.res_id}"
    return f"{residue.chain}:{label}" if residue.chain else label

