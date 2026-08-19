from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from pycurves_lib.topology.base_annotations import BASE_EDGE_ATOMS, infer_lw_strand_orientation

EQUIVALENT_AXIS_SIGN_FLIPS = (
    np.diag([1.0, 1.0, 1.0]),
    np.diag([1.0, -1.0, -1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([-1.0, -1.0, 1.0]),
)

# An observed edge defines an in-plane axis line, not a directed vector.  This
# determinant-preserving alternative reverses X and Y without changing the
# fitted base-plane normal Z.
CONTACT_IN_PLANE_SIGN_FLIP = np.diag([-1.0, -1.0, 1.0])

# Reversing X and Z together selects the opposite normal-line branch while
# preserving Y, handedness, and relative geometry when applied to both pair
# members. For contact frames Y is the directed primary-to-partner axis; for
# left_handed_cww it remains the standard fitted Y axis.
PAIR_NORMAL_SIGN_FLIP = np.diag([-1.0, 1.0, -1.0])

# Ring atoms used only to obtain a coordinate-derived base-pair center for the
# forward tangent.  Exocyclic substituents are intentionally excluded so a
# change of LW edge does not move the tangent reference sideways.
BASE_RING_ATOMS = frozenset({
    "N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9",
})

CONTACT_NORMAL_TANGENT_WEIGHT = 2.0
CONTACT_NORMAL_CONTINUITY_WEIGHT = 1.0
LEFT_HANDED_CWW_MIN_SYN_PAIRS = 3
LEFT_HANDED_CWW_MAX_EVIDENCE_GAP = 2
LEFT_HANDED_CWW_MAX_MEDIAN_STEP_DEGREES = -1.0


def apply_curvesplus_base_pair_inversion(values, invert):
    """Apply the Curves+ reverse-Z rule to intra-base-pair parameters.

    ``params.f`` changes only Shear (index 0) and Buckle (index 3) when the
    base-pair level has its Z direction inverted.  ``invert`` may be a scalar
    or an array matching the leading dimensions of ``values``.
    """
    corrected = np.array(values, dtype=float, copy=True)
    mask = np.asarray(invert, dtype=bool)
    corrected[..., 0] = np.where(mask, -corrected[..., 0], corrected[..., 0])
    corrected[..., 3] = np.where(mask, -corrected[..., 3], corrected[..., 3])
    return corrected


def build_interaction_reference_frames(ctx):
    """Build the derived frame view used for signed shape calculations.

    The fitted base frames remain available as ``params.frames``.  For a
    noncanonical pair with reliable edge contacts, or an unresolved pair with
    a stable provisional contact axis, ``params.shape_frames`` replaces both
    paired base frames with per-base interaction frames:

    * Resolved LW pairs use X along each observed interacting edge.
    * Unresolved pairs use Y along the directed atom-contact axis and derive X
      in the fitted base plane without assigning an LW edge.
    * Y is oriented consistently along the strand-1 to partner contact axis.
    * Z lies on that base's fitted normal line; for non-cWW contact pairs its
      common sign is oriented from the coordinate-derived forward tangent.
    * Origins are the centroids of the atoms defining the observed edge, or
      the participating contact atoms for an unresolved pair.

    Coordinate-confirmed left-handed cWW segments keep their standard fitted
    member frames and origins, but receive the same pair-normal branch
    selection in this derived view. The two bases never share averaged axes;
    downstream shape math still compares independent frames.
    """
    p = ctx.params
    raw_frames = np.asarray(p.frames, dtype=float)
    shape_frames = raw_frames.copy()
    pairs = _interaction_frame_pairs(ctx)
    left_handed_pairs, left_handed_segments, glycosidic_details = (
        _left_handed_cww_pairs(ctx, raw_frames, pairs)
    )
    if not pairs and not left_handed_pairs:
        p.shape_frames = shape_frames
        ctx.contact_geometry_frame_keys = set()
        ctx.contact_pair_normal_signs = {}
        ctx.contact_pair_normal_flips = []
        ctx.left_handed_cww_frame_keys = set()
        ctx.left_handed_cww_segments = []
        ctx.left_handed_cww_normal_signs = {}
        ctx.pair_normal_branch_modes = {}
        ctx.pair_normal_signs = {}
        ctx.pair_normal_flips = []
        ctx.pair_glycosidic_details = {}
        return shape_frames

    frame_keys = set()
    built_pairs = []
    for partner_strand, level, geometry in pairs:
        if not (
            _has_level(ctx, 0, level)
            and _has_level(ctx, partner_strand, level)
        ):
            continue
        pair_frames = _interaction_pair_reference_frames(ctx, 0, partner_strand, level, raw_frames, geometry)
        if pair_frames is None:
            continue
        first_frame, partner_frame = pair_frames
        shape_frames[0, level] = first_frame
        shape_frames[partner_strand, level] = partner_frame
        frame_keys.add((0, partner_strand, level))
        frame_keys.add((partner_strand, 0, level))
        mode = str(geometry.get("frame_mode") or "contact_geometry")
        built_pairs.append((partner_strand, level, mode))

    normal_pairs = built_pairs + left_handed_pairs
    branch_modes = {
        (0, int(partner), int(level)): str(mode)
        for partner, level, mode in normal_pairs
    }
    shape_frames, normal_signs = _resolve_pair_normal_branches(
        ctx,
        raw_frames,
        shape_frames,
        normal_pairs,
    )

    p.shape_frames = shape_frames
    ctx.contact_geometry_frame_keys = frame_keys
    ctx.contact_pair_normal_signs = {
        key: sign for key, sign in normal_signs.items()
        if branch_modes.get(key) in {
            "contact_geometry",
            "provisional_contact_geometry",
        }
    }
    ctx.contact_pair_normal_flips = [
        (partner + 1, level)
        for (primary, partner, level), sign in sorted(ctx.contact_pair_normal_signs.items())
        if primary == 0 and sign < 0
    ]
    left_handed_keys = {
        key for key, mode in branch_modes.items()
        if mode == "left_handed_cww"
    }
    ctx.left_handed_cww_frame_keys = {
        directed
        for primary, partner, level in left_handed_keys
        for directed in ((primary, partner, level), (partner, primary, level))
    }
    ctx.left_handed_cww_segments = left_handed_segments
    ctx.left_handed_cww_normal_signs = {
        key: sign for key, sign in normal_signs.items()
        if key in left_handed_keys
    }
    ctx.pair_normal_branch_modes = branch_modes
    ctx.pair_normal_signs = normal_signs
    ctx.pair_glycosidic_details = glycosidic_details
    ctx.pair_normal_flips = [
        {
            "partner_strand": partner + 1,
            "level": level,
            "mode": branch_modes[(primary, partner, level)],
        }
        for (primary, partner, level), sign in sorted(normal_signs.items())
        if primary == 0 and sign < 0
    ]
    _annotate_pair_normal_branches(
        ctx,
        normal_signs,
        branch_modes,
        glycosidic_details,
    )
    return shape_frames


def build_axis_reference_frames(ctx):
    """Build the frame view consumed by the legacy global-axis optimizer.

    Noncanonical edge frames can be fit on a discontinuous in-plane branch.
    Choose determinant-preserving sign equivalents for those contact-geometry
    frames only. Fitted canonical frames are fixed anchors: allowing an
    adjusted contact frame to become the reference for the following residue
    can propagate a sign branch across the rest of a strand.
    """
    p = ctx.params
    shape_frames = build_interaction_reference_frames(ctx)
    axis_frames = shape_frames.copy()

    contact_levels = {
        (int(strand), int(level))
        for strand, _partner, level in (getattr(ctx, "contact_geometry_frame_keys", set()) or set())
        if 0 <= int(strand) < ctx.nst and 1 <= int(level) <= ctx.nux
    }
    ctx.axis_reference_uses_continuity = bool(contact_levels)
    if not contact_levels:
        p.axis_frames = axis_frames
        ctx.axis_frame_adjustments = []
        return axis_frames

    adjustments = []
    for strand, level in sorted(contact_levels):
        current = shape_frames[strand, level]
        if (
            ctx.li[level, strand] < 0
            or _axis_support_weight(ctx, strand, level) <= 0.0
            or not np.all(np.isfinite(current))
        ):
            continue

        anchors = _nearest_fitted_axis_anchors(
            ctx,
            shape_frames,
            contact_levels,
            strand,
            level,
        )
        if not anchors:
            continue

        axes = current[:3].copy()
        candidates = [sign_flip @ axes for sign_flip in EQUIVALENT_AXIS_SIGN_FLIPS]
        scores = [
            sum(float(np.trace(anchor @ candidate.T)) for anchor in anchors)
            for candidate in candidates
        ]
        best_index = int(np.argmax(scores))
        axis_frames[strand, level, :3, :] = candidates[best_index]
        if best_index != 0:
            adjustments.append((strand + 1, level))

    p.axis_frames = axis_frames
    ctx.axis_frame_adjustments = adjustments
    return axis_frames


def _nearest_fitted_axis_anchors(ctx, frames, contact_levels, strand: int, level: int):
    """Return the nearest fixed frame on either side of a contact-frame run."""
    anchors = []
    for direction in (-1, 1):
        neighbor = level + direction
        while 1 <= neighbor <= ctx.nux:
            if (
                ctx.li[neighbor, strand] < 0
                or _axis_support_weight(ctx, strand, neighbor) <= 0.0
            ):
                break
            if (strand, neighbor) not in contact_levels:
                frame = frames[strand, neighbor]
                if np.all(np.isfinite(frame)):
                    anchors.append(frame[:3])
                break
            neighbor += direction
    return anchors


def _axis_support_weight(ctx, strand: int, level: int) -> float:
    weights = getattr(ctx, "axis_support_weights", None)
    if weights is None:
        return 1.0
    return float(weights[level, strand])


def _has_level(ctx, strand: int, level: int) -> bool:
    return 0 <= strand < ctx.nst and 1 <= level <= ctx.nux and ctx.li[level, strand] >= 0


def _interaction_frame_pairs(ctx):
    pairs = []
    annotations = getattr(ctx, "annotations", {}).get("base_pair_annotations", [])
    for row in annotations:
        geometry = row.get("contact_geometry") or {}
        frame_mode = str(row.get("frame_mode") or "")
        if frame_mode not in {
            "contact_geometry",
            "provisional_contact_geometry",
        }:
            continue
        # cWW is the canonical standard-frame family, including wobble or
        # mismatch identities that retain cWW geometry.  It must never enter
        # the contact-frame sign-selection path.
        if _normalized_lw_family(row, geometry) == "cWW":
            continue
        level = row.get("level")
        if level is None:
            continue
        strands = {int(row.get("strand_1", 0)), int(row.get("strand_2", 0))}
        if 1 not in strands or len(strands) != 2:
            continue
        partner = next(strand for strand in strands if strand != 1)
        pairs.append((partner - 1, int(level), geometry))
    return sorted(pairs, key=lambda item: (item[1], item[0]))


def _normalized_lw_family(annotation: dict, geometry: Optional[dict] = None) -> str:
    """Return a normalized directed LW family such as ``cWW`` or ``tSH``."""
    geometry = geometry or {}
    for source in (annotation, geometry):
        for key in (
            "manual_geometry_tag",
            "observed_lw_family",
            "reference_lw_family",
            "tag",
        ):
            value = str(source.get(key) or "").strip()
            if len(value) >= 3:
                candidate = value[0].lower() + value[1:3].upper()
                if candidate[0] in {"c", "t"} and all(
                    edge in {"W", "H", "S"} for edge in candidate[1:]
                ):
                    return candidate

    orientation = str(
        geometry.get("glycosidic_orientation")
        or annotation.get("glycosidic_orientation")
        or ""
    ).strip().lower()
    prefix = {"cis": "c", "trans": "t"}.get(orientation, "")
    edge_1 = str(geometry.get("edge_1") or annotation.get("edge_1") or "").strip().upper()
    edge_2 = str(geometry.get("edge_2") or annotation.get("edge_2") or "").strip().upper()
    if prefix and edge_1 in {"W", "H", "S"} and edge_2 in {"W", "H", "S"}:
        return f"{prefix}{edge_1}{edge_2}"
    return ""


def _left_handed_cww_pairs(ctx, raw_frames: np.ndarray, contact_pairs):
    """Return cWW levels belonging to coordinate-confirmed Z-DNA segments."""
    if not hasattr(ctx, "molecule"):
        return [], [], {}
    annotations = getattr(ctx, "annotations", {}).get("base_pair_annotations", [])
    contact_levels = {
        (int(partner), int(level)) for partner, level, _geometry in contact_pairs
    }
    cww_rows = {}
    evidence_by_partner = {}
    glycosidic_details = {}
    for row in annotations:
        geometry = row.get("contact_geometry") or {}
        if _normalized_lw_family(row, geometry) != "cWW":
            continue
        pair = _primary_partner_for_annotation(row)
        level = row.get("level")
        if pair is None or level is None:
            continue
        partner = pair
        level = int(level)
        if not (_has_level(ctx, 0, level) and _has_level(ctx, partner, level)):
            continue
        cww_rows[(partner, level)] = row
        details = {}
        has_syn_purine = False
        for strand in (0, partner):
            base = _base_symbol(ctx, strand, level)
            chi = _coordinate_glycosidic_chi(ctx, strand, level)
            state = _glycosidic_state(base, chi)
            details[strand] = {"base": base, "chi": chi, "state": state}
            has_syn_purine = has_syn_purine or (
                base in {"A", "G", "I"} and state == "syn"
            )
        glycosidic_details[(0, partner, level)] = details
        if has_syn_purine:
            evidence_by_partner.setdefault(partner, []).append(level)

    pairs = []
    segments = []
    for partner, evidence_levels in sorted(evidence_by_partner.items()):
        member_centers = _pair_member_coordinate_centers(ctx, raw_frames, partner)
        bridge_levels = contact_levels | {
            key for key in cww_rows if key[0] == partner
        }
        for evidence_run in _left_handed_evidence_runs(
            ctx,
            partner,
            sorted(set(evidence_levels)),
            bridge_levels,
        ):
            if len(evidence_run) < LEFT_HANDED_CWW_MIN_SYN_PAIRS:
                continue
            start, end = evidence_run[0], evidence_run[-1]
            step_angles = [
                angle
                for level in range(start, end)
                if (angle := _signed_pair_vector_step_degrees(
                    ctx, partner, member_centers, level, level + 1
                )) is not None
            ]
            if not step_angles:
                continue
            median_step = float(np.median(step_angles))
            if median_step >= LEFT_HANDED_CWW_MAX_MEDIAN_STEP_DEGREES:
                continue
            # Syn purines identify the Z-DNA core, but a cWW pair at a B/Z
            # junction can lie just outside that evidence while its step from
            # the core is still unambiguously left-handed.  Include such
            # connected boundary pairs in the same signed-normal run.  This
            # avoids using an oriented boundary normal merely as a DP anchor
            # while leaving the actual boundary frame on the opposite branch.
            start, end = _extend_left_handed_cww_boundaries(
                ctx,
                partner,
                start,
                end,
                cww_rows,
                member_centers,
            )
            step_angles = [
                angle
                for level in range(start, end)
                if (angle := _signed_pair_vector_step_degrees(
                    ctx, partner, member_centers, level, level + 1
                )) is not None
            ]
            median_step = float(np.median(step_angles))
            cww_levels = [
                level for level in range(start, end + 1)
                if (partner, level) in cww_rows
            ]
            if len(cww_levels) < LEFT_HANDED_CWW_MIN_SYN_PAIRS:
                continue
            pairs.extend(
                (partner, level, "left_handed_cww") for level in cww_levels
            )
            segments.append({
                "partner_strand": partner + 1,
                "start_level": start,
                "end_level": end,
                "cww_levels": cww_levels,
                "syn_evidence_levels": list(evidence_run),
                "median_pair_vector_step": median_step,
            })
    return pairs, segments, glycosidic_details


def _extend_left_handed_cww_boundaries(
    ctx,
    partner: int,
    start: int,
    end: int,
    cww_rows: dict,
    member_centers: dict,
):
    """Extend a syn-confirmed Z core across left-handed cWW boundary steps."""
    while True:
        candidate = start - 1
        if (
            (partner, candidate) not in cww_rows
            or not _pair_levels_are_connected(ctx, partner, candidate, start)
        ):
            break
        angle = _signed_pair_vector_step_degrees(
            ctx, partner, member_centers, candidate, start
        )
        if angle is None or angle >= LEFT_HANDED_CWW_MAX_MEDIAN_STEP_DEGREES:
            break
        start = candidate

    while True:
        candidate = end + 1
        if (
            (partner, candidate) not in cww_rows
            or not _pair_levels_are_connected(ctx, partner, end, candidate)
        ):
            break
        angle = _signed_pair_vector_step_degrees(
            ctx, partner, member_centers, end, candidate
        )
        if angle is None or angle >= LEFT_HANDED_CWW_MAX_MEDIAN_STEP_DEGREES:
            break
        end = candidate

    return start, end


def _primary_partner_for_annotation(annotation: dict) -> Optional[int]:
    strands = {
        int(annotation.get("strand_1", 0)),
        int(annotation.get("strand_2", 0)),
    }
    if 1 not in strands or len(strands) != 2:
        return None
    return next(strand for strand in strands if strand != 1) - 1


def _left_handed_evidence_runs(
    ctx,
    partner: int,
    levels,
    bridge_levels,
):
    runs = []
    current = []
    for level in levels:
        if current:
            previous = current[-1]
            gap = int(level) - int(previous)
            connected = all(
                _pair_levels_are_connected(ctx, partner, step, step + 1)
                for step in range(previous, level)
            )
            bridge_is_eligible = all(
                (partner, bridge) in bridge_levels
                for bridge in range(previous + 1, level)
            )
            if (
                gap > LEFT_HANDED_CWW_MAX_EVIDENCE_GAP
                or not connected
                or not bridge_is_eligible
            ):
                runs.append(current)
                current = []
        current.append(int(level))
    if current:
        runs.append(current)
    return runs


def _annotate_pair_normal_branches(
    ctx,
    signs: dict,
    modes: dict,
    glycosidic_details: dict,
) -> None:
    annotations = getattr(ctx, "annotations", {})
    rows = []
    for name in ("base_pair_annotations", "frame_base_pair_observations"):
        rows.extend(annotations.get(name, []) or [])
    for row in rows:
        partner = _primary_partner_for_annotation(row)
        level = row.get("level")
        if partner is None or level is None:
            continue
        key = (0, partner, int(level))
        if key not in modes:
            continue
        row["normal_branch_mode"] = modes[key]
        row["helical_context"] = (
            "left_handed_cww" if modes[key] == "left_handed_cww" else ""
        )
        row["pair_normal_sign"] = int(signs.get(key, 1))
        details = glycosidic_details.get(key, {})
        row_strands = (
            int(row.get("strand_1", 0)) - 1,
            int(row.get("strand_2", 0)) - 1,
        )
        for member_index, strand in enumerate(row_strands, start=1):
            member = details.get(strand)
            if not member:
                continue
            row[f"glycosidic_state_{member_index}"] = member["state"]
            row[f"glycosidic_chi_{member_index}"] = member["chi"]


def _resolve_pair_normal_branches(
    ctx,
    raw_frames: np.ndarray,
    shape_frames: np.ndarray,
    pairs,
):
    """Choose one signed normal branch for each eligible pair-frame run.

    The eligible set contains non-cWW contact frames and standard cWW frames
    inside coordinate-confirmed left-handed segments. A binary dynamic program
    orients each normal line along the increasing-level coordinate tangent
    while favoring continuity. Fitted ``params.frames`` remain immutable.
    """
    resolved = np.asarray(shape_frames, dtype=float).copy()
    signs = {}
    pairs_by_partner = {}
    for partner, level, mode in pairs:
        pairs_by_partner.setdefault(int(partner), []).append((int(level), mode))

    for partner, partner_pairs in sorted(pairs_by_partner.items()):
        centers = _pair_coordinate_centers(ctx, raw_frames, partner)
        runs = _normal_pair_runs(ctx, partner, partner_pairs)
        for run in runs:
            levels = [level for level, _mode in run]
            normals = [
                _unsigned_pair_normal(resolved, partner, level)
                for level in levels
            ]
            tangents = [
                _pair_forward_tangent(ctx, partner, centers, level)
                for level in levels
            ]
            run_signs = _binary_pair_normal_signs(
                ctx,
                raw_frames,
                partner,
                levels,
                normals,
                tangents,
                centers,
            )
            for level, sign in zip(levels, run_signs):
                signs[(0, partner, level)] = int(sign)
                if sign >= 0:
                    continue
                resolved[0, level, :3, :] = (
                    PAIR_NORMAL_SIGN_FLIP @ resolved[0, level, :3, :]
                )
                resolved[partner, level, :3, :] = (
                    PAIR_NORMAL_SIGN_FLIP @ resolved[partner, level, :3, :]
                )
    return resolved, signs


def _resolve_contact_pair_normal_branches(
    ctx,
    raw_frames: np.ndarray,
    shape_frames: np.ndarray,
    pairs,
):
    """Backward-compatible contact-only wrapper for focused unit tests."""
    normalized = [
        (partner, level, "contact_geometry")
        for partner, level, _geometry in pairs
    ]
    return _resolve_pair_normal_branches(ctx, raw_frames, shape_frames, normalized)


def _normal_pair_runs(ctx, partner: int, pairs):
    """Split eligible pair frames at missing levels and explicit breaks."""
    runs = []
    current = []
    for item in sorted(pairs, key=lambda pair: pair[0]):
        level = int(item[0])
        if current and not _pair_levels_are_connected(ctx, partner, current[-1][0], level):
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)
    return runs


def _pair_levels_are_connected(ctx, partner: int, lower: int, upper: int) -> bool:
    if int(upper) != int(lower) + 1:
        return False
    if int(getattr(ctx, "break_pt", 0) or 0) == int(upper):
        return False
    return (
        _has_level(ctx, 0, lower)
        and _has_level(ctx, 0, upper)
        and _has_level(ctx, partner, lower)
        and _has_level(ctx, partner, upper)
    )


def _unsigned_pair_normal(frames: np.ndarray, partner: int, level: int) -> np.ndarray:
    """Return one representative vector for an otherwise unsigned normal line."""
    first = _unit(frames[0, level, 2], np.array([0.0, 0.0, 1.0]))
    other = _unit(frames[partner, level, 2], first)
    if float(np.dot(first, other)) < 0.0:
        other = -other
    return _unit(first + other, first)


def _pair_coordinate_centers(ctx, raw_frames: np.ndarray, partner: int):
    member_centers = _pair_member_coordinate_centers(ctx, raw_frames, partner)
    return {
        level: np.mean(np.asarray(centers, dtype=float), axis=0)
        for level, centers in member_centers.items()
    }


def _pair_member_coordinate_centers(ctx, raw_frames: np.ndarray, partner: int):
    centers = {}
    for level in range(1, int(ctx.nux) + 1):
        if not (_has_level(ctx, 0, level) and _has_level(ctx, partner, level)):
            continue
        member_centers = []
        for strand in (0, partner):
            member_centers.append(
                _base_coordinate_center(ctx, raw_frames, strand, level)
            )
        if np.all(np.isfinite(member_centers)):
            centers[level] = tuple(member_centers)
    return centers


def _base_coordinate_center(ctx, raw_frames: np.ndarray, strand: int, level: int):
    center = None
    if hasattr(ctx, "molecule"):
        _base, atom_map = _base_atom_map(ctx, strand, level)
        points = [
            point for name, point in atom_map.items()
            if name in BASE_RING_ATOMS
        ]
        if len(points) >= 3:
            center = np.mean(np.asarray(points, dtype=float), axis=0)
    if center is None or not np.all(np.isfinite(center)):
        center = np.asarray(raw_frames[strand, level, 3], dtype=float)
    return center


def _signed_pair_vector_step_degrees(
    ctx,
    partner: int,
    member_centers: dict,
    lower: int,
    upper: int,
) -> Optional[float]:
    """Return direct coordinate handedness from successive pair vectors."""
    if not _pair_levels_are_connected(ctx, partner, lower, upper):
        return None
    if lower not in member_centers or upper not in member_centers:
        return None
    first_primary, first_partner = member_centers[lower]
    next_primary, next_partner = member_centers[upper]
    first_center = 0.5 * (first_primary + first_partner)
    next_center = 0.5 * (next_primary + next_partner)
    axis = _unit(next_center - first_center)
    if np.linalg.norm(axis) <= 1e-12:
        return None
    first_vector = first_partner - first_primary
    next_vector = next_partner - next_primary
    first_vector = first_vector - axis * float(np.dot(first_vector, axis))
    next_vector = next_vector - axis * float(np.dot(next_vector, axis))
    first_vector = _unit(first_vector)
    next_vector = _unit(next_vector)
    if min(np.linalg.norm(first_vector), np.linalg.norm(next_vector)) <= 1e-12:
        return None
    sine = float(np.dot(axis, np.cross(first_vector, next_vector)))
    cosine = float(np.clip(np.dot(first_vector, next_vector), -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def _coordinate_glycosidic_chi(ctx, strand: int, level: int) -> Optional[float]:
    base, atom_map = _base_atom_map(ctx, strand, level)
    if base in {"A", "G", "I"}:
        names = ("C4", "N9", "C1'", "O4'")
    elif base in {"C", "T", "U"}:
        names = ("C2", "N1", "C1'", "O4'")
    else:
        return None
    points = []
    for name in names:
        point = atom_map.get(name)
        if point is None and name.endswith("'"):
            point = atom_map.get(name[:-1] + "*")
        if point is None:
            return None
        points.append(point)
    return _signed_dihedral_degrees(*points)


def _signed_dihedral_degrees(p1, p2, p3, p4) -> Optional[float]:
    first = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    middle = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    last = np.asarray(p4, dtype=float) - np.asarray(p3, dtype=float)
    normal_1 = np.cross(first, middle)
    normal_2 = np.cross(last, -middle)
    if min(
        np.linalg.norm(middle),
        np.linalg.norm(normal_1),
        np.linalg.norm(normal_2),
    ) <= 1e-12:
        return None
    normal_1 = _unit(normal_1)
    normal_2 = _unit(normal_2)
    middle = _unit(middle)
    cosine = float(np.clip(np.dot(normal_1, normal_2), -1.0, 1.0))
    sine = float(np.dot(normal_1, np.cross(normal_2, middle)))
    return float(np.degrees(np.arctan2(sine, cosine)))


def _glycosidic_state(base: str, chi: Optional[float]) -> str:
    if chi is None or not np.isfinite(chi):
        return "unknown"
    if base in {"A", "G", "I"} and -90.0 <= chi <= 90.0:
        return "syn"
    if abs(float(chi)) >= 120.0:
        return "anti"
    return "ambiguous"


def _pair_forward_tangent(
    ctx,
    partner: int,
    centers: dict,
    level: int,
) -> Optional[np.ndarray]:
    previous = level - 1
    following = level + 1
    has_previous = (
        previous in centers
        and _pair_levels_are_connected(ctx, partner, previous, level)
    )
    has_following = (
        following in centers
        and _pair_levels_are_connected(ctx, partner, level, following)
    )
    if has_previous and has_following:
        delta = centers[following] - centers[previous]
    elif has_following and level in centers:
        delta = centers[following] - centers[level]
    elif has_previous and level in centers:
        delta = centers[level] - centers[previous]
    else:
        return None
    norm = float(np.linalg.norm(delta))
    return delta / norm if norm > 1e-12 else None


def _fixed_pair_normal_anchor(
    ctx,
    raw_frames: np.ndarray,
    partner: int,
    level: int,
    centers: dict,
) -> Optional[np.ndarray]:
    """Return a read-only standard-frame boundary normal oriented by coordinates."""
    if not (
        _has_level(ctx, 0, level)
        and _has_level(ctx, partner, level)
        and level in centers
    ):
        return None
    normal = _unit(raw_frames[0, level, 2], np.array([0.0, 0.0, 1.0]))
    tangent = _pair_forward_tangent(ctx, partner, centers, level)
    if tangent is not None and float(np.dot(normal, tangent)) < 0.0:
        normal = -normal
    return normal


def _binary_pair_normal_signs(
    ctx,
    raw_frames: np.ndarray,
    partner: int,
    levels,
    normals,
    tangents,
    centers: dict,
):
    """Solve the two-state normal-orientation problem for one contact run."""
    if not levels:
        return []
    states = (1, -1)  # ties preserve the provisional branch
    scores = np.full((len(levels), 2), -np.inf, dtype=float)
    back = np.zeros((len(levels), 2), dtype=int)

    left_anchor = None
    left_level = levels[0] - 1
    if _pair_levels_are_connected(ctx, partner, left_level, levels[0]):
        left_anchor = _fixed_pair_normal_anchor(
            ctx, raw_frames, partner, left_level, centers
        )

    for state_index, state in enumerate(states):
        score = _pair_normal_unary_score(state, normals[0], tangents[0])
        if left_anchor is not None:
            score += CONTACT_NORMAL_CONTINUITY_WEIGHT * float(
                np.dot(state * normals[0], left_anchor)
            )
        scores[0, state_index] = score

    for index in range(1, len(levels)):
        for state_index, state in enumerate(states):
            unary = _pair_normal_unary_score(state, normals[index], tangents[index])
            candidates = []
            for previous_index, previous_state in enumerate(states):
                continuity = CONTACT_NORMAL_CONTINUITY_WEIGHT * float(
                    np.dot(previous_state * normals[index - 1], state * normals[index])
                )
                candidates.append(scores[index - 1, previous_index] + unary + continuity)
            best_previous = int(np.argmax(candidates))
            scores[index, state_index] = candidates[best_previous]
            back[index, state_index] = best_previous

    right_anchor = None
    right_level = levels[-1] + 1
    if _pair_levels_are_connected(ctx, partner, levels[-1], right_level):
        right_anchor = _fixed_pair_normal_anchor(
            ctx, raw_frames, partner, right_level, centers
        )
    final_scores = scores[-1].copy()
    if right_anchor is not None:
        for state_index, state in enumerate(states):
            final_scores[state_index] += CONTACT_NORMAL_CONTINUITY_WEIGHT * float(
                np.dot(state * normals[-1], right_anchor)
            )

    selected = np.zeros(len(levels), dtype=int)
    selected[-1] = int(np.argmax(final_scores))
    for index in range(len(levels) - 1, 0, -1):
        selected[index - 1] = back[index, selected[index]]
    return [states[index] for index in selected]


def _pair_normal_unary_score(
    state: int,
    normal: np.ndarray,
    tangent: Optional[np.ndarray],
) -> float:
    if tangent is None:
        return 0.0
    return CONTACT_NORMAL_TANGENT_WEIGHT * float(np.dot(state * normal, tangent))


def _interaction_pair_reference_frames(
    ctx,
    first_strand: int,
    partner_strand: int,
    level: int,
    raw_frames: np.ndarray,
    geometry: dict,
):
    first_base, first_atoms = _base_atom_map(ctx, first_strand, level)
    partner_base, partner_atoms = _base_atom_map(ctx, partner_strand, level)
    first_id = first_strand + 1
    geometry_first_id = int(geometry.get("strand_1", first_id))

    if geometry_first_id == first_id:
        first_edge = geometry.get("edge_1", "")
        partner_edge = geometry.get("edge_2", "")
        first_key = "atom_1"
        partner_key = "atom_2"
    else:
        first_edge = geometry.get("edge_2", "")
        partner_edge = geometry.get("edge_1", "")
        first_key = "atom_2"
        partner_key = "atom_1"

    contact_pairs = list(geometry.get("contact_atom_pairs", []) or [])
    first_contact_atoms = []
    partner_contact_atoms = []
    first_contact_points = []
    partner_contact_points = []
    for pair in contact_pairs:
        first_atom = str(pair.get(first_key, "")).strip().upper()
        partner_atom = str(pair.get(partner_key, "")).strip().upper()
        first_point = first_atoms.get(first_atom)
        partner_point = partner_atoms.get(partner_atom)
        if first_point is None or partner_point is None:
            continue
        first_contact_atoms.append(first_atom)
        partner_contact_atoms.append(partner_atom)
        first_contact_points.append(first_point)
        partner_contact_points.append(partner_point)

    frame_basis = str(geometry.get("frame_basis") or "")
    if frame_basis == "atom_contact_axis":
        if not first_contact_points:
            return None
        first_contact_points = np.asarray(first_contact_points, dtype=float)
        partner_contact_points = np.asarray(partner_contact_points, dtype=float)
        hbond_axis_seed = np.mean(
            partner_contact_points - first_contact_points,
            axis=0,
        )
        hbond_axis = _unit(
            hbond_axis_seed,
            raw_frames[first_strand, level, 1, :],
        )
        first_frame = _atom_contact_member_reference_frame(
            raw_frames[first_strand, level],
            first_contact_points,
            hbond_axis,
        )
        partner_frame = _atom_contact_member_reference_frame(
            raw_frames[partner_strand, level],
            partner_contact_points,
            hbond_axis,
        )
        return first_frame, partner_frame

    first_points = _edge_points_for_frame(first_base, first_edge, first_atoms, first_contact_atoms)
    partner_points = _edge_points_for_frame(partner_base, partner_edge, partner_atoms, partner_contact_atoms)
    if len(first_points) < 2 or len(partner_points) < 2:
        return None

    if first_contact_points:
        first_contact_points = np.asarray(first_contact_points, dtype=float)
        partner_contact_points = np.asarray(partner_contact_points, dtype=float)
        hbond_axis_seed = np.mean(partner_contact_points - first_contact_points, axis=0)
    else:
        hbond_axis_seed = np.mean(partner_points, axis=0) - np.mean(first_points, axis=0)
    hbond_axis = _unit(hbond_axis_seed, raw_frames[first_strand, level, 1, :])
    first_frame = _interaction_member_reference_frame(
        raw_frames[first_strand, level],
        first_points,
        hbond_axis,
    )
    partner_frame = _interaction_member_reference_frame(
        raw_frames[partner_strand, level],
        partner_points,
        hbond_axis,
    )
    return first_frame, partner_frame


def _atom_contact_member_reference_frame(
    raw_frame: np.ndarray,
    contact_points: np.ndarray,
    hbond_axis: np.ndarray,
) -> np.ndarray:
    """Return a family-neutral frame from a directed atom-contact axis."""
    contact_points = np.asarray(contact_points, dtype=float)
    z_axis = _unit(raw_frame[2], np.array([0.0, 0.0, 1.0]))
    center = np.mean(contact_points, axis=0)
    y_axis = hbond_axis - z_axis * float(np.dot(hbond_axis, z_axis))
    y_axis = _unit(y_axis, raw_frame[1])
    if float(np.dot(y_axis, hbond_axis)) < 0.0:
        y_axis *= -1.0
    x_axis = _unit(np.cross(y_axis, z_axis), raw_frame[0])
    y_axis = _unit(np.cross(z_axis, x_axis), y_axis)

    frame = np.asarray(raw_frame, dtype=float).copy()
    frame[:3, :] = _orthonormalize_axes(
        np.asarray([x_axis, y_axis, z_axis], dtype=float)
    )
    frame[3, :] = center
    return frame


def _edge_points_for_frame(base: str, edge: str, atom_map: dict, contact_atoms) -> np.ndarray:
    names = []
    for atom_name in contact_atoms:
        if atom_name in atom_map and atom_name not in names:
            names.append(atom_name)
    if len(names) < 2:
        for atom_name in sorted(BASE_EDGE_ATOMS.get(base, {}).get(edge, set())):
            if atom_name in atom_map and atom_name not in names:
                names.append(atom_name)
    return np.asarray([atom_map[name] for name in names if name in atom_map], dtype=float)


def _interaction_member_reference_frame(
    raw_frame: np.ndarray,
    edge_points: np.ndarray,
    hbond_axis: np.ndarray,
) -> np.ndarray:
    """Return one base's observed-edge interaction frame."""
    edge_points = np.asarray(edge_points, dtype=float)
    z_axis = _unit(raw_frame[2], np.array([0.0, 0.0, 1.0]))
    center = np.mean(edge_points, axis=0)
    centered = edge_points - center
    centered = centered - np.outer(centered @ z_axis, z_axis)

    x_axis = None
    if len(centered) >= 2 and np.linalg.norm(centered) > 1e-10:
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            x_axis = vh[0]
        except np.linalg.LinAlgError:
            x_axis = None
    if x_axis is None or np.linalg.norm(x_axis) <= 1e-12:
        x_axis = edge_points[-1] - edge_points[0]
        x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
    x_axis = _unit(x_axis, raw_frame[0])
    if np.dot(x_axis, raw_frame[0]) < 0.0:
        x_axis *= -1.0

    y_axis = _unit(np.cross(z_axis, x_axis), raw_frame[1])
    x_axis = _unit(np.cross(y_axis, z_axis), x_axis)
    if np.dot(y_axis, hbond_axis) < 0.0:
        x_axis *= -1.0
        y_axis *= -1.0

    frame = np.asarray(raw_frame, dtype=float).copy()
    frame[:3, :] = _orthonormalize_axes(np.asarray([x_axis, y_axis, z_axis], dtype=float))
    frame[3, :] = center
    return frame


def _base_atom_map(ctx, strand: int, level: int):
    subunit = int(ctx.ni_map[strand, level - 1])
    if subunit <= 0:
        return "", {}
    start = int(ctx.molecule.subunit_boundaries[subunit - 1])
    end = int(ctx.molecule.subunit_boundaries[subunit])
    base = _base_symbol(ctx, strand, level)
    atom_map = {}
    for atom_idx in range(start, end):
        atom_name = str(ctx.molecule.atom_names[atom_idx]).strip().upper()
        atom_map.setdefault(atom_name, np.asarray(ctx.molecule.coordinates[atom_idx], dtype=float))
    return base, atom_map


def _base_symbol(ctx, strand: int, level: int) -> str:
    try:
        from pycurves_lib.data.modified_bases import parent_base_name
        subunit = int(ctx.ni_map[strand, level - 1])
        if subunit <= 0:
            return ""
        atom_idx = int(ctx.molecule.subunit_boundaries[subunit - 1])
        name = parent_base_name(ctx.molecule.residue_names[atom_idx])
    except Exception:
        return ""
    if len(name) >= 2 and name[0] in {"D", "R"} and name[1] in "GACTUI":
        return name[1]
    return name[:1]


def _orthonormalize_axes(axes: np.ndarray) -> np.ndarray:
    x_axis = _unit(axes[0], np.array([1.0, 0.0, 0.0]))
    y_axis = axes[1] - x_axis * np.dot(x_axis, axes[1])
    y_axis = _unit(y_axis, np.array([0.0, 1.0, 0.0]))
    z_axis = _unit(np.cross(x_axis, y_axis), axes[2])
    y_axis = _unit(np.cross(z_axis, x_axis), y_axis)
    return np.asarray([x_axis, y_axis, z_axis], dtype=float)


def _unit(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        return vector / norm
    if fallback is None:
        return vector
    return np.asarray(fallback, dtype=float)


@dataclass(frozen=True)
class ParameterFrame:
    """Cartesian frame used by convention-specific parameter calculators."""

    origin: np.ndarray
    axes: np.ndarray


class BaseParameterConvention:
    """Base API for convention-specific shape parameter math."""

    name = "base"

    def local_base_base_values(self, calc, partner_strand: int, level: int):
        raise NotImplementedError

    def fill_local_base_pair_steps(self, calc) -> None:
        raise NotImplementedError

    def fill_local_strand_steps(self, calc) -> None:
        return

    @staticmethod
    def finite(values) -> bool:
        return bool(np.all(np.isfinite(np.asarray(values, dtype=float))))

    @staticmethod
    def unit(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            return vector / norm
        if fallback is None:
            return vector
        return np.asarray(fallback, dtype=float)


class LegacyParameterConvention(BaseParameterConvention):
    """Legacy Curves 5.3-compatible local base-pair formulas."""

    name = "legacy"

    def local_base_base_values(self, calc, partner_strand: int, level: int):
        standard = StandardParameterConvention()
        if (
            standard._uses_contact_geometry_pair(calc, partner_strand, level)
            or standard._is_hoogsteen_pair(calc, partner_strand, level)
            or standard._uses_noncanonical_watson_pair(calc, partner_strand, level)
        ):
            return standard.local_base_base_values(calc, partner_strand, level)

        if not (calc._has_level(0, level) and calc._has_level(partner_strand, level)):
            return None

        first = calc.ctx.params.frames[0, level]
        other = calc.ctx.params.frames[partner_strand, level].copy()
        if not self.finite(first) or not self.finite(other):
            return None

        other_aligned = other.copy()
        other_aligned[1] *= -1.0
        other_aligned[2] *= -1.0

        x_axis = first[0] + other_aligned[0]
        x_axis = self.unit(x_axis, first[0])

        y_axis = first[1] + other_aligned[1]
        y_axis = y_axis - x_axis * np.dot(x_axis, y_axis)
        y_axis = self.unit(y_axis, np.cross(first[2], x_axis))

        z_axis = self.unit(np.cross(x_axis, y_axis), first[2])
        y_axis = self.unit(np.cross(z_axis, x_axis), y_axis)

        delta = first[3] - other[3]
        shear = float(np.dot(x_axis, delta))
        stretch = float(np.dot(y_axis, delta))
        stagger = float(np.dot(z_axis, delta))

        rotation = Rotation.from_matrix(first[:3] @ other_aligned[:3].T)
        buckle, propel, opening = (-rotation.as_rotvec() * calc.cdr).tolist()
        return np.array([shear, stretch, stagger, buckle, propel, opening], dtype=float)

    def fill_local_base_pair_steps(self, calc) -> None:
        p = calc.ctx.params
        nst = calc.ctx.nst
        nux = calc.ctx.n_levels
        idr_1 = calc.ctx.idr[0]

        lu1 = calc.inv[0]
        lv1 = lu1

        for k in range(1, nst):
            lu = calc.inv[0] * calc.ctx.idr[0] * calc.ctx.idr[k]
            lv = -lu

            m_uz = np.zeros((nux + 2, 3))
            m_ux = np.zeros((nux + 2, 3))
            m_uy = np.zeros((nux + 2, 3))
            m_or = np.zeros((nux + 2, 3))

            for i in range(calc.optimizer.iste, calc.optimizer.iene + 1):
                if calc.ctx.li[i, k] >= -1:
                    uz_vec = lu1 * p.frames[0, i, 2, :] + lu * p.frames[k, i, 2, :]
                    m_uz[i] = self.unit(uz_vec)

                    ux_vec = lv1 * p.frames[0, i, 0, :] + lv * p.frames[k, i, 0, :]
                    m_ux[i] = self.unit(ux_vec)

                    m_uy[i] = np.cross(m_uz[i], m_ux[i])
                    m_or[i] = (p.frames[0, i, 3, :] + p.frames[k, i, 3, :]) / 2.0

            for i in range(calc.optimizer.iste + 1, calc.optimizer.iene + 1):
                if not (
                    calc._has_level(0, i - 1)
                    and calc._has_level(0, i)
                    and calc._has_level(k, i - 1)
                    and calc._has_level(k, i)
                ):
                    continue
                nx = self.unit(m_uz[i - 1] + m_uz[i])
                qx = (m_or[i - 1] + m_or[i]) / 2.0

                v_sum = m_ux[i - 1] + m_ux[i]
                dx = v_sum - nx * np.dot(nx, v_sum)
                dx = self.unit(dx)
                fx = np.cross(nx, dx)

                dl = np.dot(nx, qx - m_or[i - 1]) / np.dot(nx, m_uz[i - 1])
                du = np.dot(nx, m_or[i] - qx) / np.dot(nx, m_uz[i])
                calc.pab[i, 2, k] = dl + du

                pl = m_or[i - 1] + m_uz[i - 1] * dl
                pu = m_or[i] - m_uz[i] * du
                diff = pu - pl

                calc.pab[i, 0, k] = np.dot(dx, diff)
                calc.pab[i, 1, k] = np.dot(fx, diff) * idr_1

                tx = np.cross(m_uz[i], dx)
                rt = np.linalg.norm(tx)
                dot_c = np.clip(np.dot(fx, tx) / rt, -1.0, 1.0)
                cln = np.arccos(dot_c) * calc.cdr
                if np.dot(np.cross(fx, tx), dx) < 0:
                    cln = -cln
                calc.pab[i, 3, k] = 2.0 * cln

                rx = np.cross(dx, tx)
                rr = np.linalg.norm(rx)
                dot_t = np.clip(np.dot(m_uz[i], rx) / rr, -1.0, 1.0)
                tip = np.arccos(dot_t) * calc.cdr
                if np.dot(np.cross(rx, m_uz[i]), tx) < 0:
                    tip = -tip
                calc.pab[i, 4, k] = 2.0 * tip * idr_1

                calc.pab[i, 5, k] = 0.0
                for l_idx, l_val in [(0, i - 1), (1, i)]:
                    sa = np.sin(calc.rdc * ((-1.0 if l_idx == 0 else 1.0) * cln))
                    ca = np.cos(calc.rdc * ((-1.0 if l_idx == 0 else 1.0) * cln))
                    fpx = dx * np.dot(dx, fx) * (1 - ca) + fx * ca + np.cross(dx, fx) * sa

                    dot_w = np.clip(np.dot(fpx, m_uy[l_val]), -1.0, 1.0)
                    wdg = np.arccos(dot_w) * calc.cdr
                    cross_w = np.cross(fpx, m_uy[l_val])
                    dot_s = np.dot(cross_w, m_uz[l_val])
                    if (l_idx == 0 and dot_s > 0) or (l_idx == 1 and dot_s < 0):
                        wdg = -wdg
                    calc.pab[i, 5, k] += wdg

                h_twist = calc.pab[i, 5, k] % 360.0
                if abs(h_twist) > 180.0:
                    h_twist -= np.copysign(360.0, h_twist)
                calc.pab[i, 5, k] = h_twist

        self._fill_contact_geometry_base_pair_steps(calc)

    def fill_local_strand_steps(self, calc) -> None:
        standard = StandardParameterConvention()
        for strand in range(calc.ctx.nst):
            _, _, iste, iene = calc._axis_bounds(strand)
            for level in range(iste + 1, iene + 1):
                previous_contact = standard._uses_contact_geometry_level(
                    calc, strand, level - 1
                )
                current_contact = standard._uses_contact_geometry_level(
                    calc, strand, level
                )
                if not (
                    previous_contact
                    or current_contact
                    or standard._is_hoogsteen_level(calc, strand, level - 1)
                    or standard._is_hoogsteen_level(calc, strand, level)
                ):
                    continue
                previous_frame = standard._oriented_strand_frame(
                    calc,
                    strand,
                    level - 1,
                    axis_reference=previous_contact,
                )
                current_frame = standard._oriented_strand_frame(
                    calc,
                    strand,
                    level,
                    axis_reference=current_contact,
                )
                if previous_frame is None or current_frame is None:
                    continue
                values = standard._rigid_body_values(
                    previous_frame,
                    current_frame,
                    calc.cdr,
                    translation_sign=1.0,
                    rotation_sign=1.0,
                )
                values[3:] = [standard._wrap_180(value) for value in values[3:]]
                calc.pal[level, :, strand] = values

    def _fill_contact_geometry_base_pair_steps(self, calc) -> None:
        standard = StandardParameterConvention()
        for partner_strand in range(1, calc.ctx.nst):
            for level in range(calc.optimizer.iste + 1, calc.optimizer.iene + 1):
                if not (
                    standard._uses_contact_geometry_pair(calc, partner_strand, level - 1)
                    or standard._uses_contact_geometry_pair(calc, partner_strand, level)
                    or standard._is_hoogsteen_pair(calc, partner_strand, level - 1)
                    or standard._is_hoogsteen_pair(calc, partner_strand, level)
                ):
                    continue
                if not (
                    calc._has_level(0, level - 1)
                    and calc._has_level(0, level)
                    and calc._has_level(partner_strand, level - 1)
                    and calc._has_level(partner_strand, level)
                ):
                    continue
                previous_pair = standard._base_pair_frame(calc, partner_strand, level - 1)
                current_pair = standard._base_pair_frame(calc, partner_strand, level)
                if previous_pair is None or current_pair is None:
                    continue
                if (
                    standard._uses_contact_geometry_pair(calc, partner_strand, level - 1)
                    or standard._uses_contact_geometry_pair(calc, partner_strand, level)
                ):
                    previous_pair, current_pair = standard._step_aligned_frames(
                        previous_pair, current_pair, calc.cdr
                    )
                values = standard._curvesplus_step_values(previous_pair, current_pair, calc.cdr)
                calc.pab[level, :, partner_strand] = values


class StandardParameterConvention(LegacyParameterConvention):
    """Standard Curves+/3DNA-style local parameter decomposition."""

    name = "standard"
    _EQUIVALENT_AXIS_SIGN_FLIPS = EQUIVALENT_AXIS_SIGN_FLIPS

    def local_base_base_values(self, calc, partner_strand: int, level: int):
        pair_frames = self._base_pair_member_frames(calc, partner_strand, level)
        if pair_frames is None:
            return None
        first, other = pair_frames
        # Standard fitted frames use the Curves partner-to-primary rotational
        # ordering.  Contact frames instead define +Y explicitly from the
        # primary base toward its partner, so their coordinate-consistent
        # rotation is primary-to-partner.  Reusing the standard negative sign
        # reverses buckle, propeller, and opening in the edge-aligned frame.
        rotation_sign = (
            1.0
            if self._uses_contact_geometry_pair(calc, partner_strand, level)
            else -1.0
        )
        values = self._rigid_body_values(
            first,
            other,
            calc.cdr,
            translation_sign=-1.0,
            rotation_sign=rotation_sign,
        )
        invert = getattr(calc, "curvesplus_invert", None)
        # A pair resolved from coordinates (non-cWW contact geometry or a
        # left_handed_cww segment) has already selected its signed normal.
        # Applying the standard reverse-Z mask again would introduce a second,
        # conflicting buckle/shear sign correction.
        if (
            not self._uses_resolved_normal_pair(calc, partner_strand, level)
            and invert is not None
            and 0 <= level < len(invert)
        ):
            values = apply_curvesplus_base_pair_inversion(values, invert[level])
        return np.array(values, dtype=float)

    def fill_local_base_pair_steps(self, calc) -> None:
        for partner_strand in range(1, calc.ctx.nst):
            for level in range(calc.optimizer.iste + 1, calc.optimizer.iene + 1):
                if not (
                    calc._has_level(0, level - 1)
                    and calc._has_level(0, level)
                    and calc._has_level(partner_strand, level - 1)
                    and calc._has_level(partner_strand, level)
                ):
                    continue

                previous_pair = self._base_pair_frame(calc, partner_strand, level - 1)
                current_pair = self._base_pair_frame(calc, partner_strand, level)
                if previous_pair is None or current_pair is None:
                    continue
                # Contact-geometry frames can require a determinant-preserving
                # branch selection. Fitted canonical frames are fixed: Curves+
                # handles a reverse Z step by changing Rise and Twist below,
                # not by rotating both base-pair frames by 180 degrees.
                if (
                    self._uses_contact_geometry_pair(calc, partner_strand, level - 1)
                    or self._uses_contact_geometry_pair(calc, partner_strand, level)
                ):
                    previous_pair, current_pair = self._step_aligned_frames(
                        previous_pair,
                        current_pair,
                        calc.cdr,
                    )

                calc.pab[level, :, partner_strand] = self._curvesplus_step_values(
                    previous_pair,
                    current_pair,
                    calc.cdr,
                )

    def fill_local_strand_steps(self, calc) -> None:
        for strand in range(calc.ctx.nst):
            _, _, iste, iene = calc._axis_bounds(strand)
            for level in range(iste + 1, iene + 1):
                previous_contact = self._uses_contact_geometry_level(
                    calc, strand, level - 1
                )
                current_contact = self._uses_contact_geometry_level(
                    calc, strand, level
                )
                previous_frame = self._oriented_strand_frame(
                    calc,
                    strand,
                    level - 1,
                    axis_reference=previous_contact,
                )
                current_frame = self._oriented_strand_frame(
                    calc,
                    strand,
                    level,
                    axis_reference=current_contact,
                )
                if previous_frame is None or current_frame is None:
                    continue
                values = self._rigid_body_values(
                    previous_frame,
                    current_frame,
                    calc.cdr,
                    translation_sign=1.0,
                    rotation_sign=1.0,
                )
                values[3:] = [self._wrap_180(value) for value in values[3:]]
                calc.pal[level, :, strand] = values

    def _base_frame(
        self,
        calc,
        strand: int,
        level: int,
        *,
        axis_reference: bool = False,
    ) -> Optional[ParameterFrame]:
        if not calc._has_level(strand, level):
            return None
        frames = (
            getattr(calc.ctx.params, "axis_frames", None)
            if axis_reference
            else getattr(calc.ctx.params, "shape_frames", None)
        )
        if (
            frames is None
            or frames.shape != calc.ctx.params.frames.shape
            or not np.any(frames)
        ):
            frames = calc.ctx.params.frames
        raw = np.asarray(frames[strand, level], dtype=float)
        if not self.finite(raw):
            return None
        return ParameterFrame(origin=raw[3].copy(), axes=raw[:3].copy())

    def _oriented_strand_frame(
        self,
        calc,
        strand: int,
        level: int,
        *,
        axis_reference: bool = False,
    ) -> Optional[ParameterFrame]:
        """Return a chemically directed frame, or an anchored contact frame."""
        frame = self._base_frame(
            calc,
            strand,
            level,
            axis_reference=axis_reference,
        )
        if frame is None:
            return None
        if calc.ctx.cfg.comb and strand > 0:
            axes = frame.axes.copy()
            if calc.ctx.idr[strand] < 0:
                axes[1] *= -1.0
                axes[2] *= -1.0
            else:
                axes[0] *= -1.0
                axes[1] *= -1.0
            frame = ParameterFrame(origin=frame.origin.copy(), axes=axes)
        return frame

    def _step_aligned_frames(
        self,
        previous: ParameterFrame,
        current: ParameterFrame,
        degrees_per_radian: float,
    ):
        """Choose signed-equivalent frames that describe one step smoothly.

        A fitted base or base-pair frame has determinant-preserving 180-degree
        sign equivalents. Noncanonical/Hoogsteen steps can otherwise report the sign
        jump as a nearly 180-degree local rotation. Select the equivalent pair
        with the smallest relative rotation, then prefer the forward-rise
        solution when the rotation score is tied.
        """
        candidates = []
        for previous_frame in self._equivalent_frame_variants(previous):
            for current_frame in self._equivalent_frame_variants(current):
                rotation_score = float(np.trace(previous_frame.axes @ current_frame.axes.T))
                values = self._rigid_body_values(
                    previous_frame,
                    current_frame,
                    degrees_per_radian,
                    translation_sign=1.0,
                    rotation_sign=1.0,
                )
                previous_x_alignment = float(np.dot(previous_frame.axes[0], previous.axes[0]))
                current_x_alignment = float(np.dot(current_frame.axes[0], current.axes[0]))
                candidates.append((
                    rotation_score,
                    values[2] >= -1e-8,
                    previous_x_alignment,
                    current_x_alignment,
                    values[2],
                    previous_frame,
                    current_frame,
                ))
        if not candidates:
            return previous, current

        best_score = max(item[0] for item in candidates)
        top = [item for item in candidates if item[0] >= best_score - 1e-8]
        best = max(top, key=lambda item: (item[1], item[2], item[3], item[4]))
        return best[5], best[6]

    def _curvesplus_step_values(
        self,
        previous: ParameterFrame,
        current: ParameterFrame,
        degrees_per_radian: float,
    ) -> np.ndarray:
        """Return Curves+ inter-base-pair parameters for two mean frames.

        ``params.f`` keeps the midpoint-frame decomposition intact. When the
        displacement points against the upper base-pair normal, only Rise and
        Twist are put on the forward-Z branch; Slide and Roll are unchanged.
        """
        values = np.asarray(
            self._rigid_body_values(
                previous,
                current,
                degrees_per_radian,
                translation_sign=1.0,
                rotation_sign=1.0,
            ),
            dtype=float,
        )
        delta = current.origin - previous.origin
        if float(np.dot(delta, current.axes[2])) < 0.0:
            values[2] = -values[2]
            values[5] = -values[5]
        values[3:] = [self._wrap_180(value) for value in values[3:]]
        return values

    def _equivalent_frame_variants(self, frame: ParameterFrame):
        for sign_flip in self._EQUIVALENT_AXIS_SIGN_FLIPS:
            yield ParameterFrame(origin=frame.origin.copy(), axes=sign_flip @ frame.axes)

    def _base_pair_member_frames(self, calc, partner_strand: int, level: int):
        contact_geometry = self._contact_geometry_for_pair(calc, partner_strand, level)
        annotation = self._base_pair_annotation(calc, partner_strand, level)
        annotated_geometry = (annotation or {}).get("contact_geometry") or {}
        pair_geometry = contact_geometry or annotated_geometry
        frame_mode = str(
            pair_geometry.get("frame_mode")
            or (annotation or {}).get("frame_mode")
            or ""
        )
        provisional_contact = frame_mode == "provisional_contact_geometry"
        frame_basis = str(
            pair_geometry.get("frame_basis")
            or (annotation or {}).get("frame_basis")
            or ""
        )
        atom_contact_basis = (
            provisional_contact and frame_basis == "atom_contact_axis"
        )
        lw_strand_orientation = str(pair_geometry.get("lw_strand_orientation") or "").lower()
        if not lw_strand_orientation and not provisional_contact:
            lw_strand_orientation = infer_lw_strand_orientation(
                pair_geometry.get("glycosidic_orientation", ""),
                pair_geometry.get("edge_1", ""),
                pair_geometry.get("edge_2", ""),
            )
        if not lw_strand_orientation and not provisional_contact:
            lw_strand_orientation = str(pair_geometry.get("strand_direction") or "").lower()
        first = self._base_frame(calc, 0, level)
        other = self._base_frame(calc, partner_strand, level)
        if first is None or other is None:
            return None
        if atom_contact_basis:
            # Both Y axes point from the primary contact atom toward its
            # partner. Select only the determinant-preserving X/Z alternative
            # so this directed axis cannot be reversed by an LW assumption.
            other = self._aligned_atom_contact_partner_frame(first, other)
        elif provisional_contact:
            # No LW family is authoritative here. Choose the member-frame
            # branch with the smaller physical rotation instead of allowing a
            # tentative cis/trans vote to imply parallel strand semantics.
            other = self._aligned_partner_frame(
                first,
                other,
                prefer_parallel=True,
            )
        elif lw_strand_orientation == "antiparallel":
            other = self._inverted_partner_frame(other)
        elif lw_strand_orientation == "parallel":
            other = ParameterFrame(origin=other.origin.copy(), axes=other.axes.copy())
        else:
            prefer_parallel = self._is_hoogsteen_pair(calc, partner_strand, level)
            other = self._aligned_partner_frame(first, other, prefer_parallel=prefer_parallel)
        if (
            self._uses_contact_geometry_pair(calc, partner_strand, level)
            and not atom_contact_basis
        ):
            other = self._aligned_contact_partner_frame(first, other)
        return first, other

    @staticmethod
    def _aligned_atom_contact_partner_frame(
        first: ParameterFrame,
        other: ParameterFrame,
    ) -> ParameterFrame:
        """Choose the minimum-rotation normal branch while preserving Y."""
        alternate = ParameterFrame(
            origin=other.origin.copy(),
            axes=PAIR_NORMAL_SIGN_FLIP @ other.axes,
        )
        direct_score = float(np.trace(first.axes @ other.axes.T))
        alternate_score = float(np.trace(first.axes @ alternate.axes.T))
        if alternate_score > direct_score + 1e-9:
            return alternate
        return other

    @staticmethod
    def _aligned_contact_partner_frame(
        first: ParameterFrame,
        other: ParameterFrame,
    ) -> ParameterFrame:
        """Choose the nearest valid in-plane branch for a contact frame.

        Edge fitting leaves the directions of X and Y ambiguous up to a
        simultaneous sign reversal.  Resolve that ambiguity only after the LW
        parallel/antiparallel orientation has been applied.  Keeping Z fixed
        preserves the fitted base normal and therefore the physical plane
        bend between the paired bases.
        """
        alternate = ParameterFrame(
            origin=other.origin.copy(),
            axes=CONTACT_IN_PLANE_SIGN_FLIP @ other.axes,
        )
        direct_score = float(np.trace(first.axes @ other.axes.T))
        alternate_score = float(np.trace(first.axes @ alternate.axes.T))
        if alternate_score > direct_score + 1e-9:
            return alternate
        return other

    def _uses_noncanonical_watson_pair(self, calc, partner_strand: int, level: int) -> bool:
        annotation = self._base_pair_annotation(calc, partner_strand, level)
        if not annotation:
            return False
        geometry = annotation.get("contact_geometry") or {}
        return self._is_noncanonical_watson_pair(annotation, geometry)

    def _base_pair_annotation(self, calc, partner_strand: int, level: int):
        base_pairs = getattr(calc.ctx, "annotations", {}).get("base_pair_annotations", [])
        strands = {1, partner_strand + 1}
        for row in base_pairs:
            if row.get("level") != level:
                continue
            annotated = {int(row.get("strand_1", 0)), int(row.get("strand_2", 0))}
            if annotated == strands:
                return row
        return None

    @staticmethod
    def _is_noncanonical_watson_pair(annotation, geometry: dict) -> bool:
        if not annotation:
            return False
        edge_1 = str((geometry or {}).get("edge_1") or annotation.get("edge_1") or "").upper()
        edge_2 = str((geometry or {}).get("edge_2") or annotation.get("edge_2") or "").upper()
        if edge_1 != "W" or edge_2 != "W":
            return False
        return bool(
            annotation.get("identity_class") not in {"watson_crick", ""}
            or annotation.get("calculation_is_hoogsteen")
        )

    def _uses_contact_geometry_pair(self, calc, partner_strand: int, level: int) -> bool:
        keys = getattr(calc.ctx, "contact_geometry_frame_keys", set()) or set()
        return (0, partner_strand, level) in keys or (partner_strand, 0, level) in keys

    @staticmethod
    def _uses_resolved_normal_pair(calc, partner_strand: int, level: int) -> bool:
        modes = getattr(calc.ctx, "pair_normal_branch_modes", {}) or {}
        if (0, partner_strand, level) in modes:
            return True
        contact_keys = getattr(calc.ctx, "contact_geometry_frame_keys", set()) or set()
        return (
            (0, partner_strand, level) in contact_keys
            or (partner_strand, 0, level) in contact_keys
        )

    def _uses_contact_geometry_level(self, calc, strand: int, level: int) -> bool:
        keys = getattr(calc.ctx, "contact_geometry_frame_keys", set()) or set()
        return any(key_strand == strand and key_level == level for key_strand, _, key_level in keys)

    def _contact_geometry_for_pair(self, calc, partner_strand: int, level: int):
        if not self._uses_contact_geometry_pair(calc, partner_strand, level):
            return None
        geometries = getattr(calc.ctx, "pair_contact_geometries", {}) or {}
        return (
            geometries.get((0, partner_strand, level))
            or geometries.get((partner_strand, 0, level))
        )

    def _is_hoogsteen_pair(self, calc, partner_strand: int, level: int) -> bool:
        base_pairs = getattr(calc.ctx, "annotations", {}).get("base_pair_annotations", [])
        strands = {1, partner_strand + 1}
        for bp in base_pairs:
            if not bp.get("calculation_is_hoogsteen") or bp.get("level") != level:
                continue
            annotated = {int(bp.get("strand_1", 0)), int(bp.get("strand_2", 0))}
            if annotated == strands:
                return True
        return False

    @staticmethod
    def _is_hoogsteen_level(calc, strand: int, level: int) -> bool:
        strand_id = strand + 1
        base_pairs = getattr(calc.ctx, "annotations", {}).get("base_pair_annotations", [])
        for bp in base_pairs:
            if not bp.get("calculation_is_hoogsteen") or bp.get("level") != level:
                continue
            strands = {int(bp.get("strand_1", 0)), int(bp.get("strand_2", 0))}
            if strand_id in strands:
                return True
        return False

    def _base_pair_frame(self, calc, partner_strand: int, level: int) -> Optional[ParameterFrame]:
        pair_frames = self._base_pair_member_frames(calc, partner_strand, level)
        if pair_frames is None:
            return None
        first, other = pair_frames
        return self._middle_frame(first, other)

    def _aligned_partner_frame(
        self,
        first: ParameterFrame,
        other: ParameterFrame,
        prefer_parallel: bool = False,
    ) -> ParameterFrame:
        inverted = self._inverted_partner_frame(other)
        if not prefer_parallel:
            return inverted

        direct_score = float(np.trace(first.axes @ other.axes.T))
        inverted_score = float(np.trace(first.axes @ inverted.axes.T))
        if inverted_score > direct_score + 1e-9:
            return inverted
        return ParameterFrame(origin=other.origin.copy(), axes=other.axes.copy())

    @staticmethod
    def _inverted_partner_frame(frame: ParameterFrame) -> ParameterFrame:
        axes = frame.axes.copy()
        axes[1] *= -1.0
        axes[2] *= -1.0
        return ParameterFrame(origin=frame.origin.copy(), axes=axes)

    def _middle_frame(self, first: ParameterFrame, second: ParameterFrame) -> ParameterFrame:
        rotation, _ = Rotation.align_vectors(second.axes, first.axes)
        half_rotation = Rotation.from_rotvec(0.5 * rotation.as_rotvec())
        axes = half_rotation.apply(first.axes)
        axes = self._orthonormalize_axes(axes)
        origin = (first.origin + second.origin) / 2.0
        return ParameterFrame(origin=origin, axes=axes)

    def _rigid_body_values(
        self,
        first: ParameterFrame,
        second: ParameterFrame,
        degrees_per_radian: float,
        translation_sign: float,
        rotation_sign: float,
    ) -> np.ndarray:
        middle = self._middle_frame(first, second)
        translation = translation_sign * (second.origin - first.origin)
        displacement = middle.axes @ translation

        rotation, _ = Rotation.align_vectors(second.axes, first.axes)
        rotvec = rotation_sign * rotation.as_rotvec() * degrees_per_radian
        angles = middle.axes @ rotvec
        return np.array([
            displacement[0],
            displacement[1],
            displacement[2],
            angles[0],
            angles[1],
            angles[2],
        ], dtype=float)

    def _orthonormalize_axes(self, axes: np.ndarray) -> np.ndarray:
        x_axis = self.unit(axes[0], np.array([1.0, 0.0, 0.0]))
        y_axis = axes[1] - x_axis * np.dot(x_axis, axes[1])
        y_axis = self.unit(y_axis, np.array([0.0, 1.0, 0.0]))
        z_axis = self.unit(np.cross(x_axis, y_axis), axes[2])
        y_axis = self.unit(np.cross(z_axis, x_axis), y_axis)
        return np.asarray([x_axis, y_axis, z_axis], dtype=float)

    @staticmethod
    def _wrap_180(value: float) -> float:
        if abs(value) > 180.0:
            value -= np.sign(value) * 360.0
        return float(value)


def convention_for_context(ctx) -> BaseParameterConvention:
    name = str(getattr(ctx.cfg, "frame_convention", "standard")).strip().lower()
    if name in {"standard", "curves_plus", "curves+", "curvesplus", "x3dna", "3dna"}:
        return StandardParameterConvention()
    return LegacyParameterConvention()
