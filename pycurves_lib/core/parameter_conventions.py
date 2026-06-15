from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

HOOGSTEEN_HBOND_ATOMS = {
    ("A", "T"): (("N7", "N3"), ("N6", "O4")),
    ("A", "U"): (("N7", "N3"), ("N6", "O4")),
    ("G", "C"): (("N7", "N3"), ("O6", "N4")),
}

EQUIVALENT_AXIS_SIGN_FLIPS = (
    np.diag([1.0, 1.0, 1.0]),
    np.diag([1.0, -1.0, -1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([-1.0, -1.0, 1.0]),
)


def build_hoogsteen_reference_frames(ctx):
    """Build Hoogsteen-aware fitted frames for shape calculations.

    The fitted base frames remain available as ``params.frames``.  For a
    Hoogsteen pair, ``params.shape_frames`` replaces both paired base frames
    with per-base Hoogsteen reference frames:

    * X follows that base's observed H-bond edge.
    * Y points from strand 1 toward the partner.
    * Z is that base's already-fitted normal from the active base convention.
    * Origins are the centroids of the atoms defining the observed H-bond edge.

    The two bases do not share averaged axes; downstream shape math still
    compares two independent fitted frames, so buckle/propeller/opening remain
    real geometric parameters instead of being collapsed to zero.
    """
    p = ctx.params
    raw_frames = np.asarray(p.frames, dtype=float)
    shape_frames = raw_frames.copy()
    pairs = _hoogsteen_pairs(ctx)
    if not pairs:
        p.shape_frames = shape_frames
        ctx.hoogsteen_reference_frames = []
        return shape_frames

    reference_rows = []
    for partner_strand, level in pairs:
        if not (
            _has_level(ctx, 0, level)
            and _has_level(ctx, partner_strand, level)
        ):
            continue
        pair_frames = _hoogsteen_pair_reference_frames(ctx, 0, partner_strand, level, raw_frames)
        if pair_frames is None:
            continue
        first_frame, partner_frame, pattern = pair_frames
        shape_frames[0, level] = first_frame
        shape_frames[partner_strand, level] = partner_frame
        reference_rows.append({
            "level": level,
            "partner_strand": partner_strand + 1,
            "atom_pairs": pattern,
        })

    p.shape_frames = shape_frames
    ctx.hoogsteen_reference_frames = reference_rows
    return shape_frames


def build_axis_reference_frames(ctx):
    """Build the frame view consumed by the legacy global-axis optimizer.

    A Hoogsteen/syn base can be fit on a discontinuous in-plane branch.  Build
    the Hoogsteen fitted frames first, then choose determinant-preserving sign
    equivalents so the legacy global-axis optimizer sees continuous reference
    frames.
    """
    p = ctx.params
    shape_frames = build_hoogsteen_reference_frames(ctx)
    axis_frames = shape_frames.copy()

    markers = getattr(ctx, "hoogsteen_markers", set()) or set()
    annotations = getattr(ctx, "annotations", {}).get("base_pair_annotations", [])
    has_hoogsteen = bool(markers) or any(bp.get("is_hoogsteen") for bp in annotations)
    ctx.axis_reference_uses_continuity = bool(has_hoogsteen)
    if not has_hoogsteen:
        p.axis_frames = axis_frames
        ctx.axis_frame_adjustments = []
        return axis_frames

    adjustments = []
    for strand in range(ctx.nst):
        previous_axes = None
        for level in range(0, ctx.nux + 2):
            if level >= ctx.li.shape[0] or ctx.li[level, strand] < 0:
                previous_axes = None
                continue
            current = shape_frames[strand, level]
            if not np.all(np.isfinite(current)):
                previous_axes = None
                continue

            axes = current[:3].copy()
            if previous_axes is not None:
                candidates = [sign_flip @ axes for sign_flip in EQUIVALENT_AXIS_SIGN_FLIPS]
                scores = [float(np.trace(previous_axes @ candidate.T)) for candidate in candidates]
                best_index = int(np.argmax(scores))
                axes = candidates[best_index]
                axis_frames[strand, level, :3, :] = axes
                if best_index != 0:
                    adjustments.append((strand + 1, level))

            previous_axes = axes

    p.axis_frames = axis_frames
    ctx.axis_frame_adjustments = adjustments
    return axis_frames


def _has_level(ctx, strand: int, level: int) -> bool:
    return 0 <= strand < ctx.nst and 1 <= level <= ctx.nux and ctx.li[level, strand] >= 0


def _hoogsteen_pairs(ctx):
    pairs = set()
    annotations = getattr(ctx, "annotations", {}).get("base_pair_annotations", [])
    for row in annotations:
        if row.get("is_hoogsteen") and row.get("level") is not None:
            strands = {int(row.get("strand_1", 0)), int(row.get("strand_2", 0))}
            if 1 in strands and len(strands) == 2:
                partner = next(strand for strand in strands if strand != 1)
                pairs.add((partner - 1, int(row["level"])))

    markers = getattr(ctx, "hoogsteen_markers", set()) or set()
    for marker in markers:
        level = None
        if isinstance(marker, tuple) and len(marker) >= 2:
            level = int(marker[-1])
        elif isinstance(marker, (int, np.integer)):
            level = int(marker)
        if level is None:
            continue
        for partner_strand in range(1, ctx.nst):
            if _has_level(ctx, 0, level) and _has_level(ctx, partner_strand, level):
                pairs.add((partner_strand, level))

    return sorted(pairs, key=lambda item: (item[1], item[0]))


def _hoogsteen_pair_reference_frames(
    ctx,
    first_strand: int,
    partner_strand: int,
    level: int,
    raw_frames: np.ndarray,
):
    first_base, first_atoms = _base_atom_map(ctx, first_strand, level)
    partner_base, partner_atoms = _base_atom_map(ctx, partner_strand, level)
    atom_pairs = _hoogsteen_atom_pairs(first_base, partner_base)
    if not atom_pairs:
        return None

    first_points = []
    partner_points = []
    used_pairs = []
    for first_atom, partner_atom in atom_pairs:
        first_point = first_atoms.get(first_atom)
        partner_point = partner_atoms.get(partner_atom)
        if first_point is None or partner_point is None:
            continue
        first_points.append(first_point)
        partner_points.append(partner_point)
        used_pairs.append((first_atom, partner_atom))

    if len(first_points) < 2:
        return None

    first_points = np.asarray(first_points, dtype=float)
    partner_points = np.asarray(partner_points, dtype=float)

    hbond_axis = _unit(np.mean(partner_points - first_points, axis=0), raw_frames[first_strand, level, 1, :])
    first_frame = _hoogsteen_member_reference_frame(
        raw_frames[first_strand, level],
        first_points,
        hbond_axis,
    )
    partner_frame = _hoogsteen_member_reference_frame(
        raw_frames[partner_strand, level],
        partner_points,
        hbond_axis,
    )
    return first_frame, partner_frame, tuple(used_pairs)


def _hoogsteen_member_reference_frame(raw_frame: np.ndarray, edge_points: np.ndarray, hbond_axis: np.ndarray) -> np.ndarray:
    """Return one base's Hoogsteen fitted frame.

    The base normal comes from the already-fitted base frame.  The in-plane
    orientation and origin are rebuilt from that base's Hoogsteen H-bond edge.
    """
    edge_points = np.asarray(edge_points, dtype=float)
    z_axis = _unit(raw_frame[2], np.array([0.0, 0.0, 1.0]))
    x_axis = edge_points[-1] - edge_points[0]
    x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
    x_axis = _unit(x_axis, raw_frame[0])
    y_axis = _unit(np.cross(z_axis, x_axis), raw_frame[1])
    x_axis = _unit(np.cross(y_axis, z_axis), x_axis)

    if np.dot(y_axis, hbond_axis) < 0.0:
        x_axis *= -1.0
        y_axis *= -1.0

    frame = np.asarray(raw_frame, dtype=float).copy()
    frame[:3, :] = _orthonormalize_axes(np.asarray([x_axis, y_axis, z_axis], dtype=float))
    frame[3, :] = np.mean(edge_points, axis=0)
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


def _hoogsteen_atom_pairs(first_base: str, partner_base: str):
    direct = HOOGSTEEN_HBOND_ATOMS.get((first_base, partner_base))
    if direct is not None:
        return direct
    reversed_pairs = HOOGSTEEN_HBOND_ATOMS.get((partner_base, first_base))
    if reversed_pairs is None:
        return ()
    return tuple((partner_atom, first_atom) for first_atom, partner_atom in reversed_pairs)


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
        if StandardParameterConvention()._is_hoogsteen_pair(calc, partner_strand, level):
            return StandardParameterConvention().local_base_base_values(calc, partner_strand, level)

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

        self._fill_hoogsteen_base_pair_steps(calc)

    def fill_local_strand_steps(self, calc) -> None:
        standard = StandardParameterConvention()
        for strand in range(calc.ctx.nst):
            _, _, iste, iene = calc._axis_bounds(strand)
            continuous_frames = {}
            previous = None
            for level in range(iste, iene + 1):
                frame = standard._oriented_strand_frame(calc, strand, level)
                if frame is None:
                    previous = None
                    continue
                if previous is not None:
                    frame = standard._most_continuous_equivalent_frame(previous, frame)
                continuous_frames[level] = frame
                previous = frame

            for level in range(iste + 1, iene + 1):
                if not (
                    standard._is_hoogsteen_level(calc, strand, level - 1)
                    or standard._is_hoogsteen_level(calc, strand, level)
                ):
                    continue
                previous_frame = continuous_frames.get(level - 1)
                current_frame = continuous_frames.get(level)
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

    def _fill_hoogsteen_base_pair_steps(self, calc) -> None:
        standard = StandardParameterConvention()
        for partner_strand in range(1, calc.ctx.nst):
            for level in range(calc.optimizer.iste + 1, calc.optimizer.iene + 1):
                if not (
                    standard._is_hoogsteen_pair(calc, partner_strand, level - 1)
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
                previous_pair, current_pair = standard._step_aligned_frames(previous_pair, current_pair, calc.cdr)
                values = standard._rigid_body_values(
                    previous_pair,
                    current_pair,
                    calc.cdr,
                    translation_sign=1.0,
                    rotation_sign=1.0,
                )
                values[3:] = [standard._wrap_180(value) for value in values[3:]]
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
        values = self._rigid_body_values(first, other, calc.cdr, translation_sign=-1.0, rotation_sign=-1.0)
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
                previous_pair, current_pair = self._step_aligned_frames(previous_pair, current_pair, calc.cdr)

                calc.pab[level, :, partner_strand] = self._rigid_body_values(
                    previous_pair,
                    current_pair,
                    calc.cdr,
                    translation_sign=1.0,
                    rotation_sign=1.0,
                )

    def fill_local_strand_steps(self, calc) -> None:
        for strand in range(calc.ctx.nst):
            _, _, iste, iene = calc._axis_bounds(strand)
            continuous_frames = {}
            previous = None
            for level in range(iste, iene + 1):
                frame = self._oriented_strand_frame(calc, strand, level)
                if frame is None:
                    previous = None
                    continue
                if previous is not None:
                    frame = self._most_continuous_equivalent_frame(previous, frame)
                continuous_frames[level] = frame
                previous = frame

            for level in range(iste + 1, iene + 1):
                previous_frame = continuous_frames.get(level - 1)
                current_frame = continuous_frames.get(level)
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

    def _base_frame(self, calc, strand: int, level: int) -> Optional[ParameterFrame]:
        if not calc._has_level(strand, level):
            return None
        frames = getattr(calc.ctx.params, "shape_frames", None)
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

    def _oriented_strand_frame(self, calc, strand: int, level: int) -> Optional[ParameterFrame]:
        frame = self._base_frame(calc, strand, level)
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

    def _most_continuous_equivalent_frame(
        self,
        previous: ParameterFrame,
        current: ParameterFrame,
    ) -> ParameterFrame:
        best_axes = max(
            (sign_flip @ current.axes for sign_flip in self._EQUIVALENT_AXIS_SIGN_FLIPS),
            key=lambda axes: float(np.trace(previous.axes @ axes.T)),
        )
        return ParameterFrame(origin=current.origin.copy(), axes=best_axes.copy())

    def _step_aligned_frames(
        self,
        previous: ParameterFrame,
        current: ParameterFrame,
        degrees_per_radian: float,
    ):
        """Choose signed-equivalent frames that describe one step smoothly.

        A fitted base or base-pair frame has determinant-preserving 180-degree
        sign equivalents. Syn/Hoogsteen steps can otherwise report the sign
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

    def _equivalent_frame_variants(self, frame: ParameterFrame):
        for sign_flip in self._EQUIVALENT_AXIS_SIGN_FLIPS:
            yield ParameterFrame(origin=frame.origin.copy(), axes=sign_flip @ frame.axes)

    def _base_pair_member_frames(self, calc, partner_strand: int, level: int):
        is_hoogsteen = self._is_hoogsteen_pair(calc, partner_strand, level)
        first = self._interaction_base_frame(calc, 0, level)
        other = self._interaction_base_frame(calc, partner_strand, level)
        if first is None or other is None:
            return None
        other = self._aligned_partner_frame(first, other, prefer_parallel=is_hoogsteen)
        return first, other

    def _interaction_base_frame(
        self,
        calc,
        strand: int,
        level: int,
    ) -> Optional[ParameterFrame]:
        frame = self._base_frame(calc, strand, level)
        if frame is None:
            return None
        # Hoogsteen-aware shape frames are installed once in params.shape_frames;
        # downstream parameter math should compare those fitted frames directly.
        return frame

    def _is_hoogsteen_pair(self, calc, partner_strand: int, level: int) -> bool:
        if self._hoogsteen_marker_matches(calc, 0, partner_strand, level):
            return True
        if self._is_hoogsteen_level(calc, 0, level) or self._is_hoogsteen_level(calc, partner_strand, level):
            return True
        base_pairs = getattr(calc.ctx, "annotations", {}).get("base_pair_annotations", [])
        strands = {1, partner_strand + 1}
        for bp in base_pairs:
            if not bp.get("is_hoogsteen") or bp.get("level") != level:
                continue
            annotated = {int(bp.get("strand_1", 0)), int(bp.get("strand_2", 0))}
            if annotated == strands:
                return True
        return False

    @staticmethod
    def _is_hoogsteen_level(calc, strand: int, level: int) -> bool:
        markers = getattr(calc.ctx, "hoogsteen_markers", set()) or set()
        strand_id = strand + 1
        if level in markers or (strand_id, level) in markers:
            return True
        for marker in markers:
            if (
                isinstance(marker, tuple)
                and len(marker) == 3
                and marker[2] == level
                and strand_id in marker[:2]
            ):
                return True
        base_pairs = getattr(calc.ctx, "annotations", {}).get("base_pair_annotations", [])
        for bp in base_pairs:
            if not bp.get("is_hoogsteen") or bp.get("level") != level:
                continue
            strands = {int(bp.get("strand_1", 0)), int(bp.get("strand_2", 0))}
            if strand_id in strands:
                return True
        return False

    @staticmethod
    def _hoogsteen_marker_matches(calc, first_strand: int, partner_strand: int, level: int) -> bool:
        markers = getattr(calc.ctx, "hoogsteen_markers", set()) or set()
        if level in markers:
            return True
        first = first_strand + 1
        second = partner_strand + 1
        return (
            (first, level) in markers
            or (second, level) in markers
            or (first, second, level) in markers
            or (second, first, level) in markers
        )

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
    name = str(getattr(ctx.cfg, "frame_convention", "legacy")).strip().lower()
    if name in {"standard", "curves_plus", "curves+", "curvesplus", "x3dna", "3dna"}:
        return StandardParameterConvention()
    return LegacyParameterConvention()
