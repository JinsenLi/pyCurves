from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


STEP_PARAMETER_COUNT = 6


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
                if calc._unsupported_shape_step(i - 1):
                    calc.pab[i, :, k] = np.nan
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


class StandardParameterConvention(LegacyParameterConvention):
    """Standard Curves+/3DNA-style local parameter decomposition."""

    name = "standard"

    def local_base_base_values(self, calc, partner_strand: int, level: int):
        first = self._base_frame(calc, 0, level)
        other = self._aligned_partner_base_frame(calc, partner_strand, level)
        if first is None or other is None:
            return None
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

                calc.pab[level, :, partner_strand] = self._rigid_body_values(
                    previous_pair,
                    current_pair,
                    calc.cdr,
                    translation_sign=1.0,
                    rotation_sign=1.0,
                )

    def _base_frame(self, calc, strand: int, level: int) -> Optional[ParameterFrame]:
        if not calc._has_level(strand, level):
            return None
        raw = np.asarray(calc.ctx.params.frames[strand, level], dtype=float)
        if not self.finite(raw):
            return None
        return ParameterFrame(origin=raw[3].copy(), axes=raw[:3].copy())

    def _aligned_partner_base_frame(self, calc, strand: int, level: int) -> Optional[ParameterFrame]:
        frame = self._base_frame(calc, strand, level)
        if frame is None:
            return None
        axes = frame.axes.copy()
        axes[1] *= -1.0
        axes[2] *= -1.0
        return ParameterFrame(origin=frame.origin.copy(), axes=axes)

    def _base_pair_frame(self, calc, partner_strand: int, level: int) -> Optional[ParameterFrame]:
        first = self._base_frame(calc, 0, level)
        other = self._aligned_partner_base_frame(calc, partner_strand, level)
        if first is None or other is None:
            return None
        return self._middle_frame(first, other)

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


class HoogsteenParameterConvention(StandardParameterConvention):
    """Placeholder for future edge-specific Hoogsteen interaction frames."""

    name = "hoogsteen"


def convention_for_context(ctx) -> BaseParameterConvention:
    name = str(getattr(ctx.cfg, "frame_convention", "legacy")).strip().lower()
    if name in {"standard", "curves_plus", "curves+", "curvesplus", "x3dna", "3dna"}:
        return StandardParameterConvention()
    return LegacyParameterConvention()
