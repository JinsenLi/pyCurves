"""Curves+/standard global-axis helpers.

This module keeps the Curves+ smooth-axis implementation separate from the
legacy Curves 5.3 global-axis/minimization code in ``curves_calculator.py``.
The methods are written as a mixin so they can still use the calculator's
existing helper methods and arrays without changing the public API.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation


class CurvesPlusAxisMixin:
    """Mixin implementing the Curves+ axis/smooth path."""

    def _calculate_curvesplus_global_parameters(self):
        """Curves+ standard global axis/BP-axis path.

        Curves+ first forms base-pair mean frames (``upm``) from standard
        reference frames, then derives and smooths a curvilinear helical axis
        (``uvw``) from adjacent base screw axes.  Hoogsteen-aware runs use the
        same axis reference frames as the legacy optimizer so syn/Hoogsteen
        discontinuities do not bend the smooth path.
        """
        if not (self.ctx.cfg.comb and self.ctx.nst > 1):
            return

        ref = self._curvesplus_reference_frames()
        upm = self._curvesplus_base_pair_frames(ref)
        uvw = self._curvesplus_smoothed_axis(ref, upm)
        axis_upm = self._curvesplus_axis_parameter_frames(upm, uvw)
        invert = self._curvesplus_inversion_flags(axis_upm)

        nux = self.ctx.n_levels
        self.curvesplus_reference_frames = ref
        self.curvesplus_base_pair_frames = upm
        self.curvesplus_axis_base_pair_frames = axis_upm
        self.curvesplus_axis_frames = uvw
        self.curvesplus_invert = invert

        self.curvesplus_bp_axis = np.full((nux + 1, 4), np.nan, dtype=float)
        for level in range(1, nux + 1):
            self.curvesplus_bp_axis[level] = self._curvesplus_bp_axis_values(axis_upm[level], uvw[level], invert[level])

        self.curvesplus_inter_base_pair = np.full((nux + 1, 6), np.nan, dtype=float)
        for level in range(2, nux + 1):
            if not (
                np.all(np.isfinite(axis_upm[level - 1]))
                and np.all(np.isfinite(axis_upm[level]))
            ):
                continue
            previous_frame, current_frame = self.parameter_convention._step_aligned_frames(
                self._frame_from_array(axis_upm[level - 1]),
                self._frame_from_array(axis_upm[level]),
                self.cdr,
            )
            values = self.parameter_convention._rigid_body_values(
                previous_frame,
                current_frame,
                self.cdr,
                translation_sign=1.0,
                rotation_sign=1.0,
            )
            if invert[level - 1]:
                values[2] = -values[2]
                values[5] = -values[5]
            values[3:] = [self._wrap_180(v) for v in values[3:]]
            self.curvesplus_inter_base_pair[level - 1] = values

    def _install_curvesplus_axis_for_groove(self):
        """Expose Curves+ smooth-axis frames through legacy axis arrays.

        Curves+ ``manta.f`` computes grooves after ``axis.f``/``smooth.f``,
        using the smoothed ``uvw`` helical-axis frames instead of Curves 5.3
        minimized ``uho/hho``.  The pyCurves groove scanner still reads the
        common ``optimizer.uho/hho`` axis arrays, so in Curves+ axis mode we
        populate those arrays from ``curvesplus_axis_frames``.
        """
        if not hasattr(self, "curvesplus_axis_frames"):
            return
        uvw = np.asarray(self.curvesplus_axis_frames, dtype=float)
        stop = min(uvw.shape[0], self.optimizer.uho.shape[0], self.optimizer.hho.shape[0])
        if stop <= 1:
            return

        directions = uvw[:stop, 2, :]
        points = uvw[:stop, 3, :]
        valid = np.all(np.isfinite(directions), axis=1) & np.all(np.isfinite(points), axis=1)
        for level in range(1, stop):
            if not valid[level]:
                continue
            for strand in range(self.ctx.nst):
                self.optimizer.uho[level, :, strand] = directions[level]
                self.optimizer.hho[level, :, strand] = points[level]
            self.ctx.params.ux[level] = directions[level]
            self.ctx.params.ox[level] = points[level]

    def _curvesplus_reference_frames(self):
        nux = self.ctx.n_levels
        nst = self.ctx.nst
        source_frames = getattr(self.ctx.params, "axis_frames", None)
        if (
            source_frames is None
            or source_frames.shape != self.ctx.params.frames.shape
            or not np.any(source_frames)
        ):
            source_frames = self.ctx.params.frames
        ref = np.full((nux + 1, nst, 4, 3), np.nan, dtype=float)
        for strand in range(nst):
            for level in range(1, nux + 1):
                if not self._has_level(strand, level):
                    continue
                frame = source_frames[strand, level].copy()
                if strand > 0:
                    if self.ctx.idr[strand] < 0:
                        frame[1] *= -1.0
                        frame[2] *= -1.0
                    else:
                        frame[0] *= -1.0
                        frame[1] *= -1.0
                ref[level, strand] = frame
        return ref

    def _curvesplus_base_pair_frames(self, ref):
        nux = self.ctx.n_levels
        upm = np.full((nux + 1, 4, 3), np.nan, dtype=float)
        for level in range(1, nux + 1):
            if self._has_level(0, level) and self._has_level(1, level):
                if self.parameter_convention._is_hoogsteen_pair(self, 1, level):
                    pair_frame = self.parameter_convention._base_pair_frame(self, 1, level)
                    if pair_frame is None:
                        continue
                    upm[level, :3] = pair_frame.axes
                    upm[level, 3] = pair_frame.origin
                else:
                    # Curves+ calls screw(r2, r1, 0): the midpoint frame between
                    # the strand-2 and strand-1 reference systems.
                    upm[level] = self._curvesplus_middle_frame(ref[level, 1], ref[level, 0])
            else:
                for strand in range(self.ctx.nst):
                    if self._has_level(strand, level):
                        upm[level] = ref[level, strand]
                        break
        return upm

    def _curvesplus_axis_parameter_frames(self, upm, uvw):
        axis_upm = upm.copy()
        nux = self.ctx.n_levels
        for level in range(1, nux + 1):
            if not (
                np.all(np.isfinite(axis_upm[level]))
                and np.all(np.isfinite(uvw[level, 2]))
                and np.all(np.isfinite(uvw[level, 3]))
            ):
                continue
            if np.dot(axis_upm[level, 2], uvw[level, 2]) < 0.0:
                axis_upm[level, 1] *= -1.0
                axis_upm[level, 2] *= -1.0
        return axis_upm

    def _curvesplus_smoothed_axis(self, ref, upm):
        nux = self.ctx.n_levels
        nst = self.ctx.nst
        upl = np.zeros((nux + 1, 9, 6), dtype=float)
        npl = np.zeros(nux + 1, dtype=int)

        for upper in range(2, nux + 1):
            lower = upper - 1
            for strand in range(nst):
                if not (self._has_level(strand, lower) and self._has_level(strand, upper)):
                    continue
                axis, point = self._curvesplus_screw_axis(ref[lower, strand], ref[upper, strand])
                for level in (lower, upper):
                    idx = npl[level] + 1
                    npl[level] = idx
                    delta = upm[level, 3] - point
                    projected = point + np.dot(delta, axis) * axis
                    upl[level, idx, :3] = axis
                    upl[level, idx, 3:] = projected

        averaged = np.zeros((nux + 1, 6), dtype=float)
        for level in range(1, nux + 1):
            count = npl[level]
            if count == 0:
                continue
            axis = self.parameter_convention.unit(np.sum(upl[level, 1:count + 1, :3], axis=0))
            point = np.mean(upl[level, 1:count + 1, 3:], axis=0)
            delta = upm[level, 3] - point
            averaged[level, :3] = axis
            averaged[level, 3:] = point + np.dot(delta, axis) * axis

        uvw = np.full((nux + 1, 4, 3), np.nan, dtype=float)
        width = 4
        weights = {0: 1.0}
        for offset in range(1, width + 1):
            weights[offset] = 1.0 - float(offset * offset) / float((width + 1) * (width + 1))
            weights[-offset] = weights[offset]

        for level in range(1, nux + 1):
            origin = upm[level, 3]
            axis_sum = np.zeros(3, dtype=float)
            point_sum = np.zeros(3, dtype=float)
            weight_sum = 0.0
            for offset in range(-width, width + 1):
                source = level + offset
                if source < 1 or source > nux:
                    continue
                axis = averaged[source, :3]
                point = averaged[source, 3:]
                if np.linalg.norm(axis) < 1e-12:
                    continue
                weight = weights[offset]
                axis_sum += axis * weight
                projected = point + np.dot(origin - point, axis) * axis
                point_sum += projected * weight
                weight_sum += weight
            axis = self.parameter_convention.unit(axis_sum, upm[level, 2])
            point = point_sum / weight_sum
            point = point + np.dot(origin - point, axis) * axis
            uvw[level, 2] = axis
            uvw[level, 3] = point
        return uvw

    def _curvesplus_inversion_flags(self, upm):
        nux = self.ctx.n_levels
        invert = np.zeros(nux + 1, dtype=bool)
        for level in range(2, nux + 1):
            previous = level - 1
            delta = upm[level, 3] - upm[previous, 3]
            invert[previous] = np.dot(delta, upm[level, 2]) < 0.0
        if nux > 1:
            invert[nux] = invert[nux - 1]
        return invert

    def _curvesplus_bp_axis_values(self, upm_frame, uvw_frame, invert):
        axis = uvw_frame[2]
        point = uvw_frame[3]
        if not (
            np.all(np.isfinite(upm_frame))
            and np.all(np.isfinite(axis))
            and np.all(np.isfinite(point))
        ):
            return np.full(4, np.nan)

        dot = np.clip(np.dot(upm_frame[2], axis), -1.0, 1.0)
        theta = math.acos(dot)
        if abs(theta) > 1e-4:
            rotation_axis = self.parameter_convention.unit(np.cross(upm_frame[2], axis))
            x_axis = self._rotate_vector_aligned(upm_frame[0], rotation_axis, math.cos(theta), math.sin(theta))
            y_axis = np.cross(axis, x_axis)
        else:
            rotation_axis = np.zeros(3, dtype=float)
            x_axis = upm_frame[0].copy()
            y_axis = upm_frame[1].copy()

        uvw_frame[0] = self.parameter_convention.unit(x_axis, upm_frame[0])
        uvw_frame[1] = self.parameter_convention.unit(y_axis, upm_frame[1])

        angle_vector = -theta * self.cdr * rotation_axis
        delta = upm_frame[3] - point
        displacements = np.array([np.dot(delta, uvw_frame[i]) for i in range(3)], dtype=float)
        angles = np.array([np.dot(angle_vector, uvw_frame[i]) for i in range(3)], dtype=float)

        if invert:
            displacements[0] = -displacements[0]
            angles[0] = -angles[0]
            angles[1] -= 180.0
        angles[0] = self._wrap_180(angles[0])
        angles[1] = self._wrap_180(angles[1])
        return np.array([displacements[0], displacements[1], angles[0], angles[1]], dtype=float)

    def _curvesplus_middle_frame(self, first, second):
        frame = self.parameter_convention._middle_frame(
            self._frame_from_array(first),
            self._frame_from_array(second),
        )
        out = np.zeros((4, 3), dtype=float)
        out[:3] = frame.axes
        out[3] = frame.origin
        return out

    def _curvesplus_screw_axis(self, first, second):
        rotation, _ = Rotation.align_vectors(second[:3], first[:3])
        rotvec = rotation.as_rotvec()
        theta = np.linalg.norm(rotvec)
        vector = second[3] - first[3]
        if theta < 1e-10:
            axis = self.parameter_convention.unit(vector, first[2])
            point = (first[3] + second[3]) / 2.0
            return axis, point
        axis = rotvec / theta
        if np.dot(axis, vector) < 0.0:
            axis = -axis
        axial_distance = np.dot(axis, vector)
        half_perp = (vector - axial_distance * axis) / 2.0
        point = (first[3] + second[3]) / 2.0 + np.cross(axis, half_perp) / math.tan(theta / 2.0)
        return axis, point

    @staticmethod
    def _rotate_vector_aligned(vector, axis, ca, sa):
        rx, ry, rz = axis
        xx, yy, zz = vector
        return np.array([
            (rx * rx + (1.0 - rx * rx) * ca) * xx
            + (rx * ry * (1.0 - ca) - rz * sa) * yy
            + (rx * rz * (1.0 - ca) + ry * sa) * zz,
            (rx * ry * (1.0 - ca) + rz * sa) * xx
            + (ry * ry + (1.0 - ry * ry) * ca) * yy
            + (ry * rz * (1.0 - ca) - rx * sa) * zz,
            (rx * rz * (1.0 - ca) - ry * sa) * xx
            + (ry * rz * (1.0 - ca) + rx * sa) * yy
            + (rz * rz + (1.0 - rz * rz) * ca) * zz,
        ], dtype=float)
