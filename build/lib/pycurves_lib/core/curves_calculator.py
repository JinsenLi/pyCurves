import numpy as np
from scipy.spatial.transform import Rotation
import math
from pycurves_lib.data.modified_bases import parent_base_name
from pycurves_lib.core.curves_groove import GrooveAnalysisMixin
from pycurves_lib.core.curvesplus_axis import CurvesPlusAxisMixin
from pycurves_lib.core.parameter_conventions import convention_for_context

class HelicalCalculator(CurvesPlusAxisMixin, GrooveAnalysisMixin):
    def __init__(self, ctx: 'CurvesContext', optimizer: 'HelicalOptimizer'):
        self.ctx = ctx
        self.optimizer = optimizer
        self.helOpt = optimizer  # Backward-compatible alias for the original port name.
        level_count = self.ctx.n_levels
        strand_count = self.ctx.nst
        self.cdr = 180.0 / np.pi
        self.rdc = np.pi / 180.0
        # Fortran inv(k): sign used to orient each local helical axis.
        self.inv = np.zeros(4, dtype = int)
        self.axis_direction_sign = self.inv
        # vkin is the Curves global inter-step/curvature work array.
        # Columns 0-5 are shift, slide, tilt, roll, displacement, angle;
        # column 6 is local axis path length.
        self.vkin = np.zeros((level_count + 2, 7, 4), dtype = float)
        self.global_axis_steps = self.vkin
        self.path_length = np.zeros(strand_count)
        self.end_to_end = np.zeros(strand_count)
        self.shortening = np.zeros(strand_count)
        self.bend = np.zeros((level_count + 2, strand_count))
        self.bend_offset = np.zeros((level_count + 2, strand_count))
        # Cached local frames from params.f.  Fortran names are shown in the
        # suffix: ulx=axis direction, vx/wx=local transverse axes, px=axis point.
        self._global_ulx = None
        self._global_vx = None
        self._global_wx = None
        self._global_px = None
        self.parameter_convention = convention_for_context(ctx)

    def _axis_bounds(self, strand: int):
        """Return Fortran ist/ien/iste/iene bounds for one output strand."""
        if self.ctx.cfg.comb:
            return (
                int(self.optimizer.ist),
                int(self.optimizer.ien),
                int(self.optimizer.iste),
                int(self.optimizer.iene),
            )
        if hasattr(self.optimizer, "ist_by_strand"):
            return (
                int(self.optimizer.ist_by_strand[strand]),
                int(self.optimizer.ien_by_strand[strand]),
                int(self.optimizer.iste_by_strand[strand]),
                int(self.optimizer.iene_by_strand[strand]),
            )
        ist = int(self.ctx.ng[strand])
        ien = int(self.ctx.nr[strand])
        iste = ist - 1 if self.ctx.cfg.ends else ist
        iene = ien + 1 if self.ctx.cfg.ends else ien
        return ist, ien, iste, iene

    def _axis_reference_frames(self):
        p = self.ctx.params
        axis_frames = getattr(p, "axis_frames", None)
        if (
            axis_frames is None
            or axis_frames.shape != p.frames.shape
            or not np.any(axis_frames)
        ):
            return p.frames
        return axis_frames

    def calculate_all(self):
        """Run the regular Curves calculation stages in Fortran call order."""
        self._set_axis_directions()
        if self._use_curvesplus_axis_convention():
            self._calculate_local_parameters()
            self._calculate_curvesplus_global_parameters()
            self._install_curvesplus_axis_for_groove()
            return
        self._calculate_global_parameters()
        self._cache_global_inter_base_parameters()
        self._calculate_local_parameters()
        self._calculate_overall_bend()

    def _set_axis_directions(self):
        """Fill inv(k), the Fortran strand-axis orientation sign."""
        p = self.ctx.params
        for k in range(self.ctx.nst):
            rise_sum = 0.0
            ist, ien, _, _ = self._axis_bounds(k)
            for i in range(ist + 1, ien + 1):
                u_sum = self.optimizer.uho[i, :, k] + self.optimizer.uho[i-1, :, k]
                h_diff = self.optimizer.hho[i, :, k] - self.optimizer.hho[i-1, :, k]
                rise_sum += np.sign(np.dot(u_sum, h_diff))
            
            self.inv[k] = 1
            if (self.ctx.idr[k] == 1 and rise_sum < 0) or (self.ctx.idr[k] == -1 and rise_sum > 0):
                self.inv[k] = -1
        self._apply_axis_direction_reference()

    def _apply_axis_direction_reference(self):
        """Optionally keep axis direction signs consistent across MD frames."""
        reference = getattr(self.ctx, "axis_direction_sign_reference", None)
        if reference is None:
            return
        reference = np.asarray(reference, dtype=float)
        if reference.size < self.ctx.nst:
            return
        for k in range(self.ctx.nst):
            if np.isfinite(reference[k]) and reference[k] != 0:
                self.inv[k] = 1 if reference[k] > 0 else -1

    def _calculate_global_parameters(self):
        """
        Calculate global base-axis and global axis-step parameters.

        The local array names are retained from params.f where the formulas are
        very dense: ulx is the local axis direction, px is the projected axis
        point, and vx/wx are the two transverse axes used for displacements.
        """
        p = self.ctx.params
        axis_frames = self._axis_reference_frames()
        cfg = self.ctx.cfg
        nst = self.ctx.nst
        n3 = self.ctx.n_levels + 2
        cdr = self.cdr  # 180 / pi
        rdc = 1.0 / cdr

        ulx_m = np.zeros((n3, nst, 3))  # Fortran ulx/uly/ulz.
        px_m = np.zeros((n3, nst, 3))  # Fortran px/py/pz.
        wx_m = np.zeros((n3, nst, 3))  # Fortran wx/wy/wz.
        vx_m = np.zeros((n3, nst, 3))  # Fortran vx/vy/vz.
        vax_all = np.zeros((n3, 3))  # Combined-strand average transverse axis.

        for k in range(nst):
            _, _, iste, iene = self._axis_bounds(k)
            
            u_sum = self.optimizer.uho[iste+1:iene+1, :, k] + self.optimizer.uho[iste:iene, :, k]
            h_diff = self.optimizer.hho[iste+1:iene+1, :, k] - self.optimizer.hho[iste:iene, :, k]
            dots = np.einsum('ij,ij->i', u_sum, h_diff)
            rise_sum = np.sum(np.sign(dots))
            
            self.inv[k] = 1
            if (self.ctx.idr[k] == 1 and rise_sum < 0) or (self.ctx.idr[k] == -1 and rise_sum > 0):
                self.inv[k] = -1
        self._apply_axis_direction_reference()

        for k in range(nst):
            is_idx = 0 if cfg.comb else k
            _, _, iste, iene = self._axis_bounds(k)
            
            if cfg.comb and k > 0:
                ulx_m[iste:iene+1, k] = -ulx_m[iste:iene+1, 0]
            else:
                ulx_m[iste:iene+1, k] = self.optimizer.uho[iste:iene+1, :, is_idx] * self.inv[k]

            rex_4 = axis_frames[k, iste:iene+1, 3, :]
            hho_is = self.optimizer.hho[iste:iene+1, :, is_idx]
            
            rel_pos = rex_4 - hho_is
            dot_p = np.einsum('ij,ij->i', rel_pos, ulx_m[iste:iene+1, k])
            px_m[iste:iene+1, k] = hho_is + ulx_m[iste:iene+1, k] * dot_p[:, np.newaxis]

            dax = axis_frames[k, iste:iene+1, 1, :]
            dot_u_dax = np.einsum('ij,ij->i', ulx_m[iste:iene+1, k], dax)
            wx = dax - ulx_m[iste:iene+1, k] * dot_u_dax[:, np.newaxis]
            wx_norm = np.linalg.norm(wx, axis=1)
            valid_wx = wx_norm > 1e-12
            wx[valid_wx] /= wx_norm[valid_wx, np.newaxis]
            wx[~valid_wx] = 0.0
            
            for i in range(iste + 1, iene + 1):
                if np.dot(wx[i-iste-1], wx[i-iste]) < 0: wx[i-iste] *= -1

            # The projected transverse axis has a 180-degree sign ambiguity.
            # Keep it on the branch closest to the current base Dy axis; otherwise
            # anti-parallel/late-MD frames can report inclinations near 180 deg
            # even though the geometric base-axis angle is small.
            opposite_branch = np.einsum('ij,ij->i', wx, dax) < 0.0
            wx[opposite_branch] *= -1.0
            wx_m[iste:iene+1, k] = wx
            
            vx_m[iste:iene+1, k] = np.cross(wx, ulx_m[iste:iene+1, k])

            # Fortran only copies the active replacement frame back to strand 1
            # for absent/substituted levels, not for every paired base.
            if cfg.comb and k > 0:
                for level in range(iste, iene + 1):
                    kc = self.ctx.iact[level] - 1 if self.ctx.li[level, k] < -1 else 0
                    if kc > 0 and kc == k:
                        dot_ul = np.dot(ulx_m[level, 0], ulx_m[level, kc])
                        sgn = math.copysign(1.0, -dot_ul)
                        vx_m[level, 0] = vx_m[level, k] * sgn
                        wx_m[level, 0] = wx_m[level, k] * sgn

            ex = rex_4 - px_m[iste:iene+1, k]
            p.helical[k, iste:iene+1, 0] = np.einsum('ij,ij->i', ex, vx_m[iste:iene+1, k])
            p.helical[k, iste:iene+1, 1] = np.einsum('ij,ij->i', ex, wx_m[iste:iene+1, k])

            dot_cln = np.clip(np.einsum('ij,ij->i', wx, dax), -1.0, 1.0)
            cln = np.arccos(dot_cln) * cdr
            tx_c = np.cross(wx, dax)
            sgn_cln = np.sign(np.einsum('ij,ij->i', tx_c, vx_m[iste:iene+1, k]))
            p.helical[k, iste:iene+1, 3] = cln * sgn_cln
            
            fx = axis_frames[k, iste:iene+1, 2, :] # Base Dz
            qx = np.cross(vx_m[iste:iene+1, k], dax)
            rq = np.linalg.norm(qx, axis=1)
            dot_tip = np.clip(np.einsum('ij,ij->i', qx, fx) / (rq + 1e-12), -1.0, 1.0)
            tip = np.arccos(dot_tip) * cdr
            tx_t = np.cross(qx, fx)
            sgn_tip = np.sign(np.einsum('ij,ij->i', tx_t, dax))
            p.helical[k, iste:iene+1, 4] = tip * sgn_tip

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        if cfg.comb:
            for i in range(self.optimizer.iste, self.optimizer.iene + 1):
                valid_k = np.where(self.ctx.li[i, :] >= -1)[0]
                if len(valid_k) == 0: continue
                dot_v = -1.0 if 0 in valid_k else np.dot(ulx_m[i, 0], ulx_m[i, valid_k[0]])
                sgn_v = math.copysign(1.0, -dot_v)
                v_sum = np.sum(vx_m[i, valid_k] * sgn_v, axis=0)
                vax_all[i] = v_sum / np.linalg.norm(v_sum)

        for k in range(nst):
            ka = 0 if cfg.comb else k
            _, _, iste, iene = self._axis_bounds(k)

            for i in range(iste + 1, iene + 1):
                if not cfg.comb:
                    nx = (ulx_m[i-1, k] + ulx_m[i, k])
                    nx /= np.linalg.norm(nx)
                    qx = (px_m[i-1, k] + px_m[i, k]) / 2.0
                    vx_avg = (vx_m[i-1, k] + vx_m[i, k])
                    self.vkin[i, 6, k] = np.linalg.norm(px_m[i-1, k] - px_m[i, k])
                else:
                    nx = (ulx_m[i-1, 0] + ulx_m[i, 0])
                    nx /= np.linalg.norm(nx)
                    qx = (p.ox[i-1] + p.ox[i]) / 2.0
                    vx_avg = (vax_all[i-1] + vax_all[i])
                    self.vkin[i, 6, 0] = np.linalg.norm(p.ox[i-1] - p.ox[i])
                    self.vkin[i, 6, k] = self.vkin[i, 6, 0]

                dx = vx_avg - nx * np.dot(nx, vx_avg)
                dx /= np.linalg.norm(dx)
                fx = np.cross(nx, dx)

                km = self.ctx.iact[i-1]-1 if (cfg.comb and self.ctx.li[i-1,k] < -1) else k
                kp = self.ctx.iact[i]-1   if (cfg.comb and self.ctx.li[i,k] < -1)   else k
                
                dl = np.dot(nx, qx - px_m[i-1, km]) / np.dot(nx, ulx_m[i-1, ka])
                du = np.dot(nx, px_m[i, kp] - qx) / np.dot(nx, ulx_m[i, ka])
                p.helical[k, i, 2] = dl + du

                # Shift, Slide, Tilt, Roll [cite: 94-96]
                diff_p = (px_m[i, kp] - ulx_m[i, ka]*du) - (px_m[i-1, km] + ulx_m[i-1, ka]*dl)
                self.vkin[i, 0, k] = np.dot(dx, diff_p)
                self.vkin[i, 1, k] = np.dot(fx, diff_p) * self.ctx.idr[k]
                
                tx_k = np.cross(ulx_m[i, ka], dx)
                cln_k = math.acos(np.clip(np.dot(fx, tx_k) / np.linalg.norm(tx_k), -1.0, 1.0)) * cdr
                if np.dot(np.cross(fx, tx_k), dx) < 0: cln_k = -cln_k
                self.vkin[i, 2, k] = 2.0 * cln_k
                
                rx_k = np.cross(dx, tx_k)
                tip_k = math.acos(np.clip(np.dot(ulx_m[i, ka], rx_k) / np.linalg.norm(rx_k), -1.0, 1.0)) * cdr
                if np.dot(np.cross(rx_k, ulx_m[i, ka]), tx_k) < 0: tip_k = -tip_k
                self.vkin[i, 3, k] = 2.0 * tip_k * self.ctx.idr[k]
                self.vkin[i, 4, k] = math.hypot(self.vkin[i, 0, k], self.vkin[i, 1, k])
                self.vkin[i, 5, k] = math.hypot(self.vkin[i, 2, k], self.vkin[i, 3, k])

                # -----------------------------------------------------------
                # -----------------------------------------------------------
                p.helical[k, i, 5] = 0.0
                for l in [i-1, i]:
                    sa = math.sin(rdc * (-cln_k if l == i-1 else cln_k))
                    ca = math.cos(rdc * cln_k)
                    fpx = (dx*dx*(1-ca)+ca)*fx + (dx*fx[1]*(1-ca)-nx[2]*sa)*fx[1]
                    fpx = (dx*np.dot(dx, fx)*(1-ca) + fx*ca + np.cross(dx, fx)*sa)
                    
                    dot_w = np.clip(np.dot(fpx, wx_m[l, k]), -1.0, 1.0)
                    wdg = math.acos(dot_w) * cdr
                    dot_sign = np.dot(np.cross(fpx, wx_m[l, k]), ulx_m[l, ka])
                    if (l == i - 1 and dot_sign > 0.0) or (l == i and dot_sign < 0.0):
                        wdg = -wdg
                    p.helical[k, i, 5] += wdg
                
                p.helical[k, i, 5] = (p.helical[k, i, 5] + 180) % 360 - 180

        if cfg.comb:
            for k in range(1, nst):
                dzl = 0.0
                dwl = 0.0
                for i in range(self.optimizer.iste, self.optimizer.iene + 1):
                    if self.ctx.li[i, k] >= -1 and self.ctx.li[i, 0] >= -1:
                        delta = px_m[i, k] - px_m[i, 0]
                        dzu = np.linalg.norm(delta)
                        if np.dot(delta, ulx_m[i, 0]) < 0.0:
                            dzu = -dzu

                        dot_w = np.clip(-np.dot(wx_m[i, 0], wx_m[i, k]), -1.0, 1.0)
                        dwu = math.acos(dot_w) * cdr
                        if np.dot(np.cross(wx_m[i, 0], wx_m[i, k]), ulx_m[i, 0]) > 0.0:
                            dwu = -dwu

                        if i == self.ctx.ng[k] and i >= self.ctx.ng[0]:
                            p.helical[k, i, 2] = p.helical[0, i, 2] + dzu
                            p.helical[k, i, 5] = p.helical[0, i, 5] + dwu
                        elif i == self.ctx.ng[0] and i >= self.ctx.ng[k]:
                            p.helical[0, i, 2] = p.helical[k, i, 2] - dzu
                            p.helical[0, i, 5] = p.helical[k, i, 5] - dwu

                        dzl = dzu
                        dwl = dwu

        self._global_ulx = ulx_m.copy()
        self._global_vx = vx_m.copy()
        self._global_wx = wx_m.copy()
        self._global_px = px_m.copy()

    def _global_inter_base_values(self, strand: int, level: int):
        """Return Section E global step values for one strand/level pair."""
        if not (self._has_level(strand, level - 1) and self._has_level(strand, level)):
            return None

        p = self.ctx.params
        source_strand = 0 if self.ctx.cfg.comb else strand
        strand_direction = self.ctx.idr[strand]

        shift = p.helical[strand, level, 0] + self.vkin[level, 0, source_strand] - p.helical[strand, level - 1, 0]
        slide = (
            p.helical[strand, level, 1]
            + self.vkin[level, 1, source_strand] * strand_direction
            - p.helical[strand, level - 1, 1]
        )
        rise = p.helical[strand, level, 2]
        tilt = p.helical[strand, level, 3] + self.vkin[level, 2, source_strand] - p.helical[strand, level - 1, 3]
        roll = (
            p.helical[strand, level, 4]
            + self.vkin[level, 3, source_strand] * strand_direction
            - p.helical[strand, level - 1, 4]
        )
        twist = self._global_interbase_twist(strand, level)
        return np.array([shift, slide, rise, tilt, roll, twist], dtype=float)

    def _global_inter_base_pair_values(self, partner_strand: int, level: int):
        """Return Section F global base-pair step values for strand 1 with partner_strand.

        This mirrors the Curves 5.3 text output formulas in outaxe Section F.
        """
        if self._use_curvesplus_axis_convention() and hasattr(self, "curvesplus_inter_base_pair") and partner_strand == 1:
            values = self.curvesplus_inter_base_pair[level - 1]
            if np.all(np.isfinite(values)):
                return values.copy()

        if not (
            self._has_level(0, level - 1)
            and self._has_level(0, level)
            and self._has_level(partner_strand, level - 1)
            and self._has_level(partner_strand, level)
        ):
            return None

        p = self.ctx.params

        xs = (p.helical[0, level, 0] + p.helical[partner_strand, level, 0]) / 2.0
        xm = (p.helical[0, level - 1, 0] + p.helical[partner_strand, level - 1, 0]) / 2.0
        ys = (p.helical[0, level, 1] - p.helical[partner_strand, level, 1]) / 2.0
        ym = (p.helical[0, level - 1, 1] - p.helical[partner_strand, level - 1, 1]) / 2.0
        ts = self._angle_aver(p.helical[0, level, 3], p.helical[partner_strand, level, 3])
        tm = self._angle_aver(p.helical[0, level - 1, 3], p.helical[partner_strand, level - 1, 3])
        ps = self._angle_aver(p.helical[0, level, 4], -p.helical[partner_strand, level, 4])
        pm = self._angle_aver(p.helical[0, level - 1, 4], -p.helical[partner_strand, level - 1, 4])

        if self.ctx.idr[0] < self.ctx.idr[partner_strand]:
            ys = -ys
            ym = -ym
            ps = -ps
            pm = -pm

        shift = xs + self.vkin[level, 0, 0] - xm
        slide = ys + self.vkin[level, 1, 0] - ym
        rise = (p.helical[0, level, 2] + p.helical[partner_strand, level, 2]) / 2.0
        tilt = ts + self.vkin[level, 2, 0] - tm
        roll = self._wrap_180(ps + self.vkin[level, 3, 0] - pm)
        twist = (p.helical[0, level, 5] + p.helical[partner_strand, level, 5]) / 2.0
        return np.array([shift, slide, rise, tilt, roll, twist], dtype=float)

    def _global_base_pair_axis_values(self, partner_strand: int, level: int):
        """Return Section C global base-pair axis values for strand 1 with partner_strand."""
        if self._use_curvesplus_axis_convention() and hasattr(self, "curvesplus_bp_axis") and partner_strand == 1:
            values = self.curvesplus_bp_axis[level]
            if np.all(np.isfinite(values)):
                return values.copy()

        if not (self._has_level(0, level) and self._has_level(partner_strand, level)):
            return None

        p = self.ctx.params
        xdisp = (p.helical[0, level, 0] + p.helical[partner_strand, level, 0]) / 2.0
        ydisp = (p.helical[0, level, 1] - p.helical[partner_strand, level, 1]) / 2.0
        inclin = (p.helical[0, level, 3] + p.helical[partner_strand, level, 3]) / 2.0
        tip = (p.helical[0, level, 4] - p.helical[partner_strand, level, 4]) / 2.0

        if self.ctx.idr[0] < self.ctx.idr[partner_strand]:
            ydisp = -ydisp
            tip = -tip
        return np.array([xdisp, ydisp, inclin, tip], dtype=float)

    def _frame_from_array(self, frame):
        from pycurves_lib.core.parameter_conventions import ParameterFrame
        return ParameterFrame(origin=np.asarray(frame[3], dtype=float), axes=np.asarray(frame[:3], dtype=float))

    def _use_curvesplus_axis_convention(self) -> bool:
        return (
            self.parameter_convention.name == "standard"
            and str(getattr(self.ctx.cfg, "axis_convention", "legacy")).lower() == "curvesplus"
        )

    def _global_base_base_values(self, partner_strand: int, level: int):
        """Return Section D global base-base values for strand 0 with partner_strand."""
        if self._use_curvesplus_axis_convention() or getattr(self.ctx, "axis_reference_uses_continuity", False):
            return self._local_base_base_values(partner_strand, level)

        if not (self._has_level(0, level) and self._has_level(partner_strand, level)):
            return None

        p = self.ctx.params
        stg = p.helical[0, level, 2] - p.helical[partner_strand, level, 2]
        opn = p.helical[0, level, 5] - p.helical[partner_strand, level, 5]

        str_val = p.helical[0, level, 1] + p.helical[partner_strand, level, 1]

        def diff_ang(a, b):
            res = a - b
            if abs(res) > 180.0:
                res -= math.copysign(360.0, res)
            return res

        pro_val = diff_ang(p.helical[0, level, 4], -p.helical[partner_strand, level, 4])

        if self.ctx.idr[0] >= self.ctx.idr[partner_strand]:
            shr_val = p.helical[0, level, 0] - p.helical[partner_strand, level, 0]
            buc_val = diff_ang(p.helical[0, level, 3], p.helical[partner_strand, level, 3])
        else:
            shr_val = p.helical[partner_strand, level, 0] - p.helical[0, level, 0]
            buc_val = diff_ang(p.helical[partner_strand, level, 3], p.helical[0, level, 3])

        return np.array([shr_val, str_val, stg, buc_val, pro_val, opn], dtype=float)

    def _local_base_base_values(self, partner_strand: int, level: int):
        """Return local intra-base-pair values from the two fitted base frames.

        Unlike Section D, these values are computed directly from the paired
        base reference frames and do not use the optimized helical axis.
        """
        return self.parameter_convention.local_base_base_values(self, partner_strand, level)

    def _cache_global_inter_base_parameters(self):
        """Store text-output Section E values for structured JSON export."""
        p = self.ctx.params
        p.inter_base.fill(0.0)
        for strand in range(self.ctx.nst):
            _, _, iste, iene = self._axis_bounds(strand)
            for level in range(iste + 1, iene + 1):
                values = self._global_inter_base_values(strand, level)
                if values is not None:
                    p.inter_base[strand, level] = values

    def _calculate_local_parameters(self):
        """Fortran-compatible implementation."""
        p = self.ctx.params
        nst = self.ctx.nst
        nux = self.ctx.n_levels
        
        self.pal = np.zeros((nux + 2, 6, nst + 1))
        self.pab = np.zeros((nux + 2, 6, nst + 1))
        self.local_inter_base = self.pal
        self.local_inter_base_pair = self.pab

        # ==========================================
        # Local Inter-Base Parameters (Section G) [cite: 86]
        # ==========================================
        for k in range(nst):
            inv_k = self.inv[k]
            idr_1 = self.ctx.idr[0]
            
            if self.ctx.cfg.comb and k > 0:
                lu = self.inv[0] * self.ctx.idr[0] * self.ctx.idr[k]
                lv = -lu
                lw = -1
            else:
                lu, lv, lw = inv_k, inv_k, 1

            _, _, iste, iene = self._axis_bounds(k)
            for i in range(iste + 1, iene + 1):
                if not (self._has_level(k, i - 1) and self._has_level(k, i)):
                    continue
                # p.frames[k, i, axis_idx, :]
                u_p, u_c = p.frames[k, i-1, 2, :], p.frames[k, i, 2, :] # Uz
                o_p, o_c = p.frames[k, i-1, 3, :], p.frames[k, i, 3, :] # Origin
                v_p, v_c = p.frames[k, i-1, 0, :], p.frames[k, i, 0, :] # Ux
                w_p, w_c = p.frames[k, i-1, 1, :], p.frames[k, i, 1, :] # Uy

                nx = lu * (u_p + u_c)
                nx /= np.linalg.norm(nx) # nx
                
                qx = (o_p + o_c) / 2.0 # qx [cite: 90]
                
                base_v = lv * (v_p + v_c)
                dx = base_v - nx * np.dot(nx, base_v)
                dx /= np.linalg.norm(dx) # dx
                fx = np.cross(nx, dx) # fx

                dl = np.dot(nx, qx - o_p) / (lu * np.dot(nx, u_p))
                du = np.dot(nx, o_c - qx) / (lu * np.dot(nx, u_c))
                
                self.pal[i, 2, k] = dl + du # Local Rise 
                
                pl = o_p + lu * u_p * dl
                pu = o_c - lu * u_c * du
                diff = pu - pl
                
                self.pal[i, 0, k] = np.dot(dx, diff) # Local Shift [cite: 92]
                self.pal[i, 1, k] = np.dot(fx, diff) * idr_1 # Local Slide [cite: 92]

                tx = lu * np.cross(u_c, dx)
                rt = np.linalg.norm(tx)
                dot_c = np.clip(np.dot(fx, tx) / rt, -1.0, 1.0)
                cln = np.arccos(dot_c) * self.cdr
                if np.dot(np.cross(fx, tx), dx) < 0: cln = -cln
                self.pal[i, 3, k] = 2.0 * cln # Local Tilt (2*cln) [cite: 93]
                
                rx = np.cross(dx, tx)
                rr = np.linalg.norm(rx)
                dot_t = np.clip(lu * np.dot(u_c, rx) / rr, -1.0, 1.0)
                tip = np.arccos(dot_t) * self.cdr
                if lu * np.dot(np.cross(rx, u_c), tx) < 0: tip = -tip
                self.pal[i, 4, k] = 2.0 * tip * idr_1 # Local Roll [cite: 93]

                self.pal[i, 5, k] = 0.0
                for l_idx, l_val in [ (0, i-1), (1, i) ]:
                    sa = np.sin(self.rdc * ((-1.0 if l_idx == 0 else 1.0) * cln))
                    ca = np.cos(self.rdc * ((-1.0 if l_idx == 0 else 1.0) * cln))
                    
                    fpx = (dx * np.dot(dx, fx) * (1 - ca) + fx * ca + np.cross(dx, fx) * sa)
                    
                    frame_w = p.frames[k, l_val, 1, :]
                    dot_w = np.clip(lw * np.dot(fpx, frame_w), -1.0, 1.0)
                    wdg = np.arccos(dot_w) * self.cdr
                    
                    cross_w = lw * np.cross(fpx, frame_w)
                    dot_s = lu * np.dot(cross_w, p.frames[k, l_val, 2, :])
                    if (l_idx == 0 and dot_s > 0) or (l_idx == 1 and dot_s < 0):
                        wdg = -wdg
                    self.pal[i, 5, k] += wdg
                
                h_twist = self.pal[i, 5, k] % 360.0
                if abs(h_twist) > 180.0:
                    h_twist -= math.copysign(360.0, h_twist)
                self.pal[i, 5, k] = h_twist

        self.parameter_convention.fill_local_strand_steps(self)

        # ==========================================
        # Local Inter-Base Pair Parameters (Section H) [cite: 95-104]
        # ==========================================
        if self.ctx.cfg.comb:
            self._calculate_local_basepair_step()

    def _calculate_local_basepair_step(self):
        """Fortran-compatible implementation."""
        self.parameter_convention.fill_local_base_pair_steps(self)

    def _calculate_overall_bend(self):
        """Fortran-compatible implementation."""
        p = self.ctx.params
        cfg = self.ctx.cfg

        for k in range(self.ctx.nst):
            if cfg.comb and k > 0:
                continue

            if cfg.comb:
                _, _, iste, iene = self._axis_bounds(k)
                axis_points = p.ox
            else:
                _, _, iste, iene = self._axis_bounds(k)
                axis_points = self.optimizer.hho[:, :, k]
            
            p_start = axis_points[iste]
            p_end = axis_points[iene]
            
            e2e_vec = p_end - p_start
            rn = np.linalg.norm(e2e_vec)
            self.end_to_end[k] = rn
            
            u_e2e = e2e_vec / (rn + 1e-12)

            for ic in range(iste + 1, iene):
                p_ic = axis_points[ic]
                u_ic = self._global_ulx[ic, k]
                v_ic = self._global_vx[ic, k]
                
                t_vec = p_ic - p_start
                
                # drd = rx*dx + ry*dy + rz*dz
                # drt = rx*tx + ry*ty + rz*tz
                drd = np.dot(u_ic, u_e2e)
                drt = np.dot(u_ic, t_vec)
                
                c_vec = (u_e2e * drt / (drd + 1e-12)) - t_vec
                rc = np.linalg.norm(c_vec)
                
                dot_bend = np.clip(np.dot(c_vec, v_ic) / (rc + 1e-12), -1.0, 1.0)
                bend = math.acos(dot_bend) * self.cdr
                if np.dot(u_ic, np.cross(v_ic, c_vec)) < 0.0:
                    bend = -bend
                self.bend[ic, k] = bend
                
                dot_proj = np.dot(t_vec, u_e2e)
                offset_vec = t_vec - u_e2e * dot_proj
                self.bend_offset[ic, k] = np.linalg.norm(offset_vec)

            u_start = self._global_ulx[iste, k]
            u_end = self._global_ulx[iene, k]
            dot_start = np.clip(np.dot(u_start, u_end), -1.0, 1.0)
            self.bend[iste, k] = np.arccos(dot_start) * self.cdr
            
            v_end = axis_points[iene] - axis_points[iene-1]
            v_start = axis_points[iste+1] - axis_points[iste]
            
            dot_end = np.clip(np.dot(v_end, v_start) / (np.linalg.norm(v_end)*np.linalg.norm(v_start)), -1.0, 1.0)
            self.bend[iene, k] = np.arccos(dot_end) * self.cdr

            pl = sum(self.vkin[i, 6, k] for i in range(iste + 1, iene + 1))
            self.path_length[k] = pl
            self.shortening[k] = 100.0 * (1.0 - rn / pl) if pl > 0 else 0.0

    def _subunit_start_atom(self, subunit_idx: int):
        if subunit_idx <= 0:
            return None
        boundaries = self.ctx.molecule.subunit_boundaries
        if boundaries is None or subunit_idx > len(boundaries) - 1:
            return None
        return int(boundaries[subunit_idx - 1])

    @staticmethod
    def _base_symbol(res_name: str) -> str:
        name = parent_base_name(res_name)
        if name == "unknown":
            return "X"
        if len(name) >= 2 and name[0] == "D" and name[1] in "GACTUIYP":
            return name[1]
        return name[:1]

    def _residue_label(self, strand: int, level: int):
        if self.ctx.cfg.ends and level in (0, self.ctx.nux + 1):
            return "VIRT", "", 0
        if level < 1 or level > self.ctx.ni_map.shape[1]:
            return None

        subunit_idx = int(self.ctx.ni_map[strand, level - 1])
        atom_idx = self._subunit_start_atom(subunit_idx)
        if atom_idx is None:
            return None

        mol = self.ctx.molecule
        base = self._base_symbol(mol.residue_names[atom_idx])
        chain = ""
        if mol.chain_ids is not None:
            chain = str(mol.chain_ids[atom_idx]).strip()
        return base, chain, int(mol.residue_ids[atom_idx])

    def _duplex_id(self, first_strand: int, other_strand: int, level: int) -> str:
        first = self._residue_label(first_strand, level)
        other = self._residue_label(other_strand, level)
        if first is None or other is None:
            return ""

        base_1, _, res_id_1 = first
        base_2, _, res_id_2 = other
        return f"{base_1}{res_id_1:3d}-{base_2}{res_id_2:3d}"

    def _base_pair_displacement(self, first_strand: int, other_strand: int, level: int):
        p = self.ctx.params
        first = p.frames[first_strand, level]
        other = p.frames[other_strand, level]

        x_axis = first[0] + other[0]
        x_axis /= np.linalg.norm(x_axis)

        y_axis = first[1] - other[1]
        y_axis /= np.linalg.norm(y_axis)

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)

        delta = first[3] - other[3]
        return (
            float(np.dot(x_axis, delta)),
            float(np.dot(y_axis, delta)),
            float(np.dot(z_axis, delta)),
        )

    def _base_pair_opening(self, first_strand: int, other_strand: int, level: int) -> float:
        first = self.ctx.params.frames[first_strand, level, :3, :]
        other = self.ctx.params.frames[other_strand, level, :3, :].copy()

        other[1] *= -1.0
        other[2] *= -1.0

        rotation = Rotation.from_matrix(first @ other.T)
        return float(-rotation.as_euler("zyx", degrees=True)[0])

    @staticmethod
    def _all_finite(values) -> bool:
        return bool(np.all(np.isfinite(np.asarray(values, dtype=float))))

    def _outaxe_local_only(self):
        """Print local/non-axis sections for Curves+ axis mode."""
        ctx = self.ctx
        cfg = ctx.cfg
        nst = ctx.nst

        print("\n  ---------------------------------")
        print("  |G| Local Inter-Base Parameters |")
        print("  ---------------------------------")
        for k in range(nst):
            print(f"\n    {self._ordinal_strand(k)} strand      Shift    Slide     Rise     Tilt     Roll    Twist   Dc")
            print("                    (Dx)     (Dy)      (Dz)     (tau)    (rho)  (Omega)")
            _, _, iste, iene = self._axis_bounds(k)
            for i in range(iste + 1, iene + 1):
                l = self.pal[i, :, k]
                step_id = self._step_label(k, i)
                if not self._all_finite(l[:6]):
                    print(f"  {i:3d})      -")
                    continue
                print(f"  {i:3d}) {step_id:11s} {l[0]:8.2f} {l[1]:8.2f} {l[2]:8.2f} "
                      f"{l[3]:8.2f} {l[4]:8.2f} {l[5]:8.2f} {self.dcod[i, k]:6d}")

        if cfg.comb and nst > 1:
            print("\n  --------------------------------------")
            print("  |H| Local Inter-Base pair Parameters |")
            print("  --------------------------------------")
            for k in range(1, nst):
                print(f"\n  Strand 1 with strand {k+1}:")
                print("\n    Duplex          Shift    Slide     Rise     Tilt     Roll    Twist   Dc")
                print("                    (Dx)     (Dy)      (Dz)     (tau)    (rho)  (Omega)")
                nav = 0
                hela = np.zeros(6)
                for i in range(self.optimizer.iste + 1, self.optimizer.iene + 1):
                    step_id = self._step_label(0, i)
                    if not (self._has_level(0, i - 1) and self._has_level(0, i) and
                            self._has_level(k, i - 1) and self._has_level(k, i)):
                        print(f"  {i:3d})      -")
                        continue
                    lb = self.pab[i, :, k]
                    if not self._all_finite(lb[:6]):
                        print(f"  {i:3d})      -")
                        continue
                    nav += 1
                    hela += lb
                    print(f"  {i:3d}) {step_id:11s} {lb[0]:8.2f} {lb[1]:8.2f} {lb[2]:8.2f} "
                          f"{lb[3]:8.2f} {lb[4]:8.2f} {lb[5]:8.2f} {self.dcod[i, 0]:6d}")

                if nav > 0:
                    avg = hela / nav
                    print(f"\n  Average:               {avg[0]:8.2f} {avg[1]:8.2f} {avg[2]:8.2f} "
                          f"{avg[3]:8.2f} {avg[4]:8.2f} {avg[5]:8.2f}")

        print("\n  -------------------------")
        print("  |J| Backbone Parameters |")
        print("  -------------------------")
        sugt = [
            "C3'-endo", "C4'-exo ", "O1'-endo", "C1'-exo ",
            "C2'-endo", "C3'-exo ", "C4'-endo", "O1'-exo ",
            "C1'-endo", "C2'-exo ",
        ]
        jtran = [12, 8, 9, 6, 7, 10, 11]
        strand_names = ["1st", "2nd", "3rd", "4th"]

        for k in range(nst):
            ist, ien, _, _ = self._axis_bounds(k)
            strand_name = strand_names[k] if k < len(strand_names) else f"{k+1}th"

            print(f"\n  {strand_name} strand   C1'-C2' C2'-C3'  Phase   Ampli   Pucker    C1'   C2'   C3' ")
            for i in range(ist, ien + 1):
                label = self._residue_unit_label(k, i)
                if not self._has_level(k, i, min_status=-2) or label is None:
                    print(f"  {i:3d})----")
                    continue

                unit, unit_no = label
                if self.ctx.backbone.flag[k, i]:
                    print(f"  {i:3d}){unit:<4s}{unit_no:3d}")
                    continue

                tor = self.ctx.backbone.torsions[k, i]
                amp = self.ctx.backbone.sugar_pucker[k, i, 0]
                ph = self.ctx.backbone.sugar_pucker[k, i, 1]
                puck_idx = int((ph % 360.0) / 36.0)
                puck_idx = max(0, min(len(sugt) - 1, puck_idx))
                print(
                    f"  {i:3d}){unit:<4s}{unit_no:3d}  "
                    f"{tor[4]:8.2f}{tor[5]:8.2f}{ph:8.2f}{amp:8.2f}  "
                    f"{sugt[puck_idx]:8s} {tor[0]:6.1f}{tor[1]:6.1f}{tor[2]:6.1f}"
                )

            print("\n  Torsions       Chi    Gamma   Delta   Epsil   Zeta    Alpha   Beta  ")
            print("                C1'-N  C5'-C4' C4'-C3' C3'-O3'  O3'-P   P-O5'  O5'-C5'")
            for i in range(ist, ien + 1):
                label = self._residue_unit_label(k, i)
                if not self._has_level(k, i, min_status=-2) or label is None:
                    print(f"  {i:3d})----")
                    continue

                unit, unit_no = label
                tor = self.ctx.backbone.torsions[k, i]
                fields = "".join(self._torsion_field(tor[j]) for j in jtran)
                print(f"  {i:3d}){unit:<4s}{unit_no:3d}  {fields}")

            print(" ")

        if cfg.grv:
            self.groove()

    def outaxe(self):
        """Print the Curves outaxe report sections."""
        ctx = self.ctx
        p = self.ctx.params
        nst = self.ctx.nst
        nux = self.ctx.n_levels
        cfg = self.ctx.cfg

        BASE_CHARS = ['G', 'A', 'C', 'T', 'I', 'U', 'P', 'Y', 'R']
        NBASE = [1, 2, 3, 4, 1, 4, 4, 2, 3]  # Fortran nbase(9): base class code.
        
        # Fortran dimer(4,4): dinucleotide lookup code used in Dc columns.
        DIMER = np.array([
            [ 1,  2,  5,  6],
            [ 3,  4, -6,  7],
            [ 8,  9, -1, -3],
            [-9, 10, -2, -4]
        ])

        # Fortran trimer(4,4,4): trinucleotide lookup code used in Tc columns.
        TRIMER = np.zeros((4, 4, 4), dtype=int)
        TRIMER[:, 0, :] = [[ 1,  2,  5,  6], [ 3,  4,  7,  8], [ 9, 10, 13, 14], [11, 12, 15, 16]]
        TRIMER[:, 1, :] = [[17, 18, 21, 22], [19, 20, 23, 24], [25, 26, 29, 30], [27, 28, 31, 32]]
        TRIMER[:, 2, :] = [[-13,-15, -5, -7], [-14,-16, -6, -8], [-9,-11, -1, -3], [-10,-12, -2, -4]]
        TRIMER[:, 3, :] = [[-29,-31,-21,-23], [-30,-32,-22,-24], [-25,-27,-17,-19], [-26,-28,-18,-20]]

        self.bcod = np.zeros((nux + 2, nst), dtype=int)
        self.dcod = np.zeros((nux + 2, nst), dtype=int)
        self.tcod = np.zeros((nux + 2, nst), dtype=int)
        self.base_code = self.bcod  # Readable alias for Fortran bcod.
        self.dinucleotide_code = self.dcod  # Readable alias for Fortran dcod.
        self.trinucleotide_code = self.tcod  # Readable alias for Fortran tcod.

        for k in range(nst):
            ist, ien = self.ctx.ng[k], self.ctx.nr[k]
            for i in range(ist, ien + 1):
                label = self._residue_label(k, i)
                if label is None:
                    continue
                char = label[0]
                for idx, b in enumerate(BASE_CHARS):
                    if char.upper() == b:
                        self.bcod[i, k] = NBASE[idx]
                        break
            
            for i in range(ist + 1, ien + 1):
                ib, ii = self.bcod[i-1, k], self.bcod[i, k]
                if ib > 0 and ii > 0:
                    self.dcod[i, k] = DIMER[ib-1, ii-1]
                    if self.ctx.idr[k] < 0: 
                        self.dcod[i, k] = DIMER[ii-1, ib-1]
                
                if i < ien:
                    ia = self.bcod[i+1, k]
                    if ib > 0 and ii > 0 and ia > 0:
                        self.tcod[i, k] = TRIMER[ib-1, ii-1, ia-1]
                        if self.ctx.idr[k] < 0:
                            self.tcod[i, k] = TRIMER[ia-1, ii-1, ib-1]

            if cfg.ends:
                self.bcod[ist-1, k] = 0
                self.bcod[ien+1, k] = 0
                self.tcod[ist-1, k] = 0
                self.tcod[ien+1, k] = 0

        if self._use_curvesplus_axis_convention():
            self._outaxe_local_only()
            return

        # -----------------------------------------------------------
        # Section B: Global Base-Axis Parameters [cite: 13-15]
        # -----------------------------------------------------------
        print("\n  ---------------------------------")
        print("  |B| Global Base-Axis Parameters |")
        print("  ---------------------------------")
        for k in range(nst):
            print(f"\n    {self._ordinal_strand(k)} strand      Xdisp    Ydisp   Inclin     Tip    Bc  Tc")
            print("                    (dx)     (dy)    (eta)    (theta)")
            _, _, iste, iene = self._axis_bounds(k)
            for i in range(iste, iene + 1):
                h = p.helical[k, i]
                bc = self.bcod[i, k]
                tc = self.tcod[i, k]
                label = self._residue_unit_label(k, i)
                if label is None or not self._all_finite([h[0], h[1], h[3], h[4]]):
                    print(f"  {i:3d})      -")
                    continue
                unit, unit_no = label
                print(f"  {i:3d}) {unit:<4s}{unit_no:3d}  {h[0]:8.2f} {h[1]:8.2f} {h[3]:8.2f} {h[4]:8.2f} {bc:3d} {tc:3d}")

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        # --- Section C: Global Base pair-Axis Parameters ---
        if cfg.comb:
            print("\n  --------------------------------------")
            print("  |C| Global Base pair-Axis Parameters |")
            print("  --------------------------------------")
            for k in range(1, self.ctx.nst):
                print(f"\n  Strand 1 with strand {k+1} ...")
                print("\n    Duplex          Xdisp    Ydisp   Inclin     Tip    Bc  Tc")
                print("                    (dx)     (dy)    (eta)    (theta)")
                
                nav = 0
                hela = np.zeros(6)

                for i in range(self.optimizer.iste, self.optimizer.iene + 1):
                    if self._has_level(0, i) and self._has_level(k, i):
                        res_name_1, _, res_num_1 = self._residue_label(0, i)
                        res_name_k, _, res_num_k = self._residue_label(k, i)

                        duplex_id = f"{res_name_1}{res_num_1:3d}-{res_name_k}{res_num_k:3d}"

                        nav += 1

                        xdi = (p.helical[0, i, 0] + p.helical[k, i, 0]) / 2.0
                        ydi = (p.helical[0, i, 1] - p.helical[k, i, 1]) / 2.0
                        cln = (p.helical[0, i, 3] + p.helical[k, i, 3]) / 2.0
                        tip = (p.helical[0, i, 4] + -p.helical[k, i, 4]) / 2.0

                        if self.ctx.idr[0] < self.ctx.idr[k]:
                            ydi, tip = -ydi, -tip
                        if not self._all_finite([xdi, ydi, cln, tip]):
                            print(f"  {i:3d})      -")
                            continue
                        
                        print(f"  {i:3d}) {duplex_id}  {xdi:8.2f} {ydi:8.2f} {cln:8.2f} {tip:8.2f} "
                              f"{self.bcod[i, 0]:7d} {self.tcod[i, 0]:7d}")
                        
                        hela[0] += xdi; hela[1] += ydi; hela[3] += cln; hela[4] += tip
                
                if nav > 0:
                    avg = hela / nav
                    print(f"\n  Average:               {avg[0]:8.2f} {avg[1]:8.2f} {avg[3]:8.2f} {avg[4]:8.2f}")

        # -----------------------------------------------------------
        # Section D: Global Base-Base Parameters [cite: 19-24]
        # -----------------------------------------------------------
        if cfg.comb and nst > 1:
            print("\n  ---------------------------------")
            print("  |D| Global Base-Base Parameters |")
            print("  ---------------------------------")
            
            def diff_ang(a, b):
                """Fortran-compatible implementation."""
                res = a - b
                if abs(res) > 180.0:
                    res -= math.copysign(360.0, res)
                return res

            for k in range(1, nst):
                print(f"\n  Strand 1 with strand {k+1} ...")
                print("\n    Duplex          Shear    Stretch  Stagger  Buckle   Propel  Opening  Bc  Tc")
                print("                    (Sx)      (Sy)     (Sz)    (kappa)  (omega) (sigma)")
                
                nav = 0
                hela = np.zeros(6)
                stg, opn = 0.0, 0.0

                for i in range(self.optimizer.iste, self.optimizer.iene + 1):
                    idx_1 = 0
                    if 1 <= i <= self.ctx.ni_map.shape[1]:
                        idx_1 = self.ctx.ni_map[0, i - 1]
                    
                    idx_k = 0
                    if 1 <= i <= self.ctx.ni_map.shape[1]:
                        idx_k = self.ctx.ni_map[k, i - 1]

                    if idx_1 > 0 and idx_k > 0:
                        res_name_1, _, res_num_1 = self._residue_label(0, i)
                        res_name_k, _, res_num_k = self._residue_label(k, i)
                        duplex_id = f"{res_name_1}{res_num_1:3d}-{res_name_k}{res_num_k:3d}"

                        if i > self.optimizer.iste:
                            if self.ctx.li[i-1, 0] < -1 or self.ctx.li[i-1, k] < -1:
                                stg, opn = 0.0, 0.0
                        
                        stg += p.helical[0, i, 2] - p.helical[k, i, 2]
                        opn += p.helical[0, i, 5] - p.helical[k, i, 5]

                        str_val = p.helical[0, i, 1] + p.helical[k, i, 1] # Stretch
                        pro_val = diff_ang(p.helical[0, i, 4], -p.helical[k, i, 4]) # Propeller

                        if self.ctx.idr[0] >= self.ctx.idr[k]:
                            shr_val = p.helical[0, i, 0] - p.helical[k, i, 0] # Shear
                            buc_val = diff_ang(p.helical[0, i, 3], p.helical[k, i, 3]) # Buckle
                        else:
                            shr_val = p.helical[k, i, 0] - p.helical[0, i, 0]
                            buc_val = diff_ang(p.helical[k, i, 3], p.helical[0, i, 3])

                        if not self._all_finite([shr_val, str_val, stg, buc_val, pro_val, opn]):
                            print(f"  {i:3d})      -")
                            continue

                        nav += 1

                        hela += [shr_val, str_val, stg, buc_val, pro_val, opn]
                        
                        print(f"  {i:3d}) {duplex_id} {shr_val:8.2f} {str_val:8.2f} {stg:8.2f} "
                              f"{buc_val:8.2f} {pro_val:8.2f} {opn:8.2f} {self.bcod[i, 0]:5d} {self.tcod[i, 0]:5d}")
                    else:
                        print(f"  {i:3d})      -")

                if nav > 0:
                    avg = hela / nav
                    print(f"\n  Average:               {avg[0]:8.2f} {avg[1]:8.2f} {avg[2]:8.2f} "
                          f"{avg[3]:8.2f} {avg[4]:8.2f} {avg[5]:8.2f}")

        # --- Section E: Global Inter-Base Parameters [cite: 25-27] ---
        print("\n  ----------------------------------")
        print("  |E| Global Inter-Base Parameters |")
        print("  ----------------------------------")
        for k in range(self.ctx.nst):
            print(f"\n    {self._ordinal_strand(k)} strand      Shift    Slide     Rise     Tilt     Roll    Twist   Dc")
            print("                    (Dx)     (Dy)      (Dz)     (tau)    (rho)  (Omega)")
            _, _, iste, iene = self._axis_bounds(k)
            for i in range(iste + 1, iene + 1):
                vals = self._global_inter_base_values(k, i)
                if vals is not None and self._all_finite(vals):
                    shif, slid, rise, tilt, roll, twis = vals
                    # shif = hel(i,1,k) + vkin(i,1,is) - hel(i-1,1,k)
                    # slid = hel(i,2,k) + vkin(i,2,is)*idr(k) - hel(i-1,2,k)
                    # rise = hel(i,3,k)
                    # tilt = hel(i,4,k) + vkin(i,3,is) - hel(i-1,4,k)
                    # roll = hel(i,5,k) + vkin(i,4,is)*idr(k) - hel(i-1,5,k)
                    # twis = hel(i,6,k)

                    step_id = self._step_label(k, i)
                    print(f"  {i:3d}) {step_id:11s} {shif:8.2f} {slid:8.2f} {rise:8.2f} "
                          f"{tilt:8.2f} {roll:8.2f} {twis:8.2f} {self.dcod[i, k]:3d}")
                else:
                    print(f"  {i:3d})      -")

        # -----------------------------------------------------------
        # Section F: Global Inter-Base pair Parameters
        # -----------------------------------------------------------
        if cfg.comb and nst > 1:
            print("\n  ---------------------------------------")
            print("  |F| Global Inter-Base pair Parameters |")
            print("  ---------------------------------------")
            for k in range(1, nst):
                print(f"\n  Strand 1 with strand {k+1} ...")
                print("\n    Duplex          Shift    Slide     Rise     Tilt     Roll    Twist   Dc")
                print("                    (Dx)     (Dy)      (Dz)     (tau)    (rho)  (Omega)")

                nav = 0
                hela = np.zeros(6)

                for i in range(self.optimizer.iste + 1, self.optimizer.iene + 1):
                    if (self._has_level(0, i - 1) and self._has_level(0, i) and
                            self._has_level(k, i - 1) and self._has_level(k, i)):
                        step_id = self._step_label(0, i)
                        nav += 1

                        xs = (p.helical[0, i, 0] + p.helical[k, i, 0]) / 2.0
                        xm = (p.helical[0, i - 1, 0] + p.helical[k, i - 1, 0]) / 2.0
                        ys = (p.helical[0, i, 1] - p.helical[k, i, 1]) / 2.0
                        ym = (p.helical[0, i - 1, 1] - p.helical[k, i - 1, 1]) / 2.0
                        ts = self._angle_aver(p.helical[0, i, 3], p.helical[k, i, 3])
                        tm = self._angle_aver(p.helical[0, i - 1, 3], p.helical[k, i - 1, 3])
                        ps = self._angle_aver(p.helical[0, i, 4], -p.helical[k, i, 4])
                        pm = self._angle_aver(p.helical[0, i - 1, 4], -p.helical[k, i - 1, 4])

                        if self.ctx.idr[0] < self.ctx.idr[k]:
                            ys = -ys
                            ym = -ym
                            ps = -ps
                            pm = -pm

                        shif = xs + self.vkin[i, 0, 0] - xm
                        slid = ys + self.vkin[i, 1, 0] - ym
                        rise = (p.helical[0, i, 2] + p.helical[k, i, 2]) / 2.0
                        tilt = ts + self.vkin[i, 2, 0] - tm
                        roll = self._wrap_180(ps + self.vkin[i, 3, 0] - pm)
                        twis = (p.helical[0, i, 5] + p.helical[k, i, 5]) / 2.0

                        vals = np.array([shif, slid, rise, tilt, roll, twis])
                        if not self._all_finite(vals):
                            print(f"  {i:3d})      -")
                            continue
                        hela += vals

                        print(f"  {i:3d}) {step_id:11s} {shif:8.2f} {slid:8.2f} {rise:8.2f} "
                              f"{tilt:8.2f} {roll:8.2f} {twis:8.2f} {self.dcod[i, 0]:6d}")
                    else:
                        print(f"  {i:3d})      -")

                if nav > 0:
                    avg = hela / nav
                    print(f"\n  Average:               {avg[0]:8.2f} {avg[1]:8.2f} {avg[2]:8.2f} "
                          f"{avg[3]:8.2f} {avg[4]:8.2f} {avg[5]:8.2f}")

        # -----------------------------------------------------------
        # Section G: Local Inter-Base Parameters [cite: 33-34]
        # -----------------------------------------------------------
        print("\n  ---------------------------------")
        print("  |G| Local Inter-Base Parameters |")
        print("  ---------------------------------")
        for k in range(nst):
            print(f"\n    {self._ordinal_strand(k)} strand      Shift    Slide     Rise     Tilt     Roll    Twist   Dc")
            print("                    (Dx)     (Dy)      (Dz)     (tau)    (rho)  (Omega)")
            _, _, iste, iene = self._axis_bounds(k)
            for i in range(iste + 1, iene + 1):
                l = self.pal[i, :, k]
                step_id = self._step_label(k, i)
                if not self._all_finite(l[:6]):
                    print(f"  {i:3d})      -")
                    continue
                print(f"  {i:3d}) {step_id:11s} {l[0]:8.2f} {l[1]:8.2f} {l[2]:8.2f} "
                      f"{l[3]:8.2f} {l[4]:8.2f} {l[5]:8.2f} {self.dcod[i, k]:6d}")

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        if cfg.comb and nst > 1:
            print("\n  --------------------------------------")
            print("  |H| Local Inter-Base pair Parameters |")
            print("  --------------------------------------")
            for k in range(1, nst):
                print(f"\n  Strand 1 with strand {k+1}:")
                print("\n    Duplex          Shift    Slide     Rise     Tilt     Roll    Twist   Dc")
                print("                    (Dx)     (Dy)      (Dz)     (tau)    (rho)  (Omega)")
                nav = 0
                hela = np.zeros(6)
                for i in range(self.optimizer.iste + 1, self.optimizer.iene + 1):
                    step_id = self._step_label(0, i)
                    if not (self._has_level(0, i - 1) and self._has_level(0, i) and
                            self._has_level(k, i - 1) and self._has_level(k, i)):
                        print(f"  {i:3d})      -")
                        continue
                    lb = self.pab[i, :, k]
                    if not self._all_finite(lb[:6]):
                        print(f"  {i:3d})      -")
                        continue
                    nav += 1
                    hela += lb
                    print(f"  {i:3d}) {step_id:11s} {lb[0]:8.2f} {lb[1]:8.2f} {lb[2]:8.2f} "
                          f"{lb[3]:8.2f} {lb[4]:8.2f} {lb[5]:8.2f} {self.dcod[i, 0]:6d}")

                if nav > 0:
                    avg = hela / nav
                    print(f"\n  Average:               {avg[0]:8.2f} {avg[1]:8.2f} {avg[2]:8.2f} "
                          f"{avg[3]:8.2f} {avg[4]:8.2f} {avg[5]:8.2f}")

        # -----------------------------------------------------------
        # Section I: Global Axis Curvature [cite: 37-40]
        # -----------------------------------------------------------
        print("\n  ---------------------------")
        print("  |I| Global Axis Curvature |")
        print("  ---------------------------")
        section_i_strands = [0] if cfg.comb else list(range(nst))
        for k in section_i_strands:
            if cfg.comb:
                _, _, iste, iene = self._axis_bounds(k)
                print("\n  Duplex          Ax      Ay     Ainc    Atip    Adis    Angle   Path   Dc")
            else:
                _, _, iste, iene = self._axis_bounds(k)
                print(f"\n  Strand {k+1}:       Ax      Ay     Ainc    Atip    Adis    Angle   Path   Dc")

            for i in range(iste + 1, iene + 1):
                v = self.vkin[i, :, k]
                label = self._step_label(k, i) if self._has_level(k, i - 1) and self._has_level(k, i) else "-   /-"
                if not self._all_finite(v[:7]):
                    print(f"  {i:2d})      -")
                    continue
                print(f"  {i:2d}) {label:9s}{v[0]:8.2f}{v[1]:8.2f}{v[2]:8.2f}{v[3]:8.2f}"
                      f"{v[4]:8.2f}{v[5]:8.2f}{v[6]:8.2f} {self.dcod[i, k]:3d}")

        # -----------------------------------------------------------
        # Overall Bending Analysis [cite: 42-45]
        # -----------------------------------------------------------
        print("")
        for k in section_i_strands:
            if cfg.comb:
                _, _, ist, ien = self._axis_bounds(k)
                axis_points = self.ctx.params.ox
            else:
                _, _, ist, ien = self._axis_bounds(k)
                axis_points = self.optimizer.hho[:, :, k]
            pl = sum(self.vkin[i, 6, k] for i in range(ist + 1, ien + 1))
            rn = np.linalg.norm(axis_points[ien] - axis_points[ist])
            shortening = 100.0 * (1.0 - rn / pl) if pl > 0 else 0.0
            
            buu = self.bend[ist, k]
            bpp = self.bend[ien, k]
            print(f"\n  Overall axis bend ... UU= {buu:7.2f} PP= {bpp:7.2f}\n")
            if cfg.comb:
                print("\n  Duplex       Offset   L.Dir  ... wrt end-to-end vector")
            else:
                print(f"\n  Strand {k+1}:   Offset   L.Dir  ... wrt end-to-end vector")
            
            p_start = axis_points[ist]
            e2e_vec = axis_points[ien] - p_start
            e2e_unit = e2e_vec / (np.linalg.norm(e2e_vec) + 1e-12)
            for i in range(ist, ien + 1):
                rel = axis_points[i] - p_start
                dot = np.dot(rel, e2e_unit)
                offset = np.linalg.norm(rel - e2e_unit * dot)
                bend = 0.0 if i == ist or i == ien else self.bend[i, k]
                label = self._residue_label(k, i)
                if label is None:
                    base, res_id = "-", 0
                else:
                    base, _, res_id = label
                print(f"  {i:2d}) {base}{res_id:3d}     {offset:8.2f}{bend:8.2f}")

            print(f"\n    Path length= {pl:8.2f}  End-to-end= {rn:8.2f}  Shortening= {shortening:8.2f} %")

        print("\n  -------------------------")
        print("  |J| Backbone Parameters |")
        print("  -------------------------")
        sugt = [
            "C3'-endo", "C4'-exo ", "O1'-endo", "C1'-exo ",
            "C2'-endo", "C3'-exo ", "C4'-endo", "O1'-exo ",
            "C1'-endo", "C2'-exo ",
        ]
        jtran = [12, 8, 9, 6, 7, 10, 11]
        strand_names = ["1st", "2nd", "3rd", "4th"]

        for k in range(nst):
            if cfg.comb:
                ist, ien, _, _ = self._axis_bounds(k)
            else:
                ist, ien, _, _ = self._axis_bounds(k)

            strand_name = strand_names[k] if k < len(strand_names) else f"{k+1}th"

            print(f"\n  {strand_name} strand   C1'-C2' C2'-C3'  Phase   Ampli   Pucker    C1'   C2'   C3' ")
            for i in range(ist, ien + 1):
                label = self._residue_unit_label(k, i)
                if not self._has_level(k, i, min_status=-2) or label is None:
                    print(f"  {i:3d})----")
                    continue

                unit, unit_no = label
                if self.ctx.backbone.flag[k, i]:
                    print(f"  {i:3d}){unit:<4s}{unit_no:3d}")
                    continue

                tor = self.ctx.backbone.torsions[k, i]
                amp = self.ctx.backbone.sugar_pucker[k, i, 0]
                ph = self.ctx.backbone.sugar_pucker[k, i, 1]
                puck_idx = int((ph % 360.0) / 36.0)
                puck_idx = max(0, min(len(sugt) - 1, puck_idx))
                print(
                    f"  {i:3d}){unit:<4s}{unit_no:3d}  "
                    f"{tor[4]:8.2f}{tor[5]:8.2f}{ph:8.2f}{amp:8.2f}  "
                    f"{sugt[puck_idx]:8s} {tor[0]:6.1f}{tor[1]:6.1f}{tor[2]:6.1f}"
                )

            print("\n  Torsions       Chi    Gamma   Delta   Epsil   Zeta    Alpha   Beta  ")
            print("                C1'-N  C5'-C4' C4'-C3' C3'-O3'  O3'-P   P-O5'  O5'-C5'")
            for i in range(ist, ien + 1):
                label = self._residue_unit_label(k, i)
                if not self._has_level(k, i, min_status=-2) or label is None:
                    print(f"  {i:3d})----")
                    continue

                unit, unit_no = label
                tor = self.ctx.backbone.torsions[k, i]
                fields = "".join(self._torsion_field(tor[j]) for j in jtran)
                print(f"  {i:3d}){unit:<4s}{unit_no:3d}  {fields}")

            print(" ")

        if cfg.grv:
            self.groove()
