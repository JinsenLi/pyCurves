from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from pycurves_lib.core.curves_analyzer import HelicalOptimizer
from typing import NamedTuple

jax.config.update('jax_enable_x64', True)

class JAXOptState(NamedTuple):
    j_li: jnp.ndarray
    j_frames: jnp.ndarray
    j_efd: jnp.ndarray
    j_efc: jnp.ndarray
    j_helical0: jnp.ndarray
    j_ux0: jnp.ndarray
    j_ox0: jnp.ndarray
    j_bx0: jnp.ndarray
    j_scale: jnp.ndarray
    j_spec_level: jnp.ndarray
    j_spec_strand: jnp.ndarray
    j_spec_col: jnp.ndarray
    j_spec_is_angle: jnp.ndarray
    j_iact: jnp.ndarray

def _jax_rotate_axis_angle(v, axis, ca, sa):
    rx, ry, rz = axis
    xx, yy, zz = v
    tx = (rx * rx + (1.0 - rx * rx) * ca) * xx + (rx * ry * (1.0 - ca) - rz * sa) * yy + (rx * rz * (1.0 - ca) + ry * sa) * zz
    ty = (rx * ry * (1.0 - ca) + rz * sa) * xx + (ry * ry + (1.0 - ry * ry) * ca) * yy + (ry * rz * (1.0 - ca) - rx * sa) * zz
    tz = (rx * rz * (1.0 - ca) - ry * sa) * xx + (ry * rz * (1.0 - ca) + rx * sa) * yy + (rz * rz + (1.0 - rz * rz) * ca) * zz
    return jnp.array([tx, ty, tz], dtype=jnp.float64)

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _compiled_jax_objective_and_grad(z, state: JAXOptState, cdr: float, nst: int, comb: bool, brk: int, bounds: tuple):
    z = jnp.asarray(z, dtype=jnp.float64)
    
    def _jax_pack_helical(z_in):
        x = z_in * state.j_scale
        values = jnp.where(state.j_spec_is_angle, x / cdr, x)
        return state.j_helical0.at[
            state.j_spec_strand,
            state.j_spec_level,
            state.j_spec_col,
        ].set(values)

    def _jax_up_logic_fast(helical):
        if not comb:
            ux_by_strand = []
            ox_by_strand = []
            sx_by_strand = []
            bx_by_strand = []

            for no in range(nst):
                init_carry = (
                    state.j_ux0[0],
                    state.j_ox0[0],
                    state.j_bx0[0, no, :]
                )

                def scan_body_noncomb(carry, loop_vars):
                    ux_prev, ox_prev, bx_prev = carry
                    _i_idx, f_i, li_i, h_i = loop_vars

                    inactive = li_i < 0

                    xdi = h_i[0]
                    ydi = h_i[1]
                    cln = h_i[3]
                    tip = h_i[4]

                    ry = f_i[1, :]
                    rz = f_i[2, :]
                    frame_pos = f_i[3, :]

                    ca = jnp.cos(cdr * (-tip))
                    sa = jnp.sin(cdr * (-tip))
                    t_vec = _jax_rotate_axis_angle(rz, ry, ca, sa)
                    dx_vec = jnp.cross(ry, t_vec)

                    ca_c = jnp.cos(cdr * (-cln))
                    sa_c = jnp.sin(cdr * (-cln))
                    ux_i = _jax_rotate_axis_angle(t_vec, dx_vec, ca_c, sa_c)

                    wx_vec = jnp.cross(ux_i, dx_vec)
                    ox_i = frame_pos - dx_vec * xdi - wx_vec * ydi

                    ux_i = jnp.where(inactive, ux_prev, ux_i)
                    ox_i = jnp.where(inactive, ox_prev + ux_prev * 3.4, ox_i)
                    sx_i = ox_i - ox_prev

                    bx_i = ox_i - frame_pos
                    bx_i = jnp.where(inactive, bx_prev, bx_i)

                    next_carry = (ux_i, ox_i, bx_i)
                    return next_carry, (ux_i, ox_i, sx_i, bx_i)

                scan_length = state.j_li[1:, no].shape[0]
                inputs = (
                    jnp.arange(scan_length),
                    state.j_frames[no, 1:, :, :],
                    state.j_li[1:, no],
                    helical[no, 1:, :],
                )
                _, (ux_stack, ox_stack, sx_stack, bx_stack) = jax.lax.scan(
                    scan_body_noncomb,
                    init_carry,
                    inputs,
                )

                ux_by_strand.append(jnp.concatenate([state.j_ux0[0:1], ux_stack], axis=0))
                ox_by_strand.append(jnp.concatenate([state.j_ox0[0:1], ox_stack], axis=0))
                sx_by_strand.append(jnp.concatenate([jnp.zeros((1, 3)), sx_stack], axis=0))
                bx_by_strand.append(jnp.concatenate([state.j_bx0[0:1, no, :], bx_stack], axis=0))

            ux = jnp.stack(ux_by_strand, axis=1)
            ox = jnp.stack(ox_by_strand, axis=1)
            sx = jnp.stack(sx_by_strand, axis=1)
            bx = jnp.stack(bx_by_strand, axis=1)
            return ux, ox, sx, bx

        init_carry = (
            state.j_ux0[0],
            state.j_ox0[0],
            state.j_bx0[0]
        )

        def scan_body(carry, loop_vars):
            ux_prev, ox_prev, bx_prev_all = carry
            i_idx, f_i, li_i, h_i = loop_vars
            
            ref = 0
            if comb:
                ref = state.j_iact[i_idx + 1] - 1
                ref = jnp.maximum(0, ref)

            inactive_ref = li_i[ref] < 0

            xdi = h_i[ref, 0]
            ydi = h_i[ref, 1]
            cln = h_i[ref, 3]
            tip = h_i[ref, 4]

            ry = f_i[ref, 1, :]
            rz = f_i[ref, 2, :]
            frame_pos_ref = f_i[ref, 3, :]

            ca = jnp.cos(cdr * (-tip))
            sa = jnp.sin(cdr * (-tip))
            t_vec = _jax_rotate_axis_angle(rz, ry, ca, sa)
            dx_vec = jnp.cross(ry, t_vec)

            ca_c = jnp.cos(cdr * (-cln))
            sa_c = jnp.sin(cdr * (-cln))
            ux_i = _jax_rotate_axis_angle(t_vec, dx_vec, ca_c, sa_c)

            wx_vec = jnp.cross(ux_i, dx_vec)
            h_local = frame_pos_ref - dx_vec * xdi - wx_vec * ydi

            if comb and nst > 1:
                frames_pos_all = f_i[:, 3, :]
                diff = frames_pos_all - h_local
                dots = jnp.einsum('ki,i->k', diff, ux_i)
                
                cond = (li_i > 0) | ((li_i[ref] == -1) & (li_i == -1))
                cond = cond.at[ref].set(False)
                
                dot_sum = jnp.sum(jnp.where(cond, dots, 0.0))
                m_count = 1.0 + jnp.sum(jnp.where(cond, 1.0, 0.0))
                ox_i = h_local + ux_i * (dot_sum / m_count)
            else:
                ox_i = h_local

            ux_i = jnp.where(inactive_ref, ux_prev, ux_i)
            ox_i = jnp.where(inactive_ref, ox_prev + ux_prev * 3.4, ox_i)
            
            sx_i = ox_i - ox_prev

            bx_all_curr = ox_i - f_i[:, 3, :]
            is_inactive_all = li_i < 0
            bx_all_curr = jnp.where(is_inactive_all[:, None], bx_prev_all, bx_all_curr)

            next_carry = (ux_i, ox_i, bx_all_curr)
            step_outputs = (ux_i, ox_i, sx_i, bx_all_curr)
            return next_carry, step_outputs

        levels_li = state.j_li[1:, :]
        scan_length = levels_li.shape[0]

        indices = jnp.arange(scan_length)
        l_frames = state.j_frames[:, 1:, :, :].transpose(1, 0, 2, 3) 
        l_helical = helical[:, 1:, :].transpose(1, 0, 2)

        inputs = (indices, l_frames, levels_li, l_helical)
        _, (ux_stack, ox_stack, sx_stack, bx_stack) = jax.lax.scan(
            scan_body, 
            init_carry, 
            inputs
        )

        ux = jnp.concatenate([state.j_ux0[0:1], ux_stack], axis=0)
        ox = jnp.concatenate([state.j_ox0[0:1], ox_stack], axis=0)
        sx = jnp.concatenate([jnp.zeros((1, 3)), sx_stack], axis=0)
        bx = jnp.concatenate([state.j_bx0[0:1], bx_stack], axis=0)

        return ux, ox, sx, bx

    def _objective(z_in):
        helical = _jax_pack_helical(z_in)
        ux, ox, sx, bx = _jax_up_logic_fast(helical)

        scp0_parts = []
        scp1_parts = []
        scp2_parts = []
        scp3_parts = []

        for no in range(nst):
            ist, ien, iste, iene = bounds[no]

            step_idx = jnp.arange(iste + 1, iene + 1)
            step_valid = (
                (step_idx != brk)
                & (state.j_li[step_idx - 1, no] >= 0)
                & (state.j_li[step_idx, no] >= 0)
            ).astype(jnp.float64)

            r1 = state.j_frames[no, step_idx - 1, :3, :]
            r2 = state.j_frames[no, step_idx, :3, :]
            ux_prev = ux[step_idx - 1] if comb else ux[step_idx - 1, no, :]
            ux_curr = ux[step_idx] if comb else ux[step_idx, no, :]
            sx_curr = sx[step_idx] if comb else sx[step_idx, no, :]

            d1_dyn = jnp.einsum("ijc,ic->ij", r1, ux_prev)
            c1_dyn = jnp.einsum("ijc,ic->ij", r1, bx[step_idx - 1, no, :])
            d2_dyn = jnp.einsum("ijc,ic->ij", r2, ux_curr)
            c2_dyn = jnp.einsum("ijc,ic->ij", r2, bx[step_idx, no, :])

            d1 = jnp.where(step_idx[:, None] > ist, d1_dyn, state.j_efd[:, 0, no])
            c1 = jnp.where(step_idx[:, None] > ist, c1_dyn, state.j_efc[:, 0, no])
            d2 = jnp.where(step_idx[:, None] <= ien, d2_dyn, state.j_efd[:, 1, no])
            c2 = jnp.where(step_idx[:, None] <= ien, c2_dyn, state.j_efc[:, 1, no])

            qr = d2 - d1
            qp = c2 - c1
            scp0_parts.append(jnp.sum(step_valid[:, None] * qr * qr))
            scp1_parts.append(jnp.sum(step_valid[:, None] * qp * qp))

            kink_idx = jnp.arange(ist + 1, ien + 1)
            kink_valid = (
                (kink_idx != brk)
                & (state.j_li[kink_idx - 1, no] >= 0)
                & (state.j_li[kink_idx, no] >= 0)
            ).astype(jnp.float64)

            ux_kink = ux[kink_idx] if comb else ux[kink_idx, no, :]
            ux_kink_prev = ux[kink_idx - 1] if comb else ux[kink_idx - 1, no, :]
            sx_kink = sx[kink_idx] if comb else sx[kink_idx, no, :]

            up = ux_kink - ux_kink_prev
            scp2_parts.append(jnp.sum(kink_valid * jnp.einsum("ic,ic->i", up, up)))

            us = ux_kink + ux_kink_prev
            um = us / (jnp.einsum("ic,ic->i", us, us)[:, None] + 1e-12)
            q = sx_kink - um * jnp.einsum("ic,ic->i", us, sx_kink)[:, None]
            scp3_parts.append(jnp.sum(kink_valid * jnp.einsum("ic,ic->i", q, q)))

        scp0 = jnp.sum(jnp.asarray(scp0_parts, dtype=jnp.float64))
        scp1 = jnp.sum(jnp.asarray(scp1_parts, dtype=jnp.float64))
        scp2 = jnp.sum(jnp.asarray(scp2_parts, dtype=jnp.float64))
        scp3 = jnp.sum(jnp.asarray(scp3_parts, dtype=jnp.float64))

        f = 10.0 * (scp0 + scp2) + scp1 + scp3
        return f, (helical, ux, ox, sx, bx, (scp0, scp1, scp2, scp3))

    (f, aux), g = jax.value_and_grad(_objective, has_aux=True)(z)
    return f, g, aux

class HelicalOptimizerJAX(HelicalOptimizer):
    """JAX-backed helical-axis optimizer."""

    def enable_jax_mode(self):
        p = self.ctx.params

        self._j_li = jnp.asarray(self.ctx.li, dtype=jnp.int32)
        self._j_frames = jnp.asarray(p.frames, dtype=jnp.float64)
        self._j_efd = jnp.asarray(getattr(p, 'efd'), dtype=jnp.float64)
        self._j_efc = jnp.asarray(getattr(p, 'efc'), dtype=jnp.float64)
        self._j_helical0 = jnp.asarray(p.helical, dtype=jnp.float64)
        self._j_ux0 = jnp.asarray(p.ux, dtype=jnp.float64)
        self._j_ox0 = jnp.asarray(p.ox, dtype=jnp.float64)
        self._j_sx0 = jnp.asarray(p.sx, dtype=jnp.float64)
        self._j_bx0 = jnp.asarray(p.bx, dtype=jnp.float64)
        self._j_scale = jnp.asarray(self._min_scale, dtype=jnp.float64)
        spec = np.asarray(self._min_spec, dtype=np.int32)
        self._j_spec_level = jnp.asarray(spec[:, 0], dtype=jnp.int32)
        self._j_spec_strand = jnp.asarray(spec[:, 1], dtype=jnp.int32)
        self._j_spec_col = jnp.asarray(spec[:, 2], dtype=jnp.int32)
        self._j_spec_is_angle = jnp.asarray(spec[:, 3].astype(bool))
        self._j_iact = jnp.asarray(self.iact, dtype = jnp.int32)
        
        self._jax_opt_state = JAXOptState(
            j_li=self._j_li,
            j_frames=self._j_frames,
            j_efd=self._j_efd,
            j_efc=self._j_efc,
            j_helical0=self._j_helical0,
            j_ux0=self._j_ux0,
            j_ox0=self._j_ox0,
            j_bx0=self._j_bx0,
            j_scale=self._j_scale,
            j_spec_level=self._j_spec_level,
            j_spec_strand=self._j_spec_strand,
            j_spec_col=self._j_spec_col,
            j_spec_is_angle=self._j_spec_is_angle,
            j_iact=self._j_iact
        )
        self._jax_bounds = tuple(
            (int(self.ist_by_strand[no]), int(self.ien_by_strand[no]), int(self.iste_by_strand[no]), int(self.iene_by_strand[no]))
            for no in range(self.nst)
        )

        self._jax_ready = True
        self._last_eval_z = None
        self._last_eval_result = None

    def minimise(self):
        if self.ctx.cfg.dinu or self.ctx.cfg.line:
            raise NotImplementedError('JAX optimizer currently supports cfg.dinu=False and cfg.line=False only.')
            
        self._build_min_spec()
        self.enable_jax_mode()

        z0 = self._pack_min_vars()

        f0, _ = self._evaluate(z0.copy(), log=False)
        print(f"\n  FIRST SUM={f0:10.3f}    CPTS: {self.scp[0]*10:.3f} {self.scp[1]:.3f} {self.scp[2]*10:.3f} {self.scp[3]:.3f}")
        self.prev_sum = float(f0)
        self.step_count = 0
        maxiter = int(self.ctx.cfg.maxn)
        if not self.ctx.cfg.comb:
            # Curves 5.3 calls the optimizer once per strand in comb=.f. mode.
            # The JAX path solves the same independent blocks together, so give
            # the aggregate solve the equivalent total iteration budget.
            maxiter *= int(self.ctx.nst)
        print(f"\n  MINIMISATION: ACC = {self.ctx.cfg.acc:10.3E} MAXN= {maxiter:4d} NVAR= {len(z0):4d}")

        def f(z):
            return self._evaluate(z, log=False)[0]

        def g(z):
            return self._evaluate(z, log=False)[1]

        from scipy.optimize import fmin_bfgs

        zopt = fmin_bfgs(
            f=f,
            x0=z0,
            fprime=g,
            gtol=self.ctx.cfg.acc,
            maxiter=maxiter,
            callback=lambda zk: self._evaluate(zk, log=True),
            disp=False,
            xrtol=0.0,
        )
        return zopt

    def _evaluate(self, z, log=True):
        if not getattr(self, '_jax_ready', False):
            raise RuntimeError("JAX optimizer state has not been initialized.")
        z_np = np.asarray(z, dtype=float)
        if (
            self._last_eval_z is not None
            and z_np.shape == self._last_eval_z.shape
            and np.array_equal(z_np, self._last_eval_z)
        ):
            f, g, aux = self._last_eval_result
        else:
            f, g, aux = _compiled_jax_objective_and_grad(
                z_np,
                self._jax_opt_state,
                float(self.cdr),
                int(self.nst),
                bool(self.ctx.cfg.comb),
                int(getattr(self.ctx.cfg, 'break_lvl', 0)),
                self._jax_bounds
            )
            self._last_eval_z = z_np.copy()
            self._last_eval_result = (f, g, aux)
            
        f = float(f)
        g = np.asarray(g, dtype=float)
        self.gra = np.asarray(g) 
        helical, ux, ox, sx, bx, scp = aux
        self._jax_writeback_state(helical, ux, ox, sx, bx, scp)
        if log:
            self.step_count += 1
            delta = f - self.prev_sum if self.step_count > 1 else 0.0
            print(f"  STEP {self.step_count:4d} SUM={f:10.3f} DEL={delta:12.3E}")
            self.prev_sum = f
        return f, g

    def _jax_writeback_state(self, helical, ux, ox, sx, bx, scp):
        p = self.ctx.params
        p.helical = np.asarray(helical).copy()
        ux_np = np.asarray(ux).copy()
        ox_np = np.asarray(ox).copy()
        sx_np = np.asarray(sx).copy()
        if ux_np.ndim == 3:
            self._axis_ux_by_strand = ux_np
            self._axis_ox_by_strand = ox_np
            self._axis_sx_by_strand = sx_np
            p.ux = ux_np[:, 0, :].copy()
            p.ox = ox_np[:, 0, :].copy()
            p.sx = sx_np[:, 0, :].copy()
        else:
            self._axis_ux_by_strand = None
            self._axis_ox_by_strand = None
            self._axis_sx_by_strand = None
            p.ux = ux_np
            p.ox = ox_np
            p.sx = sx_np
        p.bx = np.asarray(bx).copy()
        self.scp = np.array(scp)
