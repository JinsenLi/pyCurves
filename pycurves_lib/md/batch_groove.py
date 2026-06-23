"""Batched groove helpers for the experimental Curves+ MD path.

The legacy Curves groove routine is a stateful scanner over two interpolated
backbone splines.  This module keeps the scanner faithful to the translated
Curves routine while moving the axis and backbone interpolation into reusable
NumPy array kernels for the batch-MD path.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

try:
    from numba import njit as _numba_njit
except Exception:  # pragma: no cover - optional accelerator
    _numba_njit = None


GROOVE_ATOMS = ["C1'", "C2'", "C3'", "C4'", "O1'", "O3'", "P", "O5'", "C5'"]
GROOVE_RADII = [1.6, 1.6, 1.6, 1.6, 1.4, 1.4, 2.9, 1.4, 1.6]


def _unit(values: np.ndarray, fallback=None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    out = np.divide(arr, norm, out=np.zeros_like(arr), where=norm > 1e-12)
    if fallback is not None:
        mask = np.squeeze(norm <= 1e-12, axis=-1)
        if np.any(mask):
            out[mask] = np.asarray(fallback, dtype=float)[mask]
    return out


def _norm3(vector: np.ndarray) -> float:
    return math.sqrt(float(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]))


def _unit1(vector: np.ndarray) -> np.ndarray:
    norm = _norm3(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros(3, dtype=float)
    return vector / norm


def _cross3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.empty(3, dtype=float)
    out[0] = left[1] * right[2] - left[2] * right[1]
    out[1] = left[2] * right[0] - left[0] * right[2]
    out[2] = left[0] * right[1] - left[1] * right[0]
    return out


def _cross_left_many(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.empty_like(right, dtype=float)
    out[:, 0] = left[1] * right[:, 2] - left[2] * right[:, 1]
    out[:, 1] = left[2] * right[:, 0] - left[0] * right[:, 2]
    out[:, 2] = left[0] * right[:, 1] - left[1] * right[:, 0]
    return out



def _parameter_or_none(value):
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _helical_value(local_inter_base: np.ndarray, strand: int, scalar_level: int, parameter: int) -> np.ndarray:
    # Scalar Curves p.helical[strand, level] stores the step ending at ``level``.
    # Batch local_inter_base stores that same step at index ``level - 1``.
    index = int(scalar_level) - 1
    if index < 0 or index >= local_inter_base.shape[2]:
        return np.full(local_inter_base.shape[0], np.nan, dtype=float)
    return local_inter_base[:, strand, index, parameter]


def compute_batch_grooves(analyzer, coordinates: np.ndarray, frames: np.ndarray, axis_tables: Dict[str, np.ndarray], local_inter_base: np.ndarray) -> List[List[Dict]]:
    ctx = analyzer.ctx
    if not ctx.cfg.comb or ctx.nst < 2:
        return [[] for _ in range(coordinates.shape[0])]

    paired_levels = [
        level for level in range(1, ctx.n_levels + 1)
        if ctx.ni_map[0, level - 1] > 0 and ctx.ni_map[1, level - 1] > 0
    ]
    if len(paired_levels) < 4:
        return [[] for _ in range(coordinates.shape[0])]

    nbac = int(getattr(ctx.cfg, "nbac", 7))
    nlevel = int(getattr(ctx.cfg, "nlevel", 3))
    if nbac < 1 or nbac > len(GROOVE_ATOMS):
        raise ValueError("nbac too big, cannot exceed 9")
    atom_name = GROOVE_ATOMS[nbac - 1]
    vdw = GROOVE_RADII[nbac - 1]
    nat_fortran = nbac + 2
    if nat_fortran == 11:
        nat_fortran = 13
    nat_index = nat_fortran - 1

    num = paired_levels[0]
    numa = paired_levels[-1]
    # The scalar Curves+ groove path installs the Curves+ smooth axis but does
    # not populate ctx.params.helical with local rise/twist.  Use the same zero
    # work array here so the batch groove table is numerically comparable to the
    # current scalar output.
    groove_helical = np.zeros_like(local_inter_base)
    uxb, cor, dya, ind, nma = _axis_interpolation_batch(
        analyzer,
        frames,
        axis_tables["axis_frames"],
        groove_helical,
        num,
        numa,
        nlevel,
    )
    box, nsu, spline_start = _backbone_interpolation_batch(
        analyzer,
        coordinates,
        groove_helical,
        nat_index,
        num,
        numa,
        uxb,
    )

    rows_by_frame: List[List[Dict]] = []
    total_levels = abs(int(ctx.nu[0])) if hasattr(ctx, "nu") else int(ctx.nux)
    for frame_index in range(coordinates.shape[0]):
        rows = _scan_groove_frame(
            analyzer,
            axis_tables["bp_axis"][frame_index],
            uxb[frame_index],
            cor[frame_index],
            dya[frame_index],
            ind[frame_index],
            nma[frame_index],
            box[frame_index],
            nsu,
            spline_start,
            num,
            numa,
            atom_name,
            total_levels,
            nlevel,
            vdw,
        )
        rows_by_frame.append(rows)
    return rows_by_frame


def _axis_interpolation_batch(analyzer, frames, axis_frames, local_inter_base, num: int, numa: int, nlevel: int):
    ctx = analyzer.ctx
    batch = frames.shape[0]
    nux = ctx.n_levels
    max_sub = max(30, nlevel + 8)
    max_points = (nux + 3) * (max_sub + 24)

    uxb = np.zeros((batch, nux + 2, max_sub, 3), dtype=float)
    cor = np.zeros((batch, max_points, 3), dtype=float)
    dya = np.zeros((batch, max_points, 3), dtype=float)
    ind = np.zeros((batch, nux + 2, max_sub), dtype=int)
    nma = np.ones((batch, nux + 2), dtype=int)

    uxb[:, num:numa + 1, 0, :] = axis_frames[:, num:numa + 1, 2, :]
    for level in range(num + 1, numa + 1):
        flip = np.sum(uxb[:, level - 1, 0, :] * uxb[:, level, 0, :], axis=1) < 0.0
        uxb[flip, level, 0, :] *= -1.0

    kpt = np.ones(batch, dtype=int)
    break_lvl = int(getattr(ctx.cfg, "break_lvl", -1))
    for level in range(num, numa):
        p0 = axis_frames[:, level, 3, :]
        p1 = axis_frames[:, level + 1, 3, :]
        u0 = uxb[:, level, 0, :]
        u1 = uxb[:, level + 1, 0, :]
        delta = p1 - p0
        gl = np.linalg.norm(delta, axis=1)
        finite = (
            np.isfinite(gl)
            & (gl > 1.0e-8)
            & np.all(np.isfinite(p0), axis=1)
            & np.all(np.isfinite(p1), axis=1)
            & np.all(np.isfinite(u0), axis=1)
            & np.all(np.isfinite(u1), axis=1)
        )
        if level + 1 == break_lvl:
            finite[:] = False

        valid_1 = analyzer._has_level(0, level) and analyzer._has_level(0, level + 1)
        valid_2 = analyzer._has_level(1, level) and analyzer._has_level(1, level + 1)
        if valid_1 and valid_2:
            rise = (_helical_value(local_inter_base, 0, level + 1, 2) + _helical_value(local_inter_base, 1, level + 1, 2)) / 2.0
        elif valid_1:
            rise = _helical_value(local_inter_base, 0, level + 1, 2)
        elif valid_2:
            rise = _helical_value(local_inter_base, 1, level + 1, 2)
        else:
            rise = gl.copy()
        rise = np.where(np.isfinite(rise), rise, gl)
        nmi = np.asarray(np.abs(rise) * (nlevel + 1) / 3.4 + 0.5, dtype=int)
        nmi = np.clip(nmi, 1, max_sub - 1)
        nmi = np.where(finite, nmi, 1)
        nma[:, level] = nmi

        invalid = ~finite
        if np.any(invalid):
            bidx = np.where(invalid)[0]
            ind[bidx, level, 0] = kpt[bidx]
            point = np.where(np.all(np.isfinite(p0[bidx]), axis=1)[:, None], p0[bidx], p1[bidx])
            point = np.where(np.all(np.isfinite(point), axis=1)[:, None], point, 0.0)
            cor[bidx, kpt[bidx], :] = point
            kpt[bidx] += 1

        if np.any(finite):
            bidx = np.where(finite)[0]
            glv = gl[bidx]
            g1 = p0[bidx]
            g2 = u0[bidx]
            g3 = 3.0 * (p1[bidx] - p0[bidx]) / glv[:, None] ** 2 - (u1[bidx] + 2.0 * u0[bidx]) / glv[:, None]
            g4 = -2.0 * (p1[bidx] - p0[bidx]) / glv[:, None] ** 3 + (u1[bidx] + u0[bidx]) / glv[:, None] ** 2
            for sub in range(int(np.max(nmi[bidx]))):
                active = bidx[sub < nmi[bidx]]
                if active.size == 0:
                    continue
                local_pos = np.searchsorted(bidx, active)
                s = gl[active] * sub / nmi[active]
                ind[active, level, sub] = kpt[active]
                cor[active, kpt[active], :] = (
                    g1[local_pos]
                    + g2[local_pos] * s[:, None]
                    + g3[local_pos] * s[:, None] ** 2
                    + g4[local_pos] * s[:, None] ** 3
                )
                kpt[active] += 1

    bidx = np.arange(batch)
    cor[bidx, kpt, :] = axis_frames[:, numa, 3, :]
    ind[:, numa, 0] = kpt
    nma[:, numa] = 1

    for level in range(num, numa + 1):
        k0 = ind[:, level, 0]
        rx = frames[:, 0, level, 0, :] + frames[:, 1, level, 0, :]
        axis = uxb[:, level, 0, :]
        vx = rx - axis * np.sum(axis * rx, axis=1, keepdims=True)
        dya[bidx, k0, :] = _unit(vx)

    for level in range(num, numa):
        max_n = int(np.max(nma[:, level]))
        for sub in range(1, max_n):
            active = np.where(sub < nma[:, level])[0]
            if active.size == 0:
                continue
            nmi = nma[active, level]
            uxb[active, level, sub, :] = _unit(
                uxb[active, level, 0, :] * ((nmi - sub) / nmi)[:, None]
                + uxb[active, level + 1, 0, :] * (sub / nmi)[:, None]
            )
            dx = np.zeros((active.size, 2, 3), dtype=float)
            for offset in range(2):
                ik = level + offset
                k0 = ind[active, ik, 0]
                co = np.sum(uxb[active, ik, 0, :] * uxb[active, level, sub, :], axis=1)
                r = np.cross(uxb[active, level, sub, :], uxb[active, ik, 0, :])
                dy = dya[active, k0, :]
                dp = np.sum(r * dy, axis=1)
                denom = 1.0 + co
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
                dx[:, offset, :] = dy * co[:, None] + dp[:, None] * r / denom[:, None] + np.cross(r, dy)
            co = np.clip(np.sum(dx[:, 0, :] * dx[:, 1, :], axis=1), -1.0, 1.0)
            w = _unit(dx[:, 1, :] - dx[:, 0, :] * co[:, None])
            an = np.arccos(co)
            k0 = ind[active, level, sub]
            dya[active, k0, :] = dx[:, 0, :] * np.cos(sub * an / nmi)[:, None] + w * np.sin(sub * an / nmi)[:, None]

    return uxb, cor, dya, ind, nma


def _backbone_interpolation_batch(analyzer, coordinates, local_inter_base, nat_index: int, num: int, numa: int, uxb):
    ctx = analyzer.ctx
    batch = coordinates.shape[0]
    max_back_populated = 20 * (numa - num + 2) + 1
    max_back_accessed = 20 * (numa + 5) + 1
    max_back = max(max_back_populated, max_back_accessed, 901)
    box = np.zeros((batch, max_back, 2, 3), dtype=float)
    nsu = np.zeros(2, dtype=int)
    spline_start = np.ones(2, dtype=int)

    for strand in range(2):
        im = num
        ix = numa
        spline_start[strand] = im
        nsu[strand] = 20 * (ix - im)
        pts: Dict[int, np.ndarray] = {}
        for level in range(im, ix + 1):
            atom_idx = int(analyzer.backbone_atom_map[strand, level, nat_index])
            if atom_idx >= 0:
                pts[level] = coordinates[:, atom_idx, :]
        for level in range(im, ix + 1):
            if level in pts:
                continue
            prev_level = next((j for j in range(level - 1, im - 1, -1) if j in pts), None)
            next_level = next((j for j in range(level + 1, ix + 1) if j in pts), None)
            if prev_level is not None and next_level is not None:
                weight = (level - prev_level) / (next_level - prev_level)
                pts[level] = pts[prev_level] + weight * (pts[next_level] - pts[prev_level])
            elif prev_level is not None:
                pts[level] = pts[prev_level].copy()
            elif next_level is not None:
                pts[level] = pts[next_level].copy()
            else:
                pts[level] = np.zeros((batch, 3), dtype=float)

        tang: Dict[int, np.ndarray] = {}
        for level in range(im + 1, ix):
            a = np.sum((pts[level + 1] - pts[level]) ** 2, axis=1)
            b = np.sum((pts[level - 1] - pts[level]) ** 2, axis=1)
            tang[level] = _unit(a[:, None] * (pts[level] - pts[level - 1]) + b[:, None] * (pts[level + 1] - pts[level]))

        for level in (im, ix):
            lsgn = 1 if level == im else -1
            nua = num + 1 if level == im else numa - 1
            twist1 = _helical_value(local_inter_base, strand, nua, 5)
            twist2 = _helical_value(local_inter_base, strand, nua + 1, 5)
            angle = (twist1 + twist2) / 2.0
            angle = np.where(np.isfinite(angle), angle, 0.0)
            uax = uxb[:, nua, 0, :]
            ax = tang[level + lsgn]
            ap = np.sum(ax * uax, axis=1)
            ca = np.cos(np.radians(angle))
            sa = np.sin(np.radians(angle))
            tang[level] = ap[:, None] * uax + (ax - ap[:, None] * uax) * ca[:, None] + np.cross(ax, uax) * sa[:, None] * lsgn

        for level in range(im, ix):
            p0 = pts[level]
            p1 = pts[level + 1]
            t0 = tang[level]
            t1 = tang[level + 1]
            d = np.linalg.norm(p1 - p0, axis=1)
            f = 3.0 * (p1 - p0) - (t1 + 2.0 * t0) * d[:, None]
            g = -2.0 * (p1 - p0) + (t1 + t0) * d[:, None]
            for sample in range(20):
                idx = 20 * (level - im) + sample
                r = sample / 20.0
                box[:, idx, strand, :] = p0 + t0 * d[:, None] * r + f * r**2 + g * r**3
        box[:, 20 * (ix - im), strand, :] = pts[ix]

    return box, nsu, spline_start


def _window_projections(box, strand: int, start: int, end: int, center, dyad, uvec, tvec):
    indices = np.arange(start, end + 1, dtype=int)
    points = box[indices, strand]
    rel = points - center
    return {
        "start": int(start),
        "indices": indices,
        "points": points,
        "rel": rel,
        "norm": np.sqrt(np.sum(rel * rel, axis=1)),
        "dyad": rel @ dyad,
        "axis": rel @ uvec,
        "transverse": rel @ tvec,
    }



def _window_crossing_matrices(first_rel: np.ndarray, first_normals: np.ndarray, second_rel: np.ndarray, second_normals: np.ndarray):
    """Precompute line-plane crossing tests for one groove scan window.

    The scalar Curves scanner tests each sampled point against many candidate
    points on the opposite strand.  These matrices keep the same stateful
    acceptance logic but move the repeated dot products into two dense NumPy
    kernels:

    * ``first_to_second[i, j]`` is ``dot(second_rel[j], first_normals[i])``.
    * ``second_to_first[j, i]`` is ``dot(first_rel[i], second_normals[j])``.
    """
    return first_normals @ second_rel.T, second_normals @ first_rel.T



def _scan_window_values_kernel(
    first_points,
    first_norm,
    first_dyad,
    first_axis,
    first_transverse,
    first_normals,
    first_to_second,
    first_side,
    first_hp,
    second_points,
    second_norm,
    second_dyad,
    second_axis,
    second_transverse,
    second_to_first,
    second_side,
    second_gq,
    center,
    dyad,
    uvec,
    tvec,
    inin,
    inma,
    jnin,
    jnma,
    nsu0,
    nsu1,
    oa0,
    ob0,
    oa1,
    ob1,
    nmi,
    sub,
    vdw,
):
    """Numeric Curves groove scanner for one level/sublevel window.

    This deliberately mirrors the legacy scanner's state transitions.  It is
    written with simple arrays/scalars so it can be optionally JIT-compiled by
    numba while remaining usable as a pure-Python fallback.
    """
    hpm = np.empty(2, dtype=np.float64)
    gqm = np.empty(2, dtype=np.float64)
    pqn = np.empty(2, dtype=np.float64)
    pqi = np.empty(2, dtype=np.float64)
    dm = np.empty(2, dtype=np.float64)
    dep = np.empty(2, dtype=np.float64)
    hpn = np.empty(2, dtype=np.float64)
    gqn = np.empty(2, dtype=np.float64)
    hpi = np.empty(2, dtype=np.float64)
    gqi = np.empty(2, dtype=np.float64)
    inm = np.zeros(2, dtype=np.int64)
    jnm = np.zeros(2, dtype=np.int64)
    inn = np.zeros(2, dtype=np.int64)
    jnn = np.zeros(2, dtype=np.int64)
    ini = np.zeros(2, dtype=np.int64)
    jni = np.zeros(2, dtype=np.int64)
    ian = np.zeros(2, dtype=np.int64)
    for idx in range(2):
        hpm[idx] = 99.0
        gqm[idx] = 99.0
        pqn[idx] = 99.0
        pqi[idx] = 99.0
        dm[idx] = 99.0
        dep[idx] = 99.0
        hpn[idx] = 99.0
        gqn[idx] = 99.0
        hpi[idx] = 99.0
        gqi[idx] = 99.0

    rad1 = 0.0
    rad2 = 0.0

    pqee, pqe, dee, de = 1.0, 2.0, 1.0, 2.0
    hpee, hpe = 1.0, 2.0
    lee = le = jee = je = 0
    pqex, inex = 99.0, -1
    jnee = jne = jnex = jnt = 0
    ogee = oge = ogex = 99.0
    gqee = gqe = gqex = 99.0
    ohee = ohe = 99.0
    jex = 0
    scaue = float(first_axis[0])
    jm_stop = max(jnin - 1, jnma - 10)
    for in_idx in range(inin, inma + 1, 2):
        in_off = in_idx - inin
        om_in = float(first_norm[in_off])
        oh = float(first_dyad[in_off])
        scu = float(first_axis[in_off])
        scat_in = float(first_transverse[in_off])
        if scu * scaue < 0.0:
            x1 = (float(first_norm[in_off - 2]) * scu - om_in * scaue) / (scu - scaue)
            if x1 > rad1:
                rad1 = x1
        inters = False
        j = 0
        side = int(first_side[in_off])
        sgn1 = side == le and le == lee
        hp = float(first_hp[in_off])
        if sgn1 and hp > hpe and hpe < hpee and hpe < hpm[side - 1]:
            hpm[side - 1] = hpe
            inm[side - 1] = in_idx - 2

        normal_dot_j = first_to_second[in_off]
        s = float(normal_dot_j[0])
        for jm in range(jnin, jm_stop + 1, 10):
            r = s
            s = float(normal_dot_j[jm + 10 - jnin])
            if r * s < 0.0:
                s = r
                jn = jm
                while r * s > 0.0 and jn < jm + 10:
                    jn += 1
                    r = s
                    s = float(normal_dot_j[jn - jnin])
                joff = jn - jnin
                og = float(second_dyad[joff])
                scau_j = float(second_axis[joff])
                scat_j = float(second_transverse[joff])
                if scu * scau_j <= 0.0 and scat_in * scat_j <= 0.0:
                    inters = True
                    jnt = jn
                    j = int(second_side[joff])
                    sgn2 = j == je and je == jee
                    gq = float(second_gq[joff])
                    dx = float(first_points[in_off, 0] - second_points[joff, 0])
                    dy = float(first_points[in_off, 1] - second_points[joff, 1])
                    dz = float(first_points[in_off, 2] - second_points[joff, 2])
                    pq = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if in_idx == inex and pq > pqex:
                        jnt, og, j, gq, pq = jnex, ogex, jex, gqex, pqex
                    inex, jnex, ogex, jex, gqex, pqex = in_idx, jn, og, j, gq, pq
                    break
        if inters:
            d = pq - pqee
            if side == j and sgn1 and sgn2:
                jj = j - 1
                if pq > pqe and pqe < pqee and pqe < pqn[jj]:
                    pqn[jj] = pqe
                    inn[jj] = in_idx - 2
                    jnn[jj] = jne
                    hpn[jj] = hpe
                    gqn[jj] = gqe
                if d * de > 0.0 and d * dee > 0.0 and abs(d) > abs(de) and abs(de) < abs(dee) and abs(de) < dm[jj]:
                    pqi[jj] = pqee
                    ini[jj] = in_idx - 4
                    jni[jj] = jnee
                    hpi[jj] = hpee
                    gqi[jj] = gqee
                    dm[jj] = abs(de)
            pqee, pqe = pqe, pq
            jnee, jne = jne, jnt
            ogee, oge = oge, og
            gqee, gqe = gqe, gq
            dee, de = de, d
        hpee, hpe = hpe, hp
        ohee, ohe = ohe, oh
        lee, le = le, side
        jee, je = je, j
        scaue = scu

    pqee, pqe, dee, de = 1.0, 2.0, 1.0, 2.0
    gqee, gqe = 1.0, 2.0
    lee = le = jee = je = 0
    pqex, jnex = 99.0, -1
    inee = ine = inex = intr = 0
    ohee = ohe = ohex = 99.0
    hpee = hpe = hpex = 99.0
    ogee = oge = 99.0
    lex = 0
    scaue = float(second_axis[0])
    im_stop = max(inin - 1, inma - 10)
    for jn in range(jnin, jnma + 1, 2):
        joff = jn - jnin
        om_j = float(second_norm[joff])
        og = float(second_dyad[joff])
        scu = float(second_axis[joff])
        scat_j = float(second_transverse[joff])
        if scu * scaue < 0.0:
            x2 = (float(second_norm[joff - 2]) * scu - om_j * scaue) / (scu - scaue)
            if x2 > rad2:
                rad2 = x2
        inters = False
        side = 0
        j = int(second_side[joff])
        sgn2 = j == je and je == jee
        gq = float(second_gq[joff])
        if sgn2 and gq > gqe and gqe < gqee and gqe < gqm[j - 1]:
            gqm[j - 1] = gqe
            jnm[j - 1] = jn - 2

        normal_dot_i = second_to_first[joff]
        s = float(normal_dot_i[0])
        for im in range(inin, im_stop + 1, 10):
            r = s
            s = float(normal_dot_i[im + 10 - inin])
            if r * s < 0.0:
                s = r
                in_idx = im
                while r * s > 0.0 and in_idx < im + 10:
                    in_idx += 1
                    r = s
                    s = float(normal_dot_i[in_idx - inin])
                in_off = in_idx - inin
                oh = float(first_dyad[in_off])
                scau_i = float(first_axis[in_off])
                scat_i = float(first_transverse[in_off])
                if scau_i * scu <= 0.0 and scat_i * scat_j <= 0.0:
                    inters = True
                    intr = in_idx
                    side = int(first_side[in_off])
                    sgn1 = side == le and le == lee
                    hp = float(first_hp[in_off])
                    dx = float(first_points[in_off, 0] - second_points[joff, 0])
                    dy = float(first_points[in_off, 1] - second_points[joff, 1])
                    dz = float(first_points[in_off, 2] - second_points[joff, 2])
                    pq = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if jn == jnex and pq > pqex:
                        intr, oh, side, hp, pq = inex, ohex, lex, hpex, pqex
                    inex, jnex, ohex, lex, hpex, pqex = in_idx, jn, oh, side, hp, pq
                    break
        if inters:
            d = pq - pqee
            if side == j and sgn1 and sgn2:
                jj = j - 1
                if pq > pqe and pqe < pqee and pqe < pqn[jj]:
                    pqn[jj] = pqe
                    inn[jj] = ine
                    jnn[jj] = jn - 2
                    hpn[jj] = hpe
                    gqn[jj] = gqe
                if d * de > 0.0 and d * dee > 0.0 and abs(d) > abs(de) and abs(de) < abs(dee) and abs(de) < dm[jj]:
                    pqi[jj] = pqee
                    ini[jj] = inee
                    jni[jj] = jn - 4
                    hpi[jj] = hpee
                    gqi[jj] = gqee
                    dm[jj] = abs(de)
            pqee, pqe = pqe, pq
            inee, ine = ine, intr
            ohee, ohe = ohe, oh
            hpee, hpe = hpe, hp
            dee, de = de, d
        ogee, oge = oge, og
        gqee, gqe = gqe, gq
        lee, le = le, side
        jee, je = je, j
        scaue = scu

    minor_interpolated = 0
    major_interpolated = 0
    width0 = math.nan
    depth0 = math.nan
    angle0 = math.nan
    width1 = math.nan
    depth1 = math.nan
    angle1 = math.nan
    for side_index in range(2):
        pq = pqn[side_index]
        if pq < 99.0 and (inn[side_index] == 0 or inn[side_index] == nsu0 or jnn[side_index] == 0 or jnn[side_index] == nsu1):
            pqn[side_index] = 99.0
        if pqn[side_index] == 99.0:
            hp = hpm[side_index]
            gq = gqm[side_index]
            if hp != 99.0 and (inm[side_index] == 0 or inm[side_index] == nsu0):
                hpm[side_index] = 99.0
            if gq != 99.0 and (jnm[side_index] == 0 or jnm[side_index] == nsu1):
                gqm[side_index] = 99.0
            if hpm[side_index] != 99.0 and gqm[side_index] != 99.0:
                if hpm[side_index] <= gqm[side_index]:
                    gqm[side_index] = 99.0
                else:
                    hpm[side_index] = 99.0

        if side_index == 0:
            oa = oa0 * (nmi - sub) / nmi + oa1 * sub / nmi
        else:
            oa = ob0 * (nmi - sub) / nmi + ob1 * sub / nmi
        if pqn[side_index] + dm[side_index] < 198.0:
            if pqn[side_index] < 99.0:
                im = inn[side_index]
                jm = jnn[side_index]
                r = hpn[side_index] / (hpn[side_index] + gqn[side_index])
            else:
                im = ini[side_index]
                jm = jni[side_index]
                r = hpi[side_index] / (hpi[side_index] + gqi[side_index])
            ioff = im - inin
            joff = jm - jnin
            oh0 = r * second_points[joff, 0] + (1.0 - r) * first_points[ioff, 0] - center[0]
            oh1 = r * second_points[joff, 1] + (1.0 - r) * first_points[ioff, 1] - center[1]
            oh2 = r * second_points[joff, 2] + (1.0 - r) * first_points[ioff, 2] - center[2]
            oh = oh0 * dyad[0] + oh1 * dyad[1] + oh2 * dyad[2]
            if side_index == 0:
                dep[side_index] = oa - oh
            else:
                dep[side_index] = oh - oa

            nx = -first_normals[ioff, 0]
            ny = -first_normals[ioff, 1]
            nz = -first_normals[ioff, 2]
            nn = math.sqrt(nx * nx + ny * ny + nz * nz)
            if math.isfinite(nn) and nn > 1.0e-12:
                nx /= nn
                ny /= nn
                nz /= nn
            else:
                nx = ny = nz = 0.0
            co = tvec[0] * nx + tvec[1] * ny + tvec[2] * nz
            si = uvec[0] * nx + uvec[1] * ny + uvec[2] * nz
            if co < 0.0:
                co = -co
            if co > 1.0:
                an = 0.0
            else:
                cc = co
                if cc < -1.0:
                    cc = -1.0
                elif cc > 1.0:
                    cc = 1.0
                an = math.acos(cc) * 57.29577951308232
            if co * si > 0.0:
                an = -an
            ian[side_index] = int(round(an))

        if pqn[side_index] < 99.0:
            raw_width = pqn[side_index] - 2.0 * vdw
            raw_depth = dep[side_index]
            raw_angle = float(ian[side_index])
        elif pqn[side_index] == 99.0 and dm[side_index] < 99.0:
            raw_width = pqi[side_index] - 2.0 * vdw
            raw_depth = dep[side_index]
            raw_angle = float(ian[side_index])
            if side_index == 0:
                minor_interpolated = 1
            else:
                major_interpolated = 1
        else:
            raw_width = math.nan
            raw_depth = math.nan
            raw_angle = math.nan

        if side_index == 0:
            width0 = raw_width
            depth0 = raw_depth
            angle0 = raw_angle
        else:
            width1 = raw_width
            depth1 = raw_depth
            angle1 = raw_angle

    diameter = math.nan
    if rad1 * rad2 != 0.0:
        diameter = rad1 + rad2
    return width0, depth0, angle0, width1, depth1, angle1, diameter, minor_interpolated, major_interpolated



def _scan_window_from_box_kernel(
    box,
    center,
    dyad,
    uvec,
    tvec,
    inin,
    inma,
    jnin,
    jnma,
    nsu0,
    nsu1,
    oa0,
    ob0,
    oa1,
    ob1,
    nmi,
    sub,
    vdw,
):
    """JIT-friendly groove window setup plus scanner.

    This folds the projection, normal, and crossing-matrix setup into the same
    compiled call as the branch-heavy scanner.  The pure NumPy setup remains the
    fallback path for environments without the optional numba accelerator.
    """
    nfirst = inma - inin + 1
    nsecond = jnma - jnin + 1
    first_points = np.empty((nfirst, 3), dtype=np.float64)
    first_rel = np.empty((nfirst, 3), dtype=np.float64)
    first_norm = np.empty(nfirst, dtype=np.float64)
    first_dyad = np.empty(nfirst, dtype=np.float64)
    first_axis = np.empty(nfirst, dtype=np.float64)
    first_transverse = np.empty(nfirst, dtype=np.float64)
    first_normals = np.empty((nfirst, 3), dtype=np.float64)
    first_side = np.empty(nfirst, dtype=np.int64)
    first_hp = np.empty(nfirst, dtype=np.float64)

    second_points = np.empty((nsecond, 3), dtype=np.float64)
    second_rel = np.empty((nsecond, 3), dtype=np.float64)
    second_norm = np.empty(nsecond, dtype=np.float64)
    second_dyad = np.empty(nsecond, dtype=np.float64)
    second_axis = np.empty(nsecond, dtype=np.float64)
    second_transverse = np.empty(nsecond, dtype=np.float64)
    second_normals = np.empty((nsecond, 3), dtype=np.float64)
    second_side = np.empty(nsecond, dtype=np.int64)
    second_gq = np.empty(nsecond, dtype=np.float64)

    for i in range(nfirst):
        source = inin + i
        px = float(box[source, 0, 0])
        py = float(box[source, 0, 1])
        pz = float(box[source, 0, 2])
        rx = px - center[0]
        ry = py - center[1]
        rz = pz - center[2]
        first_points[i, 0] = px
        first_points[i, 1] = py
        first_points[i, 2] = pz
        first_rel[i, 0] = rx
        first_rel[i, 1] = ry
        first_rel[i, 2] = rz
        norm = math.sqrt(rx * rx + ry * ry + rz * rz)
        dy = rx * dyad[0] + ry * dyad[1] + rz * dyad[2]
        first_norm[i] = norm
        first_dyad[i] = dy
        first_axis[i] = rx * uvec[0] + ry * uvec[1] + rz * uvec[2]
        first_transverse[i] = rx * tvec[0] + ry * tvec[1] + rz * tvec[2]
        first_normals[i, 0] = dyad[1] * rz - dyad[2] * ry
        first_normals[i, 1] = dyad[2] * rx - dyad[0] * rz
        first_normals[i, 2] = dyad[0] * ry - dyad[1] * rx
        first_side[i] = 1 if dy < 0.0 else 2
        hp2 = norm * norm - dy * dy
        first_hp[i] = math.sqrt(hp2 if hp2 > 0.0 else 0.0)

    for j in range(nsecond):
        source = jnin + j
        px = float(box[source, 1, 0])
        py = float(box[source, 1, 1])
        pz = float(box[source, 1, 2])
        rx = px - center[0]
        ry = py - center[1]
        rz = pz - center[2]
        second_points[j, 0] = px
        second_points[j, 1] = py
        second_points[j, 2] = pz
        second_rel[j, 0] = rx
        second_rel[j, 1] = ry
        second_rel[j, 2] = rz
        norm = math.sqrt(rx * rx + ry * ry + rz * rz)
        dy = rx * dyad[0] + ry * dyad[1] + rz * dyad[2]
        second_norm[j] = norm
        second_dyad[j] = dy
        second_axis[j] = rx * uvec[0] + ry * uvec[1] + rz * uvec[2]
        second_transverse[j] = rx * tvec[0] + ry * tvec[1] + rz * tvec[2]
        second_normals[j, 0] = dyad[1] * rz - dyad[2] * ry
        second_normals[j, 1] = dyad[2] * rx - dyad[0] * rz
        second_normals[j, 2] = dyad[0] * ry - dyad[1] * rx
        second_side[j] = 1 if dy < 0.0 else 2
        gq2 = norm * norm - dy * dy
        second_gq[j] = math.sqrt(gq2 if gq2 > 0.0 else 0.0)

    first_to_second = np.empty((nfirst, nsecond), dtype=np.float64)
    for i in range(nfirst):
        nx = first_normals[i, 0]
        ny = first_normals[i, 1]
        nz = first_normals[i, 2]
        for j in range(nsecond):
            first_to_second[i, j] = second_rel[j, 0] * nx + second_rel[j, 1] * ny + second_rel[j, 2] * nz

    second_to_first = np.empty((nsecond, nfirst), dtype=np.float64)
    for j in range(nsecond):
        nx = second_normals[j, 0]
        ny = second_normals[j, 1]
        nz = second_normals[j, 2]
        for i in range(nfirst):
            second_to_first[j, i] = first_rel[i, 0] * nx + first_rel[i, 1] * ny + first_rel[i, 2] * nz

    return _SCAN_WINDOW_VALUES_FAST(
        first_points,
        first_norm,
        first_dyad,
        first_axis,
        first_transverse,
        first_normals,
        first_to_second,
        first_side,
        first_hp,
        second_points,
        second_norm,
        second_dyad,
        second_axis,
        second_transverse,
        second_to_first,
        second_side,
        second_gq,
        center,
        dyad,
        uvec,
        tvec,
        inin,
        inma,
        jnin,
        jnma,
        nsu0,
        nsu1,
        oa0,
        ob0,
        oa1,
        ob1,
        nmi,
        sub,
        vdw,
    )


_SCAN_WINDOW_VALUES_FAST = None
_SCAN_WINDOW_FROM_BOX_FAST = None
if _numba_njit is not None:  # pragma: no cover - depends on optional numba
    _SCAN_WINDOW_VALUES_FAST = _numba_njit(cache=True)(_scan_window_values_kernel)
    _SCAN_WINDOW_FROM_BOX_FAST = _numba_njit(cache=True)(_scan_window_from_box_kernel)


def _scan_window_values(*args):
    global _SCAN_WINDOW_VALUES_FAST
    if _SCAN_WINDOW_VALUES_FAST is None:
        return _scan_window_values_kernel(*args)
    try:
        return _SCAN_WINDOW_VALUES_FAST(*args)
    except Exception:  # pragma: no cover - defensive fallback for optional JIT
        _SCAN_WINDOW_VALUES_FAST = None
        return _scan_window_values_kernel(*args)


def _scan_window_from_box(*args):
    global _SCAN_WINDOW_FROM_BOX_FAST
    if _SCAN_WINDOW_FROM_BOX_FAST is None:
        return None
    try:
        return _SCAN_WINDOW_FROM_BOX_FAST(*args)
    except Exception:  # pragma: no cover - defensive fallback for optional JIT
        _SCAN_WINDOW_FROM_BOX_FAST = None
        return None


def _depth_reference(bp_axis: np.ndarray, level: int) -> Tuple[float, float]:
    lookup = min(max(int(level), 1), bp_axis.shape[0] - 1)
    values = bp_axis[lookup]
    if np.all(np.isfinite(values[[0, 3]])):
        return float(values[0]), float(values[3])
    return 0.0, 0.0


def _scan_groove_frame(
    analyzer,
    bp_axis: np.ndarray,
    uxb: np.ndarray,
    cor: np.ndarray,
    dya: np.ndarray,
    ind: np.ndarray,
    nma: np.ndarray,
    box: np.ndarray,
    nsu: np.ndarray,
    spline_start: np.ndarray,
    num: int,
    numa: int,
    atom_name: str,
    total_levels: int,
    nlevel: int,
    vdw: float,
) -> List[Dict]:
    rows: List[Dict] = []
    ea, eb = -3.0, 4.0
    xdi, tip = _depth_reference(bp_axis, num)
    if math.cos(math.radians(tip)) < 0.0:
        xdi = -xdi
    oa1 = xdi + ea
    ob1 = xdi + eb

    for level in range(num, numa + 1):
        oa0, ob0 = oa1, ob1
        if level + 1 < bp_axis.shape[0]:
            xdi, tip = _depth_reference(bp_axis, level + 1)
            if math.cos(math.radians(tip)) < 0.0:
                xdi = -xdi
        oa1, ob1 = xdi + ea, xdi + eb
        nmi = int(nma[level])

        for sub in range(nmi):
            kpt = int(ind[level, sub])
            center = cor[kpt]
            dyad = dya[kpt]
            uvec = uxb[level, sub]
            tvec = _cross3(uvec, dyad)
            inin = max(0, 20 * (level - int(spline_start[0])) - 110)
            inma = min(int(nsu[0]), 20 * (level - int(spline_start[0])) + 90)
            jnin = max(0, 20 * (level - int(spline_start[1])) - 110)
            jnma = min(int(nsu[1]), 20 * (level - int(spline_start[1])) + 90)
            raw_values = _scan_window_from_box(
                box,
                center,
                dyad,
                uvec,
                tvec,
                inin,
                inma,
                jnin,
                jnma,
                int(nsu[0]),
                int(nsu[1]),
                oa0,
                ob0,
                oa1,
                ob1,
                nmi,
                sub,
                vdw,
            )
            if raw_values is None:
                first_backbone = _window_projections(box, 0, inin, inma, center, dyad, uvec, tvec)
                second_backbone = _window_projections(box, 1, jnin, jnma, center, dyad, uvec, tvec)
                first_points = first_backbone["points"]
                first_rel = first_backbone["rel"]
                first_norm = first_backbone["norm"]
                first_dyad = first_backbone["dyad"]
                first_axis = first_backbone["axis"]
                first_transverse = first_backbone["transverse"]
                second_points = second_backbone["points"]
                second_rel = second_backbone["rel"]
                second_norm = second_backbone["norm"]
                second_dyad = second_backbone["dyad"]
                second_axis = second_backbone["axis"]
                second_transverse = second_backbone["transverse"]
                first_normals = _cross_left_many(dyad, first_rel)
                second_normals = _cross_left_many(dyad, second_rel)
                first_to_second, second_to_first = _window_crossing_matrices(first_rel, first_normals, second_rel, second_normals)
                first_side = np.where(first_dyad < 0.0, 1, 2)
                second_side = np.where(second_dyad < 0.0, 1, 2)
                first_hp = np.sqrt(np.maximum(first_norm * first_norm - first_dyad * first_dyad, 0.0))
                second_gq = np.sqrt(np.maximum(second_norm * second_norm - second_dyad * second_dyad, 0.0))

                raw_values = _scan_window_values(
                    first_points,
                    first_norm,
                    first_dyad,
                    first_axis,
                    first_transverse,
                    first_normals,
                    first_to_second,
                    first_side,
                    first_hp,
                    second_points,
                    second_norm,
                    second_dyad,
                    second_axis,
                    second_transverse,
                    second_to_first,
                    second_side,
                    second_gq,
                    center,
                    dyad,
                    uvec,
                    tvec,
                    inin,
                    inma,
                    jnin,
                    jnma,
                    int(nsu[0]),
                    int(nsu[1]),
                    oa0,
                    ob0,
                    oa1,
                    ob1,
                    nmi,
                    sub,
                    vdw,
                )
            width = [None, None]
            depth = [None, None]
            angle = [None, None]
            for side_index, offset in enumerate((0, 3)):
                raw_width = raw_values[offset]
                raw_depth = raw_values[offset + 1]
                raw_angle = raw_values[offset + 2]
                if np.isfinite(raw_width):
                    width[side_index] = round(float(raw_width), 2)
                if np.isfinite(raw_depth):
                    depth[side_index] = round(float(raw_depth), 2)
                if np.isfinite(raw_angle):
                    angle[side_index] = float(raw_angle)
            raw_diameter = raw_values[6]
            diameter = round(float(raw_diameter), 2) if np.isfinite(raw_diameter) else None
            minor_interpolated = bool(raw_values[7])
            major_interpolated = bool(raw_values[8])

            label = analyzer._residue_labels.get((0, level))
            label1 = analyzer._residue_labels.get((1, level))
            if label and label1:
                bp_name = f"{label[0]}-{label1[0]}"
            elif label:
                bp_name = f"{label[0]}-"
            elif label1:
                bp_name = f"-{label1[0]}"
            else:
                bp_name = ""
            rows.append({
                "atom_defining_backbone": atom_name.strip(),
                "total_levels": total_levels,
                "total_sub_levels": nlevel,
                "level": level,
                "base_pair": bp_name,
                "sub_level": sub,
                "minor_width": _parameter_or_none(width[0]),
                "minor_depth": _parameter_or_none(depth[0]),
                "minor_angle": _parameter_or_none(angle[0]),
                "major_width": _parameter_or_none(width[1]),
                "major_depth": _parameter_or_none(depth[1]),
                "major_angle": _parameter_or_none(angle[1]),
                "diameter": _parameter_or_none(diameter),
                "_minor_interpolated": minor_interpolated,
                "_major_interpolated": major_interpolated,
            })

    _clear_terminal_interpolated(rows, "minor")
    _clear_terminal_interpolated(rows, "major")
    for row in rows:
        row.pop("_minor_interpolated", None)
        row.pop("_major_interpolated", None)
    return rows


def _clear_terminal_interpolated(rows: List[Dict], groove_name: str) -> None:
    width_key = f"{groove_name}_width"
    depth_key = f"{groove_name}_depth"
    angle_key = f"{groove_name}_angle"
    interp_key = f"_{groove_name}_interpolated"

    still_terminal = True
    for row in rows:
        if row.get(width_key) is None:
            continue
        if not row.get(interp_key, False):
            still_terminal = False
        elif still_terminal:
            row[width_key] = None
            row[depth_key] = None
            row[angle_key] = None

    still_terminal = True
    for row in reversed(rows):
        if row.get(width_key) is None:
            continue
        if not row.get(interp_key, False):
            still_terminal = False
        elif still_terminal:
            row[width_key] = None
            row[depth_key] = None
            row[angle_key] = None












