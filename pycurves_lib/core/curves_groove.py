"""Groove analysis helpers for pyCurves.

This mixin contains the translated Curves groove scanner and the small
Curves+ smooth-axis adapter hooks used by the main calculator.
"""

import math

import numpy as np


class GrooveAnalysisMixin:
    @staticmethod
    def _cross3(left, right):
        """Fast cross product for the scalar 3-vectors used in groove scans."""
        return np.array([
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ], dtype=float)

    @staticmethod
    def _groove_window_projections(box, strand: int, start: int, end: int, center, dyad, uvec, tvec):
        """Precompute local groove-frame coordinates for one backbone window."""
        indices = np.arange(start, end + 1, dtype=int)
        rel = box[indices, strand] - center
        return {
            "start": int(start),
            "indices": indices,
            "rel": rel,
            "norm": np.linalg.norm(rel, axis=1),
            "dyad": rel @ dyad,
            "axis": rel @ uvec,
            "transverse": rel @ tvec,
        }

    @staticmethod
    def _groove_proj_at(projections, point: int):
        offset = int(point) - projections["start"]
        return (
            projections["rel"][offset],
            float(projections["norm"][offset]),
            float(projections["dyad"][offset]),
            float(projections["axis"][offset]),
            float(projections["transverse"][offset]),
        )

    def _global_interbase_twist(self, strand: int, level: int) -> float:
        p = self.ctx.params
        return float(p.helical[strand, level, 5])

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Fortran diff(a,b): signed angular difference in degrees."""
        ar = math.radians(a)
        br = math.radians(b)
        dot = math.cos(ar) * math.cos(br) + math.sin(ar) * math.sin(br)
        dot = max(-1.0, min(1.0, dot))
        val = math.degrees(math.acos(dot))
        if math.cos(br) * math.sin(ar) - math.sin(br) * math.cos(ar) < 0.0:
            val = -val
        return val

    @classmethod
    def _angle_aver(cls, a: float, b: float) -> float:
        """Fortran aver(a,b): midpoint angle following Curves wrap rules."""
        return b + cls._angle_diff(a, b) / 2.0

    @staticmethod
    def _wrap_180(value: float) -> float:
        if abs(value) > 180.0:
            value -= math.copysign(360.0, value)
        return value

    def _has_level(self, strand: int, level: int, min_status: int = -1) -> bool:
        if strand < 0 or strand >= self.ctx.nst:
            return False
        if level < 0 or level >= self.ctx.li.shape[0]:
            return False
        if 1 <= level <= self.ctx.ni_map.shape[1]:
            if int(self.ctx.ni_map[strand, level - 1]) <= 0:
                return False
        return self.ctx.li[level, strand] >= min_status

    def _residue_unit_label(self, strand: int, level: int):
        """Return the Curves munit/nunit-style residue label for output rows."""
        if self.ctx.cfg.ends and level in (0, self.ctx.nux + 1):
            return "VIRT", 0
        if level < 1 or level > self.ctx.ni_map.shape[1]:
            return None

        subunit_idx = int(self.ctx.ni_map[strand, level - 1])
        atom_idx = self._subunit_start_atom(subunit_idx)
        if atom_idx is None:
            return None

        mol = self.ctx.molecule
        unit = str(mol.residue_names[atom_idx]).strip()
        chain = ""
        if mol.chain_ids is not None:
            chain = str(mol.chain_ids[atom_idx]).strip()
        return f"{unit}{chain}".strip(), int(mol.residue_ids[atom_idx])

    def _step_label(self, strand: int, level: int) -> str:
        prev_label = self._residue_label(strand, level - 1)
        curr_label = self._residue_label(strand, level)
        if prev_label is None or curr_label is None:
            return "-"

        prev_base, _, prev_id = prev_label
        curr_base, _, curr_id = curr_label
        return f"{prev_base}{prev_id:3d}/{curr_base}{curr_id:3d}"

    @staticmethod
    def _torsion_field(value: float) -> str:
        if np.isfinite(value) and value < 990.0:
            return f"{value:8.2f}"
        return "  ......"

    @staticmethod
    def _ordinal_strand(strand: int) -> str:
        names = ["1st", "2nd", "3rd", "4th"]
        if strand < len(names):
            return names[strand]
        return f"{strand + 1}th"

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        if n == 0.0:
            return v * 0.0
        return v / n

    @staticmethod
    def _clear_groove_measure(row: dict, groove_name: str):
        """Clear a groove measurement and its optional visualization geometry."""
        row[f"{groove_name}_width"] = None
        row[f"{groove_name}_depth"] = None
        row[f"{groove_name}_angle"] = None
        geometry = row.get("geometry")
        if isinstance(geometry, dict):
            geometry[groove_name] = None

    def _groove_axeint(self, num: int, numa: int, nlevel: int):
        p = self.ctx.params
        nux = self.ctx.n_levels
        max_sub = max(30, nlevel + 8)
        max_points = (nux + 3) * (max_sub + 24)

        uxb = np.zeros((nux + 2, max_sub, 3), dtype=float)
        cor = np.zeros((max_points, 3), dtype=float)
        dya = np.zeros((max_points, 3), dtype=float)
        ind = np.zeros((nux + 2, max_sub), dtype=int)
        nma = np.ones(nux + 2, dtype=int)

        for i in range(num, numa + 1):
            uxb[i, 0] = self.optimizer.uho[i, :, 0]
        for i in range(num + 1, numa + 1):
            if np.dot(uxb[i - 1, 0], uxb[i, 0]) < 0.0:
                uxb[i, 0] *= -1.0

        kpt = 1
        break_lvl = getattr(self.ctx.cfg, "break_lvl", -1)
        for i in range(num, numa):
            if i + 1 == break_lvl:
                nma[i] = 1
                ind[i, 0] = kpt
                cor[kpt] = self.optimizer.hho[i, :, 0]
                kpt += 1
                continue

            p0 = self.optimizer.hho[i, :, 0]
            p1 = self.optimizer.hho[i + 1, :, 0]
            u0 = uxb[i, 0]
            u1 = uxb[i + 1, 0]
            gl = np.linalg.norm(p1 - p0)
            g1 = p0
            g2 = u0
            g3 = 3.0 * (p1 - p0) / gl**2 - (u1 + 2.0 * u0) / gl
            g4 = -2.0 * (p1 - p0) / gl**3 + (u1 + u0) / gl**2
            valid_1 = self._has_level(0, i) and self._has_level(0, i + 1)
            valid_2 = self._has_level(1, i) and self._has_level(1, i + 1)
            
            if valid_1 and valid_2:
                rise = (p.helical[0, i + 1, 2] + p.helical[1, i + 1, 2]) / 2.0
            elif valid_1:
                rise = p.helical[0, i + 1, 2]
            elif valid_2:
                rise = p.helical[1, i + 1, 2]
            else:
                rise = gl
                
            nmi = int(abs(rise) * (nlevel + 1) / 3.4 + 0.5)
            nmi = max(1, min(max_sub - 1, nmi))
            nma[i] = nmi
            for n in range(nmi):
                s = gl * n / nmi
                ind[i, n] = kpt
                cor[kpt] = g1 + g2 * s + g3 * s**2 + g4 * s**3
                kpt += 1

        cor[kpt] = self.optimizer.hho[numa, :, 0]
        ind[numa, 0] = kpt
        nma[numa] = 1

        for i in range(num, numa + 1):
            k0 = ind[i, 0]
            rx = p.frames[0, i, 0, :] + p.frames[1, i, 0, :]
            vx = rx - uxb[i, 0] * np.dot(uxb[i, 0], rx)
            dya[k0] = self._unit(vx)

        for i in range(num, numa):
            nmi = nma[i]
            if nmi <= 1:
                continue
            for n in range(1, nmi):
                uxb[i, n] = self._unit(uxb[i, 0] * (nmi - n) / nmi + uxb[i + 1, 0] * n / nmi)
                dx = np.zeros((2, 3), dtype=float)
                for l in range(2):
                    ik = i + l
                    k0 = ind[ik, 0]
                    co = np.dot(uxb[ik, 0], uxb[i, n])
                    r = np.cross(uxb[i, n], uxb[ik, 0])
                    dp = np.dot(r, dya[k0])
                    dx[l] = dya[k0] * co + dp * r / (1.0 + co) + np.cross(r, dya[k0])
                co = np.clip(np.dot(dx[0], dx[1]), -1.0, 1.0)
                w = self._unit(dx[1] - dx[0] * co)
                an = math.acos(co)
                k0 = ind[i, n]
                dya[k0] = dx[0] * math.cos(n * an / nmi) + w * math.sin(n * an / nmi)

        return uxb, cor, dya, ind, nma

    def _groove_bacint(self, nat_index: int, num: int, numa: int, uxb, ind):
        p = self.ctx.params
        coords = self.ctx.molecule.coordinates
        max_back_populated = 20 * (numa - num + 2) + 1
        max_back_accessed = 20 * (numa + 5) + 1
        max_back = max(max_back_populated, max_back_accessed, 901)
        box = np.zeros((max_back, 2, 3), dtype=float)
        nsu = np.zeros(2, dtype=int)
        spline_start = np.ones(2, dtype=int)

        for k in range(2):
            im = num
            ix = numa
            spline_start[k] = im
            nsu[k] = 20 * (ix - im)
            pts = {}
            for i in range(im, ix + 1):
                atom_idx = self.ctx.backbone.atom_map[k, i, nat_index]
                if atom_idx >= 0:
                    pts[i] = coords[atom_idx]

            missing = [i for i in range(im, ix + 1) if i not in pts]
            for i in missing:
                prev_i = next((j for j in range(i - 1, im - 1, -1) if j in pts), None)
                next_i = next((j for j in range(i + 1, ix + 1) if j in pts), None)
                if prev_i is not None and next_i is not None:
                    weight = (i - prev_i) / (next_i - prev_i)
                    pts[i] = pts[prev_i] + weight * (pts[next_i] - pts[prev_i])
                elif prev_i is not None:
                    pts[i] = pts[prev_i]
                elif next_i is not None:
                    pts[i] = pts[next_i]
                else:
                    pts[i] = np.zeros(3)

            tang = {}
            for i in range(im + 1, ix):
                a = np.sum((pts[i + 1] - pts[i]) ** 2)
                b = np.sum((pts[i - 1] - pts[i]) ** 2)
                tang[i] = self._unit(a * (pts[i] - pts[i - 1]) + b * (pts[i + 1] - pts[i]))

            for i in (im, ix):
                lsgn = 1 if i == im else -1
                nua = num + 1 if i == im else numa - 1
                angle = (p.helical[k, nua, 5] + p.helical[k, nua + 1, 5]) / 2.0
                uax = uxb[nua, 0]
                ax = tang[i + lsgn]
                ap = np.dot(ax, uax)
                ca = math.cos(math.radians(angle))
                sa = math.sin(math.radians(angle))
                tang[i] = ap * uax + (ax - ap * uax) * ca + np.cross(ax, uax) * sa * lsgn

            for i in range(im, ix):
                p0 = pts[i]
                p1 = pts[i + 1]
                t0 = tang[i]
                t1 = tang[i + 1]
                d = np.linalg.norm(p1 - p0)
                f = 3.0 * (p1 - p0) - (t1 + 2.0 * t0) * d
                g = -2.0 * (p1 - p0) + (t1 + t0) * d
                for m in range(20):
                    idx = 20 * (i - im) + m
                    r = m / 20.0
                    box[idx, k] = p0 + t0 * d * r + f * r**2 + g * r**3
            box[20 * (ix - im), k] = pts[ix]

        return box, nsu, spline_start

    def _groove_depth_reference(self, level: int):
        if self._use_curvesplus_axis_convention() and hasattr(self, "curvesplus_bp_axis"):
            lookup_level = min(max(int(level), 1), self.curvesplus_bp_axis.shape[0] - 1)
            values = self.curvesplus_bp_axis[lookup_level]
            if np.all(np.isfinite(values[[0, 3]])):
                return float(values[0]), float(values[3])
        xdi = (self.ctx.params.helical[0, level, 0] + self.ctx.params.helical[1, level, 0]) / 2.0
        tip = self._wrap_180(self._angle_aver(self.ctx.params.helical[0, level, 4], -self.ctx.params.helical[1, level, 4]))
        return float(xdi), float(tip)

    def groove(self):
        if not self.ctx.cfg.comb or self.ctx.nst < 2:
            print("\n  -----------------------")
            print("  |K| Groove parameters |")
            print("  -----------------------")
            print("\n  Groove analysis requires combined two-strand input.")
            self.groove_params = {}
            return

        paired_levels = [
            level for level in range(1, self.ctx.n_levels + 1)
            if self.ctx.ni_map[0, level - 1] > 0 and self.ctx.ni_map[1, level - 1] > 0
        ]
        if len(paired_levels) < 4:
            print("\n  -----------------------")
            print("  |K| Groove parameters |")
            print("  -----------------------")
            print("\n  Groove analysis requires at least four paired levels.")
            self.groove_params = {}
            return

        atoms = ["C1'", "C2'", "C3'", "C4'", "O1'", "O3'", "P", "O5'", "C5'"]
        radius = [1.6, 1.6, 1.6, 1.6, 1.4, 1.4, 2.9, 1.4, 1.6]
        nbac = int(getattr(self.ctx.cfg, "nbac", 7))
        nlevel = int(getattr(self.ctx.cfg, "nlevel", 3))
        if nbac < 1 or nbac > len(atoms):
            raise ValueError("nbac too big, cannot exceed 9")
        bato = atoms[nbac - 1]
        vdw = radius[nbac - 1]
        nat_fortran = nbac + 2
        if nat_fortran == 11:
            nat_fortran = 13
        nat_index = nat_fortran - 1

        num = None
        numa = None
        for i in range(1, self.ctx.n_levels + 1):
            present = int(self.ctx.ni_map[0, i - 1] * self.ctx.ni_map[1, i - 1] != 0)
            if present:
                if num is None:
                    num = i
                numa = i
        uxb, cor, dya, ind, nma = self._groove_axeint(num, numa, nlevel)
        box, nsu, spline_start = self._groove_bacint(nat_index, num, numa, uxb, ind)
        self.groove_backbone_splines = []
        for strand in range(2):
            spline_points = []
            for sample_index in range(int(nsu[strand]) + 1):
                coords = box[sample_index, strand]
                spline_points.append({
                    "sample_index": sample_index,
                    "level_position": float(spline_start[strand] + sample_index / 20.0),
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "z": float(coords[2]),
                })
            self.groove_backbone_splines.append({
                "strand": strand + 1,
                "atom_defining_backbone": bato.strip(),
                "spline_start_level": int(spline_start[strand]),
                "spline_points": spline_points,
            })

        print("\n  -----------------------")
        print("  |K| Groove parameters |")
        print("  -----------------------")
        print(f"\n  Atom defining backbone: {bato:<4s} {abs(int(self.ctx.nu[0])):3d} levels, {nlevel:2d} sub-levels")
        print("\n   Levels          Minor groove                Major groove")
        print("     i  n      Width    Depth  Angle       Width    Depth  Angle    Diam\n")

        lines = []
        self.groove_params = {
            "atom_defining_backbone": bato.strip(),
            "levels": abs(int(self.ctx.nu[0])),
            "sub_levels": nlevel,
            "data": {}
        }
        ea, eb = -3.0, 4.0
        xdi, tip = self._groove_depth_reference(num)
        if math.cos(math.radians(tip)) < 0.0:
            xdi = -xdi
        oa1 = xdi + ea
        ob1 = xdi + eb

        def rel(point, strand, center):
            return box[point, strand] - center

        def f6(value):
            return f"{value:6.2f}"

        for i in range(num, numa + 1):
            oa0, ob0 = oa1, ob1
            if i + 1 < self.ctx.params.helical.shape[1]:
                xdi, tip = self._groove_depth_reference(i + 1)
                if math.cos(math.radians(tip)) < 0.0:
                    xdi = -xdi
            oa1, ob1 = xdi + ea, xdi + eb
            nmi = nma[i]

            for n in range(nmi):
                kpt = ind[i, n]
                center = cor[kpt]
                dyad = dya[kpt]
                uvec = uxb[i, n]
                tvec = self._cross3(uvec, dyad)
                inin = max(0, 20 * (i - spline_start[0]) - 110)
                inma = min(nsu[0], 20 * (i - spline_start[0]) + 90)
                jnin = max(0, 20 * (i - spline_start[1]) - 110)
                jnma = min(nsu[1], 20 * (i - spline_start[1]) + 90)
                first_backbone = self._groove_window_projections(box, 0, inin, inma, center, dyad, uvec, tvec)
                second_backbone = self._groove_window_projections(box, 1, jnin, jnma, center, dyad, uvec, tvec)

                hpm = [99.0, 99.0]
                gqm = [99.0, 99.0]
                pqn = [99.0, 99.0]
                pqi = [99.0, 99.0]
                dm = [99.0, 99.0]
                dep = [99.0, 99.0]
                inm = [0, 0]
                jnm = [0, 0]
                inn = [0, 0]
                jnn = [0, 0]
                ini = [0, 0]
                jni = [0, 0]
                hpn = [99.0, 99.0]
                gqn = [99.0, 99.0]
                hpi = [99.0, 99.0]
                gqi = [99.0, 99.0]
                ian = [0, 0]
                ca = 99.0
                bd = 99.0

                def projected_point(point, strand):
                    return self._groove_proj_at(first_backbone if strand == 0 else second_backbone, point)

                pqee, pqe, dee, de = 1.0, 2.0, 1.0, 2.0
                hpee, hpe = 1.0, 2.0
                lee = le = jee = je = 0
                pqex, inex = 99.0, -1
                jnee = jne = jnex = jnt = 0
                ogee = oge = ogex = 99.0
                gqee = gqe = gqex = 99.0
                ohee = ohe = 99.0
                scaue = projected_point(inin, 0)[3]
                rad1 = 0.0
                for in_idx in range(inin, inma + 1, 2):
                    o_in, om_in, oh, scu, _ = projected_point(in_idx, 0)
                    if scu * scaue < 0.0:
                        x1 = (projected_point(in_idx - 2, 0)[1] * scu - om_in * scaue) / (scu - scaue)
                        rad1 = max(rad1, x1)
                    inters = False
                    j = 0
                    l = 1 if oh < 0.0 else 2
                    sgn1 = (l == le and le == lee)
                    hp = math.sqrt(max(om_in**2 - oh**2, 0.0))
                    if sgn1 and hp > hpe and hpe < hpee and hpe < hpm[l - 1]:
                        hpm[l - 1] = hpe
                        inm[l - 1] = in_idx - 2

                    normal = self._cross3(dyad, o_in)
                    normal_dot_j = second_backbone["rel"] @ normal
                    s = normal_dot_j[0]
                    for jm in range(jnin, max(jnin - 1, jnma - 10) + 1, 10):
                        r = s
                        s = normal_dot_j[jm + 10 - jnin]
                        if r * s < 0.0:
                            s = r
                            jn = jm
                            while r * s > 0.0 and jn < jm + 10:
                                jn += 1
                                r = s
                                s = normal_dot_j[jn - jnin]
                            _, _, og, scau_j, scat_j = projected_point(jn, 1)
                            if scu * scau_j <= 0.0 and projected_point(in_idx, 0)[4] * scat_j <= 0.0:
                                inters = True
                                jnt = jn
                                j = 1 if og < 0.0 else 2
                                sgn2 = (j == je and je == jee)
                                gq = math.sqrt(max(projected_point(jn, 1)[1] ** 2 - og**2, 0.0))
                                pq = np.linalg.norm(box[in_idx, 0] - box[jn, 1])
                                if in_idx == inex and pq > pqex:
                                    jnt, og, j, gq, pq = jnex, ogex, jex, gqex, pqex
                                inex, jnex, ogex, jex, gqex, pqex = in_idx, jn, og, j, gq, pq
                                break
                    if inters:
                        d = pq - pqee
                        if l == j and sgn1 and sgn2:
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
                    lee, le = le, l
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
                scaue = projected_point(jnin, 1)[3]
                rad2 = 0.0
                for jn in range(jnin, jnma + 1, 2):
                    o_j, om_j, og, scu, _ = projected_point(jn, 1)
                    if scu * scaue < 0.0:
                        x2 = (projected_point(jn - 2, 1)[1] * scu - om_j * scaue) / (scu - scaue)
                        rad2 = max(rad2, x2)
                    inters = False
                    l = 0
                    j = 1 if og < 0.0 else 2
                    sgn2 = (j == je and je == jee)
                    gq = math.sqrt(max(om_j**2 - og**2, 0.0))
                    if sgn2 and gq > gqe and gqe < gqee and gqe < gqm[j - 1]:
                        gqm[j - 1] = gqe
                        jnm[j - 1] = jn - 2

                    normal = self._cross3(dyad, o_j)
                    normal_dot_i = first_backbone["rel"] @ normal
                    s = normal_dot_i[0]
                    for im in range(inin, max(inin - 1, inma - 10) + 1, 10):
                        r = s
                        s = normal_dot_i[im + 10 - inin]
                        if r * s < 0.0:
                            s = r
                            in_idx = im
                            while r * s > 0.0 and in_idx < im + 10:
                                in_idx += 1
                                r = s
                                s = normal_dot_i[in_idx - inin]
                            _, _, oh, scau_i, scat_i = projected_point(in_idx, 0)
                            if scau_i * scu <= 0.0 and scat_i * projected_point(jn, 1)[4] <= 0.0:
                                inters = True
                                intr = in_idx
                                l = 1 if oh < 0.0 else 2
                                sgn1 = (l == le and le == lee)
                                hp = math.sqrt(max(projected_point(in_idx, 0)[1] ** 2 - oh**2, 0.0))
                                pq = np.linalg.norm(box[in_idx, 0] - box[jn, 1])
                                if jn == jnex and pq > pqex:
                                    intr, oh, l, hp, pq = inex, ohex, lex, hpex, pqex
                                inex, jnex, ohex, lex, hpex, pqex = in_idx, jn, oh, l, hp, pq
                                break
                    if inters:
                        d = pq - pqee
                        if l == j and sgn1 and sgn2:
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
                    lee, le = le, l
                    jee, je = je, j
                    scaue = scu

                c5, c6, c7, c8 = ["   -- ", "   -- "], [" ", " "], ["   -- ", "   -- "], ["  --", "  --"]
                groove_measure_geometry = [None, None]
                for l in range(2):
                    pq = pqn[l]
                    if pq < 99.0 and (inn[l] == 0 or inn[l] == nsu[0] or jnn[l] == 0 or jnn[l] == nsu[1]):
                        pq = pqn[l] = 99.0
                    if pq == 99.0:
                        hp = hpm[l]
                        gq = gqm[l]
                        if hp != 99.0 and (inm[l] == 0 or inm[l] == nsu[0]):
                            hp = hpm[l] = 99.0
                        if gq != 99.0 and (jnm[l] == 0 or jnm[l] == nsu[1]):
                            gq = gqm[l] = 99.0
                        if hp != 99.0 and gq != 99.0:
                            if hp <= gq:
                                gq = gqm[l] = 99.0
                            else:
                                hp = hpm[l] = 99.0

                    oa = (oa0 if l == 0 else ob0) * (nmi - n) / nmi + (oa1 if l == 0 else ob1) * n / nmi
                    if pq + dm[l] < 198.0:
                        if pq < 99.0:
                            im = inn[l]
                            jm = jnn[l]
                            r = hpn[l] / (hpn[l] + gqn[l])
                        else:
                            im = ini[l]
                            jm = jni[l]
                            r = hpi[l] / (hpi[l] + gqi[l])
                        oh_vec = r * box[jm, 1] + (1.0 - r) * box[im, 0] - center
                        oh = np.dot(oh_vec, dyad)
                        dep[l] = oa - oh if l == 0 else oh - oa
                        normal = self._cross3(rel(im, 0, center), dyad)
                        normal = self._unit(normal)
                        co = np.dot(tvec, normal)
                        si = np.dot(uvec, normal)
                        if co < 0.0:
                            co = -co
                        an = 0.0 if co > 1.0 else math.degrees(math.acos(np.clip(co, -1.0, 1.0)))
                        if co * si > 0.0:
                            an = -an
                        ian[l] = int(round(an))
                        raw_width = pq if pq < 99.0 else pqi[l]
                        groove_measure_geometry[l] = {
                            "width_endpoint_1": box[im, 0].tolist(),
                            "width_endpoint_2": box[jm, 1].tolist(),
                            "depth_reference": (center + dyad * oa).tolist(),
                            "depth_point": (center + oh_vec).tolist(),
                            "axis_point": center.tolist(),
                            "raw_width": float(raw_width),
                            "display_width": float(raw_width - 2.0 * vdw),
                            "depth": float(dep[l]),
                            "angle": float(ian[l]),
                            "interpolated": bool(pq >= 99.0 and dm[l] < 99.0),
                        }

                    if pqn[l] < 99.0:
                        c5[l] = f6(pqn[l] - 2.0 * vdw)
                        c7[l] = f6(dep[l])
                        c8[l] = f"{ian[l]:4d}"
                    elif pqn[l] == 99.0 and dm[l] < 99.0:
                        c5[l] = f6(pqi[l] - 2.0 * vdw)
                        c6[l] = "*"
                        c7[l] = f6(dep[l])
                        c8[l] = f"{ian[l]:4d}"

                label = self._residue_label(0, i)
                label1 = self._residue_label(1, i)
                c1 = label[0] if (n == 0 and label is not None) else " "
                
                bp_name = ""
                if label and label1:
                    bp_name = f"{label[0]}-{label1[0]}"
                elif label:
                    bp_name = f"{label[0]}-"
                elif label1:
                    bp_name = f"-{label1[0]}"
                
                if str(i) not in self.groove_params["data"]:
                    self.groove_params["data"][str(i)] = {
                        "base_pair": bp_name,
                        "sub_levels": {}
                    }
                
                c9 = "   -- "
                val_c9 = None
                if rad1 * rad2 != 0.0:
                    val_c9 = round(rad1 + rad2, 2)
                    c9 = f6(val_c9)
                line = (f"  {c1}{i:3d}{n:3d}    {c5[0]}{c6[0]}  {c7[0]}  {c8[0]}"
                        f"    {c1}  {c5[1]}{c6[1]}  {c7[1]}  {c8[1]}    {c9}")
                lines.append(line.ljust(72))
                
                # capture raw values
                val_minor_w = round(pqn[0] - 2.0 * vdw, 2) if pqn[0] < 99.0 else (round(pqi[0] - 2.0 * vdw, 2) if pqn[0] == 99.0 and dm[0] < 99.0 else None)
                val_minor_d = round(dep[0], 2) if (pqn[0] < 99.0 or (pqn[0] == 99.0 and dm[0] < 99.0)) else None
                val_minor_a = float(ian[0]) if (pqn[0] < 99.0 or (pqn[0] == 99.0 and dm[0] < 99.0)) else None
                
                val_major_w = round(pqn[1] - 2.0 * vdw, 2) if pqn[1] < 99.0 else (round(pqi[1] - 2.0 * vdw, 2) if pqn[1] == 99.0 and dm[1] < 99.0 else None)
                val_major_d = round(dep[1], 2) if (pqn[1] < 99.0 or (pqn[1] == 99.0 and dm[1] < 99.0)) else None
                val_major_a = float(ian[1]) if (pqn[1] < 99.0 or (pqn[1] == 99.0 and dm[1] < 99.0)) else None
                
                sub_level_data = {
                    "minor_width": val_minor_w,
                    "minor_depth": val_minor_d,
                    "minor_angle": val_minor_a,
                    "major_width": val_major_w,
                    "major_depth": val_major_d,
                    "major_angle": val_major_a,
                    "diameter": val_c9,
                    "geometry": {
                        "minor": groove_measure_geometry[0],
                        "major": groove_measure_geometry[1],
                    }
                }
                self.groove_params["data"][str(i)]["sub_levels"][str(n)] = sub_level_data
                
                if not hasattr(self, "_groove_flat_refs"):
                    self._groove_flat_refs = []
                self._groove_flat_refs.append(sub_level_data)

        clear = "   --       --     --"
        st1 = True
        st2 = True
        for idx, line in enumerate(lines):
            chars = list(line)
            if line[16:18] != "--":
                if line[19:20] != "*":
                    st1 = False
                if st1 and line[19:20] == "*":
                    chars[13:34] = list(clear)
                    self._clear_groove_measure(self._groove_flat_refs[idx], "minor")
            if line[44:46] != "--":
                if line[47:48] != "*":
                    st2 = False
                if st2 and line[47:48] == "*":
                    chars[41:62] = list(clear)
                    self._clear_groove_measure(self._groove_flat_refs[idx], "major")
            lines[idx] = "".join(chars)

        st1 = True
        st2 = True
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            chars = list(line)
            if line[16:18] != "--":
                if line[19:20] != "*":
                    st1 = False
                if st1 and line[19:20] == "*":
                    chars[13:34] = list(clear)
                    self._clear_groove_measure(self._groove_flat_refs[idx], "minor")
            if line[44:46] != "--":
                if line[47:48] != "*":
                    st2 = False
                if st2 and line[47:48] == "*":
                    chars[41:62] = list(clear)
                    self._clear_groove_measure(self._groove_flat_refs[idx], "major")
            lines[idx] = "".join(chars)
        for line in lines:
            print(line.rstrip())

