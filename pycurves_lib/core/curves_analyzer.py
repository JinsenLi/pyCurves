import numpy as np
import sys
from pycurves_lib.data.modified_bases import parent_base_name

class BackboneAnalyzer:
    """
    """
    def __init__(self):
        self.ITA_RAW = np.array([
            [7, 3, 4, 0],
            [3, 4, 5, 0],
            [4, 5, 6, 0],
            [1, 2, 3, 7],
            [7, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 3],
            [6, 7, 3, 4],
            [6, 5, 8, 9],
            [5, 8, 9, 10],
            [14, 13, 6, 5],
            [13, 6, 5, 8],
            [8, 9, 10, 11],
            [9, 10, 11, 12],
            [15, 2, 3, 7],
        ], dtype=int)

        self.END_SET = np.array([106.2540, 100.7339, 102.3817, 78.1441,
                                 39.4467, -34.6199, -133.1470, -156.8818], dtype=float)
        self.END_PSET = np.array([39.70, 154.76], dtype=float)

    def analyze(self, ctx: 'CurvesContext'):
        """Fortran-compatible implementation."""
        self._find_all_atoms(ctx)
        self._calculate_geometry_and_pucker(ctx)
        self._apply_end(ctx)

    def _apply_end(self, ctx: 'CurvesContext'):
        if not ctx.cfg.ends:
            return
        
        for k in range(ctx.n_strands):
            m = ctx.ng[k] - 1
            n = ctx.nr[k] + 1

            ctx.backbone.torsions[k, m, 0:8] = self.END_SET
            ctx.backbone.torsions[k, n, 0:8] = self.END_SET

            if ctx.idr[k] == 1:
                ctx.backbone.torsions[k, n, 6:8] = 0.0
                ctx.backbone.torsions[k, n - 1, 6:8] = self.END_SET[6:8]
            else:
                ctx.backbone.torsions[k, m, 6:8] = 0.0
                ctx.backbone.torsions[k, m + 1, 6:8] = self.END_SET[6:8]

            ctx.backbone.sugar_pucker[k, m, 0:2] = self.END_PSET
            ctx.backbone.sugar_pucker[k, n, 0:2] = self.END_PSET
       
    def _find_all_atoms(self, ctx):
        mol = ctx.molecule

        for k in range(ctx.n_strands):
            forward = (ctx.idr[k] == 1)

            for i in range(ctx.ng[k], ctx.nr[k] + 1):
                if ctx.li[i, k] < -2:
                    continue

                ctx.backbone.flag[k, i] = False
                # nat(i,k,3) = C1'
                c1 = ctx.backbone.atom_map[k, i, 2]
                if c1 < 0:
                    continue

                # C1' -> O4'/O1* , C2'/C2*
                self._match_neighbors(ctx, k, i, c1, {
                    "O1'": 6, "O1*": 6, "O4'": 6, "O4*": 6,
                    "C2'": 3, "C2*": 3
                })

                c2 = ctx.backbone.atom_map[k, i, 3]
                if c2 >= 0:
                    self._match_neighbors(ctx, k, i, c2, {"C3'": 4, "C3*": 4})

                c3 = ctx.backbone.atom_map[k, i, 4]
                if c3 >= 0:
                    self._match_neighbors(ctx, k, i, c3, {
                        "C4'": 5, "C4*": 5, "O3'": 7, "O3*": 7
                    })

                c4 = ctx.backbone.atom_map[k, i, 5]
                if c4 >= 0:
                    self._match_neighbors(ctx, k, i, c4, {"C5'": 12, "C5*": 12})

                c5 = ctx.backbone.atom_map[k, i, 12]
                if c5 >= 0:
                    self._match_neighbors(ctx, k, i, c5, {"O5'": 13, "O5*": 13})

                if (forward and i < ctx.nr[k]) or ((not forward) and i > ctx.ng[k]):
                    o3 = ctx.backbone.atom_map[k, i, 7]
                    if o3 >= 0:
                        p_idx = self._find_neighbor_by_name(mol, o3, {"P"})
                        if p_idx >= 0:
                            ctx.backbone.atom_map[k, i, 8] = p_idx

                            o5_next = self._find_neighbor_by_name(mol, p_idx, {"O5'", "O5*"})
                            if o5_next >= 0:
                                ctx.backbone.atom_map[k, i, 9] = o5_next

                                c5_next = self._find_neighbor_by_name(mol, o5_next, {"C5'", "C5*"})
                                if c5_next >= 0:
                                    ctx.backbone.atom_map[k, i, 10] = c5_next

                                    c4_next = self._find_neighbor_by_name(mol, c5_next, {"C4'", "C4*"})
                                    if c4_next >= 0:
                                        ctx.backbone.atom_map[k, i, 11] = c4_next
                #print(ctx.backbone.atom_map[k, i])
                #print(ctx.molecule.atom_names[ ctx.backbone.atom_map[k, i] ])

    def _calculate_geometry_and_pucker(self, ctx):
        mol = ctx.molecule

        for k in range(ctx.n_strands):
            for i in range(ctx.ng[k], ctx.nr[k] + 1):
                if ctx.li[i, k] < -2:
                    continue

                atom_map = ctx.backbone.atom_map[k, i]
                store = np.full(16, 999.0, dtype=float)
                isg = 0

                for l in range(16):
                    row = self.ITA_RAW[l]

                    i1, i2, i3, i4 = row

                    if i1 <= 0 or i2 <= 0 or i3 <= 0:
                        continue

                    idx1 = atom_map[i1 - 1]
                    idx2 = atom_map[i2 - 1]
                    idx3 = atom_map[i3 - 1]

                    if idx1 < 0 or idx2 < 0 or idx3 < 0:
                        continue

                    p1 = mol.coordinates[idx1]
                    p2 = mol.coordinates[idx2]
                    p3 = mol.coordinates[idx3]

                    if l == 10:
                        ctx.backbone.angles[k, i, 0] = self.torp(p1, p2, p3, None)
                    elif l == 13:
                        ctx.backbone.angles[k, i, 1] = self.torp(p1, p2, p3, None)

                    if l <= 2:
                        if i4 > 0 and atom_map[i4 - 1] >= 0:
                            p4 = mol.coordinates[atom_map[i4 - 1]]
                            store[l] = self.torp(p1, p2, p3, p4)
                        else:
                            store[l] = self.torp(p1, p2, p3, None)
                    elif i4 > 0 and atom_map[i4 - 1] >= 0:
                        p4 = mol.coordinates[atom_map[i4 - 1]]
                        store[l] = self.torp(p1, p2, p3, p4)
                        if 4 <= l <= 8:
                            isg += 1

                ctx.backbone.flag[k, i] = (isg < 5)

                ctx.backbone.torsions[k, i, 0:6] = store[0:6]
                ctx.backbone.torsions[k, i, 6:13] = store[9:16]
                #print(ctx.backbone.torsions[k, i])

                v = np.array([store[5], store[6], store[7], store[8], store[4]], dtype=float)
                if not np.all(np.isfinite(v)) or np.any(v >= 900.0):
                    continue

                a = 0.0
                b = 0.0
                for l in range(5):
                    theta = np.radians(144.0 * l)
                    a += v[l] * np.cos(theta)
                    b += v[l] * np.sin(theta)

                a *= 2.0 / 5.0
                b *= -2.0 / 5.0

                amp = np.sqrt(a * a + b * b)
                if amp > 0.0:
                    cp = np.clip(a / amp, -1.0, 1.0)
                    pha = np.degrees(np.arccos(cp))
                    if b < 0.0:
                        pha = 360.0 - pha
                else:
                    pha = 0.0

                ctx.backbone.sugar_pucker[k, i, 0] = amp
                ctx.backbone.sugar_pucker[k, i, 1] = pha
                #print(amp, pha)

    def _match_neighbors(self, ctx, k, i, start_idx, name_map):
        mol = ctx.molecule
        row = mol.connectivity[start_idx]
        nnb = int(row[6])

        for nb in row[:nnb]:
            if nb == 0:
                continue
            nb0 = nb - 1
            name = mol.atom_names[nb0].strip().upper()
            for target, pos in name_map.items():
                if name == target.upper():
                    ctx.backbone.atom_map[k, i, pos] = nb0

    def _find_neighbor_by_name(self, mol, start, targets):
        row = mol.connectivity[start]
        nnb = int(row[6])

        target_set = {t.upper() for t in targets}
        for nb in row[:nnb]:
            if nb == 0:
                continue
            nb0 = nb - 1
            name = mol.atom_names[nb0].strip().upper()
            if name in target_set:
                return nb0
        return -1
        
    @staticmethod
    def torp(p1, p2, p3, p4=None):
        """
        """
        v1 = p2 - p1
        v2 = p3 - p2

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return np.nan

        if p4 is None:
            c = -(np.dot(v1, v2) / (n1 * n2))
            c = np.clip(c, -1.0, 1.0)
            return np.degrees(np.arccos(c))

        u1 = np.cross(v1, v2)

        v3 = p4 - p3
        u2 = np.array([
            v3[2] * v2[1] - v3[1] * v2[2],
            v3[0] * v2[2] - v3[2] * v2[0],
            v3[1] * v2[0] - v3[0] * v2[1],
        ])

        n1u = np.linalg.norm(u1)
        n2u = np.linalg.norm(u2)
        if n1u == 0.0 or n2u == 0.0:
            return np.nan

        ctor = np.dot(u1, u2) / (n1u * n2u)
        ctor = np.clip(ctor, -1.0, 1.0)

        ang = np.degrees(np.arccos(ctor))

        triple = np.dot(u1, np.cross(u2, v2))
        if triple < 0:
            ang = -ang

        return ang



class HelicalOptimizer:
    """
    1:1 Mathematical Port of Curves 5.3 Optimization Engine.
    Integrates up.f, calc.f, move.f, and analy.f [cite: 241-306].
    """
    def __init__(self, ctx: 'CurvesContext'):
        self.ctx = ctx
        self.n = ctx.n_levels  # Fortran n: number of helical levels.
        self.nst = ctx.n_strands  # Fortran nst: number of strands.
        self.nl = 0  # Fortran nl: current strand index in optimizer loops.
        self.level_count = self.n
        self.strand_count = self.nst
        self.cdr = np.pi / 180.0
        
        self.gra = np.zeros(4 * self.n)  # Fortran gra: optimization gradient.
        self.gradient = self.gra
        self.scp = np.zeros(4)  # Fortran scp: objective component sums.
        self.objective_components = self.scp
        self.initial_origins = None

        self.step_count = 0
        self.prev_sum = 0.0

        self.uho = np.zeros((ctx.n_levels + 2, 3, ctx.n_strands))  # Fortran uho: strand axis directions.
        self.hho = np.zeros((ctx.n_levels + 2, 3, ctx.n_strands))  # Fortran hho: strand axis points.
        self.axis_directions_by_strand = self.uho
        self.axis_points_by_strand = self.hho
        self.oz = np.zeros(ctx.n_levels + 2)  # Fortran oz scratch array for z-axis mode.

        self.prepare()

    def print_fortran_setup_report(self, ctx, file=None):
        """
        - break point
        - Strand / Nucleo / Atoms / Units

        ctx.kam       : atoms
        ctx.kcen      : units
        """
        if file is None:
            file = sys.stdout

        def p(*args):
            print(*args, file=file)

        def dir_str(val):
            """
            """
            if hasattr(ctx, "dir") and callable(ctx.dir):
                return ctx.dir(val)
            if hasattr(ctx, "dir") and not callable(ctx.dir):
                try:
                    return ctx.dir[val]
                except Exception:
                    return str(val)
            if val == 1:
                return "5'-3'"
            if val == -1:
                return "3'-5'"
            return str(val)

        def join_na_for_strand(nl):
            """Build the Curves sequence string for one strand."""
            chars = []
            for i in range(ctx.nux):
                subunit = int(ctx.ni_map[nl, i])
                if subunit <= 0:
                    chars.append("-")
                    continue
                atom_idx = ctx.molecule.subunit_boundaries[subunit - 1]
                name = parent_base_name(ctx.molecule.residue_names[atom_idx])
                chars.append(name[1] if len(name) >= 2 and name[0] in {"D", "R"} and name[1] in "GACTUIYP" else name[:1])
            return "".join(chars)

        # if(break.gt.0) write(6,18) break-1,break
        if getattr(ctx, "break_pt", 0) > 0:
            p()
            p(f"  Break point between levels {ctx.break_pt - 1:2d} and {ctx.break_pt:2d}")

        # write(6,20) nst,nt,kam,kcen
        p(
            f"  Strand= {ctx.nst:4d} Nucleo= {ctx.nt:4d} "
            f"Atoms = {ctx.molecule.kam:4d} Units = {ctx.molecule.kcen:4d}"
        )

        if ctx.cfg.zaxe:
            if ctx.cfg.comb:
                p()
                p(f"  Combined strands have {ctx.nux:4d} levels ...")
                p()

                # do nl=1,max(ns,nst)
                #   write(6,30) nl,nu(nl),dir(idr(nl)),(na(i,nl),i=1,nux)
                # enddo
                for nl in range(ctx.nst):
                    seq = join_na_for_strand(nl)
                    p(
                        f"  Strand {nl + 1:2d} has {ctx.nu[nl]:3d} bases "
                        f"({dir_str(ctx.idr[nl])}): {seq}"
                    )
            else:
                for nl in range(ctx.nst):
                    p()
                    seq = join_na_for_strand(nl)
                    p(
                        f"  Strand {nl + 1:2d} has {ctx.nu[nl]:3d} bases "
                        f"({dir_str(ctx.idr[nl])}): {seq}"
                    )

            return

        p()
        for k in range(ctx.cfg.inpv):
            p(
                f"  Input {k + 1:3d}) "
                f"Xdisp= {ctx.cfg.xdi[k]:7.2f} "
                f"Ydisp= {ctx.cfg.ydi[k]:7.2f} "
                f"Inclin= {ctx.cfg.cln[k]:7.2f} "
                f"Tip= {ctx.cfg.tip[k]:7.2f}"
            )

        if ctx.cfg.comb:
            p()
            p(f"  Combined strands have {ctx.nux:4d} levels ...")
            p()
            for nl in range(ctx.nst):
                seq = join_na_for_strand(nl)
                p(
                    f"  Strand {nl + 1:2d} has {ctx.nu[nl]:3d} bases "
                    f"({dir_str(ctx.idr[nl])}): {seq}"
                )
        else:
            for nl in range(ctx.nst):
                p()
                seq = join_na_for_strand(nl)
                p(
                    f"  Strand {nl + 1:2d} has {ctx.nu[nl]:3d} bases "
                    f"({dir_str(ctx.idr[nl])}): {seq}"
                )

        if ctx.cfg.ends:
            p()
            start = ctx.cfg.end_start
            stop = ctx.cfg.end_stop
            p(
                f"  ENDS  {0:3d}) "
                f"Xd={start[0]:7.2f} "
                f"Yd={start[1]:7.2f} "
                f"Ri={start[2]:7.2f} "
                f"In={start[3]:7.2f} "
                f"Tp={start[4]:7.2f} "
                f"Tw={start[5]:7.2f}"
            )
            p(
                f"  ENDS  {ctx.nux + 1:3d}) "
                f"Xd={stop[0]:7.2f} "
                f"Yd={stop[1]:7.2f} "
                f"Ri={stop[2]:7.2f} "
                f"In={stop[3]:7.2f} "
                f"Tp={stop[4]:7.2f} "
                f"Tw={stop[5]:7.2f}"
            )

    def print_final_report(self, file=None):
        """

        write(6,10) sum,scp(1)*10,scp(2),scp(3)*10,scp(4)
        write(6,20) (grm(i),i=1,nvar)
        call title(...)
        do i=ist,ien ...
        """

        if file is None:
            file = sys.stdout

        def emit(*args, **kwargs):
            print(*args, file=file, **kwargs)

        ctx = self.ctx
        p = ctx.params

        ist = self.ist
        ien = self.ien

        # ----------------------------
        # FINAL SUM + SCP
        # ----------------------------
        total_sum = getattr(self, "prev_sum", 0.0)

        scp = self.scp  # length 4

        emit()
        emit(f"  FINAL SUM= {total_sum:8.3f} CPTS: "
            f"{scp[0]*10:8.3f}{scp[1]:8.3f}{scp[2]*10:8.3f}{scp[3]:8.3f}")
        emit()

        # ----------------------------
        # GRA (gradient)
        # ----------------------------
        gra = self.gra
        nvar = len(self._min_spec)

        emit("  GRA=", end="")
        for i in range(nvar):
            if i % 8 == 0 and i != 0:
                emit()
                emit("      ", end="")
            emit(f"{gra[i]:9.2E}", end="")
        emit("\n")

        # ----------------------------
        # TITLE
        # ----------------------------
        emit("  ----------------------------")
        emit("  |A| Global axis parameters |")
        emit("  ----------------------------")
        emit()

        # ----------------------------
        # LOOP over levels
        # ----------------------------
        for i in range(ist, ien + 1):

            # U = axis direction
            ux, uy, uz = p.ux[i]

            # P = position
            if ctx.cfg.comb:
                x, y, z = p.ox[i]
            else:
                x, y, z = p.hx[i]

            dif = getattr(p, "dif", None)

            if i < ien and dif is not None:
                emit(f"  {i:3d}) U: "
                    f"{ux:8.3f}{uy:8.3f}{uz:8.3f}  "
                    f"P: {x:8.3f}{y:8.3f}{z:8.3f}  "
                    f"D: {dif[i]:8.3f}")
            else:
                emit(f"  {i:3d}) U: "
                    f"{ux:8.3f}{uy:8.3f}{uz:8.3f}  "
                    f"P: {x:8.3f}{y:8.3f}{z:8.3f}")
                
    def prepare(self):
        """Fortran-compatible implementation."""
        self.ns = self.ctx.nst
        if self.ctx.cfg.comb:
            self.active_nst = 1
        else:
            self.active_nst = self.ctx.nst
            
        self._map_active_strands()
        
        self._seed_initial_parameters()
        
        if self.ctx.cfg.zaxe:
            self._setup_z_axis_reference()

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

    def _map_active_strands(self):
        ctx = self.ctx
        nux = ctx.n_levels
        ns = ctx.n_strands

        self.iact = np.zeros(nux + 2, dtype=int)

        if not ctx.cfg.comb:
            return

        for i in range(1, nux + 1):
            found = False

            for j in range(ns):
                if ctx.li[i, j] > 0:
                    self.iact[i] = j + 1
                    found = True
                    break

            if not found:
                for j in range(ns):
                    if ctx.li[i, j] == -1:
                        self.iact[i] = j + 1
                        found = True
                        break

            if not found:
                raise ValueError(f"Level {i} with no bases not allowed.")
            
    def _seed_initial_parameters(self):
        ctx = self.ctx
        cfg = ctx.cfg
        nux = ctx.n_levels
        nst = ctx.n_strands

        m = 0
        for k in range(nst):
            if cfg.comb:
                ist = 1
                ien = nux
            else:
                ist = ctx.ng[k]
                ien = ctx.nr[k]

            for i in range(ist, ien + 1):
                m += 1

                if cfg.comb:
                    l = i if cfg.rest else 1
                else:
                    l = m if cfg.rest else (k + 1)

                idx = l - 1  # Python 0-based

                ctx.params.helical[k, i, 0] = cfg.xdi[idx]   # hel(i,1,k)
                ctx.params.helical[k, i, 1] = cfg.ydi[idx]   # hel(i,2,k)
                ctx.params.helical[k, i ,3] = cfg.cln[idx]   # hel(i,4,k)
                ctx.params.helical[k, i, 4] = cfg.tip[idx]   # hel(i,5,k)

    def _setup_z_axis_reference(self):
        """
        """
        ctx = self.ctx
        nux = ctx.n_levels
        
        for nl in range(ctx.n_strands):
            ist = 1 if ctx.cfg.comb else ctx.ng[nl]
            ien = nux if ctx.cfg.comb else ctx.nr[nl]
            
            for i in range(ist, ien + 1):
                self.uho[i, 0, nl] = 0.0
                self.uho[i, 1, nl] = 0.0
                self.uho[i, 2, nl] = float(ctx.idr[nl]) # uho(i,3,nl)=idr(nl) 
                
                nn = nl
                if ctx.li[i-1, nl] < 0:
                    nn = self.iact[i]
                
                res_z = ctx.base_frames.rez[nn, i, 3]
                
                self.hho[i, 0, nl] = 0.0
                self.hho[i, 1, nl] = 0.0
                self.hho[i, 2, nl] = res_z # hho(i,3,nl)=rez(i,4,nn) 
                
                self.oz[i] = res_z

    def _prepare_before_analy(self):
        """
        """
        ctx = self.ctx
        cfg = ctx.cfg
        nux = ctx.n_levels
        nst = ctx.n_strands

        self.iact = np.zeros(nux + 2, dtype=int)

        self.nvar_by_strand = np.zeros(nst, dtype=int)

        self.ist_by_strand = np.zeros(nst, dtype=int)
        self.ien_by_strand = np.zeros(nst, dtype=int)
        self.iste_by_strand = np.zeros(nst, dtype=int)
        self.iene_by_strand = np.zeros(nst, dtype=int)

        ns = nst if cfg.comb else 1

        for nl in range(nst):
            # -----------------------
            # nvar
            # -----------------------
            n = ctx.nu[nl]
            if cfg.comb:
                n = nux

            nvar = 4 * n
            if cfg.line:
                nvar = 4
            if cfg.line and cfg.break_lvl > 0:
                nvar = 8

            self.nvar_by_strand[nl] = nvar

            # -----------------------
            # non-comb
            # -----------------------
            if not cfg.comb:
                ist = ctx.ng[nl]
                ien = ctx.nr[nl]
                iste = ist - 1 if cfg.ends else ist
                iene = ien + 1 if cfg.ends else ien

                self.ist_by_strand[nl] = ist
                self.ien_by_strand[nl] = ien
                self.iste_by_strand[nl] = iste
                self.iene_by_strand[nl] = iene

                for i in range(iste, iene + 1):
                    self.iact[i] = nl + 1

            # -----------------------
            # comb
            # -----------------------
            else:
                ist = 1
                ien = nux
                iste = ist - 1 if cfg.ends else ist
                iene = ien + 1 if cfg.ends else ien

                self.ist_by_strand[nl] = ist
                self.ien_by_strand[nl] = ien
                self.iste_by_strand[nl] = iste
                self.iene_by_strand[nl] = iene

                for i in range(iste, iene + 1):
                    found = False

                    for j in range(ns):  # j = 0..nst-1
                        if ctx.li[i, j] > 0:
                            self.iact[i] = j + 1
                            found = True
                            break

                    if not found:
                        for j in range(ns):
                            if ctx.li[i, j] == -1:
                                self.iact[i] = j + 1
                                found = True
                                break

                    if not found:
                        raise ValueError("Level with no bases not allowed.")
            self.iste = iste
            self.ist = ist
            self.ien = ien
            self.iene = iene
        self.icyc = 0

    def run(self, mini = True):
        """Fortran-compatible implementation."""
        self._prepare_before_analy()
        self.initial_origins = self._axis_reference_frames()[0, :, 3].copy()
        #print(self.ctx.params.helical)
        #first_sum = self._calc_physics_logic()
        #self._compute_derivs() 
        #self._compute_grads_logic()
        if mini:
            self.minimise()
        self.saveparams()

    def saveparams(self):
        '''
        do i=ist,ien
        uho(i,1,nl)=ux(i)
        uho(i,2,nl)=uy(i)
        uho(i,3,nl)=uz(i)
        hho(i,1,nl)=ox(i)
        hho(i,2,nl)=oy(i)
        hho(i,3,nl)=oz(i)
        enddo
        enddo
        '''
        ctx = self.ctx
        p = self.ctx.params
        if not ctx.cfg.comb and getattr(self, "_axis_ux_by_strand", None) is not None:
            for nl in range(ctx.nst):
                ist = int(self.ist_by_strand[nl])
                ien = int(self.ien_by_strand[nl])
                for i in range(ist, ien + 1):
                    self.uho[i, :, nl] = self._axis_ux_by_strand[i, nl]
                    self.hho[i, :, nl] = self._axis_ox_by_strand[i, nl]
            self._apply_end_axis_extensions()
            return

        for nl in range(ctx.nst):
            for i in range(self.ist, self.ien + 1):
                self.uho[i, : , nl] = p.ux[i]
                self.hho[i, : , nl] = p.ox[i]
        self._apply_end_axis_extensions()

    @staticmethod
    def _normalize_vector(vector, fallback=None):
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            return np.asarray(vector, dtype=float) / norm
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        return np.asarray(fallback, dtype=float).copy()

    @staticmethod
    def _rotate_axis_angle(vector, axis, angle_degrees):
        axis = HelicalOptimizer._normalize_vector(axis)
        vector = np.asarray(vector, dtype=float)
        angle = np.deg2rad(angle_degrees)
        ca = np.cos(angle)
        sa = np.sin(angle)
        return (
            vector * ca
            + np.cross(axis, vector) * sa
            + axis * np.dot(axis, vector) * (1.0 - ca)
        )

    def _store_end_frame(self, strand, level, origin, x_axis, y_axis, z_axis):
        p = self.ctx.params
        reference_frames = self._axis_reference_frames()
        y_axis = self._normalize_vector(y_axis, fallback=reference_frames[strand, max(1, min(self.n, level)), 1])
        z_axis = self._normalize_vector(z_axis, fallback=reference_frames[strand, max(1, min(self.n, level)), 2])
        x_axis = self._normalize_vector(x_axis, fallback=np.cross(y_axis, z_axis))
        p.frames[strand, level, 0, :] = x_axis
        p.frames[strand, level, 1, :] = y_axis
        p.frames[strand, level, 2, :] = z_axis
        p.frames[strand, level, 3, :] = origin
        if hasattr(p, "axis_frames"):
            p.axis_frames[strand, level, 0, :] = x_axis
            p.axis_frames[strand, level, 1, :] = y_axis
            p.axis_frames[strand, level, 2, :] = z_axis
            p.axis_frames[strand, level, 3, :] = origin

    def _axis_at(self, level, strand):
        if not self.ctx.cfg.comb and getattr(self, "_axis_ux_by_strand", None) is not None:
            return self._axis_ux_by_strand[level, strand]
        return self.ctx.params.ux[level]

    def _apply_end_axis_extensions(self):
        """Fortran setend.f: build virtual terminal axis points and frames."""
        ctx = self.ctx
        if not ctx.cfg.ends:
            return

        p = ctx.params
        reference_frames = self._axis_reference_frames()
        nux = ctx.n_levels
        work_strands = [0] if ctx.cfg.comb else list(range(ctx.nst))

        for nl in work_strands:
            for level, adjacent, id_step, end_values in (
                (0, 1, 1, ctx.cfg.end_start),
                (nux + 1, nux, -1, ctx.cfg.end_stop),
            ):
                xdi, ydi, rise, cln, tip, twis = map(float, end_values)
                is_sign = int(ctx.idr[nl]) if id_step == 1 else -int(ctx.idr[nl])

                axis = self._normalize_vector(self._axis_at(adjacent, nl), fallback=reference_frames[nl, adjacent, 2])
                point = self.hho[adjacent, :, nl] - axis * rise * is_sign

                strands_to_mark = range(ctx.nst) if ctx.cfg.comb else [nl]
                for strand in strands_to_mark:
                    self.uho[level, :, strand] = axis
                    self.hho[level, :, strand] = point
                if ctx.cfg.comb or nl == 0:
                    p.ux[level, :] = axis
                    p.ox[level, :] = point

                adjacent_y = reference_frames[nl, adjacent, 1, :]
                wx0 = adjacent_y - axis * np.dot(axis, adjacent_y)
                wx0 = self._normalize_vector(wx0, fallback=reference_frames[nl, adjacent, 0, :])
                wx = self._normalize_vector(self._rotate_axis_angle(wx0, axis, -is_sign * twis), fallback=wx0)
                vx = self._normalize_vector(np.cross(wx, axis), fallback=reference_frames[nl, adjacent, 0, :])

                origin = point + vx * xdi + wx * ydi
                y_axis = self._normalize_vector(self._rotate_axis_angle(wx, vx, cln), fallback=wx)
                q_axis = self._normalize_vector(self._rotate_axis_angle(axis, vx, cln), fallback=axis)
                z_axis = self._normalize_vector(self._rotate_axis_angle(q_axis, y_axis, tip), fallback=q_axis)
                x_axis = self._normalize_vector(np.cross(y_axis, z_axis), fallback=vx)
                self._store_end_frame(nl, level, origin, x_axis, y_axis, z_axis)

                if ctx.cfg.comb and ctx.nst > 1:
                    paired_origin = point + vx * xdi - wx * ydi
                    paired_y = -y_axis
                    paired_q = -q_axis
                    paired_z = self._normalize_vector(self._rotate_axis_angle(paired_q, paired_y, tip), fallback=paired_q)
                    paired_x = self._normalize_vector(np.cross(paired_y, paired_z), fallback=vx)
                    self._store_end_frame(1, level, paired_origin, paired_x, paired_y, paired_z)


    def _build_min_spec(self):
        ctx = self.ctx
        p = ctx.params
        cfg = ctx.cfg

        comb = bool(getattr(cfg, "comb", False))
        line = bool(getattr(cfg, "line", False))
        brk = int(getattr(cfg, "break_lvl", 0))

        spec = []
        scales = []

        if comb:
            strands = [0]
        else:
            strands = list(range(ctx.nst))

        for nl in strands:
            ist = int(self.ist_by_strand[nl]) if hasattr(self, "ist_by_strand") else int(getattr(self, "ist", 1))
            ien = int(self.ien_by_strand[nl]) if hasattr(self, "ien_by_strand") else int(getattr(self, "ien", self.n))
            iup = 1 if line else ien

            for i in range(ist, iup + 1):
                nch = int(self.iact[i]) - 1 if comb else nl
                items = [
                    (i, nch, 0, False),  # dx
                    (i, nch, 1, False),  # dy
                    (i, nch, 3, True),   # cln
                    (i, nch, 4, True),   # tip
                ]
                spec.extend(items)
                scales.extend([0.5, 0.5, 1.5, 1.5])

            if line and brk > 0:
                nch = int(self.iact[ist]) - 1 if comb else nl
                spec.extend([
                    (brk, nch, 0, False),
                    (brk, nch, 1, False),
                    (brk, nch, 3, True),
                    (brk, nch, 4, True),
                ])
                scales.extend([0.5, 0.5, 1.5, 1.5])

        self._min_spec = spec
        self._min_scale = np.array(scales, dtype=float)
        return spec

    
    def _pack_min_vars(self):
        p = self.ctx.params
        x = np.zeros(len(self._min_spec), dtype=float)
        for m, (i, nch, col, _) in enumerate(self._min_spec):
            x[m] = float(p.helical[nch, i, col])
        return x / self._min_scale
