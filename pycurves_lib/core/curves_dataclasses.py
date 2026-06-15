import numpy as np
from dataclasses import dataclass, field
from typing import List

from pycurves_lib.topology.base_annotations import is_modified_base, parent_base_name
from pycurves_lib.io.base_reference import BaseFrameFitter, BaseReferenceLibrary, canonical_atom_name

def get_standard_bref():
    """
    Returns the bref(10,3,5) array hardcoded from setup.f .
    Dimensions: (5 types, 10 atoms, 3 coordinates)
    """
    bref = np.zeros((5, 10, 3))

    # --- Type 1: Guanine (G) [cite: 91-92] ---
    bref[0, :10] = [
        [1.58195, -2.39594, -0.12320], [-0.10165, 2.41153, -0.05020],
        [1.18367, 1.92662, -0.18720],  [1.47916, 0.62851, -0.18920],
        [0.37551, -0.14975, -0.04020], [-0.93480, 0.24071, 0.10280],
        [-1.25149, 1.62390, 0.10480],  [-1.76708, -0.87080, 0.22880],
        [-0.94239, -1.88689, 0.15880], [0.37714, -1.52785, -0.00520]
    ]

    # --- Type 2: Adenine (A) [cite: 92-93] ---
    bref[1, :10] = [
        [1.57340, -2.41044, -0.12190], [-0.11123, 2.43959, -0.04964],
        [1.10298, 1.90641, -0.17768],  [1.45731, 0.64079, -0.18694],
        [0.38110, -0.16006, -0.04011], [-0.91754, 0.23706, 0.10110],
        [-1.16848, 1.61858, 0.09523],  [-1.75109, -0.86494, 0.22638],
        [-0.94428, -1.87750, 0.15888], [0.37786, -1.52945, -0.00531]
    ]

    # --- Type 3: Cytosine (C) [cite: 93-94] ---
    bref[2, :7] = [
        [1.94866, -1.45161, -0.19029], [0.74385, -0.58352, -0.07229],
        [0.93020, 0.79520, -0.12929],   [-0.15547, 1.60399, -0.02329],
        [-1.37969, 1.08626, 0.13371],  [-1.59254, -0.32937, 0.19471],
        [-0.49500, -1.12092, 0.08671]
    ]

    # --- Type 4: Thymine (T) [cite: 94] ---
    bref[3, :7] = [
        [1.95363, -1.45659, -0.19073], [0.75787, -0.57587, -0.07419],
        [0.97215, 0.78019, -0.13405],   [-0.15841, 1.56515, -0.02137],
        [-1.45350, 1.11665, 0.14102],  [-1.57773, -0.32139, 0.19302],
        [-0.49400, -1.10811, 0.08633]
    ]

    # --- Type 5: Pseudo-uracil (P) [cite: 94-95] ---
    bref[4, :7] = [
        [-1.24071, -2.10001, -0.01015], [1.59022, 0.32075, -0.00729],
        [0.91535, 1.59602, 0.00679],    [-0.46861, 1.50106, 0.01545],
        [-1.20770, 0.33858, 0.00979],   [-0.48272, -0.82819, -0.00729],
        [0.89420, -0.82819, -0.00729]
    ]
    return bref

@dataclass
class BaseGeometryConstants:
    """
    1:1 Mirror of standard base geometry constants in setup.f.
    """
    th1: float = 132.193     # Rotation angle for origin placement
    th2: float = -54.512     # Rotation angle for dyad axis
    dis: float = 4.5033      # Displacement distance for origin (dist)
    rs2: float = 3.3         # Squared distance for sugar retry (rs2)
    cdr: float = np.pi / 180.0 # Degrees to Radians
    
    iequ: List[int] = field(default_factory=lambda: [1, 2, 1, 1, 3, 4, 4, 3, 5])
    ibref: List[int] = field(default_factory=lambda: [10, 10, 7, 7, 7])

    base_list: list = field(default_factory=lambda: ['G', 'A', 'I', 'Y', 'C', 'T', 'U', 'R', 'P'])
    ban: list = field(default_factory=lambda: ['N1','C2','N3','C4','C5','C6','N7','C8','N9'])

class BaseLocator:
    def __init__(self, constants: BaseGeometryConstants, reference_library: BaseReferenceLibrary = None):
        self.const = constants
        self.reference_library = reference_library or BaseReferenceLibrary.load("legacy")
        self.frame_fitter = BaseFrameFitter(self.reference_library)

    def locate_all(self, ctx: 'CurvesContext'):
        """
        Orchestrates the location process[cite: 51].
        Implements Goto 20 skip logic for missing bases.
        """
        if ctx.cfg.fit:
            print("\n  Least squares fitting of standard bases ...")
            print("\n   Str   Pos  Base          Rms (ang)\n")

        ctx.annotations.setdefault("base_fit_quality", [])
        ctx.annotations["base_fit_quality"].clear()

        for n in range(ctx.n_strands):
            for m in range(ctx.n_levels):
                atom_indices = self._identify_base_atoms(n, m, ctx)
                #for resid in atom_indices["nind"]:
                #    print(resid, ctx.molecule.atom_names[resid], ctx.molecule.coordinates[resid])
                if atom_indices is None:
                    continue
                lvl = m + 1
            
                ctx.backbone.atom_map[n, lvl, 0] = atom_indices['i4'] # nat(1)
                ctx.backbone.atom_map[n, lvl, 1] = atom_indices['i2'] # nat(2)
                ctx.backbone.atom_map[n, lvl, 2] = atom_indices['i1'] # nat(3) <- C1' 
                ctx.backbone.atom_map[n, lvl, 14] = atom_indices['i3'] # nat(15)
                    
                rms = self._calculate_base_frame(n, m, atom_indices, ctx)
                ctx.annotations["base_fit_quality"].append(
                    self._base_fit_quality_record(n, m, atom_indices, rms, ctx)
                )

                if ctx.cfg.fit:
                    res_label = self._fit_report_label(atom_indices, ctx)
                    residue_id = int(ctx.molecule.residue_ids[atom_indices['i1']])
                    print(f"   {n+1:2d} : {m+1:3d})  {res_label:<12s} {residue_id:5d}      {rms:7.3f}")
        if ctx.cfg.ends:
            self._build_end_extensions(ctx)

    def _fit_report_label(self, atom_indices, ctx: 'CurvesContext') -> str:
        """Human-readable residue label for the least-squares fit report."""
        atom_idx = atom_indices["i1"]
        residue_name = atom_indices.get("residue_name", "").strip().upper()
        parent_base = atom_indices.get("parent_base", "").strip().upper()
        chain = ""
        if ctx.molecule.chain_ids is not None:
            chain = str(ctx.molecule.chain_ids[atom_idx]).strip()

        if residue_name and parent_base and residue_name != parent_base and residue_name != f"D{parent_base}":
            base_text = f"{residue_name}->{parent_base}"
        else:
            base_text = residue_name or parent_base or "?"
        return f"{base_text}:{chain}" if chain else base_text

    def _build_end_extensions(self, ctx: 'CurvesContext'):
        cdr = self.const.cdr
        nst = ctx.n_strands
        nux = ctx.n_levels
        p = ctx.params

        for n in range(nst):
            # Fortran copies hel(0,*,1) and hel(nux+1,*,1) to all strands.
            p.helical[n, 0, :] = ctx.cfg.end_start
            p.helical[n, nux + 1, :] = ctx.cfg.end_stop

        for n in range(nst):
            # Fortran: id=1; if(n.gt.1) id=-1
            id_sign = 1 if n == 0 else -1

            for m in range(2):
                # Fortran: l=ng(n)-1; if(m.eq.2) l=nr(n)+1
                if m == 0:
                    l = ctx.ng[n] - 1
                else:
                    l = ctx.nr[n] + 1

                xdi = p.helical[n, l, 0]
                ydi = p.helical[n, l, 1]
                cln = p.helical[n, l, 3]
                tip = p.helical[n, l, 4]

                ct = np.cos(cdr * cln)
                st = np.sin(cdr * cln)
                cp = np.cos(cdr * tip)
                sp = np.sin(cdr * (id_sign * tip))

                rx3 = sp * id_sign
                ry3 = -st * cp * id_sign
                rz3 = ct * cp * id_sign

                rx2 = 0.0
                ry2 = ct * id_sign
                rz2 = st * id_sign

                rx1 = ry2 * rz3 - rz2 * ry3
                ry1 = rz2 * rx3 - rx2 * rz3
                rz1 = rx2 * ry3 - ry2 * rx3

                if ctx.cfg.comb:
                    p.efd[0, m, n] = rz1
                    p.efd[1, m, n] = rz2
                    p.efd[2, m, n] = rz3
                else:
                    p.efd[0, m, n] = rz1 * id_sign
                    p.efd[1, m, n] = rz2 * id_sign
                    p.efd[2, m, n] = rz3 * id_sign

                p.efc[0, m, n] = -rx1 * xdi - ry1 * ydi
                p.efc[1, m, n] = -rx2 * xdi - ry2 * ydi
                p.efc[2, m, n] = -rx3 * xdi - ry3 * ydi
                
    def _get_base_type_index(self, res_name: str) -> int:
        """
        """
        name = res_name.strip().upper()
        parent = parent_base_name(name)
        mapping = {
            'G': 1, 'DG': 1, 'GUA': 1,
            'A': 2, 'DA': 2, 'ADE': 2,
            'I': 3,
            'Y': 4,
            'C': 5, 'DC': 5, 'CYT': 5,
            'T': 6, 'DT': 6, 'THY': 6,
            'U': 7,
            'R': 8,
            'P': 9
        }
        return mapping.get(name, mapping.get(parent, mapping.get(parent[:1], 10)))
    
    def _identify_base_atoms(self, strand: int, level: int, ctx: 'CurvesContext'):
        """
        """
        mol = ctx.molecule
        subunit_idx = ctx.ni_map[strand, level]
        if subunit_idx == 0: return None

        start_idx = mol.subunit_boundaries[subunit_idx - 1]
        end_idx = mol.subunit_boundaries[subunit_idx]
        res_name = mol.residue_names[start_idx].strip().upper()
        
        lsav = self._get_base_type_index(res_name)
        lequ = self.const.iequ[lsav - 1] if lsav <= 9 else 10
        purine = lsav <= 4 
        mn = 9 if lequ <= 2 else 6 

        res_atoms = {}
        for i in range(start_idx, end_idx):
            name = mol.atom_names[i]
            while len(name) > 0 and (name[0] == ' ' or '0' <= name[0] <= '9'):
                name = name[1:]
            clean_name = canonical_atom_name(name)
            res_atoms[clean_name] = i

        i1 = res_atoms.get("C1'", res_atoms.get("C1*", -1))

        ref_i2 = res_atoms.get('N9', -1) if purine else res_atoms.get('N1', -1)
        if i1 == -1 and ref_i2 != -1:
            c0 = mol.coordinates[ref_i2]
            for name in ["C1'", "C1*"]:
                if name in res_atoms:
                    idx = res_atoms[name]
                    if np.sum((c0 - mol.coordinates[idx])**2) < self.const.rs2:
                        i1 = idx; break

        nind = [-1] * (mn + 1)
        kb = 0
        if i1 != -1:
            nind[0] = i1
            kb = 1

        for mm in range(mn):
            target_name = self.const.ban[mm] # 'N1','C2','N3'...
            target_idx = res_atoms.get(target_name, -1)
            if target_idx != -1:
                nind[mm + 1] = target_idx
                kb += 1

        expected_fit_atoms = ["C1'"] + self.const.ban[:mn]
        present_fit_atoms = [
            atom_name for atom_name, atom_idx in zip(expected_fit_atoms, nind)
            if atom_idx != -1
        ]
        missing_fit_atoms = [
            atom_name for atom_name, atom_idx in zip(expected_fit_atoms, nind)
            if atom_idx == -1
        ]
        parent_base = parent_base_name(res_name)
        template_base = parent_base if parent_base != "unknown" else res_name[:1]
        reference_template = self.reference_library.template_for_base(template_base)
        reference_atom_names = list(reference_template.atom_names) if reference_template is not None else expected_fit_atoms
        reference_present_atoms = [
            atom_name for atom_name in reference_atom_names
            if any(alias in res_atoms for alias in self._atom_aliases(atom_name))
        ]
        reference_missing_atoms = [
            atom_name for atom_name in reference_atom_names
            if not any(alias in res_atoms for alias in self._atom_aliases(atom_name))
        ]
        ignored_base_atoms = self._ignored_base_atoms(res_atoms, expected_fit_atoms, parent_base)

        if purine:
            i2, i3, i4 = res_atoms.get('N9', -1), res_atoms.get('C4', -1), res_atoms.get('C8', -1)
        else:
            if lsav < 9:
                i2, i3, i4 = res_atoms.get('N1', -1), res_atoms.get('C2', -1), res_atoms.get('C6', -1)
            else: # P (Pseudo-U)
                i2, i3, i4 = res_atoms.get('C5', -1), res_atoms.get('C4', -1), res_atoms.get('N1', -1)

        if kb < 3:
            if i1 != -1:
                self._handle_isolation(strand, level, i1, ctx)
                return None
            raise ValueError(
                f"Strand {strand + 1} level {level + 1} ({res_name.strip()} subunit {subunit_idx}) "
                f"lacks enough atoms for base fitting (count={kb})."
            )

        return {
            'i1': i1,
            'i2': i2,
            'i3': i3,
            'i4': i4,
            'nind': nind,
            'subunit': int(subunit_idx),
            'residue_name': res_name,
            'parent_base': parent_base,
            'template_base': template_base,
            'res_atoms': res_atoms,
            'reference_template': reference_template,
            'reference_atom_names': reference_atom_names,
            'reference_present_atoms': reference_present_atoms,
            'reference_missing_atoms': reference_missing_atoms,
            'is_modified': is_modified_base(res_name),
            'expected_fit_atoms': expected_fit_atoms,
            'present_fit_atoms': present_fit_atoms,
            'missing_fit_atoms': missing_fit_atoms,
            'ignored_base_atoms': ignored_base_atoms,
        }

    @staticmethod
    def _atom_aliases(atom_name: str):
        clean = canonical_atom_name(atom_name)
        aliases = [clean]
        if clean.endswith("*"):
            aliases.append(clean[:-1] + "'")
        elif clean.endswith("'"):
            aliases.append(clean[:-1] + "*")
        return aliases

    def _ignored_base_atoms(self, res_atoms, expected_fit_atoms, parent_base):
        """Return non-hydrogen base atoms not used by the parent-base template."""
        parent_base_atoms = {
            "A": {"N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"},
            "G": {"N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"},
            "C": {"N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"},
            "T": {"N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "C7"},
            "U": {"N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"},
            "I": {"N1", "C2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"},
        }
        expected = set(expected_fit_atoms) | {"C1*"} | parent_base_atoms.get(parent_base, set())
        backbone_or_sugar = {
            "P", "OP1", "OP2", "O1P", "O2P", "O3P", "O5'", "O5*", "C5'", "C5*",
            "C4'", "C4*", "O4'", "O4*", "C3'", "C3*", "O3'", "O3*", "C2'",
            "C2*", "O2'", "O2*", "C1'", "C1*",
        }
        ignored = []
        for atom_name in sorted(res_atoms):
            if atom_name in expected or atom_name in backbone_or_sugar:
                continue
            if atom_name.startswith(("H", "D")):
                continue
            ignored.append(atom_name)
        return ignored

    def _base_fit_quality_record(self, strand, level, atom_indices, rms, ctx):
        """Record pyCurves-native fit provenance; Fortran names remain in arrays."""
        mol = ctx.molecule
        atom_idx = atom_indices['i1']
        if atom_idx == -1:
            atom_idx = next((idx for idx in atom_indices.get("nind", []) if idx != -1), -1)
        return {
            "strand": int(strand + 1),
            "level": int(level + 1),
            "subunit": int(atom_indices.get("subunit", 0)),
            "chain_id": str(mol.chain_ids[atom_idx]).strip() if atom_idx != -1 and mol.chain_ids is not None else "",
            "residue_id": int(mol.residue_ids[atom_idx]) if atom_idx != -1 else 0,
            "residue_name": atom_indices.get("residue_name", "").strip().upper(),
            "parent_base": atom_indices.get("parent_base", "unknown"),
            "template_used": atom_indices.get("template_base", "unknown"),
            "frame_convention": getattr(ctx.cfg, "frame_convention", "legacy"),
            "fit_strategy": (
                "standard_base_library"
                if getattr(ctx.cfg, "frame_convention", "legacy") == "standard"
                else ("parent_base_atoms" if atom_indices.get("is_modified") else "standard_base_atoms")
            ),
            "rmsd": float(rms),
            "is_modified": bool(atom_indices.get("is_modified")),
            "expected_fit_atoms": list(atom_indices.get("expected_fit_atoms", [])),
            "present_fit_atoms": list(atom_indices.get("present_fit_atoms", [])),
            "missing_fit_atoms": list(atom_indices.get("missing_fit_atoms", [])),
            "reference_fit_atoms": list(atom_indices.get("reference_atom_names", [])),
            "reference_present_atoms": list(atom_indices.get("reference_present_atoms", [])),
            "reference_missing_atoms": list(atom_indices.get("reference_missing_atoms", [])),
            "ignored_base_atoms": list(atom_indices.get("ignored_base_atoms", [])),
        }

    def _handle_isolation(self, strand, level, i1, ctx):
        """Fortran-compatible implementation."""
        if ctx.li[level + 1, strand] != -2:
            ctx.li[level + 1, strand] = -2
            print(f".... no base for residue {level+1} in strand {strand+1}") # [cite: 4, 12]
        ctx.backbone.atom_map[strand, level + 1, 2] = i1

    def lsfit(self, lequ, n, nind, ctx: 'CurvesContext'):
        """Fortran-compatible implementation."""
        mol = ctx.molecule
        bref = ctx.params.bref
        sq2 = np.sqrt(2.0)

        valid_indices = []
        valid_bref = []
        for l in range(n):
            if nind[l] != -1:
                valid_indices.append(nind[l])
                valid_bref.append(bref[lequ - 1, l])

        if len(valid_indices) < 3:
            return None, 0, 1

        fit_indices = valid_indices
        fit_coords = mol.coordinates[fit_indices]
        valid_bref = np.array(valid_bref)

        cg = np.mean(fit_coords, axis=0)

        u = np.zeros((3, 3))
        for l in range(len(fit_indices)):
            idx = fit_indices[l]
            standard_pos = valid_bref[l]
            real_pos_shifted = mol.coordinates[idx] - cg
            u += np.outer(standard_pos, real_pos_shifted) / len(fit_indices)
        det = np.linalg.det(u)
        if abs(det) < 1e-9: return None, 0, 1

        w_mat = np.zeros((6, 6))
        w_mat[0:3, 3:6] = u
        w_mat[3:6, 0:3] = u.T
        eig_vals, eig_vecs = np.linalg.eigh(w_mat)
        order = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]

        if det < 0.0 and abs(eig_vals[2] - eig_vals[5]) < 1e-6:
            return None, 0, 1
        
        h = sq2 * eig_vecs[0:3, 0:3].copy()
        k = sq2 * eig_vecs[3:6, 0:3].copy()

        sn = np.linalg.det(h)
        if sn < 0:
            h[:, 2] *= -1
            k[:, 2] *= -1

        rot_u = k @ np.diag([1.0, 1.0, np.sign(det)]) @ h.T

        dcor = (bref[lequ - 1, :n] @ rot_u.T) + cg
        rms = np.sqrt(np.mean(np.sum((fit_coords - dcor)**2, axis=1)))
        return dcor, rms, 0
    
    def _calculate_base_frame(self, strand: int, level: int, atom_data: dict, ctx: 'CurvesContext'):
        """
        """
        if getattr(ctx.cfg, "frame_convention", "legacy") == "standard" and ctx.cfg.fit:
            result = self._calculate_standard_base_frame(strand, level, atom_data, ctx)
            if result is not None:
                return result

        mol = ctx.molecule
        nind = atom_data['nind']
        
        res_name = mol.residue_names[nind[0]].strip().upper()
        lsav = self._get_base_type_index(res_name)
        lequ = self.const.iequ[lsav - 1]
        purine = lsav <= 4

        rms = 0.0
        if ctx.cfg.fit:
            dcor, rms, key = self.lsfit(lequ, len(nind), nind, ctx)
            
            if key == 0:
                c1 = dcor[0] # i1 = 1 (C1') [cite: 15]
                if purine:
                    # i2 = 10 (N9), i3 = 5 (C4) [cite: 15]
                    c0, c3 = dcor[9], dcor[4] 
                else:
                    if lsav < 9:
                        # i2 = 2 (N1), i3 = 3 (C2) [cite: 15]
                        c0, c3 = dcor[1], dcor[2]
                    else:
                        # i2 = 6 (C5), i3 = 5 (C4) [cite: 16]
                        c0, c3 = dcor[5], dcor[4]
            else:
                c0, c1, c3 = self._get_raw_coords(atom_data, mol)
        else:
            c0, c1, c3 = self._get_raw_coords(atom_data, mol)

        # ax = corm(i1,1)-x0, cx = corm(i3,1)-x0
        ax = c1 - c0
        cx = c3 - c0

        # rx=ay*cz-az*cy, ry=az*cx-ax*cz, rz=ax*cy-ay*cx
        rx = ax[1]*cx[2] - ax[2]*cx[1]
        ry = ax[2]*cx[0] - ax[0]*cx[2]
        rz = ax[0]*cx[1] - ax[1]*cx[0]
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        z_axis = np.array([rx, ry, rz]) / r

        ra = np.linalg.norm(ax)
        ca = np.cos(self.const.cdr * self.const.th1)
        sa = np.sin(self.const.cdr * self.const.th1)
        fac = self.const.dis / ra
        
        xx, yy, zz = ax * fac
        origin = self._rotate_vector_aligned([xx, yy, zz], z_axis, ca, sa) + c0

        cb = np.cos(self.const.cdr * self.const.th2)
        sb = np.sin(self.const.cdr * self.const.th2)
        xx_u, yy_u, zz_u = ax / ra
        y_axis = self._rotate_vector_aligned([xx_u, yy_u, zz_u], z_axis, cb, sb)

        x_axis = np.cross(y_axis, z_axis)

        lvl = level + 1
        ctx.params.frames[strand, lvl, 0] = x_axis
        ctx.params.frames[strand, lvl, 1] = y_axis
        ctx.params.frames[strand, lvl, 2] = z_axis
        ctx.params.frames[strand, lvl, 3] = origin

        return rms

    def _calculate_standard_base_frame(self, strand: int, level: int, atom_data: dict, ctx: 'CurvesContext'):
        """Fit the Curves+ template, then build the standard base frame.

        Curves+ `locate.f` does not use the fitted standard-base coordinate
        axes directly.  After `lsfit` it reconstructs the reference point and
        axes from the fitted C1*/glycosidic/normal-defining atoms using the
        standard constants th1=141.47, th2=-54.41, dis=4.7024.
        """
        template = atom_data.get("reference_template")
        if template is None:
            return None

        fit = self.frame_fitter.fit(
            template,
            atom_data.get("res_atoms", {}),
            ctx.molecule.coordinates,
            atom_order=atom_data.get("reference_atom_names"),
        )
        if fit is None:
            return None

        fitted_by_atom = fit.get("fitted_by_atom", {})
        try:
            c1 = np.asarray(fitted_by_atom[template.atom_names[0]], dtype=float)
            c0 = np.asarray(fitted_by_atom[template.atom_names[1]], dtype=float)
            c3 = np.asarray(fitted_by_atom[template.atom_names[2]], dtype=float)
        except KeyError:
            return None

        glycosidic_axis = c1 - c0
        glycosidic_axis = glycosidic_axis / (np.linalg.norm(glycosidic_axis) + 1e-12)
        normal_seed = c3 - c0
        z_axis = np.cross(glycosidic_axis, normal_seed)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

        standard_th1 = 141.47
        standard_th2 = -54.41
        standard_dis = 4.7024
        ca = np.cos(self.const.cdr * standard_th1)
        sa = np.sin(self.const.cdr * standard_th1)
        origin = self._rotate_vector_aligned(glycosidic_axis * standard_dis, z_axis, ca, sa) + c0

        cb = np.cos(self.const.cdr * standard_th2)
        sb = np.sin(self.const.cdr * standard_th2)
        y_axis = self._rotate_vector_aligned(glycosidic_axis, z_axis, cb, sb)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)

        lvl = level + 1
        ctx.params.frames[strand, lvl, 0] = x_axis
        ctx.params.frames[strand, lvl, 1] = y_axis
        ctx.params.frames[strand, lvl, 2] = z_axis
        ctx.params.frames[strand, lvl, 3] = origin

        for key, atom_name in (("i1", "C1*"), ("i2", template.glycosidic_atom), ("i3", template.normal_atom), ("i4", template.groove_atom)):
            if atom_name in fitted_by_atom:
                atom_data[f"{key}_fitted_coord"] = fitted_by_atom[atom_name]
        atom_data["fit_atom_names"] = fit.get("fit_atom_names", [])
        return float(fit["rmsd"])

    def _get_raw_coords(self, atom_data, mol):
        """Fortran-compatible implementation."""
        c1 = mol.coordinates[atom_data['i1']]
        c0 = mol.coordinates[atom_data['i2']]
        c3 = mol.coordinates[atom_data['i3']]
        return c0, c1, c3

    def _rotate_vector_aligned(self, v, axis, ca, sa):
        """1:1 Port of rotation matrix in locate.f[cite: 66]."""
        rx, ry, rz = axis
        xx, yy, zz = v
        tx = (rx*rx+(1-rx*rx)*ca)*xx+(rx*ry*(1-ca)-rz*sa)*yy+(rx*rz*(1-ca)+ry*sa)*zz
        ty = (rx*ry*(1-ca)+rz*sa)*xx+(ry*ry+(1-ry*ry)*ca)*yy+(ry*rz*(1-ca)-rx*sa)*zz
        tz = (rx*rz*(1-ca)-ry*sa)*xx+(ry*rz*(1-ca)+rx*sa)*yy+(rz*rz+(1-rz*rz)*ca)*zz
        return np.array([tx, ty, tz])
    
@dataclass
class MolecularStructure:
    """
    Equivalent to common/mac and common/cha.
    Stores raw atomic data and file metadata[cite: 1, 28].
    """
    mcode: str = "No Title"
    kam: int = 0  # Total number of atoms (Fortran n1) [cite: 65, 89]
    kcen: int = 0 # Total number of subunits/residues (Fortran n2) [cite: 65, 89]
    
    # Atomic properties
    atom_names: np.ndarray = None  # mnam [cite: 1, 89]
    coordinates: np.ndarray = None # corm (kam, 3) [cite: 1, 89]
    residue_names: np.ndarray = None # munit [cite: 1, 89]
    residue_ids: np.ndarray = None   # nunit [cite: 1, 89]
    chain_ids: np.ndarray = None
    
    # Topology
    atom_types: np.ndarray = None     # imch [cite: 1, 89]
    connectivity: np.ndarray = None   # matd (kam, 7) [cite: 1, 89]
    subunit_boundaries: np.ndarray = None # ncen [cite: 1, 89]
    source_base_pairs: list = None
    crystal_cell: tuple = None
    spacegroup_hm: str = ""

@dataclass
class BackboneTopology:
    """
    Stores atom indices and backbone geometry.
    """
    torsions: np.ndarray        # shape: (nst, n3, 13)
    
    angles: np.ndarray          # shape: (nst, n3, 2)
    
    sugar_pucker: np.ndarray    # shape: (nst, n3, 2)
    
    atom_map: np.ndarray        # shape: (nst, n3, 15)
    
    flag: np.ndarray # shape: (nst, n3)

    dif: np.ndarray

    ATOM_NAMES = [
        'C6C8', 'C2N9', 'C1\'', 'C2\'', 'C3\'', 'C4\'', 'O4\'', 
        'O3\'', 'P', 'O5\'', 'C5\'', 'C4\'', 'C5\'', 'O5\'', 'C2C4'
    ]

@dataclass
class HelicalParameters:
    """
    1:1 Mirror of Curves 5.3 COMMON blocks /vec/ and /der/ .
    Names are kept identical to Fortran source to prevent AttributeErrors.
    """
    # Fortran /dat/ block.
    frames: np.ndarray        # Fortran rex/rey/rez: base frame axes and origin
    shape_frames: np.ndarray  # Shape-parameter frame view, including noncanonical contact frames
    axis_frames: np.ndarray   # Sign-continuous reference frames for the global-axis optimizer
    helical: np.ndarray       # Fortran hel: global base-axis parameters
    inter_base: np.ndarray    # Cached Section E global inter-base step parameters
    
    # Fortran /vec/ block.
    ux: np.ndarray            # Fortran ux/uy/uz: optimized axis direction
    hx: np.ndarray            # Fortran hx/hy/hz: axis point
    sx: np.ndarray            # Fortran sx/sy/sz: axis displacement between levels
    bx: np.ndarray            # Fortran bx/by/bz: vector from base origin to axis point
    ox: np.ndarray            # Fortran ox/oy/oz: optimized axis coordinates
    qr: np.ndarray            # Fortran qr: rotational residual terms
    qp: np.ndarray            # Fortran qp: positional residual terms
    
    # Fortran /der/ block.
    up: np.ndarray            # Fortran upx/upy/upz: axis direction difference
    us: np.ndarray            # Fortran usx/usy/usz: axis direction sum
    um: np.ndarray            # Fortran umx/umy/umz: normalized axis direction sum
    q:  np.ndarray            # Fortran qx/qy/qz: axis path residual

    initial_origins: np.ndarray

    efd: np.ndarray
    efc: np.ndarray


@dataclass
class HelicalConfig:
    """
    1:1 Mirror of Curves 5.3 control parameters[cite: 1, 2].
    Values represent the official Fortran defaults.
    """
    # Fortran common/drl logical flags.
    line: bool = False    # Linear-axis mode.
    comb: bool = False    # Combined-strand analysis.
    fit: bool = False     # Least-squares fit bases to standard geometry.
    grv: bool = False     # Groove analysis.
    mini: bool = True     # Run optimizer.
    ends: bool = False    # Add terminal extension levels.
    supp: bool = True     # Supplemental report sections.
    dinu: bool = False    # Dinucleotide mode.
    rest: bool = False    # Restricted analysis.
    zaxe: bool = False    # Force the global Z axis.
    test: bool = False    # Test/debug mode from Curves input.
    old: bool = True      # Curves 5.3 compatibility mode.
    axonly: bool = False  # Axis-only calculation.
    frame_convention: str = "legacy"  # Base reference-frame convention: legacy or standard.
    axis_convention: str = "legacy"  # Global axis convention: legacy optimizer or curvesplus axis/smooth.

    # Fortran hel(0,*,*) and hel(nux+1,*,*) terminal rows used when ends=.t.
    # Columns are Xdisp, Ydisp, Rise, Inclination, Tip, Twist.
    end_start: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 3.4, 0.0, 0.0, 0.0], dtype=float))
    end_stop: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 3.4, 0.0, 0.0, 0.0], dtype=float))

    # Fortran common/drl integer settings.
    break_lvl: int = -1   # Fortran break: step break level.
    nlevel: int = 3       # Fortran nlevel: groove/output sampling level.
    nbac: int = 7         # Fortran nbac: backbone atom selector for groove analysis.
    spline: int = 3       # Fortran splin/spline: spline order.
    ior: int = 0          # Fortran ior: initial orientation mode.
    ibond: int = 0        # Fortran ibond: bonding/connectivity mode.
    maxn: int = 500       # Fortran maxn: optimizer iteration cap.

    # Fortran common/drl real settings.
    acc: float = 1e-6     # Optimizer convergence tolerance.
    wid: float = 0.75     # Groove sampling width.

    # Fortran xytp input vectors.
    inpv: int = 1
    xdi: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # Initial X displacement.
    ydi: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # Initial Y displacement.
    cln: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # Initial inclination.
    tip: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # Initial tip.

class CurvesContext:
    """
    Runtime state for one Curves analysis.

    The original Curves 5.3 names are kept as compatibility aliases for the
    translated numerical routines.  Readable aliases are initialized beside the
    Fortran names at construction time.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg['config']
        self.nst = int(cfg['n_strands'])
        self.nux = int(cfg['n_levels'])
        self.n_strands = self.nst
        self.n_levels = self.nux
        self.strand_count = self.nst  # Readable alias for Fortran nst.
        self.level_count = self.nux  # Readable alias for Fortran nux.
        n3 = self.nux + 2  # levels 0 and nux+1 are optional terminal extensions
        
        nu_raw = np.array(cfg.get('nu_raw', [self.nux] * self.nst))
        
        self.idr = cfg.get('idr', np.sign(nu_raw).astype(int))
        self.nu = cfg.get('nu', np.abs(nu_raw).astype(int))
        self.nt = cfg.get('nt', np.sum(self.nu))
        self.strand_directions = self.idr  # Readable alias for Fortran idr.
        self.strand_lengths = self.nu  # Readable alias for Fortran nu.
        self.total_nucleotides = self.nt  # Readable alias for Fortran nt.
        
        self.ng = cfg.get('ng', np.zeros(self.nst, dtype=int))
        self.nr = cfg.get('nr', np.zeros(self.nst, dtype=int))
        self.ni_map = np.array(cfg['ni_map'])   # Fortran ni: subunit index per strand/level.
        self.hoogsteen_markers = set(cfg.get('hoogsteen_markers', set()) or set())
        self.pair_geometry_markers = dict(cfg.get('pair_geometry_markers', {}) or {})
        self.active_start_levels = self.ng  # Readable alias for Fortran ng.
        self.active_end_levels = self.nr  # Readable alias for Fortran nr.
        self.subunit_map = self.ni_map  # Readable alias for Fortran ni.
        self._calculate_active_range()

        self.iact = np.zeros(n3, dtype=int)
        self.li = np.zeros((n3, self.nst), dtype=int)
        self.active_strand_by_level = self.iact  # Readable alias for Fortran iact.
        self.level_status = self.li  # Readable alias for Fortran li.
        self._initialize_li_status()
        self.level_status = self.li

        
        self.molecule = MolecularStructure()
        self.backbone = BackboneTopology(
            # 999.0 is the Curves sentinel for missing torsion/angle values.
            torsions = np.full((self.n_strands, n3, 13), 999.0),
            
            angles = np.full((self.n_strands, n3, 2), 999.0),
            
            sugar_pucker = np.zeros((self.n_strands, n3, 2)),
            
            # Atom indices are Python 0-based; -1 means "not found".
            atom_map = np.full((self.n_strands, n3, 15), -1, dtype=int),
            
            flag = np.zeros((self.n_strands, n3), dtype=bool),
            
            dif = np.zeros(n3) 
        )
        self.backbone.atom_map.fill(-1)

        self.params = HelicalParameters(
            frames = np.zeros((self.n_strands, n3, 4, 3)),
            shape_frames = np.zeros((self.n_strands, n3, 4, 3)),
            axis_frames = np.zeros((self.n_strands, n3, 4, 3)),
            helical = np.zeros((self.n_strands, n3, 6)),
            inter_base = np.zeros((self.n_strands, n3, 6)),
            
            ux = np.zeros((n3, 3)),
            hx = np.zeros((n3, 3)),
            sx = np.zeros((n3, 3)),
            bx = np.zeros((n3, self.n_strands, 3)),
            ox = np.zeros((n3, 3)),
            qr = np.zeros((n3, 3, self.n_strands)),
            qp = np.zeros((n3, 3, self.n_strands)),
            
            up = np.zeros((n3, 3)),
            us = np.zeros((n3, 3)),
            um = np.zeros((n3, 3)),
            q  = np.zeros((n3, 3)),
            
            initial_origins = np.zeros((n3, 3)),
            efd = np.zeros((3, 2, self.nst), dtype=float),
            efc = np.zeros((3, 2, self.nst), dtype=float)
        )

        self.lsf_const = BaseGeometryConstants()
        
        self.params.bref = get_standard_bref()
        
        self.iequ = self.lsf_const.iequ
        self.annotations = {
            # pyCurves-native provenance added on top of the Curves 5.3 arrays.
            "base_fit_quality": [],
        }

    def _calculate_active_range(self):
        """Populate Fortran ng/nr active level ranges from ni_map."""
        for k in range(self.nst):
            for j in range(self.nux):
                if self.ni_map[k, j] > 0:
                    self.nr[k] = j + 1
                    if self.ng[k] == 0:
                        self.ng[k] = j + 1

    def _initialize_li_status(self):
        """Initialize Fortran li level status: 1 active, -1 missing/isolated."""
        self.li = np.zeros((self.nux + 2, self.nst), dtype=int)

        for k in range(self.nst):
            for j in range(self.nux):
                self.li[j + 1, k] = 1 if self.ni_map[k, j] > 0 else -1

        if self.cfg.ends:
            self.li[0, :] = 1
            self.li[self.nux + 1, :] = 1
