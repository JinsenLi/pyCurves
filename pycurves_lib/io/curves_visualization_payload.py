from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pycurves_lib.data.modified_bases import parent_base_name


def _all_finite(values) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=float))))


class VisualizationPayloadMixin:
    def _visualization_payload(self, records: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Geometry overlays for external viewers.

        The core result tables intentionally stay analysis-oriented.  This
        payload stores display-ready coordinates so a viewer can render the
        optimized helical axis, backbone traces, and base-pair connectors
        without re-running pyCurves.
        """
        ctx = self.runner.ctx
        calc = self.runner.calc
        p = ctx.params
        mol = ctx.molecule
        annotations = self._annotations()

        def point_payload(coords):
            return {
                "x": float(coords[0]),
                "y": float(coords[1]),
                "z": float(coords[2]),
            }

        def vector_payload(coords):
            return point_payload(coords)

        def residue_payload(strand: int, level: int) -> Optional[Dict[str, Any]]:
            if level < 1 or level > ctx.nux:
                return None
            if strand < 0 or strand >= ctx.nst:
                return None
            subunit = int(ctx.ni_map[strand, level - 1])
            if subunit <= 0:
                return None
            atom_idx = int(mol.subunit_boundaries[subunit - 1])
            return {
                "strand": strand + 1,
                "level": level,
                "subunit": subunit,
                "residue_name": str(mol.residue_names[atom_idx]).strip(),
                "parent_base": parent_base_name(str(mol.residue_names[atom_idx]).strip()),
                "residue_id": int(mol.residue_ids[atom_idx]),
                "chain_id": str(mol.chain_ids[atom_idx]).strip() if mol.chain_ids is not None else "",
            }

        base_excluded_atoms = {
            "P", "OP1", "OP2", "O1P", "O2P", "O3P", "O5'", "O5*", "C5'", "C5*",
            "C4'", "C4*", "O4'", "O4*", "C3'", "C3*", "O3'", "O3*", "C2'",
            "C2*", "O2'", "O2*", "C1'", "C1*",
        }

        def clean_atom_name(atom_name) -> str:
            return str(atom_name).strip().upper()

        hbond_edge_atoms = {
            "A": {"N1", "N6"},
            "C": {"N3", "N4", "O2"},
            "G": {"N1", "N2", "O6"},
            "T": {"N3", "O2", "O4"},
            "U": {"N3", "O2", "O4"},
            "I": {"N1", "O6"},
        }
        hoogsteen_edge_atoms = {
            **hbond_edge_atoms,
            "A": {"N6", "N7"},
            "G": {"O6", "N7"},
            "I": {"O6", "N7"},
        }

        def base_heavy_atoms(subunit: int):
            start = int(mol.subunit_boundaries[subunit - 1])
            end = int(mol.subunit_boundaries[subunit])
            atoms = []
            for atom_idx in range(start, end):
                atom_name = clean_atom_name(mol.atom_names[atom_idx])
                if not atom_name or atom_name in base_excluded_atoms:
                    continue
                if atom_name.startswith(("H", "D")):
                    continue
                atoms.append({
                    "name": atom_name,
                    "coords": np.asarray(mol.coordinates[atom_idx], dtype=float),
                })
            return atoms

        def residue_atom(subunit: int, preferred_names):
            start = int(mol.subunit_boundaries[subunit - 1])
            end = int(mol.subunit_boundaries[subunit])
            atoms = {}
            for atom_idx in range(start, end):
                atoms[clean_atom_name(mol.atom_names[atom_idx])] = atom_idx
            for atom_name in preferred_names:
                if atom_name in atoms:
                    return atoms[atom_name], atom_name
            return -1, ""

        def base_plate_geometry(base_atoms, edge_names, fallback_center, fallback_x, fallback_y, fallback_z, is_purine: bool):
            def safe_unit(vec, fallback):
                vec = np.asarray(vec, dtype=float)
                if _all_finite(vec):
                    norm = np.linalg.norm(vec)
                    if norm > 1e-8:
                        return vec / norm
                return np.asarray(fallback, dtype=float)

            frame_z = safe_unit(fallback_z, [0.0, 0.0, 1.0])
            frame_y_seed = np.asarray(fallback_y, dtype=float)
            if not _all_finite(frame_y_seed):
                frame_y_seed = np.array([0.0, 1.0, 0.0], dtype=float)
            frame_y = frame_y_seed - frame_z * np.dot(frame_y_seed, frame_z)
            if np.linalg.norm(frame_y) <= 1e-8:
                frame_y = np.array([1.0, 0.0, 0.0], dtype=float) - frame_z * frame_z[0]
            frame_y = safe_unit(frame_y, [0.0, 1.0, 0.0])
            frame_x = safe_unit(fallback_x, np.cross(frame_y, frame_z))
            frame_x = frame_x - frame_z * np.dot(frame_x, frame_z)
            frame_x = safe_unit(frame_x, np.cross(frame_y, frame_z))

            if len(base_atoms) >= 3:
                atom_coords = np.asarray([atom["coords"] for atom in base_atoms], dtype=float)
                center = np.mean(atom_coords, axis=0)
                centered = atom_coords - center
                edge_coords = [atom["coords"] for atom in base_atoms if atom["name"] in edge_names]
                edge_center = np.mean(np.asarray(edge_coords, dtype=float), axis=0) if edge_coords else center + frame_y
                hbond_axis = edge_center - center
                hbond_axis = hbond_axis - frame_z * np.dot(hbond_axis, frame_z)
                if np.linalg.norm(hbond_axis) < 1e-8:
                    hbond_axis = frame_y
                hbond_axis = hbond_axis / np.linalg.norm(hbond_axis)

                candidates = [frame_x, -frame_x, frame_y, -frame_y]
                x_axis = max(candidates, key=lambda axis: float(np.dot(axis, hbond_axis)))
                y_axis = np.cross(frame_z, x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)
                z_axis = frame_z

                projections_x = centered @ x_axis
                projections_y = centered @ y_axis
                min_length = 5.7 if is_purine else 5.0
                min_width = 3.2 if is_purine else 2.8
                max_width = 3.9 if is_purine else 3.4
                length = max(min_length, float(np.ptp(projections_x) + 1.3))
                width = min(max_width, max(min_width, float(np.ptp(projections_y) + 0.8)))
                return center, x_axis, y_axis, z_axis, length, width, point_payload(edge_center), sorted(edge_names)

            return (
                fallback_center,
                frame_y,
                frame_x,
                frame_z,
                5.7 if is_purine else 5.0,
                3.2 if is_purine else 2.8,
                point_payload(fallback_center + frame_y),
                sorted(edge_names),
            )

        axis = []
        if ctx.cfg.comb:
            _, _, axis_start, axis_end = calc._axis_bounds(0)
            for level in range(axis_start, axis_end + 1):
                coords = np.asarray(p.ox[level], dtype=float)
                direction = np.asarray(p.ux[level], dtype=float)
                if not _all_finite(coords) or not _all_finite(direction):
                    continue
                if not np.any(coords):
                    continue
                axis.append({
                    "level": level,
                    "strand": 0,
                    "axis_scope": "combined",
                    **point_payload(coords),
                    "direction": point_payload(direction),
                })
        else:
            for strand in range(ctx.nst):
                _, _, axis_start, axis_end = calc._axis_bounds(strand)
                for level in range(axis_start, axis_end + 1):
                    coords = np.asarray(calc.optimizer.hho[level, :, strand], dtype=float)
                    direction = np.asarray(calc.optimizer.uho[level, :, strand], dtype=float)
                    if not _all_finite(coords) or not _all_finite(direction):
                        continue
                    if not np.any(coords):
                        continue
                    axis.append({
                        "level": level,
                        "strand": strand + 1,
                        "axis_scope": "strand",
                        **point_payload(coords),
                        "direction": point_payload(direction),
                    })

        base_origins_by_key = {}
        base_origins = []
        for strand in range(ctx.nst):
            for level in range(1, ctx.nux + 1):
                if ctx.ni_map[strand, level - 1] <= 0:
                    continue
                coords = np.asarray(p.frames[strand, level, 3, :], dtype=float)
                if not _all_finite(coords):
                    continue
                residue = residue_payload(strand, level)
                if residue is None:
                    continue
                base_atoms = base_heavy_atoms(residue["subunit"])
                frame_origin = np.asarray(coords, dtype=float)
                frame_x = np.asarray(p.frames[strand, level, 0, :], dtype=float)
                frame_y = np.asarray(p.frames[strand, level, 1, :], dtype=float)
                frame_z = np.asarray(p.frames[strand, level, 2, :], dtype=float)
                is_purine = residue["parent_base"] in {"A", "G", "I"}
                (
                    display_center,
                    plate_x,
                    plate_y,
                    plate_z,
                    plate_length,
                    plate_width,
                    hbond_center,
                    hbond_atoms,
                ) = base_plate_geometry(
                    base_atoms,
                    hbond_edge_atoms.get(residue["parent_base"], set()),
                    frame_origin,
                    frame_x,
                    frame_y,
                    frame_z,
                    is_purine,
                )
                (
                    _,
                    hoogsteen_plate_x,
                    hoogsteen_plate_y,
                    hoogsteen_plate_z,
                    hoogsteen_plate_length,
                    hoogsteen_plate_width,
                    hoogsteen_hbond_center,
                    hoogsteen_hbond_atoms,
                ) = base_plate_geometry(
                    base_atoms,
                    hoogsteen_edge_atoms.get(residue["parent_base"], hbond_edge_atoms.get(residue["parent_base"], set())),
                    frame_origin,
                    frame_x,
                    frame_y,
                    frame_z,
                    is_purine,
                )
                entry = {
                    **residue,
                    **point_payload(display_center),
                    "frame_origin": point_payload(frame_origin),
                    "x_axis": vector_payload(frame_x),
                    "y_axis": vector_payload(frame_y),
                    "z_axis": vector_payload(frame_z),
                    "plate_x_axis": vector_payload(plate_x),
                    "plate_y_axis": vector_payload(plate_y),
                    "plate_z_axis": vector_payload(plate_z),
                    "plate_length": plate_length,
                    "plate_width": plate_width,
                    "hbond_edge_center": hbond_center,
                    "hbond_edge_atoms": hbond_atoms,
                    "hoogsteen_plate_x_axis": vector_payload(hoogsteen_plate_x),
                    "hoogsteen_plate_y_axis": vector_payload(hoogsteen_plate_y),
                    "hoogsteen_plate_z_axis": vector_payload(hoogsteen_plate_z),
                    "hoogsteen_plate_length": hoogsteen_plate_length,
                    "hoogsteen_plate_width": hoogsteen_plate_width,
                    "hoogsteen_hbond_edge_center": hoogsteen_hbond_center,
                    "hoogsteen_hbond_edge_atoms": hoogsteen_hbond_atoms,
                }
                base_origins.append(entry)
                base_origins_by_key[(strand + 1, level)] = entry

        backbones = []
        groove_splines = {
            int(item.get("strand", 0)): item
            for item in getattr(calc, "groove_backbone_splines", []) or []
        }
        for strand in range(ctx.nst):
            points = []
            for level in range(1, ctx.nux + 1):
                if ctx.ni_map[strand, level - 1] <= 0:
                    continue
                subunit = int(ctx.ni_map[strand, level - 1])
                atom_idx, atom_name = residue_atom(
                    subunit,
                    ["P", "O5'", "O5*", "C5'", "C5*", "C4'", "C4*", "C3'", "C3*", "O3'", "O3*", "C1'", "C1*"],
                )
                if atom_idx < 0:
                    continue
                coords = np.asarray(mol.coordinates[atom_idx], dtype=float)
                if not _all_finite(coords):
                    continue
                residue = residue_payload(strand, level)
                if residue is None:
                    continue
                points.append({
                    **residue,
                    "atom_name": atom_name,
                    **point_payload(coords),
                })
            backbone_record = {
                "strand": strand + 1,
                "points": points,
            }
            if strand + 1 in groove_splines:
                backbone_record.update(groove_splines[strand + 1])
            backbones.append(backbone_record)

        base_pairs = []
        for pair in annotations.get("base_pair_annotations", []):
            level = int(pair.get("level", 0) or 0)
            strand_1 = int(pair.get("strand_1", 1) or 1)
            strand_2 = int(pair.get("strand_2", 2) or 2)
            first = base_origins_by_key.get((strand_1, level))
            second = base_origins_by_key.get((strand_2, level))
            if not first or not second:
                continue
            first_xyz = np.array([first["x"], first["y"], first["z"]], dtype=float)
            second_xyz = np.array([second["x"], second["y"], second["z"]], dtype=float)
            midpoint = (first_xyz + second_xyz) / 2.0
            base_pairs.append({
                "level": level,
                "first": first,
                "second": second,
                "midpoint": point_payload(midpoint),
                "pair_family": pair.get("pair_family", ""),
                "pair_subtype": pair.get("pair_subtype", ""),
                "is_canonical": bool(pair.get("is_canonical", False)),
                "is_hoogsteen": bool(pair.get("is_hoogsteen", False)),
                "is_mismatch": bool(pair.get("is_mismatch", False)),
                "has_modified_base": bool(pair.get("has_modified_base", False)),
                "shape_parameters_supported": bool(pair.get("shape_parameters_supported", True)),
                "label": f"{pair.get('residue_1', '')} - {pair.get('residue_2', '')}".strip(" -"),
            })

        def groove_parameter_rows():
            groove = (records or {}).get("groove", {})
            rows = []
            for level_text, level_data in groove.get("data", {}).items():
                for sub_level_text, values in level_data.get("sub_levels", {}).items():
                    row = {
                        "level": int(level_text),
                        "sub_level": int(sub_level_text),
                        "base_pair": level_data.get("base_pair", ""),
                        "minor_width": values.get("minor_width"),
                        "minor_depth": values.get("minor_depth"),
                        "minor_angle": values.get("minor_angle"),
                        "major_width": values.get("major_width"),
                        "major_depth": values.get("major_depth"),
                        "major_angle": values.get("major_angle"),
                        "diameter": values.get("diameter"),
                        "geometry": values.get("geometry", {}),
                    }
                    rows.append(row)
            return rows

        result_records = records or {}
        global_base_pair_axis = result_records.get("global_base_pair_axis", [])
        curvesplus_base_pair_axis = result_records.get("curvesplus_base_pair_axis", [])
        display_base_pair_axis = global_base_pair_axis or curvesplus_base_pair_axis
        global_inter_base_pair = result_records.get("global_inter_base_pair", [])
        local_inter_base_pair = result_records.get("local_inter_base_pair", [])
        global_inter_base = result_records.get("global_inter_base", [])
        local_inter_base = result_records.get("local_inter_base", [])

        return {
            "schema": "pycurves-visualization-v10",
            "axis_mode": "combined" if ctx.cfg.comb else "per_strand",
            "analyzed_residue_names": sorted({
                item["residue_name"] for item in base_origins if item.get("residue_name")
            }),
            "structure_chains": sorted({
                str(chain).strip() for chain in np.asarray(mol.chain_ids if mol.chain_ids is not None else [])
                if str(chain).strip()
            }),
            "structure_residue_names": sorted({
                str(residue).strip() for residue in np.asarray(mol.residue_names if mol.residue_names is not None else [])
                if str(residue).strip()
            }),
            "axis": axis,
            "base_origins": base_origins,
            "backbones": backbones,
            "base_pairs": base_pairs,
            "parameters": {
                "global_base_base": result_records.get("global_base_base", []),
                "local_base_base": result_records.get("local_base_base", []),
                "global_base_pair_axis": global_base_pair_axis,
                "curvesplus_base_pair_axis": curvesplus_base_pair_axis,
                "global_base_axis": result_records.get("global_base_axis", []),
                "global_inter_base_pair": global_inter_base_pair,
                "local_inter_base_pair": local_inter_base_pair,
                "global_inter_base": global_inter_base,
                "local_inter_base": local_inter_base,
                "base_pair": result_records.get("global_base_base", []),
                "local_base_pair": result_records.get("local_base_base", []),
                "base_pair_axis": display_base_pair_axis,
                "base_axis": result_records.get("global_base_axis", []),
                "global_step": global_inter_base_pair if ctx.cfg.comb else global_inter_base,
                "local_step": local_inter_base_pair if ctx.cfg.comb else local_inter_base,
                "global_strand_step": global_inter_base,
                "local_strand_step": local_inter_base,
                "groove": groove_parameter_rows(),
            },
        }

