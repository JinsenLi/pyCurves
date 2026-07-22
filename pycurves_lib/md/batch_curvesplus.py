from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from pycurves_lib.core.curves_analyzer import BackboneAnalyzer
from pycurves_lib.core.curves_dataclasses import (
    BaseGeometryConstants,
    BaseLocator,
    CurvesContext,
    MolecularStructure,
)
from pycurves_lib.data.modified_bases import parent_base_name
from pycurves_lib.io.base_reference import (
    BaseReferenceLibrary,
    atom_aliases,
)
from pycurves_lib.io.curves_config_loader import ConfigLoader
from pycurves_lib.io.curves_mol_loader import MolecularLoader
from pycurves_lib.curves_wrapper import CurvesWrapper
from pycurves_lib.md.batch_groove import compute_batch_grooves
from pycurves_lib.md.trajectory_statistics import (
    ALPHA_GAMMA_STATES,
    BI_BII_STATES,
    SUGAR_PUCKERS as SUMMARY_SUGAR_PUCKERS,
    alpha_gamma_counts,
    bi_bii_counts,
    sugar_pucker_counts,
)


STEP_PARAMETERS = ("shift", "slide", "rise", "tilt", "roll", "twist")
BASE_BASE_PARAMETERS = ("shear", "stretch", "stagger", "buckle", "propel", "opening")
AXIS_PARAMETERS = ("xdisp", "ydisp", "inclin", "tip")
SUGAR_PUCKERS = (
    "C3'-endo", "C4'-exo", "O1'-endo", "C1'-exo", "C2'-endo",
    "C3'-exo", "C4'-endo", "O1'-exo", "C1'-endo", "C2'-exo",
)

SIGN_FLIPS = np.asarray(
    [
        np.diag([1.0, 1.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ],
    dtype=float,
)


@dataclass(frozen=True)
class BaseFitTemplate:
    strand: int
    level: int
    subunit: int
    residue_name: str
    parent_base: str
    atom_indices: np.ndarray
    reference_coords: np.ndarray
    template_coords: np.ndarray
    template_atom_names: Tuple[str, ...]
    c1_template_index: int
    glycosidic_template_index: int
    normal_template_index: int


@dataclass(frozen=True)
class BaseFitTemplateGroup:
    strands: np.ndarray
    levels: np.ndarray
    atom_indices: np.ndarray
    reference_shifted: np.ndarray
    template_key_shifted: np.ndarray
    ref_centroids: np.ndarray


@dataclass(frozen=True)
class BackboneTaskSet:
    torsion_strands: np.ndarray
    torsion_levels: np.ndarray
    torsion_indices: np.ndarray
    torsion_atoms: np.ndarray
    angle_strands: np.ndarray
    angle_levels: np.ndarray
    angle_indices: np.ndarray
    angle_atoms: np.ndarray
    sugar_strands: np.ndarray
    sugar_levels: np.ndarray


def _unit(values: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    out = np.divide(arr, norm, out=np.zeros_like(arr, dtype=float), where=norm > 1e-12)
    if fallback is not None:
        fallback_arr = np.asarray(fallback, dtype=float)
        mask = np.squeeze(norm <= 1e-12, axis=-1)
        out[mask] = fallback_arr[mask] if fallback_arr.shape == out.shape else fallback_arr
    return out


def _wrap_180(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    mask = np.abs(values) > 180.0
    values[mask] -= np.sign(values[mask]) * 360.0
    return values


def _rotate_axis_angle_batch(vectors: np.ndarray, axes: np.ndarray, ca: np.ndarray, sa: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    ca = np.asarray(ca, dtype=float)[..., None]
    sa = np.asarray(sa, dtype=float)[..., None]
    return vectors * ca + np.cross(axes, vectors) * sa + axes * np.sum(axes * vectors, axis=-1, keepdims=True) * (1.0 - ca)


def _angle_batch(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    v1 = p2 - p1
    v2 = p3 - p2
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    denom = n1 * n2
    values = np.full(denom.shape, np.nan, dtype=float)
    valid = denom > 1e-12
    if np.any(valid):
        c = -np.sum(v1[valid] * v2[valid], axis=-1) / denom[valid]
        values[valid] = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    return values


def _torsion_batch(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    u1 = np.cross(v1, v2)
    u2 = np.empty_like(u1)
    u2[..., 0] = v3[..., 2] * v2[..., 1] - v3[..., 1] * v2[..., 2]
    u2[..., 1] = v3[..., 0] * v2[..., 2] - v3[..., 2] * v2[..., 0]
    u2[..., 2] = v3[..., 1] * v2[..., 0] - v3[..., 0] * v2[..., 1]
    n1 = np.linalg.norm(u1, axis=-1)
    n2 = np.linalg.norm(u2, axis=-1)
    denom = n1 * n2
    values = np.full(denom.shape, np.nan, dtype=float)
    valid = denom > 1e-12
    if np.any(valid):
        c = np.sum(u1[valid] * u2[valid], axis=-1) / denom[valid]
        angle = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        triple = np.sum(u1[valid] * np.cross(u2[valid], v2[valid]), axis=-1)
        angle[triple < 0.0] *= -1.0
        values[valid] = angle
    return values


def _rotation_matrix_first_to_second(first_axes: np.ndarray, second_axes: np.ndarray) -> np.ndarray:
    # Row-vector axes are rotated by v @ R.T.  Solve first_axes @ R.T = second_axes.
    return np.einsum("...ji,...jk->...ik", second_axes, first_axes)


def _orthonormalize_axes_batch(axes: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=float)
    x_axis = _unit(axes[..., 0, :])
    y_axis = axes[..., 1, :] - x_axis * np.sum(x_axis * axes[..., 1, :], axis=-1, keepdims=True)
    y_axis = _unit(y_axis)
    z_axis = _unit(np.cross(x_axis, y_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=-2)


def _relative_rotation_and_middle_axes(first_axes: np.ndarray, second_axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix = _rotation_matrix_first_to_second(first_axes, second_axes)
    rotvec = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_rotvec().reshape(matrix.shape[:-2] + (3,))
    half_matrix = Rotation.from_rotvec((0.5 * rotvec).reshape(-1, 3)).as_matrix().reshape(matrix.shape)
    middle_axes = np.einsum("...ki,...ji->...kj", first_axes, half_matrix)
    middle_axes = _orthonormalize_axes_batch(middle_axes)
    return rotvec, middle_axes


def _middle_frame_batch(
    first_axes: np.ndarray,
    first_origin: np.ndarray,
    second_axes: np.ndarray,
    second_origin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    _, middle_axes = _relative_rotation_and_middle_axes(first_axes, second_axes)
    return middle_axes, (first_origin + second_origin) / 2.0


def _rigid_body_values_from_middle_axes(
    first_origin: np.ndarray,
    second_origin: np.ndarray,
    rotvec: np.ndarray,
    middle_axes: np.ndarray,
    degrees_per_radian: float,
    translation_sign: float,
    rotation_sign: float,
) -> np.ndarray:
    translation = translation_sign * (second_origin - first_origin)
    displacement = np.einsum("...ij,...j->...i", middle_axes, translation)
    scaled_rotvec = rotation_sign * rotvec * degrees_per_radian
    angles = np.einsum("...ij,...j->...i", middle_axes, scaled_rotvec)
    return np.concatenate([displacement, angles], axis=-1)


def _rigid_body_values_batch(
    first_axes: np.ndarray,
    first_origin: np.ndarray,
    second_axes: np.ndarray,
    second_origin: np.ndarray,
    degrees_per_radian: float,
    translation_sign: float,
    rotation_sign: float,
) -> np.ndarray:
    rotvec, middle_axes = _relative_rotation_and_middle_axes(first_axes, second_axes)
    return _rigid_body_values_from_middle_axes(
        first_origin,
        second_origin,
        rotvec,
        middle_axes,
        degrees_per_radian,
        translation_sign,
        rotation_sign,
    )


def _rigid_body_values_and_middle_frame_batch(
    first_axes: np.ndarray,
    first_origin: np.ndarray,
    second_axes: np.ndarray,
    second_origin: np.ndarray,
    degrees_per_radian: float,
    translation_sign: float,
    rotation_sign: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotvec, middle_axes = _relative_rotation_and_middle_axes(first_axes, second_axes)
    values = _rigid_body_values_from_middle_axes(
        first_origin,
        second_origin,
        rotvec,
        middle_axes,
        degrees_per_radian,
        translation_sign,
        rotation_sign,
    )
    return values, middle_axes, (first_origin + second_origin) / 2.0


def _step_aligned_values_batch(
    previous_axes: np.ndarray,
    previous_origin: np.ndarray,
    current_axes: np.ndarray,
    current_origin: np.ndarray,
    degrees_per_radian: float,
) -> np.ndarray:
    batch = previous_axes.shape[0]
    best_score = np.full(batch, -np.inf, dtype=float)
    best_forward = np.zeros(batch, dtype=bool)
    best_prev_x = np.full(batch, -np.inf, dtype=float)
    best_curr_x = np.full(batch, -np.inf, dtype=float)
    best_rise = np.full(batch, -np.inf, dtype=float)
    best_values = np.full((batch, 6), np.nan, dtype=float)

    previous_candidates = np.einsum("fij,bjk->fbik", SIGN_FLIPS, previous_axes)
    current_candidates = np.einsum("fij,bjk->fbik", SIGN_FLIPS, current_axes)
    p_axes = np.repeat(previous_candidates, len(SIGN_FLIPS), axis=0)
    c_axes = np.tile(current_candidates, (len(SIGN_FLIPS), 1, 1, 1))

    candidate_scores = np.einsum("cbij,cbij->cb", p_axes, c_axes)
    max_score = np.max(candidate_scores, axis=0)
    candidate_prev_x = np.einsum("cbi,bi->cb", p_axes[:, :, 0, :], previous_axes[:, 0, :])
    candidate_curr_x = np.einsum("cbi,bi->cb", c_axes[:, :, 0, :], current_axes[:, 0, :])

    for candidate_index in range(p_axes.shape[0]):
        score = candidate_scores[candidate_index]
        score_tied = np.abs(score - max_score) <= 1e-8
        if not np.any(score_tied):
            continue

        values = _rigid_body_values_batch(
            p_axes[candidate_index, score_tied],
            previous_origin[score_tied],
            c_axes[candidate_index, score_tied],
            current_origin[score_tied],
            degrees_per_radian,
            translation_sign=1.0,
            rotation_sign=1.0,
        )
        forward = values[:, 2] >= -1e-8
        prev_x = candidate_prev_x[candidate_index, score_tied]
        curr_x = candidate_curr_x[candidate_index, score_tied]
        rise = values[:, 2]

        subset = np.flatnonzero(score_tied)
        better = score[score_tied] > best_score[subset] + 1e-8
        tied = np.abs(score[score_tied] - best_score[subset]) <= 1e-8
        better |= tied & (forward & ~best_forward[subset])
        tied_forward = tied & (forward == best_forward[subset])
        better |= tied_forward & (prev_x > best_prev_x[subset] + 1e-8)
        tied_prev = tied_forward & (np.abs(prev_x - best_prev_x[subset]) <= 1e-8)
        better |= tied_prev & (curr_x > best_curr_x[subset] + 1e-8)
        tied_curr = tied_prev & (np.abs(curr_x - best_curr_x[subset]) <= 1e-8)
        better |= tied_curr & (rise > best_rise[subset])

        target = subset[better]
        best_score[target] = score[target]
        best_forward[target] = forward[better]
        best_prev_x[target] = prev_x[better]
        best_curr_x[target] = curr_x[better]
        best_rise[target] = rise[better]
        best_values[target] = values[better]

    best_values[:, 3:] = _wrap_180(best_values[:, 3:])
    return best_values


def _parameter_row(values: Optional[Sequence[float]], names: Sequence[str], **metadata) -> Dict:
    row = dict(metadata)
    if values is None:
        for name in names:
            row[name] = None
        return row
    values = np.asarray(values, dtype=float)
    for name, value in zip(names, values):
        row[name] = float(value) if np.isfinite(value) else None
    return row


class BatchCurvesPlusMDAnalyzer:
    """Experimental vectorized MD path for standard-frame Curves+ analyses.

    This intentionally supports a narrow subset first: combined two-strand
    duplexes with standard base fitting and Curves+ smooth-axis output.  It is
    designed as a validation target before these kernels are moved into the
    regular static-structure path.
    """

    def __init__(
        self,
        topology_file: str,
        inpfile: Optional[str] = None,
        output_dir: str = ".",
        continuous_strands: bool = False,
        fit_override: Optional[bool] = None,
        comb_override: Optional[bool] = None,
        ends_override: Optional[bool] = None,
        include_grooves: Optional[bool] = None,
        include_curvesplus_axis_steps: bool = False,
        include_fit_quality: bool = False,
    ):
        self.topology_file = topology_file
        self.output_dir = output_dir
        self.include_requested_grooves = include_grooves
        self.include_curvesplus_axis_steps = include_curvesplus_axis_steps
        self.include_fit_quality = include_fit_quality
        self.inpfile, self.generated_inpfiles = self._resolve_inp(
            topology_file,
            inpfile,
            output_dir,
            continuous_strands,
            fit_override=fit_override,
            comb_override=comb_override,
            ends_override=ends_override,
        )
        self.config_dict = ConfigLoader.parse_inp(
            self.inpfile,
            config_overrides={
                "frame_convention": "standard",
                "axis_convention": "curvesplus",
                "fit": True if fit_override is None else fit_override,
                "comb": True if comb_override is None else comb_override,
                "ends": False if ends_override is None else ends_override,
                "grv": include_grooves,
            },
        )
        self.ctx = CurvesContext(self.config_dict)
        self.include_grooves = bool(getattr(self.ctx.cfg, "grv", False))
        self.template_molecule = self._load_template_molecule(topology_file)
        self.ctx.molecule = self.template_molecule
        self.library = BaseReferenceLibrary.load("standard")
        self.constants = BaseGeometryConstants()
        self.degrees_per_radian = 180.0 / np.pi
        self.radians_per_degree = np.pi / 180.0

        self._validate_supported()
        self.fit_templates = self._build_fit_templates()
        self.fit_template_groups = self._build_fit_template_groups()
        self.backbone_analyzer = BackboneAnalyzer()
        self.backbone_analyzer._find_all_atoms(self.ctx)
        self.backbone_atom_map = self.ctx.backbone.atom_map.copy()
        self.backbone_ita = self.backbone_analyzer.ITA_RAW.copy()
        self.backbone_tasks = self._build_backbone_tasks()
        self.primary_levels = [level for level in range(1, self.ctx.nux + 1) if self._has_level(0, level)]
        self._residue_labels = {
            (strand, level): self._residue_label(strand, level)
            for strand in range(self.ctx.nst)
            for level in range(1, self.ctx.nux + 1)
            if self._has_level(strand, level)
        }

    @staticmethod
    def _resolve_inp(
        topology_file: str,
        inpfile: Optional[str],
        output_dir: str,
        continuous_strands: bool,
        **overrides,
    ) -> Tuple[str, List[str]]:
        if inpfile is not None:
            return inpfile, []
        runner = CurvesWrapper(
            pdbfile=topology_file,
            output_dir=output_dir,
            continuous_strands=continuous_strands,
            frame_convention="standard",
            axis_convention="curvesplus",
            fit_override=overrides.get("fit_override"),
            comb_override=overrides.get("comb_override"),
            ends_override=overrides.get("ends_override"),
        )
        return runner.inpfile, list(runner.generated_inpfiles)

    @staticmethod
    def _load_template_molecule(topology_file: str) -> MolecularStructure:
        holder = type("MoleculeHolder", (), {"molecule": MolecularStructure()})()
        MolecularLoader.load(topology_file, holder)
        return holder.molecule

    def _validate_supported(self) -> None:
        cfg = self.ctx.cfg
        if not cfg.comb:
            raise NotImplementedError("pycurves-md-batch currently requires comb=true.")
        if self.ctx.nst != 2:
            raise NotImplementedError("pycurves-md-batch currently supports exactly two strands.")
        if cfg.ends:
            raise NotImplementedError("pycurves-md-batch currently does not support ends=true.")
        if not cfg.fit:
            raise NotImplementedError("pycurves-md-batch currently requires least-squares base fitting.")
        if self.ctx.hoogsteen_markers or self.ctx.pair_geometry_markers:
            raise NotImplementedError(
                "pycurves-md-batch currently supports canonical frame geometry only; "
                "use pycurves-md for Hoogsteen/contact-geometry inputs."
            )

    def _build_fit_templates(self) -> List[BaseFitTemplate]:
        locator = BaseLocator(self.constants, reference_library=self.library)
        items: List[BaseFitTemplate] = []
        for strand in range(self.ctx.nst):
            for zero_level in range(self.ctx.nux):
                level = zero_level + 1
                if not self._has_level(strand, level):
                    continue
                atom_data = locator._identify_base_atoms(strand, zero_level, self.ctx)
                if atom_data is None:
                    continue
                self.ctx.backbone.atom_map[strand, level, 0] = atom_data["i4"]
                self.ctx.backbone.atom_map[strand, level, 1] = atom_data["i2"]
                self.ctx.backbone.atom_map[strand, level, 2] = atom_data["i1"]
                self.ctx.backbone.atom_map[strand, level, 14] = atom_data["i3"]
                template = atom_data.get("reference_template")
                if template is None:
                    raise ValueError(f"No standard template for strand {strand + 1} level {level}.")
                atom_order = list(atom_data.get("reference_atom_names") or template.atom_names)
                atom_indices = []
                reference_coords = []
                for atom_name in atom_order:
                    if atom_name not in template.atom_map:
                        continue
                    idx = self._residue_atom_index(atom_data["res_atoms"], atom_name)
                    if idx is None:
                        continue
                    atom_indices.append(idx)
                    reference_coords.append(template.atom_map[atom_name])
                if len(atom_indices) < 3:
                    raise ValueError(f"Strand {strand + 1} level {level} has fewer than three fitting atoms.")
                names = tuple(template.atom_names)
                items.append(
                    BaseFitTemplate(
                        strand=strand,
                        level=level,
                        subunit=int(atom_data["subunit"]),
                        residue_name=str(atom_data["residue_name"]).strip().upper(),
                        parent_base=str(atom_data["parent_base"]).strip().upper(),
                        atom_indices=np.asarray(atom_indices, dtype=int),
                        reference_coords=np.asarray(reference_coords, dtype=float),
                        template_coords=np.asarray(template.coordinates, dtype=float),
                        template_atom_names=names,
                        c1_template_index=0,
                        glycosidic_template_index=1,
                        normal_template_index=2,
                    )
                )
        return items

    def _build_fit_template_groups(self) -> List[BaseFitTemplateGroup]:
        groups: Dict[int, List[BaseFitTemplate]] = {}
        for item in self.fit_templates:
            groups.setdefault(int(len(item.atom_indices)), []).append(item)

        built: List[BaseFitTemplateGroup] = []
        for atom_count in sorted(groups):
            items = groups[atom_count]
            strands = np.asarray([item.strand for item in items], dtype=int)
            levels = np.asarray([item.level for item in items], dtype=int)
            atom_indices = np.asarray([item.atom_indices for item in items], dtype=int)
            references = np.asarray([item.reference_coords for item in items], dtype=float)
            ref_centroids = references.mean(axis=1)
            reference_shifted = references - ref_centroids[:, None, :]
            template_key_shifted = np.asarray(
                [
                    item.template_coords[
                        [item.c1_template_index, item.glycosidic_template_index, item.normal_template_index]
                    ] - ref_centroids[index][None, :]
                    for index, item in enumerate(items)
                ],
                dtype=float,
            )
            built.append(
                BaseFitTemplateGroup(
                    strands=strands,
                    levels=levels,
                    atom_indices=atom_indices,
                    reference_shifted=reference_shifted,
                    template_key_shifted=template_key_shifted,
                    ref_centroids=ref_centroids,
                )
            )
        return built

    def _build_backbone_tasks(self) -> BackboneTaskSet:
        torsion_records = []
        angle_records = []
        sugar_counts: Dict[Tuple[int, int], int] = {}

        for strand in range(self.ctx.nst):
            for level in range(1, self.ctx.nux + 1):
                if self.ctx.li[level, strand] < -2:
                    continue
                atom_map = self.backbone_atom_map[strand, level]

                for torsion_index, row in enumerate(self.backbone_ita):
                    i1, i2, i3, i4 = [int(value) for value in row]
                    idx1 = int(atom_map[i1 - 1])
                    idx2 = int(atom_map[i2 - 1])
                    idx3 = int(atom_map[i3 - 1])
                    if idx1 < 0 or idx2 < 0 or idx3 < 0:
                        continue

                    if i4 == 0:
                        angle_records.append((strand, level, torsion_index, idx1, idx2, idx3))
                        continue

                    idx4 = int(atom_map[i4 - 1])
                    if idx4 < 0:
                        continue
                    torsion_records.append((strand, level, torsion_index, idx1, idx2, idx3, idx4))
                    if 4 <= torsion_index <= 8:
                        sugar_counts[(strand, level)] = sugar_counts.get((strand, level), 0) + 1

        sugar_records = [(strand, level) for (strand, level), count in sugar_counts.items() if count >= 5]

        def task_parts(records, atom_count):
            arr = np.asarray(records, dtype=int) if records else np.empty((0, 3 + atom_count), dtype=int)
            return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3:]

        torsion_strands, torsion_levels, torsion_indices, torsion_atoms = task_parts(torsion_records, 4)
        angle_strands, angle_levels, angle_indices, angle_atoms = task_parts(angle_records, 3)
        if sugar_records:
            sugar = np.asarray(sugar_records, dtype=int)
            sugar_strands = sugar[:, 0]
            sugar_levels = sugar[:, 1]
        else:
            sugar_strands = np.empty(0, dtype=int)
            sugar_levels = np.empty(0, dtype=int)

        return BackboneTaskSet(
            torsion_strands=torsion_strands,
            torsion_levels=torsion_levels,
            torsion_indices=torsion_indices,
            torsion_atoms=torsion_atoms,
            angle_strands=angle_strands,
            angle_levels=angle_levels,
            angle_indices=angle_indices,
            angle_atoms=angle_atoms,
            sugar_strands=sugar_strands,
            sugar_levels=sugar_levels,
        )

    @staticmethod
    def _residue_atom_index(residue_atoms: Dict[str, int], atom_name: str) -> Optional[int]:
        for alias in atom_aliases(atom_name):
            if alias in residue_atoms:
                return int(residue_atoms[alias])
        return None

    def analyze_batch(
        self,
        coordinates: np.ndarray,
        frame_indices: Sequence[int],
        times: Sequence[Optional[float]],
        accumulator=None,
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 3 or coordinates.shape[1:] != self.template_molecule.coordinates.shape:
            raise ValueError(
                "Batch coordinates must have shape (frames, atoms, 3); "
                f"got {coordinates.shape}, expected (*, {self.template_molecule.coordinates.shape[0]}, 3)."
            )
        frames, rmsd = self._fit_frames(coordinates)
        local_inter_base = self._local_inter_base(frames)
        local_base_base, pair_frames = self._local_base_base_and_pair_frames(frames)
        axis_tables = self._curvesplus_axis_tables(frames, include_inter_bp=self.include_curvesplus_axis_steps)
        backbone_torsions, backbone_pucker = self._backbone_values(coordinates)
        groove_rows_by_frame = (
            compute_batch_grooves(self, coordinates, frames, axis_tables, local_inter_base)
            if self.include_grooves else None
        )

        batch_rows = []
        table_records: Dict[str, List[Dict]] = {
            "local_inter_base": [],
            "local_base_base": [],
            "local_inter_base_pair": [],
            "curvesplus_base_pair_axis": [],
            "backbone": [],
        }
        if self.include_grooves:
            table_records["groove"] = []
        if self.include_curvesplus_axis_steps:
            table_records["curvesplus_inter_base_pair"] = []
        if self.include_fit_quality:
            table_records["base_fit_quality"] = []
        local_inter_bp = self._local_inter_base_pair(pair_frames)
        if accumulator is not None:
            self._accumulate_precomputed_summary(
                accumulator,
                local_inter_base,
                local_base_base,
                local_inter_bp,
                axis_tables,
                backbone_torsions,
                backbone_pucker,
                rmsd,
                groove_rows_by_frame,
            )

        for batch_index, (frame_id, time_value) in enumerate(zip(frame_indices, times)):
            dataframes = {
                "local_inter_base": self._local_inter_base_rows(local_inter_base[batch_index]),
                "local_base_base": self._local_base_base_rows(local_base_base[batch_index]),
                "local_inter_base_pair": self._local_inter_base_pair_rows(local_inter_bp[batch_index]),
                "curvesplus_base_pair_axis": self._curvesplus_base_pair_axis_rows(axis_tables["bp_axis"][batch_index]),
                "backbone": self._backbone_rows(backbone_torsions[batch_index], backbone_pucker[batch_index]),
            }
            if self.include_grooves:
                dataframes["groove"] = groove_rows_by_frame[batch_index] if groove_rows_by_frame is not None else []
            if self.include_curvesplus_axis_steps:
                dataframes["curvesplus_inter_base_pair"] = self._curvesplus_inter_base_pair_rows(axis_tables["inter_bp"][batch_index])
            if self.include_fit_quality:
                dataframes["base_fit_quality"] = self._base_fit_quality_rows(rmsd[batch_index])
            for rows in dataframes.values():
                for row in rows:
                    row["frame"] = int(frame_id)
                    row["time"] = None if time_value is None else float(time_value)
            for name, rows in dataframes.items():
                table_records[name].extend(rows)
            batch_rows.append({"frame": int(frame_id), "time": None if time_value is None else float(time_value), "dataframes": dataframes})
        return batch_rows, table_records

    def accumulate_batch_summary(
        self,
        coordinates: np.ndarray,
        frame_indices: Sequence[int],
        times: Sequence[Optional[float]],
        accumulator,
    ) -> int:
        """Accumulate summary statistics directly from vectorized arrays.

        This is used by ``pycurves-md-batch --mode summary`` to avoid building
        per-frame row dictionaries that are immediately reduced again.
        """
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 3 or coordinates.shape[1:] != self.template_molecule.coordinates.shape:
            raise ValueError(
                "Batch coordinates must have shape (frames, atoms, 3); "
                f"got {coordinates.shape}, expected (*, {self.template_molecule.coordinates.shape[0]}, 3)."
            )
        batch = coordinates.shape[0]
        frames, rmsd = self._fit_frames(coordinates)
        local_inter_base = self._local_inter_base(frames)
        local_base_base, pair_frames = self._local_base_base_and_pair_frames(frames)
        axis_tables = self._curvesplus_axis_tables(frames, include_inter_bp=self.include_curvesplus_axis_steps)
        backbone_torsions, backbone_pucker = self._backbone_values(coordinates)
        local_inter_bp = self._local_inter_base_pair(pair_frames)
        groove_rows_by_frame = (
            compute_batch_grooves(self, coordinates, frames, axis_tables, local_inter_base)
            if self.include_grooves else None
        )
        self._accumulate_precomputed_summary(
            accumulator,
            local_inter_base,
            local_base_base,
            local_inter_bp,
            axis_tables,
            backbone_torsions,
            backbone_pucker,
            rmsd,
            groove_rows_by_frame,
        )
        return batch

    def _accumulate_precomputed_summary(
        self,
        accumulator,
        local_inter_base: np.ndarray,
        local_base_base: np.ndarray,
        local_inter_bp: np.ndarray,
        axis_tables: Dict[str, np.ndarray],
        backbone_torsions: np.ndarray,
        backbone_pucker: np.ndarray,
        rmsd: np.ndarray,
        groove_rows_by_frame: Optional[List[List[Dict]]],
    ) -> None:
        batch = local_inter_base.shape[0]
        self._accumulate_local_inter_base_summary(accumulator, local_inter_base)
        self._accumulate_local_base_base_summary(accumulator, local_base_base, batch)
        self._accumulate_local_inter_base_pair_summary(accumulator, "local_inter_base_pair", local_inter_bp, batch)
        self._accumulate_axis_summary(accumulator, axis_tables["bp_axis"], batch)
        self._accumulate_backbone_summary(accumulator, backbone_torsions, backbone_pucker)
        if self.include_grooves:
            accumulator.ensure_table("groove")
            groove_names = (
                "minor_width", "minor_depth", "minor_angle",
                "major_width", "major_depth", "major_angle", "diameter",
            )
            for rows in groove_rows_by_frame or ():
                accumulator.add_rows("groove", rows, groove_names)
        if self.include_curvesplus_axis_steps:
            self._accumulate_local_inter_base_pair_summary(
                accumulator,
                "curvesplus_inter_base_pair",
                axis_tables["inter_bp"],
                batch,
            )
        if self.include_fit_quality:
            self._accumulate_fit_quality_summary(accumulator, rmsd)

    def _accumulate_local_inter_base_summary(self, accumulator, values: np.ndarray) -> None:
        accumulator.ensure_table("local_inter_base")
        for strand in range(self.ctx.nst):
            for level in range(1, self.ctx.nux):
                if not (self._has_level(strand, level) and self._has_level(strand, level + 1)):
                    continue
                accumulator.add_values(
                    "local_inter_base",
                    {"strand": strand + 1, "level": level, "step": self._step_label(strand, level + 1)},
                    STEP_PARAMETERS,
                    values[:, strand, level, :],
                )

    def _accumulate_local_base_base_summary(self, accumulator, values: np.ndarray, batch: int) -> None:
        accumulator.ensure_table("local_base_base")
        empty = np.full((batch, len(BASE_BASE_PARAMETERS)), np.nan, dtype=float)
        for sequence_index, level in enumerate(self.primary_levels, start=1):
            data = values[:, level, :] if self._has_level(1, level) else empty
            accumulator.add_values(
                "local_base_base",
                {
                    "partner_strand": 2,
                    "sequence_index": sequence_index,
                    "level": level,
                    "duplex": self._duplex_id(level),
                },
                BASE_BASE_PARAMETERS,
                data,
            )

    def _accumulate_local_inter_base_pair_summary(self, accumulator, table_name: str, values: np.ndarray, batch: int) -> None:
        accumulator.ensure_table(table_name)
        empty = np.full((batch, len(STEP_PARAMETERS)), np.nan, dtype=float)
        for sequence_index, (level, next_level) in enumerate(zip(self.primary_levels, self.primary_levels[1:]), start=1):
            has_step = (
                next_level == level + 1
                and self._has_level(1, level)
                and self._has_level(1, next_level)
            )
            data = values[:, level, :] if has_step else empty
            accumulator.add_values(
                table_name,
                {
                    "partner_strand": 2,
                    "sequence_index": sequence_index,
                    "level": level,
                    "next_level": next_level,
                    "step": self._step_label_between(level, next_level),
                    "duplex": f"{self._duplex_id(level)}/{self._duplex_id(next_level)}",
                },
                STEP_PARAMETERS,
                data,
            )

    def _accumulate_axis_summary(self, accumulator, values: np.ndarray, batch: int) -> None:
        accumulator.ensure_table("curvesplus_base_pair_axis")
        empty = np.full((batch, len(AXIS_PARAMETERS)), np.nan, dtype=float)
        for sequence_index, level in enumerate(self.primary_levels, start=1):
            data = values[:, level, :] if self._has_level(1, level) else empty
            accumulator.add_values(
                "curvesplus_base_pair_axis",
                {
                    "partner_strand": 2,
                    "sequence_index": sequence_index,
                    "level": level,
                    "duplex": self._duplex_id(level),
                },
                AXIS_PARAMETERS,
                data,
            )

    def _accumulate_backbone_summary(self, accumulator, torsions: np.ndarray, sugar_pucker: np.ndarray) -> None:
        accumulator.ensure_table("backbone")
        names = (
            "c1_c2", "c2_c3", "phase", "amplitude", "c1_prime", "c2_prime",
            "c3_prime", "chi", "gamma", "delta", "epsilon", "zeta", "alpha", "beta",
        )
        for strand in range(self.ctx.nst):
            for level in range(1, self.ctx.nux + 1):
                if not self._has_level(strand, level):
                    continue
                label = self._residue_unit_label(strand, level)
                if label is None:
                    continue
                residue_name, residue_id = label
                tor = torsions[:, strand, level, :]
                pucker = sugar_pucker[:, strand, level, :]
                values = np.stack(
                    [
                        tor[:, 4], tor[:, 5], pucker[:, 1], pucker[:, 0],
                        tor[:, 0], tor[:, 1], tor[:, 2], tor[:, 12],
                        tor[:, 8], tor[:, 9], tor[:, 6], tor[:, 7],
                        tor[:, 10], tor[:, 11],
                    ],
                    axis=1,
                )
                metadata = {
                    "strand": strand + 1,
                    "level": level,
                    "residue_name": residue_name,
                    "residue_id": residue_id,
                }
                phase = pucker[:, 1]
                pucker_index = np.zeros(phase.shape[0], dtype=int)
                finite = np.isfinite(phase)
                if np.any(finite):
                    pucker_index[finite] = np.asarray((phase[finite] % 360.0) / 36.0, dtype=int)
                    pucker_index = np.clip(pucker_index, 0, len(SUGAR_PUCKERS) - 1)
                seen = []
                for idx in pucker_index:
                    value = int(idx)
                    if value not in seen:
                        seen.append(value)
                for idx in seen:
                    mask = pucker_index == idx
                    row_metadata = dict(metadata)
                    row_metadata["pucker"] = SUGAR_PUCKERS[idx]
                    accumulator.add_values("backbone", row_metadata, names, values[mask])

                sugar_counts, sugar_total = sugar_pucker_counts(phase)
                for label, count in zip(SUMMARY_SUGAR_PUCKERS, sugar_counts):
                    accumulator.add_population_counts(
                        "backbone_sugar_pucker_distribution",
                        metadata,
                        {"pucker": label},
                        int(count),
                        sugar_total,
                    )

                bi_counts, bi_total = bi_bii_counts(tor[:, 6], tor[:, 7])
                for label, count in zip(BI_BII_STATES, bi_counts):
                    accumulator.add_population_counts(
                        "backbone_bi_bii_distribution",
                        metadata,
                        {"conformer": label},
                        int(count),
                        bi_total,
                    )

                ag_counts, ag_total = alpha_gamma_counts(tor[:, 10], tor[:, 8])
                for (alpha_state, gamma_state), count in zip(ALPHA_GAMMA_STATES, ag_counts):
                    accumulator.add_population_counts(
                        "backbone_alpha_gamma_distribution",
                        metadata,
                        {
                            "alpha_state": alpha_state,
                            "gamma_state": gamma_state,
                            "conformer": f"{alpha_state}/{gamma_state}",
                        },
                        int(count),
                        ag_total,
                    )

    def _accumulate_fit_quality_summary(self, accumulator, rmsd: np.ndarray) -> None:
        accumulator.ensure_table("base_fit_quality")
        for item in self.fit_templates:
            label = self._residue_labels.get((item.strand, item.level))
            if label is None:
                continue
            base, chain_id, residue_id = label
            accumulator.add_values(
                "base_fit_quality",
                {
                    "strand": item.strand + 1,
                    "level": item.level,
                    "subunit": item.subunit,
                    "chain_id": chain_id,
                    "residue_id": residue_id,
                    "residue_name": item.residue_name,
                    "parent_base": item.parent_base,
                    "base": base,
                    "frame_convention": "standard",
                    "fit_atom_count": int(len(item.atom_indices)),
                },
                ("rmsd",),
                rmsd[:, item.strand, item.level],
            )

    def _fit_frames(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = coordinates.shape[0]
        n3 = self.ctx.nux + 2
        frames = np.full((batch, self.ctx.nst, n3, 4, 3), np.nan, dtype=float)
        rmsd = np.full((batch, self.ctx.nst, n3), np.nan, dtype=float)
        ca_origin = math.cos(self.radians_per_degree * 141.47)
        sa_origin = math.sin(self.radians_per_degree * 141.47)
        ca_y = math.cos(self.radians_per_degree * -54.41)
        sa_y = math.sin(self.radians_per_degree * -54.41)
        standard_dis = 4.7024

        for group in self.fit_template_groups:
            observed = coordinates[:, group.atom_indices, :]
            obs_centroid = observed.mean(axis=2)
            obs_shifted = observed - obs_centroid[:, :, None, :]
            covariance = np.einsum("gmi,bgmj->bgij", group.reference_shifted, obs_shifted)
            flat_covariance = covariance.reshape(-1, 3, 3)
            u_mat, _, vh_mat = np.linalg.svd(flat_covariance)
            vh_group = vh_mat.reshape(batch, len(group.levels), 3, 3)
            u_group = u_mat.reshape(batch, len(group.levels), 3, 3)
            handedness = np.sign(
                np.linalg.det(np.einsum("bgji,bgjk->bgik", vh_group, np.swapaxes(u_group, -1, -2)))
            )
            handedness[handedness == 0.0] = 1.0
            diag = np.zeros((batch, len(group.levels), 3, 3), dtype=float)
            diag[:, :, 0, 0] = 1.0
            diag[:, :, 1, 1] = 1.0
            diag[:, :, 2, 2] = handedness
            rotation = np.einsum("bgji,bgjk,bgkl->bgil", vh_group, diag, np.swapaxes(u_group, -1, -2))

            fitted = np.einsum("gmi,bgji->bgmj", group.reference_shifted, rotation) + obs_centroid[:, :, None, :]
            residual = observed - fitted
            rmsd[:, group.strands, group.levels] = np.sqrt(np.mean(np.sum(residual * residual, axis=3), axis=2))

            fitted_key = np.einsum("gmi,bgji->bgmj", group.template_key_shifted, rotation) + obs_centroid[:, :, None, :]
            c1 = fitted_key[:, :, 0, :]
            c0 = fitted_key[:, :, 1, :]
            c3 = fitted_key[:, :, 2, :]

            glycosidic_axis = _unit(c1 - c0)
            z_axis = _unit(np.cross(glycosidic_axis, c3 - c0))
            origin = _rotate_axis_angle_batch(glycosidic_axis * standard_dis, z_axis, ca_origin, sa_origin) + c0
            y_axis = _unit(_rotate_axis_angle_batch(glycosidic_axis, z_axis, ca_y, sa_y))
            x_axis = _unit(np.cross(y_axis, z_axis))
            y_axis = _unit(np.cross(z_axis, x_axis))

            frames[:, group.strands, group.levels, 0, :] = x_axis
            frames[:, group.strands, group.levels, 1, :] = y_axis
            frames[:, group.strands, group.levels, 2, :] = z_axis
            frames[:, group.strands, group.levels, 3, :] = origin
        return frames, rmsd

    def _local_inter_base(self, frames: np.ndarray) -> np.ndarray:
        batch = frames.shape[0]
        values = np.full((batch, self.ctx.nst, self.ctx.nux + 1, 6), np.nan, dtype=float)
        oriented = self._continuous_oriented_strand_frames(frames)
        records = [
            (strand, level)
            for strand in range(self.ctx.nst)
            for level in range(2, self.ctx.nux + 1)
            if self._has_level(strand, level - 1) and self._has_level(strand, level)
        ]
        if not records:
            return values

        strands = np.asarray([strand for strand, _ in records], dtype=int)
        levels = np.asarray([level for _, level in records], dtype=int)
        previous_axes = oriented[:, strands, levels - 1, :3, :].reshape(-1, 3, 3)
        current_axes = oriented[:, strands, levels, :3, :].reshape(-1, 3, 3)
        previous_origin = oriented[:, strands, levels - 1, 3, :].reshape(-1, 3)
        current_origin = oriented[:, strands, levels, 3, :].reshape(-1, 3)
        step = _rigid_body_values_batch(
            previous_axes,
            previous_origin,
            current_axes,
            current_origin,
            self.degrees_per_radian,
            translation_sign=1.0,
            rotation_sign=1.0,
        ).reshape(batch, len(records), 6)
        step[:, :, 3:] = _wrap_180(step[:, :, 3:])
        for index, (strand, level) in enumerate(records):
            values[:, strand, level - 1, :] = step[:, index, :]
        return values

    def _continuous_oriented_strand_frames(self, frames: np.ndarray) -> np.ndarray:
        oriented = frames.copy()
        for strand in range(1, self.ctx.nst):
            if self.ctx.idr[strand] < 0:
                oriented[:, strand, :, 1, :] *= -1.0
                oriented[:, strand, :, 2, :] *= -1.0
            else:
                oriented[:, strand, :, 0, :] *= -1.0
                oriented[:, strand, :, 1, :] *= -1.0

        batch = frames.shape[0]
        for strand in range(self.ctx.nst):
            previous_axes = None
            previous_valid = np.zeros(batch, dtype=bool)
            for level in range(1, self.ctx.nux + 1):
                if not self._has_level(strand, level):
                    previous_axes = None
                    previous_valid[:] = False
                    continue
                axes = oriented[:, strand, level, :3, :]
                finite = np.all(np.isfinite(axes), axis=(1, 2))
                if previous_axes is not None:
                    candidates = np.einsum("fij,bjk->fbik", SIGN_FLIPS, axes)
                    scores = np.einsum("bij,fbij->bf", previous_axes, candidates)
                    best = np.argmax(scores, axis=1)
                    selected = candidates[best, np.arange(batch)]
                    use = previous_valid & finite
                    oriented[use, strand, level, :3, :] = selected[use]
                    axes = oriented[:, strand, level, :3, :]
                previous_axes = axes.copy()
                previous_valid = finite
        return oriented

    def _local_base_base_and_pair_frames(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = frames.shape[0]
        base_base = np.full((batch, self.ctx.nux + 1, 6), np.nan, dtype=float)
        pair_frames = np.full((batch, self.ctx.nux + 1, 4, 3), np.nan, dtype=float)
        levels = np.asarray(
            [level for level in range(1, self.ctx.nux + 1) if self._has_level(0, level) and self._has_level(1, level)],
            dtype=int,
        )
        if levels.size == 0:
            return base_base, pair_frames

        first_axes = frames[:, 0, levels, :3, :]
        first_origin = frames[:, 0, levels, 3, :]
        other_axes = frames[:, 1, levels, :3, :].copy()
        other_axes[:, :, 1, :] *= -1.0
        other_axes[:, :, 2, :] *= -1.0
        other_origin = frames[:, 1, levels, 3, :]
        values, middle_axes, middle_origin = _rigid_body_values_and_middle_frame_batch(
            first_axes,
            first_origin,
            other_axes,
            other_origin,
            self.degrees_per_radian,
            translation_sign=-1.0,
            rotation_sign=-1.0,
        )
        base_base[:, levels, :] = values
        pair_frames[:, levels, :3, :] = middle_axes
        pair_frames[:, levels, 3, :] = middle_origin
        return base_base, pair_frames

    def _local_inter_base_pair(self, pair_frames: np.ndarray) -> np.ndarray:
        batch = pair_frames.shape[0]
        values = np.full((batch, self.ctx.nux + 1, 6), np.nan, dtype=float)
        levels = [
            level
            for level in range(2, self.ctx.nux + 1)
            if (
                self._has_level(0, level - 1)
                and self._has_level(1, level - 1)
                and self._has_level(0, level)
                and self._has_level(1, level)
            )
        ]
        if not levels:
            return values

        previous_levels = np.asarray([level - 1 for level in levels], dtype=int)
        current_levels = np.asarray(levels, dtype=int)
        previous = pair_frames[:, previous_levels]
        current = pair_frames[:, current_levels]
        step = _rigid_body_values_batch(
            previous[:, :, :3, :].reshape(-1, 3, 3),
            previous[:, :, 3, :].reshape(-1, 3),
            current[:, :, :3, :].reshape(-1, 3, 3),
            current[:, :, 3, :].reshape(-1, 3),
            self.degrees_per_radian,
            translation_sign=1.0,
            rotation_sign=1.0,
        ).reshape(batch, len(levels), 6)
        delta = current[:, :, 3, :] - previous[:, :, 3, :]
        invert = np.sum(delta * current[:, :, 2, :], axis=2) < 0.0
        step[invert, 2] *= -1.0
        step[invert, 5] *= -1.0
        step[:, :, 3:] = _wrap_180(step[:, :, 3:])
        values[:, previous_levels, :] = step
        return values

    def _curvesplus_axis_tables(self, frames: np.ndarray, include_inter_bp: bool = False) -> Dict[str, np.ndarray]:
        ref = self._curvesplus_reference_frames(frames)
        upm = self._curvesplus_base_pair_frames(ref)
        uvw = self._curvesplus_smoothed_axis(ref, upm)
        invert = self._curvesplus_inversion_flags(upm)
        bp_axis = self._curvesplus_bp_axis_values(upm, uvw, invert)
        tables = {"bp_axis": bp_axis, "axis_frames": uvw}
        if include_inter_bp:
            tables["inter_bp"] = self._curvesplus_inter_base_pair(upm, invert)
        return tables

    def _curvesplus_reference_frames(self, frames: np.ndarray) -> np.ndarray:
        ref = np.full((frames.shape[0], self.ctx.nux + 1, self.ctx.nst, 4, 3), np.nan, dtype=float)
        for strand in range(self.ctx.nst):
            for level in range(1, self.ctx.nux + 1):
                if not self._has_level(strand, level):
                    continue
                frame = frames[:, strand, level].copy()
                if strand > 0:
                    if self.ctx.idr[strand] < 0:
                        frame[:, 1, :] *= -1.0
                        frame[:, 2, :] *= -1.0
                    else:
                        frame[:, 0, :] *= -1.0
                        frame[:, 1, :] *= -1.0
                ref[:, level, strand] = frame
        return ref

    def _curvesplus_base_pair_frames(self, ref: np.ndarray) -> np.ndarray:
        batch = ref.shape[0]
        upm = np.full((batch, self.ctx.nux + 1, 4, 3), np.nan, dtype=float)
        paired_levels = np.asarray(
            [level for level in range(1, self.ctx.nux + 1) if self._has_level(0, level) and self._has_level(1, level)],
            dtype=int,
        )
        if paired_levels.size:
            axes, origin = _middle_frame_batch(
                ref[:, paired_levels, 1, :3, :],
                ref[:, paired_levels, 1, 3, :],
                ref[:, paired_levels, 0, :3, :],
                ref[:, paired_levels, 0, 3, :],
            )
            upm[:, paired_levels, :3, :] = axes
            upm[:, paired_levels, 3, :] = origin

        first_only = np.asarray(
            [level for level in range(1, self.ctx.nux + 1) if self._has_level(0, level) and not self._has_level(1, level)],
            dtype=int,
        )
        second_only = np.asarray(
            [level for level in range(1, self.ctx.nux + 1) if self._has_level(1, level) and not self._has_level(0, level)],
            dtype=int,
        )
        if first_only.size:
            upm[:, first_only] = ref[:, first_only, 0]
        if second_only.size:
            upm[:, second_only] = ref[:, second_only, 1]
        return upm

    def _curvesplus_smoothed_axis(self, ref: np.ndarray, upm: np.ndarray) -> np.ndarray:
        batch = ref.shape[0]
        axis_sum = np.zeros((batch, self.ctx.nux + 1, 3), dtype=float)
        point_sum = np.zeros((batch, self.ctx.nux + 1, 3), dtype=float)
        counts = np.zeros((batch, self.ctx.nux + 1), dtype=float)
        for upper in range(2, self.ctx.nux + 1):
            lower = upper - 1
            for strand in range(self.ctx.nst):
                if not (self._has_level(strand, lower) and self._has_level(strand, upper)):
                    continue
                axis, point = self._curvesplus_screw_axis(ref[:, lower, strand], ref[:, upper, strand])
                for level in (lower, upper):
                    origin = upm[:, level, 3, :]
                    projected = point + np.sum((origin - point) * axis, axis=1, keepdims=True) * axis
                    axis_sum[:, level, :] += axis
                    point_sum[:, level, :] += projected
                    counts[:, level] += 1.0

        averaged = np.zeros((batch, self.ctx.nux + 1, 6), dtype=float)
        valid = counts > 0
        averaged[:, :, :3] = _unit(axis_sum, upm[:, :, 2, :])
        averaged[:, :, 3:] = np.divide(point_sum, counts[:, :, None], out=np.zeros_like(point_sum), where=counts[:, :, None] > 0)
        for level in range(1, self.ctx.nux + 1):
            mask = valid[:, level]
            if not np.any(mask):
                continue
            origin = upm[mask, level, 3, :]
            axis = averaged[mask, level, :3]
            point = averaged[mask, level, 3:]
            averaged[mask, level, 3:] = point + np.sum((origin - point) * axis, axis=1, keepdims=True) * axis

        uvw = np.full((batch, self.ctx.nux + 1, 4, 3), np.nan, dtype=float)
        width = 4
        weights = {0: 1.0}
        for offset in range(1, width + 1):
            weight = 1.0 - float(offset * offset) / float((width + 1) * (width + 1))
            weights[offset] = weight
            weights[-offset] = weight

        for level in range(1, self.ctx.nux + 1):
            origin = upm[:, level, 3, :]
            level_axis_sum = np.zeros((batch, 3), dtype=float)
            level_point_sum = np.zeros((batch, 3), dtype=float)
            weight_sum = np.zeros(batch, dtype=float)
            for offset in range(-width, width + 1):
                source = level + offset
                if source < 1 or source > self.ctx.nux:
                    continue
                axis = averaged[:, source, :3]
                point = averaged[:, source, 3:]
                source_valid = np.linalg.norm(axis, axis=1) > 1e-12
                if not np.any(source_valid):
                    continue
                weight = weights[offset]
                projected = point + np.sum((origin - point) * axis, axis=1, keepdims=True) * axis
                level_axis_sum[source_valid] += axis[source_valid] * weight
                level_point_sum[source_valid] += projected[source_valid] * weight
                weight_sum[source_valid] += weight
            axis = _unit(level_axis_sum, upm[:, level, 2, :])
            point = np.divide(level_point_sum, weight_sum[:, None], out=np.full_like(level_point_sum, np.nan), where=weight_sum[:, None] > 0)
            point = point + np.sum((origin - point) * axis, axis=1, keepdims=True) * axis
            uvw[:, level, 2, :] = axis
            uvw[:, level, 3, :] = point
        return uvw

    def _curvesplus_screw_axis(self, first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        matrix = _rotation_matrix_first_to_second(first[:, :3, :], second[:, :3, :])
        rotvec = Rotation.from_matrix(matrix).as_rotvec()
        theta = np.linalg.norm(rotvec, axis=1)
        vector = second[:, 3, :] - first[:, 3, :]
        axis = np.zeros_like(vector)
        small = theta < 1e-10
        if np.any(~small):
            axis[~small] = rotvec[~small] / theta[~small, None]
        if np.any(small):
            axis[small] = _unit(vector[small], first[small, 2, :])
        axial_distance = np.sum(axis * vector, axis=1, keepdims=True)
        half_perp = (vector - axial_distance * axis) / 2.0
        point = (first[:, 3, :] + second[:, 3, :]) / 2.0
        if np.any(~small):
            point[~small] += np.cross(axis[~small], half_perp[~small]) / np.tan(theta[~small, None] / 2.0)
        return axis, point

    def _curvesplus_axis_parameter_frames(self, upm: np.ndarray, uvw: np.ndarray) -> np.ndarray:
        axis_upm = upm.copy()
        for level in range(1, self.ctx.nux + 1):
            dot = np.sum(axis_upm[:, level, 2, :] * uvw[:, level, 2, :], axis=1)
            flip = dot < 0.0
            axis_upm[flip, level, 1, :] *= -1.0
            axis_upm[flip, level, 2, :] *= -1.0
        return axis_upm

    def _curvesplus_inversion_flags(self, upm: np.ndarray) -> np.ndarray:
        batch = upm.shape[0]
        invert = np.zeros((batch, self.ctx.nux + 1), dtype=bool)
        for level in range(2, self.ctx.nux + 1):
            previous = level - 1
            delta = upm[:, level, 3, :] - upm[:, previous, 3, :]
            invert[:, previous] = np.sum(delta * upm[:, level, 2, :], axis=1) < 0.0
        if self.ctx.nux > 1:
            invert[:, self.ctx.nux] = invert[:, self.ctx.nux - 1]
        return invert

    def _curvesplus_bp_axis_values(self, upm: np.ndarray, uvw: np.ndarray, invert: np.ndarray) -> np.ndarray:
        batch = upm.shape[0]
        values = np.full((batch, self.ctx.nux + 1, 4), np.nan, dtype=float)
        for level in range(1, self.ctx.nux + 1):
            axis = uvw[:, level, 2, :]
            point = uvw[:, level, 3, :]
            upm_frame = upm[:, level]
            finite = np.all(np.isfinite(upm_frame), axis=(1, 2)) & np.all(np.isfinite(axis), axis=1) & np.all(np.isfinite(point), axis=1)
            if not np.any(finite):
                continue
            dot = np.clip(np.sum(upm_frame[:, 2, :] * axis, axis=1), -1.0, 1.0)
            theta = np.arccos(dot)
            rotation_axis = _unit(np.cross(upm_frame[:, 2, :], axis))
            ca = np.cos(theta)
            sa = np.sin(theta)
            x_axis = _rotate_axis_angle_batch(upm_frame[:, 0, :], rotation_axis, ca, sa)
            small = np.abs(theta) <= 1e-4
            x_axis[small] = upm_frame[small, 0, :]
            y_axis = np.cross(axis, x_axis)
            axis_frame = uvw[:, level].copy()
            axis_frame[:, 0, :] = _unit(x_axis, upm_frame[:, 0, :])
            axis_frame[:, 1, :] = _unit(y_axis, upm_frame[:, 1, :])
            angle_vector = -theta[:, None] * self.degrees_per_radian * rotation_axis
            delta = upm_frame[:, 3, :] - point
            displacements = np.einsum("bij,bj->bi", axis_frame[:, :3, :], delta)
            angles = np.einsum("bij,bj->bi", axis_frame[:, :3, :], angle_vector)
            inv = invert[:, level]
            displacements[inv, 0] *= -1.0
            angles[inv, 0] *= -1.0
            angles[inv, 1] -= 180.0
            angles[:, 0] = _wrap_180(angles[:, 0])
            angles[:, 1] = _wrap_180(angles[:, 1])
            values[finite, level, :] = np.stack(
                [displacements[:, 0], displacements[:, 1], angles[:, 0], angles[:, 1]],
                axis=1,
            )[finite]
        return values

    def _curvesplus_inter_base_pair(self, upm: np.ndarray, invert: np.ndarray) -> np.ndarray:
        batch = upm.shape[0]
        values = np.full((batch, self.ctx.nux + 1, 6), np.nan, dtype=float)
        if self.ctx.nux < 2:
            return values

        previous_levels = np.arange(1, self.ctx.nux, dtype=int)
        current_levels = previous_levels + 1
        previous = upm[:, previous_levels]
        current = upm[:, current_levels]
        finite = np.all(np.isfinite(previous), axis=(2, 3)) & np.all(np.isfinite(current), axis=(2, 3))
        if not np.any(finite):
            return values

        step = _rigid_body_values_batch(
            previous[:, :, :3, :][finite],
            previous[:, :, 3, :][finite],
            current[:, :, :3, :][finite],
            current[:, :, 3, :][finite],
            self.degrees_per_radian,
            translation_sign=1.0,
            rotation_sign=1.0,
        )
        inv = invert[:, previous_levels][finite]
        step[inv, 2] *= -1.0
        step[inv, 5] *= -1.0
        step[:, 3:] = _wrap_180(step[:, 3:])
        target = values[:, previous_levels, :]
        target[finite] = step
        values[:, previous_levels, :] = target
        return values

    def _backbone_values(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = coordinates.shape[0]
        n3 = self.ctx.nux + 2
        store = np.full((batch, self.ctx.nst, n3, 16), 999.0, dtype=float)
        tasks = self.backbone_tasks

        if tasks.torsion_atoms.size:
            atoms = tasks.torsion_atoms
            values = _torsion_batch(
                coordinates[:, atoms[:, 0], :],
                coordinates[:, atoms[:, 1], :],
                coordinates[:, atoms[:, 2], :],
                coordinates[:, atoms[:, 3], :],
            )
            store[:, tasks.torsion_strands, tasks.torsion_levels, tasks.torsion_indices] = values

        if tasks.angle_atoms.size:
            atoms = tasks.angle_atoms
            values = _angle_batch(
                coordinates[:, atoms[:, 0], :],
                coordinates[:, atoms[:, 1], :],
                coordinates[:, atoms[:, 2], :],
            )
            store[:, tasks.angle_strands, tasks.angle_levels, tasks.angle_indices] = values

        torsions = np.full((batch, self.ctx.nst, n3, 13), 999.0, dtype=float)
        sugar_pucker = np.zeros((batch, self.ctx.nst, n3, 2), dtype=float)
        torsions[:, :, :, 0:6] = store[:, :, :, 0:6]
        torsions[:, :, :, 6:13] = store[:, :, :, 9:16]

        if tasks.sugar_strands.size:
            strands = tasks.sugar_strands
            levels = tasks.sugar_levels
            v = np.stack(
                [
                    store[:, strands, levels, 5],
                    store[:, strands, levels, 6],
                    store[:, strands, levels, 7],
                    store[:, strands, levels, 8],
                    store[:, strands, levels, 4],
                ],
                axis=2,
            )
            valid = np.all(np.isfinite(v), axis=2) & ~np.any(v >= 900.0, axis=2)
            if np.any(valid):
                theta = np.radians(144.0 * np.arange(5, dtype=float))
                a = (2.0 / 5.0) * np.sum(v * np.cos(theta), axis=2)
                b = (-2.0 / 5.0) * np.sum(v * np.sin(theta), axis=2)
                amp = np.sqrt(a * a + b * b)
                phase = np.zeros_like(amp)
                nonzero = valid & (amp > 0.0)
                if np.any(nonzero):
                    cp = np.clip(a[nonzero] / amp[nonzero], -1.0, 1.0)
                    phase_nonzero = np.degrees(np.arccos(cp))
                    phase_nonzero[b[nonzero] < 0.0] = 360.0 - phase_nonzero[b[nonzero] < 0.0]
                    phase[nonzero] = phase_nonzero
                amp_out = np.zeros_like(amp)
                amp_out[valid] = amp[valid]
                sugar_pucker[:, strands, levels, 0] = amp_out
                sugar_pucker[:, strands, levels, 1] = phase

        return torsions, sugar_pucker

    def _backbone_rows(self, torsions: np.ndarray, sugar_pucker: np.ndarray) -> List[Dict]:
        rows = []
        for strand in range(self.ctx.nst):
            for level in range(1, self.ctx.nux + 1):
                if not self._has_level(strand, level):
                    continue
                label = self._residue_unit_label(strand, level)
                if label is None:
                    continue
                residue_name, residue_id = label
                tor = torsions[strand, level]
                pucker = sugar_pucker[strand, level]
                phase = self._json_number(pucker[1])
                pucker_index = 0
                if phase is not None:
                    pucker_index = int((phase % 360.0) / 36.0)
                    pucker_index = max(0, min(len(SUGAR_PUCKERS) - 1, pucker_index))
                rows.append({
                    "strand": strand + 1,
                    "level": level,
                    "residue_name": residue_name,
                    "residue_id": residue_id,
                    "c1_c2": self._json_number(tor[4]),
                    "c2_c3": self._json_number(tor[5]),
                    "phase": phase,
                    "amplitude": self._json_number(pucker[0]),
                    "pucker": SUGAR_PUCKERS[pucker_index],
                    "c1_prime": self._json_number(tor[0]),
                    "c2_prime": self._json_number(tor[1]),
                    "c3_prime": self._json_number(tor[2]),
                    "chi": self._json_number(tor[12]),
                    "gamma": self._json_number(tor[8]),
                    "delta": self._json_number(tor[9]),
                    "epsilon": self._json_number(tor[6]),
                    "zeta": self._json_number(tor[7]),
                    "alpha": self._json_number(tor[10]),
                    "beta": self._json_number(tor[11]),
                })
        return rows

    def _residue_unit_label(self, strand: int, level: int):
        if level < 1 or level > self.ctx.ni_map.shape[1]:
            return None
        subunit_idx = int(self.ctx.ni_map[strand, level - 1])
        if subunit_idx <= 0:
            return None
        atom_idx = int(self.template_molecule.subunit_boundaries[subunit_idx - 1])
        residue_name = str(self.template_molecule.residue_names[atom_idx]).strip()
        chain = ""
        if self.template_molecule.chain_ids is not None:
            chain = str(self.template_molecule.chain_ids[atom_idx]).strip()
        return f"{residue_name}{chain}".strip(), int(self.template_molecule.residue_ids[atom_idx])

    @staticmethod
    def _json_number(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def _curvesplus_inter_base_pair_rows(self, values: np.ndarray) -> List[Dict]:
        return self._local_inter_base_pair_rows(values)

    def _base_fit_quality_rows(self, rmsd: np.ndarray) -> List[Dict]:
        rows = []
        for item in self.fit_templates:
            label = self._residue_labels.get((item.strand, item.level))
            if label is None:
                continue
            base, chain_id, residue_id = label
            rows.append({
                "strand": item.strand + 1,
                "level": item.level,
                "subunit": item.subunit,
                "chain_id": chain_id,
                "residue_id": residue_id,
                "residue_name": item.residue_name,
                "parent_base": item.parent_base,
                "base": base,
                "frame_convention": "standard",
                "fit_atom_count": int(len(item.atom_indices)),
                "rmsd": self._json_number(rmsd[item.strand, item.level]),
            })
        return rows

    def _local_inter_base_rows(self, values: np.ndarray) -> List[Dict]:
        rows = []
        for strand in range(self.ctx.nst):
            for level in range(1, self.ctx.nux):
                if not (self._has_level(strand, level) and self._has_level(strand, level + 1)):
                    continue
                rows.append(_parameter_row(
                    values[strand, level],
                    STEP_PARAMETERS,
                    strand=strand + 1,
                    level=level,
                    step=self._step_label(strand, level + 1),
                ))
        return rows

    def _local_base_base_rows(self, values: np.ndarray) -> List[Dict]:
        rows = []
        for sequence_index, level in enumerate(self.primary_levels, start=1):
            rows.append(_parameter_row(
                values[level] if self._has_level(1, level) else None,
                BASE_BASE_PARAMETERS,
                partner_strand=2,
                sequence_index=sequence_index,
                level=level,
                duplex=self._duplex_id(level),
            ))
        return rows

    def _local_inter_base_pair_rows(self, values: np.ndarray) -> List[Dict]:
        rows = []
        for sequence_index, (level, next_level) in enumerate(zip(self.primary_levels, self.primary_levels[1:]), start=1):
            has_step = (
                next_level == level + 1
                and self._has_level(1, level)
                and self._has_level(1, next_level)
            )
            rows.append(_parameter_row(
                values[level] if has_step else None,
                STEP_PARAMETERS,
                partner_strand=2,
                sequence_index=sequence_index,
                level=level,
                next_level=next_level,
                step=self._step_label_between(level, next_level),
                duplex=f"{self._duplex_id(level)}/{self._duplex_id(next_level)}",
            ))
        return rows

    def _curvesplus_base_pair_axis_rows(self, values: np.ndarray) -> List[Dict]:
        rows = []
        for sequence_index, level in enumerate(self.primary_levels, start=1):
            rows.append(_parameter_row(
                values[level] if self._has_level(1, level) else None,
                AXIS_PARAMETERS,
                partner_strand=2,
                sequence_index=sequence_index,
                level=level,
                duplex=self._duplex_id(level),
            ))
        return rows

    def _has_level(self, strand: int, level: int) -> bool:
        return 0 <= strand < self.ctx.nst and 1 <= level <= self.ctx.nux and self.ctx.li[level, strand] >= 0

    def _residue_label(self, strand: int, level: int):
        subunit_idx = int(self.ctx.ni_map[strand, level - 1])
        if subunit_idx <= 0:
            return None
        start = int(self.template_molecule.subunit_boundaries[subunit_idx - 1])
        residue_name = str(self.template_molecule.residue_names[start]).strip().upper()
        base = parent_base_name(residue_name)
        if base == "unknown":
            base = residue_name[:1]
        if len(base) >= 2 and base[0] == "D" and base[1] in "GACTUIYP":
            base = base[1]
        else:
            base = base[:1]
        chain = str(self.template_molecule.chain_ids[start]).strip() if self.template_molecule.chain_ids is not None else ""
        return base, chain, int(self.template_molecule.residue_ids[start])

    def _duplex_id(self, level: int) -> str:
        first = self._residue_labels.get((0, level))
        second = self._residue_labels.get((1, level))
        if first is None or second is None:
            return ""
        return f"{first[0]}{first[2]:3d}-{second[0]}{second[2]:3d}"

    def _step_label(self, strand: int, level: int) -> str:
        first = self._residue_labels.get((strand, level - 1))
        second = self._residue_labels.get((strand, level))
        if first is None or second is None:
            return "-"
        return f"{first[0]}{first[2]:3d}/{second[0]}{second[2]:3d}"

    def _step_label_between(self, first_level: int, second_level: int) -> str:
        first = self._residue_labels.get((0, first_level))
        second = self._residue_labels.get((0, second_level))
        if first is None or second is None:
            return "-"
        return f"{first[0]}{first[2]:3d}/{second[0]}{second[2]:3d}"










