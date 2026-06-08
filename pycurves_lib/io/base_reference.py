from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ATOM_ALIASES = {
    "C1*": ("C1*", "C1'"),
    "C2*": ("C2*", "C2'"),
    "C3*": ("C3*", "C3'"),
    "C4*": ("C4*", "C4'"),
    "C5*": ("C5*", "C5'"),
    "O3*": ("O3*", "O3'"),
    "O4*": ("O4*", "O4'"),
    "O5*": ("O5*", "O5'"),
}

# Coordinates from the standard reference frame table.  They are used
# to express the final base-frame origin and axes in the coordinates of the
# Curves+ fitting templates from `standard_b.lib`.
STANDARD_BASE_COORDS = {
    "A": {
        "C1*": (-2.479, 5.346, 0.000),
        "N9": (-1.291, 4.498, 0.000),
        "C8": (0.024, 4.897, 0.000),
        "N7": (0.877, 3.902, 0.000),
        "C5": (0.071, 2.771, 0.000),
        "C6": (0.369, 1.398, 0.000),
        "N1": (-0.668, 0.532, 0.000),
        "C2": (-1.912, 1.023, 0.000),
        "N3": (-2.320, 2.290, 0.000),
        "C4": (-1.267, 3.124, 0.000),
    },
    "C": {
        "C1*": (-2.477, 5.402, 0.000),
        "N1": (-1.285, 4.542, 0.000),
        "C2": (-1.472, 3.158, 0.000),
        "N3": (-0.391, 2.344, 0.000),
        "C4": (0.837, 2.868, 0.000),
        "C5": (1.056, 4.275, 0.000),
        "C6": (-0.023, 5.068, 0.000),
    },
    "G": {
        "C1*": (-2.477, 5.399, 0.000),
        "N9": (-1.289, 4.551, 0.000),
        "C8": (0.023, 4.962, 0.000),
        "N7": (0.870, 3.969, 0.000),
        "C5": (0.071, 2.833, 0.000),
        "C6": (0.424, 1.460, 0.000),
        "N1": (-0.700, 0.641, 0.000),
        "C2": (-1.999, 1.087, 0.000),
        "N3": (-2.342, 2.364, 0.001),
        "C4": (-1.265, 3.177, 0.000),
    },
    "T": {
        "C1*": (-2.481, 5.354, 0.000),
        "N1": (-1.284, 4.500, 0.000),
        "C2": (-1.462, 3.135, 0.000),
        "N3": (-0.298, 2.407, 0.000),
        "C4": (0.994, 2.897, 0.000),
        "C5": (1.106, 4.338, 0.000),
        "C6": (-0.024, 5.057, 0.000),
    },
    "U": {
        "C1*": (-2.481, 5.354, 0.000),
        "N1": (-1.284, 4.500, 0.000),
        "C2": (-1.462, 3.131, 0.000),
        "N3": (-0.302, 2.397, 0.000),
        "C4": (0.989, 2.884, 0.000),
        "C5": (1.089, 4.311, 0.000),
        "C6": (-0.024, 5.053, 0.000),
    },
}

STANDARD_FRAME_EQUIVALENTS = {
    "I": "G",
    "Y": "G",
    "P": "U",
}


def canonical_atom_name(name: str) -> str:
    """Normalize PDB/mmCIF atom names without losing prime/star aliases."""
    clean = str(name).strip().strip('"').upper()
    return clean


def atom_aliases(name: str) -> Tuple[str, ...]:
    clean = canonical_atom_name(name)
    if clean in ATOM_ALIASES:
        return ATOM_ALIASES[clean]
    if clean.endswith("'"):
        return (clean, clean[:-1] + "*")
    return (clean,)


@dataclass(frozen=True)
class BaseReferenceTemplate:
    code: str
    family: str
    full_name: str
    atom_names: Tuple[str, ...]
    coordinates: np.ndarray
    frame_origin: Optional[np.ndarray] = None
    frame_axes: Optional[np.ndarray] = None

    @property
    def atom_map(self) -> Dict[str, np.ndarray]:
        return dict(zip(self.atom_names, self.coordinates))

    @property
    def glycosidic_atom(self) -> str:
        return self.atom_names[1]

    @property
    def normal_atom(self) -> str:
        return self.atom_names[2]

    @property
    def groove_atom(self) -> str:
        return self.atom_names[3]

    @property
    def reference_axes(self) -> np.ndarray:
        """Return row-vector standard axes expressed in template coordinates."""
        # Curves+ `standard_b.lib` defines the fitting template coordinate
        # system directly: the first two atoms define the glycosidic bond and
        # the third atom defines the standard base normal.  The optional
        # frame_origin maps the standard origin into this template system, but
        # the axes should stay tied to the intrinsic Curves+ template axes.
        glycosidic_vector = self.coordinates[0] - self.coordinates[1]
        normal_vector = np.cross(glycosidic_vector, self.coordinates[2] - self.coordinates[1])
        z_axis = self._unit(normal_vector)
        x_axis = np.array([-1.0, 0.0, 0.0], dtype=float)
        x_axis = self._unit(x_axis - np.dot(x_axis, z_axis) * z_axis)
        y_axis = self._unit(np.cross(z_axis, x_axis))
        x_axis = self._unit(np.cross(y_axis, z_axis))
        return np.asarray([x_axis, y_axis, z_axis], dtype=float)

    @staticmethod
    def _unit(vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return arr
        return arr / norm


class BaseReferenceLibrary:
    """Convention-aware standard base templates used by the base-frame fitter."""

    def __init__(self, templates: Dict[str, BaseReferenceTemplate], source: str, convention: str):
        self.templates = templates
        self.source = source
        self.convention = convention

    @classmethod
    def load(cls, convention: str = "legacy", path: Optional[Path | str] = None) -> "BaseReferenceLibrary":
        name = convention.strip().lower().replace("-", "_")
        if name in {"legacy"}:
            return cls({}, "setup.f hardcoded bref", "legacy")
        if name in {"standard", "curves_plus", "curves+", "curvesplus", "x3dna", "3dna"}:
            if path is not None:
                lib_path = Path(path)
            else:
                package_path = Path(__file__).resolve().parents[1] / "data" / "reference" / "standard_b.lib"
                legacy_path = Path(__file__).resolve().parents[2] / "reference" / "standard_b.lib"
                lib_path = package_path if package_path.exists() else legacy_path
            return cls(cls._parse_standard_b_lib(lib_path), str(lib_path), "standard")
        raise ValueError(f"Unknown base frame convention {convention!r}; use 'standard' or 'legacy'.")

    @staticmethod
    def _parse_standard_b_lib(path: Path) -> Dict[str, BaseReferenceTemplate]:
        if not path.exists():
            raise FileNotFoundError(f"Standard base library not found: {path}")

        templates: Dict[str, BaseReferenceTemplate] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        idx = 0
        header_re = re.compile(r"^([A-Za-z])\s+([RY])\s+(\d+)\s+'?([^']*)'?", re.I)
        atom_re = re.compile(
            r"^\s*([-+]?\d+(?:\.\d*)?)\s+([-+]?\d+(?:\.\d*)?)\s+([-+]?\d+(?:\.\d*)?)\s+'?([^']+)'?\s*$"
        )

        while idx < len(lines):
            line = lines[idx].strip()
            idx += 1
            if not line or line.startswith("#"):
                continue
            match = header_re.match(line)
            if not match:
                continue
            code, family, count_text, full_name = match.groups()
            count = int(count_text)
            atom_names: List[str] = []
            coords: List[List[float]] = []
            while idx < len(lines) and len(atom_names) < count:
                atom_line = lines[idx]
                idx += 1
                atom_match = atom_re.match(atom_line)
                if not atom_match:
                    continue
                x, y, z, atom_name = atom_match.groups()
                atom_names.append(canonical_atom_name(atom_name))
                coords.append([float(x), float(y), float(z)])
            if len(atom_names) != count:
                raise ValueError(f"Incomplete base template {code!r} in {path}")
            code = code.upper()
            coordinates_array = np.asarray(coords, dtype=float)
            frame_origin, _frame_axes = BaseReferenceLibrary._standard_frame_in_template_coordinates(
                code,
                tuple(atom_names),
                coordinates_array,
            )
            templates[code] = BaseReferenceTemplate(
                code=code.upper(),
                family=family.upper(),
                full_name=full_name.strip(),
                atom_names=tuple(atom_names),
                coordinates=coordinates_array,
                frame_origin=frame_origin,
                frame_axes=None,
            )
        return templates

    @staticmethod
    def _standard_frame_in_template_coordinates(
        code: str,
        atom_names: Tuple[str, ...],
        coordinates: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        standard_code = STANDARD_FRAME_EQUIVALENTS.get(code, code)
        standard = STANDARD_BASE_COORDS.get(standard_code)
        if standard is None:
            return None, None

        template_map = dict(zip(atom_names, coordinates))
        reference_coords = []
        standard_coords = []
        for atom_name, coord in template_map.items():
            if atom_name not in standard:
                continue
            reference_coords.append(coord)
            standard_coords.append(standard[atom_name])

        if len(reference_coords) < 3:
            return None, None

        reference = np.asarray(reference_coords, dtype=float)
        standard_array = np.asarray(standard_coords, dtype=float)
        ref_centroid = reference.mean(axis=0)
        std_centroid = standard_array.mean(axis=0)
        ref_shifted = reference - ref_centroid
        std_shifted = standard_array - std_centroid

        covariance = ref_shifted.T @ std_shifted
        u_mat, _, vt_mat = np.linalg.svd(covariance)
        handedness = np.sign(np.linalg.det(vt_mat.T @ u_mat.T))
        if handedness == 0.0:
            handedness = 1.0
        rotation = vt_mat.T @ np.diag([1.0, 1.0, handedness]) @ u_mat.T

        # standard = (template - ref_centroid) @ rotation.T + std_centroid.
        # Solve the standard origin and basis vectors back into template
        # coordinates so the later residue fit can map them to the structure.
        origin = (-std_centroid) @ rotation + ref_centroid
        axes = np.eye(3, dtype=float) @ rotation
        return origin, axes

    def template_for_base(self, base: str) -> Optional[BaseReferenceTemplate]:
        code = str(base).strip().upper()
        if code == "R":
            code = "A"
        if code == "Y":
            code = "C"
        return self.templates.get(code)

    def reference_origin_for_base(self, template: BaseReferenceTemplate) -> np.ndarray:
        """Return the standard base reference point in template coordinates.

        `standard_b.lib` coordinates are already expressed relative to the
        standard base-frame origin, so the origin is the coordinate-system
        origin. Curves 5.3 uses a different constructed origin and is handled
        by the legacy `locate.f`-style path instead of this library path.
        """
        if self.convention != "standard":
            return np.zeros(3, dtype=float)
        if template.frame_origin is not None:
            return np.asarray(template.frame_origin, dtype=float)
        return np.zeros(3, dtype=float)


class BaseFrameFitter:
    """Fit a residue to a standard base and return the mapped reference frame."""

    def __init__(self, library: BaseReferenceLibrary):
        self.library = library

    def fit(
        self,
        template: BaseReferenceTemplate,
        residue_atoms: Dict[str, int],
        coordinates: np.ndarray,
        atom_order: Optional[Iterable[str]] = None,
    ) -> Optional[Dict[str, object]]:
        names = list(atom_order) if atom_order is not None else list(template.atom_names)
        template_map = template.atom_map
        fit_indices: List[int] = []
        reference_coords: List[np.ndarray] = []
        used_names: List[str] = []

        for ref_name in names:
            if ref_name not in template_map:
                continue
            atom_idx = self._residue_atom_index(residue_atoms, ref_name)
            if atom_idx is None:
                continue
            fit_indices.append(atom_idx)
            reference_coords.append(template_map[ref_name])
            used_names.append(ref_name)

        if len(fit_indices) < 3:
            return None

        reference = np.asarray(reference_coords, dtype=float)
        observed = np.asarray(coordinates[fit_indices], dtype=float)
        ref_centroid = reference.mean(axis=0)
        obs_centroid = observed.mean(axis=0)
        ref_shifted = reference - ref_centroid
        obs_shifted = observed - obs_centroid

        covariance = ref_shifted.T @ obs_shifted
        u_mat, _, vt_mat = np.linalg.svd(covariance)
        handedness = np.sign(np.linalg.det(vt_mat.T @ u_mat.T))
        if handedness == 0.0:
            handedness = 1.0
        rotation = vt_mat.T @ np.diag([1.0, 1.0, handedness]) @ u_mat.T

        fitted = ref_shifted @ rotation.T + obs_centroid
        residual = observed - fitted
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

        reference_origin = self.library.reference_origin_for_base(template)
        origin = (reference_origin - ref_centroid) @ rotation.T + obs_centroid
        reference_axes = template.reference_axes
        x_axis = reference_axes[0] @ rotation.T
        y_axis = reference_axes[1] @ rotation.T
        z_axis = reference_axes[2] @ rotation.T

        x_axis = self._unit(x_axis)
        y_axis = self._unit(y_axis - np.dot(y_axis, x_axis) * x_axis)
        z_axis = self._unit(np.cross(x_axis, y_axis))
        y_axis = self._unit(np.cross(z_axis, x_axis))

        all_fitted = (template.coordinates - ref_centroid) @ rotation.T + obs_centroid
        fitted_by_atom = dict(zip(template.atom_names, all_fitted))

        return {
            "origin": origin,
            "axes": np.asarray([x_axis, y_axis, z_axis], dtype=float),
            "fitted_by_atom": fitted_by_atom,
            "rmsd": rms,
            "fit_atom_names": used_names,
            "fit_indices": fit_indices,
        }

    @staticmethod
    def _residue_atom_index(residue_atoms: Dict[str, int], atom_name: str) -> Optional[int]:
        for alias in atom_aliases(atom_name):
            if alias in residue_atoms:
                return residue_atoms[alias]
        return None

    @staticmethod
    def _unit(vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return arr
        return arr / norm
