"""Conservative Leontis-Westhof classification from fitted base frames.

The bundled envelopes are a compact, data-only extraction of the official
FR3D classifier. They let pyCurves recognize all 18 directed LW families
without importing FR3D or depending on raw atom-coordinate edge votes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from importlib.resources import files
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


LW_FAMILIES = frozenset(
    f"{orientation}{first}{second}"
    for orientation in "ct"
    for first in "WHS"
    for second in "WHS"
)

_BASE_EQUIVALENTS = {
    "A": "A",
    "DA": "A",
    "R": "A",
    "C": "C",
    "DC": "C",
    "Y": "C",
    "G": "G",
    "DG": "G",
    "I": "G",
    "DI": "G",
    "P": "G",
    "U": "U",
    "DU": "U",
    "T": "U",
    "DT": "U",
}
_FRAME_KEYS = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "T"}
_GLYCOSIDIC_ATOMS = {"A": "N9", "G": "N9", "C": "N1", "U": "N1", "T": "N1"}


def reverse_lw_tag(tag: str) -> str:
    """Reverse a directed LW tag while preserving cis/trans orientation."""
    tag = str(tag).strip()
    if len(tag) != 3 or tag[0].lower() not in {"c", "t"}:
        raise ValueError(f"Invalid LW tag: {tag!r}")
    edges = tag[1:].upper()
    if any(edge not in "WHS" for edge in edges):
        raise ValueError(f"Invalid LW tag: {tag!r}")
    return f"{tag[0].lower()}{edges[1]}{edges[0]}"


@dataclass(frozen=True)
class LWDescriptor:
    """FR3D-style relative geometry for one directed base pair."""

    x: float
    y: float
    z: float
    normal: float
    angle: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "normal": float(self.normal),
            "angle": float(self.angle),
        }


@dataclass(frozen=True)
class LWClassification:
    """Best family assignment and its conservative confidence evidence."""

    tag: str
    score: float
    margin: float
    confident: bool
    variant: str
    subcategory: int
    direction: str
    descriptor: LWDescriptor
    candidate_scores: Mapping[str, float]
    source: Mapping[str, Any]

    @property
    def glycosidic_orientation(self) -> str:
        return "cis" if self.tag.startswith("c") else "trans"

    @property
    def edge_1(self) -> str:
        return self.tag[1]

    @property
    def edge_2(self) -> str:
        return self.tag[2]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "score": float(self.score),
            "margin": None if not math.isfinite(self.margin) else float(self.margin),
            "confident": bool(self.confident),
            "variant": self.variant,
            "subcategory": int(self.subcategory),
            "direction": self.direction,
            "descriptor": self.descriptor.as_dict(),
            "candidate_scores": dict(self.candidate_scores),
            "source": dict(self.source),
        }


class LWExemplarLibrary:
    """Classify fitted standard frames against FR3D family envelopes.

    Only exact envelope membership is eligible. The normalized centre score
    and family separation margin make boundary-like or ambiguous assignments
    diagnostic rather than confident.
    """

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        max_center_score: float = 0.95,
        min_family_margin: float = 0.10,
    ):
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported LW exemplar schema")
        self.source = dict(payload.get("source", {}))
        self.max_center_score = float(max_center_score)
        self.min_family_margin = float(min_family_margin)
        self.records = tuple(dict(record) for record in payload.get("records", ()))
        families = {str(record.get("family", "")) for record in self.records}
        if families != LW_FAMILIES:
            missing = sorted(LW_FAMILIES - families)
            extra = sorted(families - LW_FAMILIES)
            raise ValueError(
                f"Incomplete LW exemplar library; missing={missing}, extra={extra}"
            )

        conversions = payload.get("frame_conversions", {})
        self.frame_rotations = {
            str(base): np.asarray(value["rotation"], dtype=float)
            for base, value in conversions.items()
        }
        for base in ("A", "C", "G", "U", "T"):
            matrix = self.frame_rotations.get(base)
            if (
                matrix is None
                or matrix.shape != (3, 3)
                or not np.all(np.isfinite(matrix))
            ):
                raise ValueError(f"Invalid frame conversion for {base}")

    @classmethod
    def load(cls) -> "LWExemplarLibrary":
        path = files("pycurves_lib.data").joinpath("reference/lw_exemplars.json")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        library = cls(payload)
        library.source.setdefault(
            "bundled_resource",
            "pycurves_lib.data/reference/lw_exemplars.json",
        )
        return library

    def classify_fits(
        self,
        base_1: str,
        fit_1: Optional[Mapping[str, object]],
        base_2: str,
        fit_2: Optional[Mapping[str, object]],
    ) -> Optional[LWClassification]:
        """Classify two pyCurves standard-frame fits in the given direction."""
        normalized_1 = self._normalize_base(base_1)
        normalized_2 = self._normalize_base(base_2)
        frame_1 = self._frame_key(base_1)
        frame_2 = self._frame_key(base_2)
        if not normalized_1 or not normalized_2 or not frame_1 or not frame_2:
            return None
        descriptor = self.descriptor_from_fits(frame_1, fit_1, frame_2, fit_2)
        reverse_descriptor = self.descriptor_from_fits(frame_2, fit_2, frame_1, fit_1)
        if descriptor is None or reverse_descriptor is None:
            return None
        return self.classify_descriptors(
            normalized_1,
            normalized_2,
            descriptor,
            reverse_descriptor=reverse_descriptor,
        )

    def descriptor_from_fits(
        self,
        base_1: str,
        fit_1: Optional[Mapping[str, object]],
        base_2: str,
        fit_2: Optional[Mapping[str, object]],
    ) -> Optional[LWDescriptor]:
        """Convert two fitted pyCurves frames to FR3D relative descriptors."""
        if fit_1 is None or fit_2 is None:
            return None
        try:
            axes_1 = np.asarray(fit_1["axes"], dtype=float)
            axes_2 = np.asarray(fit_2["axes"], dtype=float)
            fitted_1 = fit_1["fitted_by_atom"]
            fitted_2 = fit_2["fitted_by_atom"]
            gly_1 = np.asarray(fitted_1[_GLYCOSIDIC_ATOMS[base_1]], dtype=float)
            gly_2 = np.asarray(fitted_2[_GLYCOSIDIC_ATOMS[base_2]], dtype=float)
            conversion_1 = self.frame_rotations[base_1]
            conversion_2 = self.frame_rotations[base_2]
        except (KeyError, TypeError, ValueError):
            return None
        if axes_1.shape != (3, 3) or axes_2.shape != (3, 3):
            return None

        # pyCurves stores basis vectors as rows. FR3D stores the corresponding
        # local basis as columns after a base-specific frame conversion.
        basis_1 = axes_1.T @ conversion_1.T
        basis_2 = axes_2.T @ conversion_2.T
        displacement = (gly_2 - gly_1) @ basis_1
        rotation = basis_1.T @ basis_2
        angle = math.degrees(math.atan2(rotation[1, 1], rotation[0, 1])) - 90.0
        if angle <= -90.0:
            angle += 360.0
        values = np.asarray([*displacement, rotation[2, 2], angle], dtype=float)
        if not np.all(np.isfinite(values)):
            return None
        return LWDescriptor(*map(float, values))

    def classify_descriptors(
        self,
        base_1: str,
        base_2: str,
        descriptor: LWDescriptor,
        *,
        reverse_descriptor: Optional[LWDescriptor] = None,
    ) -> Optional[LWClassification]:
        """Classify already-computed descriptors; useful for tests and audits."""
        base_1 = self._normalize_base(base_1) or ""
        base_2 = self._normalize_base(base_2) or ""
        if not base_1 or not base_2:
            return None

        candidates = list(
            self._matching_records(base_1 + base_2, descriptor, "forward")
        )
        if reverse_descriptor is not None:
            for record, score, _direction in self._matching_records(
                base_2 + base_1, reverse_descriptor, "reverse"
            ):
                reversed_record = dict(record)
                reversed_record["family"] = reverse_lw_tag(str(record["family"]))
                candidates.append((reversed_record, score, "reverse"))
        if not candidates:
            return None

        best_by_family: Dict[str, Tuple[float, Mapping[str, Any], str]] = {}
        for record, score, direction in candidates:
            family = str(record["family"])
            current = best_by_family.get(family)
            if current is None or score < current[0]:
                best_by_family[family] = (score, record, direction)

        ranked = sorted(
            best_by_family.items(),
            key=lambda item: (item[1][0], item[0]),
        )
        tag, (score, record, direction) = ranked[0]
        margin = float("inf") if len(ranked) == 1 else float(ranked[1][1][0] - score)
        confident = score <= self.max_center_score and margin >= self.min_family_margin
        return LWClassification(
            tag=tag,
            score=float(score),
            margin=margin,
            confident=confident,
            variant=str(record.get("variant", tag)),
            subcategory=int(record.get("subcategory", 0)),
            direction=direction,
            descriptor=descriptor,
            candidate_scores={family: float(values[0]) for family, values in ranked},
            source=self.source,
        )

    def _matching_records(
        self,
        bases: str,
        descriptor: LWDescriptor,
        direction: str,
    ) -> Iterable[Tuple[Mapping[str, Any], float, str]]:
        for record in self.records:
            if record["bases"] != bases:
                continue
            if not self._inside(record, descriptor):
                continue
            yield record, self._center_score(record, descriptor), direction

    @classmethod
    def _inside(
        cls,
        record: Mapping[str, Any],
        descriptor: LWDescriptor,
    ) -> bool:
        return (
            math.hypot(descriptor.x, descriptor.y)
            <= float(record.get("radiusmax", float("inf")))
            and float(record["xmin"]) <= descriptor.x <= float(record["xmax"])
            and float(record["ymin"]) <= descriptor.y <= float(record["ymax"])
            and float(record["zmin"]) <= descriptor.z <= float(record["zmax"])
            and float(record["normalmin"])
            <= descriptor.normal
            <= float(record["normalmax"])
            and cls._angle_inside(
                descriptor.angle,
                float(record["anglemin"]),
                float(record["anglemax"]),
            )
        )

    @classmethod
    def _center_score(
        cls,
        record: Mapping[str, Any],
        descriptor: LWDescriptor,
    ) -> float:
        terms = []
        for name, value in (
            ("x", descriptor.x),
            ("y", descriptor.y),
            ("z", descriptor.z),
            ("normal", descriptor.normal),
        ):
            low = float(record[f"{name}min"])
            high = float(record[f"{name}max"])
            half_width = max((high - low) / 2.0, 1.0e-12)
            terms.append((value - (low + high) / 2.0) / half_width)

        angle_center, angle_half_width = cls._angle_center_and_half_width(
            float(record["anglemin"]),
            float(record["anglemax"]),
        )
        terms.append(
            cls._signed_angle_delta(descriptor.angle, angle_center) / angle_half_width
        )
        return float(np.sqrt(np.mean(np.square(terms))))

    @staticmethod
    def _angle_inside(angle: float, low: float, high: float) -> bool:
        if low < high:
            return low <= angle <= high
        return angle >= low or angle <= high

    @staticmethod
    def _angle_center_and_half_width(
        low: float,
        high: float,
    ) -> Tuple[float, float]:
        if low < high:
            return (
                (low + high) / 2.0,
                max((high - low) / 2.0, 1.0e-12),
            )
        adjusted_high = high + 360.0
        center = (low + adjusted_high) / 2.0
        if center > 270.0:
            center -= 360.0
        return center, max((adjusted_high - low) / 2.0, 1.0e-12)

    @staticmethod
    def _signed_angle_delta(angle: float, center: float) -> float:
        return (angle - center + 180.0) % 360.0 - 180.0

    @staticmethod
    def _normalize_base(base: str) -> Optional[str]:
        return _BASE_EQUIVALENTS.get(str(base).strip().upper())

    @staticmethod
    def _frame_key(base: str) -> Optional[str]:
        text = str(base).strip().upper()
        if text in {"DT", "T"}:
            return "T"
        normalized = _BASE_EQUIVALENTS.get(text)
        return _FRAME_KEYS.get(normalized or "")


@lru_cache(maxsize=1)
def get_lw_exemplar_library() -> LWExemplarLibrary:
    """Return the process-wide LW exemplar library."""
    return LWExemplarLibrary.load()
