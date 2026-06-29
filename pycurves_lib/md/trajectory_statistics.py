from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


CIRCULAR_DEGREE_COLUMNS = frozenset({
    "tilt", "roll", "twist",
    "buckle", "propel", "opening",
    "inclin", "tip",
    "ainc", "atip", "angle",
    "minor_angle", "major_angle",
    "local_direction", "overall_bend_uu", "overall_bend_pp",
    "c1_c2", "c2_c3", "phase",
    "c1_prime", "c2_prime", "c3_prime",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi",
})

SUGAR_PUCKERS = (
    "C3'-endo", "C4'-exo", "O1'-endo", "C1'-exo", "C2'-endo",
    "C3'-exo", "C4'-endo", "O1'-exo", "C1'-endo", "C2'-exo",
)

BI_BII_STATES = ("BI", "BII")
ALPHA_GAMMA_STATES = (
    ("g-", "g-"), ("g-", "t"), ("g-", "g+"),
    ("t", "g-"), ("t", "t"), ("t", "g+"),
    ("g+", "g-"), ("g+", "t"), ("g+", "g+"),
)


class SummaryStats(NamedTuple):
    mean: Optional[float]
    stddev: Optional[float]
    variance: Optional[float]


def is_circular_degree_column(column_name: str) -> bool:
    return str(column_name).lower() in CIRCULAR_DEGREE_COLUMNS


def wrap_degrees_180(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if np.isclose(wrapped, -180.0) else wrapped


def _stddev_from_variance(variance: Optional[float]) -> Optional[float]:
    if variance is None:
        return None
    return float(np.sqrt(max(float(variance), 0.0)))


def linear_summary(values: np.ndarray) -> SummaryStats:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return SummaryStats(None, None, None)
    mean = float(np.mean(vals))
    variance = float(np.var(vals))
    if abs(variance) < 1e-15:
        variance = 0.0
    return SummaryStats(mean, _stddev_from_variance(variance), variance)


def circular_degree_mean_from_sums(
    sin_sum: float,
    cos_sum: float,
    count: int,
) -> Optional[float]:
    if count <= 0:
        return None
    resultant = float(np.hypot(sin_sum, cos_sum))
    if resultant <= 1e-12:
        return None
    return wrap_degrees_180(np.degrees(np.arctan2(sin_sum, cos_sum)))


def circular_degree_summary(values: np.ndarray) -> SummaryStats:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return SummaryStats(None, None, None)
    radians = np.radians(vals)
    mean = circular_degree_mean_from_sums(
        float(np.sum(np.sin(radians))),
        float(np.sum(np.cos(radians))),
        int(vals.size),
    )
    if mean is None:
        return SummaryStats(None, None, None)
    residuals = np.asarray([wrap_degrees_180(value - mean) for value in vals], dtype=float)
    variance = float(np.mean(residuals * residuals))
    if abs(variance) < 1e-15:
        variance = 0.0
    return SummaryStats(mean, _stddev_from_variance(variance), variance)


def sugar_pucker_counts(phase_values: Sequence[float]) -> Tuple[np.ndarray, int]:
    phase = np.asarray(phase_values, dtype=float)
    valid = np.isfinite(phase)
    counts = np.zeros(len(SUGAR_PUCKERS), dtype=int)
    if np.any(valid):
        indices = np.asarray((phase[valid] % 360.0) / 36.0, dtype=int)
        indices = np.clip(indices, 0, len(SUGAR_PUCKERS) - 1)
        counts += np.bincount(indices, minlength=len(SUGAR_PUCKERS))[:len(SUGAR_PUCKERS)]
    return counts, int(np.sum(valid))


def bi_bii_counts(epsilon_values: Sequence[float], zeta_values: Sequence[float]) -> Tuple[np.ndarray, int]:
    epsilon = np.asarray(epsilon_values, dtype=float)
    zeta = np.asarray(zeta_values, dtype=float)
    valid = np.isfinite(epsilon) & np.isfinite(zeta)
    counts = np.zeros(len(BI_BII_STATES), dtype=int)
    if np.any(valid):
        diff = epsilon[valid] - zeta[valid]
        diff = np.where(np.abs(diff) > 180.0, diff - np.sign(diff) * 360.0, diff)
        counts[0] = int(np.sum(diff < 0.0))
        counts[1] = int(np.sum(diff >= 0.0))
    return counts, int(np.sum(valid))


def _gauche_trans_state(values: np.ndarray) -> np.ndarray:
    state = np.full(values.shape, 1, dtype=int)
    within_gauche = np.abs(values) < 120.0
    state[within_gauche & (values < 0.0)] = 0
    state[within_gauche & (values >= 0.0)] = 2
    return state


def alpha_gamma_counts(alpha_values: Sequence[float], gamma_values: Sequence[float]) -> Tuple[np.ndarray, int]:
    alpha = np.asarray(alpha_values, dtype=float)
    gamma = np.asarray(gamma_values, dtype=float)
    valid = np.isfinite(alpha) & np.isfinite(gamma)
    counts = np.zeros(len(ALPHA_GAMMA_STATES), dtype=int)
    if np.any(valid):
        alpha_state = _gauche_trans_state(alpha[valid])
        gamma_state = _gauche_trans_state(gamma[valid])
        indices = alpha_state * 3 + gamma_state
        counts += np.bincount(indices, minlength=len(ALPHA_GAMMA_STATES))[:len(ALPHA_GAMMA_STATES)]
    return counts, int(np.sum(valid))


def population_rows(
    metadata: Dict[str, Any],
    category_rows: Sequence[Dict[str, Any]],
    counts: Sequence[int],
    total_count: int,
) -> List[Dict[str, Any]]:
    if total_count <= 0:
        return []
    rows = []
    for category, count_value in zip(category_rows, counts):
        count = int(count_value)
        fraction = count / float(total_count)
        row = dict(metadata)
        row.update(category)
        row["count"] = count
        row["total_count"] = int(total_count)
        row["fraction"] = float(fraction)
        row["percent"] = float(100.0 * fraction)
        rows.append(row)
    return rows


def sugar_pucker_population_rows(metadata: Dict[str, Any], phase_values: Sequence[float]) -> List[Dict[str, Any]]:
    counts, total = sugar_pucker_counts(phase_values)
    categories = [{"pucker": label} for label in SUGAR_PUCKERS]
    return population_rows(metadata, categories, counts, total)


def bi_bii_population_rows(
    metadata: Dict[str, Any],
    epsilon_values: Sequence[float],
    zeta_values: Sequence[float],
) -> List[Dict[str, Any]]:
    counts, total = bi_bii_counts(epsilon_values, zeta_values)
    categories = [{"conformer": label} for label in BI_BII_STATES]
    return population_rows(metadata, categories, counts, total)


def alpha_gamma_population_rows(
    metadata: Dict[str, Any],
    alpha_values: Sequence[float],
    gamma_values: Sequence[float],
) -> List[Dict[str, Any]]:
    counts, total = alpha_gamma_counts(alpha_values, gamma_values)
    categories = [
        {"alpha_state": alpha, "gamma_state": gamma, "conformer": f"{alpha}/{gamma}"}
        for alpha, gamma in ALPHA_GAMMA_STATES
    ]
    return population_rows(metadata, categories, counts, total)


def _number_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def backbone_conformer_population_tables(backbone_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: "OrderedDict[Tuple[Any, ...], List[Dict[str, Any]]]" = OrderedDict()
    metadata_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in backbone_rows:
        key = (
            row.get("strand"),
            row.get("level"),
            row.get("residue_name"),
            row.get("residue_id"),
        )
        grouped.setdefault(key, []).append(row)
        metadata_by_key.setdefault(
            key,
            {
                "strand": row.get("strand"),
                "level": row.get("level"),
                "residue_name": row.get("residue_name"),
                "residue_id": row.get("residue_id"),
            },
        )

    sugar_rows: List[Dict[str, Any]] = []
    bi_bii_rows: List[Dict[str, Any]] = []
    alpha_gamma_rows: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        metadata = metadata_by_key[key]
        phase = [_number_or_nan(row.get("phase")) for row in rows]
        epsilon = [_number_or_nan(row.get("epsilon")) for row in rows]
        zeta = [_number_or_nan(row.get("zeta")) for row in rows]
        alpha = [_number_or_nan(row.get("alpha")) for row in rows]
        gamma = [_number_or_nan(row.get("gamma")) for row in rows]
        sugar_rows.extend(sugar_pucker_population_rows(metadata, phase))
        bi_bii_rows.extend(bi_bii_population_rows(metadata, epsilon, zeta))
        alpha_gamma_rows.extend(alpha_gamma_population_rows(metadata, alpha, gamma))

    return {
        "backbone_sugar_pucker_distribution": sugar_rows,
        "backbone_bi_bii_distribution": bi_bii_rows,
        "backbone_alpha_gamma_distribution": alpha_gamma_rows,
    }
