from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
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


@dataclass(slots=True)
class LinearSummaryAccumulator:
    """Mergeable population moments using the Chan-Welford algorithm."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, values) -> None:
        vals = np.asarray(values, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return

        batch_mean = float(np.mean(vals))
        centered = vals - batch_mean
        batch_m2 = float(np.dot(centered, centered))
        self._merge_moments(int(vals.size), batch_mean, batch_m2)

    def merge(self, other: "LinearSummaryAccumulator") -> None:
        self._merge_moments(other.count, other.mean, other.m2)

    def summary(self) -> SummaryStats:
        if self.count == 0:
            return SummaryStats(None, None)
        variance = self.m2 / self.count
        if abs(variance) < 1e-15:
            variance = 0.0
        return SummaryStats(float(self.mean), _stddev_from_variance(variance))

    def _merge_moments(self, count: int, mean: float, m2: float) -> None:
        if count <= 0:
            return
        if self.count == 0:
            self.count = int(count)
            self.mean = float(mean)
            self.m2 = float(m2)
            return

        combined_count = self.count + count
        delta = mean - self.mean
        self.mean += delta * (count / combined_count)
        self.m2 += m2 + delta * delta * self.count * count / combined_count
        self.count = combined_count


@dataclass(slots=True)
class CircularDegreeSummaryAccumulator:
    """Merge circular moments using compensated resultant-vector sums.

    The standard deviation is ``sqrt(-2 log(R))`` in degrees, where ``R`` is
    the mean resultant length. This definition needs only the two first
    trigonometric moments and remains well behaved across the degree wrap.
    """

    count: int = 0
    _sin_sum: float = 0.0
    _sin_compensation: float = 0.0
    _cos_sum: float = 0.0
    _cos_compensation: float = 0.0

    def add(self, values) -> None:
        vals = np.asarray(values, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return

        radians = np.radians(np.remainder(vals, 360.0))
        self._add_sin(float(np.sum(np.sin(radians), dtype=np.float64)))
        self._add_cos(float(np.sum(np.cos(radians), dtype=np.float64)))
        self.count += int(vals.size)

    def merge(self, other: "CircularDegreeSummaryAccumulator") -> None:
        self._add_sin(other._sin_sum)
        self._add_sin(other._sin_compensation)
        self._add_cos(other._cos_sum)
        self._add_cos(other._cos_compensation)
        self.count += other.count

    def summary(self) -> SummaryStats:
        if self.count == 0:
            return SummaryStats(None, None)

        sin_sum = self._sin_sum + self._sin_compensation
        cos_sum = self._cos_sum + self._cos_compensation
        resultant = float(np.hypot(sin_sum, cos_sum))
        mean_resultant_length = resultant / self.count
        if mean_resultant_length <= 1e-12:
            return SummaryStats(None, None)

        mean = wrap_degrees_180(np.degrees(np.arctan2(sin_sum, cos_sum)))
        mean_resultant_length = min(mean_resultant_length, 1.0)
        if 1.0 - mean_resultant_length <= 8.0 * np.finfo(float).eps:
            mean_resultant_length = 1.0
        circular_variance = -2.0 * np.log(mean_resultant_length)
        stddev = float(np.degrees(np.sqrt(max(float(circular_variance), 0.0))))
        return SummaryStats(mean, stddev)

    def _add_sin(self, value: float) -> None:
        self._sin_sum, self._sin_compensation = _neumaier_add(
            self._sin_sum,
            self._sin_compensation,
            value,
        )

    def _add_cos(self, value: float) -> None:
        self._cos_sum, self._cos_compensation = _neumaier_add(
            self._cos_sum,
            self._cos_compensation,
            value,
        )


def _neumaier_add(total: float, compensation: float, value: float) -> Tuple[float, float]:
    updated = total + value
    if abs(total) >= abs(value):
        compensation += (total - updated) + value
    else:
        compensation += (value - updated) + total
    return updated, compensation


def is_circular_degree_column(column_name: str) -> bool:
    return str(column_name).lower() in CIRCULAR_DEGREE_COLUMNS


class BatchSummaryAccumulator:
    """Accumulate grouped trajectory statistics without materializing frames."""

    def __init__(self) -> None:
        self._tables: Dict[str, Dict[tuple, Dict]] = {}
        self._population_tables: Dict[str, Dict[tuple, Dict]] = {}

    def ensure_table(self, table_name: str) -> None:
        self._tables.setdefault(table_name, {})

    def add_population_counts(
        self,
        table_name: str,
        metadata: Dict,
        category_metadata: Dict,
        count: int,
        total_count: int,
    ) -> None:
        if total_count <= 0:
            return
        table = self._population_tables.setdefault(table_name, {})
        row_metadata = dict(metadata)
        row_metadata.update(category_metadata)
        key = tuple(row_metadata.items())
        group = table.get(key)
        if group is None:
            group = {"metadata": row_metadata, "count": 0, "total_count": 0}
            table[key] = group
        group["count"] += int(count)
        group["total_count"] += int(total_count)

    @staticmethod
    def _new_stats(name: str):
        if is_circular_degree_column(name):
            return CircularDegreeSummaryAccumulator()
        return LinearSummaryAccumulator()

    def add_values(self, table_name: str, metadata: Dict, parameter_names, values) -> None:
        self.ensure_table(table_name)
        parameter_names = tuple(parameter_names)
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0:
            return
        if arr.ndim != 2 or arr.shape[1] != len(parameter_names):
            raise ValueError(
                "Summary values must have one column per parameter name; "
                f"got shape {arr.shape} for {len(parameter_names)} parameters."
            )
        row_count = int(arr.shape[0])
        metadata = dict(metadata)
        key = tuple(metadata.items())
        group = self._tables[table_name].get(key)
        if group is None:
            group = {
                "metadata": metadata,
                "count": 0,
                "stats": {
                    name: self._new_stats(name)
                    for name in parameter_names
                },
            }
            self._tables[table_name][key] = group
        group["count"] += row_count
        for col_index, name in enumerate(parameter_names):
            group["stats"][name].add(arr[:, col_index])

    def add_rows(self, table_name: str, rows: List[Dict], numeric_names) -> None:
        self.ensure_table(table_name)
        numeric_names = tuple(numeric_names)
        numeric_set = set(numeric_names)
        for row in rows:
            metadata = {
                key: value
                for key, value in row.items()
                if key not in numeric_set and key not in {"frame", "time"}
            }
            key = tuple(metadata.items())
            group = self._tables[table_name].get(key)
            if group is None:
                group = {
                    "metadata": metadata,
                    "count": 0,
                    "stats": {
                        name: self._new_stats(name)
                        for name in numeric_names
                    },
                }
                self._tables[table_name][key] = group
            group["count"] += 1
            for name in numeric_names:
                value = row.get(name)
                if value is not None:
                    group["stats"][name].add(value)

    def to_summary(self) -> Dict[str, List[Dict]]:
        output: Dict[str, List[Dict]] = {}
        for table_name, groups in self._tables.items():
            rows = []
            for group in groups.values():
                out = dict(group["metadata"])
                out["count"] = int(group["count"])
                for name, stats in group["stats"].items():
                    summary = stats.summary()
                    out[f"{name}_mean"] = summary.mean
                    out[f"{name}_stddev"] = summary.stddev
                rows.append(out)
            output[table_name] = rows

        for table_name, groups in self._population_tables.items():
            rows = []
            for group in groups.values():
                total_count = int(group["total_count"])
                count = int(group["count"])
                fraction = count / float(total_count) if total_count > 0 else 0.0
                row = dict(group["metadata"])
                row["count"] = count
                row["total_count"] = total_count
                row["fraction"] = float(fraction)
                row["percent"] = float(100.0 * fraction)
                rows.append(row)
            output[table_name] = rows
        return output


def wrap_degrees_180(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if np.isclose(wrapped, -180.0) else wrapped


def wrap_degrees_180_array(values: np.ndarray) -> np.ndarray:
    wrapped = (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0
    return np.where(np.isclose(wrapped, -180.0), 180.0, wrapped)


def _stddev_from_variance(variance: Optional[float]) -> Optional[float]:
    if variance is None:
        return None
    return float(np.sqrt(max(float(variance), 0.0)))


def linear_summary(values: np.ndarray) -> SummaryStats:
    accumulator = LinearSummaryAccumulator()
    accumulator.add(values)
    return accumulator.summary()


def circular_degree_mean_from_sums(
    sin_sum: float,
    cos_sum: float,
    count: int,
) -> Optional[float]:
    if count <= 0:
        return None
    resultant = float(np.hypot(sin_sum, cos_sum))
    if resultant / count <= 1e-12:
        return None
    return wrap_degrees_180(np.degrees(np.arctan2(sin_sum, cos_sum)))


def circular_degree_summary(values: np.ndarray) -> SummaryStats:
    accumulator = CircularDegreeSummaryAccumulator()
    accumulator.add(values)
    return accumulator.summary()


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
