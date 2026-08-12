"""Geometry-derived confidence weights for helical-axis construction."""

import numpy as np


FULL_SUPPORT_DISTANCE = 4.0
ZERO_SUPPORT_DISTANCE = 8.0


def axis_support_weights(frames, level_status, enabled=False):
    """Return per-strand axis weights from paired fitted-base origins.

    ``frames`` is level-major with shape ``(..., levels, strands, 4, 3)``.
    Standard Curves base origins nearly coincide for an intact pair, so their
    separation is a permissive, annotation-independent measure of whether the
    two bases still define one duplex cross-section.
    """
    frames = np.asarray(frames, dtype=float)
    status = np.asarray(level_status)
    if frames.ndim < 4 or frames.shape[-2:] != (4, 3):
        raise ValueError("Axis frames must have shape (..., levels, strands, 4, 3).")
    if frames.shape[-4:-2] != status.shape:
        raise ValueError("Axis-frame levels/strands do not match level status.")

    leading = frames.shape[:-4]
    active = status > 0
    weights = np.broadcast_to(active, leading + active.shape).astype(float).copy()
    if not enabled or status.shape[1] < 2:
        return weights

    for level in range(status.shape[0]):
        strands = np.flatnonzero(active[level])
        if strands.size < 2:
            continue
        primary = int(strands[0])
        pair_weights = []
        for partner_value in strands[1:]:
            partner = int(partner_value)
            delta = frames[..., level, partner, 3, :] - frames[..., level, primary, 3, :]
            distance = np.linalg.norm(delta, axis=-1)
            weight = _smooth_distance_weight(distance)
            weight = np.where(np.isfinite(distance), weight, 0.0)
            weights[..., level, partner] = weight
            pair_weights.append(weight)
        weights[..., level, primary] = np.maximum.reduce(pair_weights)
    return weights


def _smooth_distance_weight(distance):
    fraction = np.clip(
        (ZERO_SUPPORT_DISTANCE - np.asarray(distance, dtype=float))
        / (ZERO_SUPPORT_DISTANCE - FULL_SUPPORT_DISTANCE),
        0.0,
        1.0,
    )
    return fraction * fraction * (3.0 - 2.0 * fraction)
