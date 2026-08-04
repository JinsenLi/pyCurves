"""Public, linear-time DSSR topology builder."""

from __future__ import annotations

from pycurves_lib.topology.dssr_topology_core import (
    DSSRBuildResult,
    DSSRSelectionError,
    DSSRTopologyBuilder as _DSSRTopologyBuilder,
    DSSRTopologyError,
    OrientedPairs,
)


class DSSRTopologyBuilder(_DSSRTopologyBuilder):
    def _orient_unit(self, unit):
        if len(unit.pairs) < 2:
            raise DSSRTopologyError(
                f"DSSR {unit.selector} has {len(unit.pairs)} pair; Curves topology requires at least two."
            )
        resolved = tuple(self._resolve_pair(pair) for pair in unit.pairs)
        flattened = [subunit for pair in resolved for subunit in (pair.first, pair.second)]
        if len(flattened) != len(set(flattened)):
            raise DSSRTopologyError(
                f"DSSR {unit.selector} assigns one nucleotide to multiple pairs within the same unit."
            )

        best = None
        for direction1 in (1, -1):
            for direction2 in (1, -1):
                scores = {
                    0: (0, 0, (0,)),
                    1: (0, -1, (1,)),
                }
                for pair_index in range(1, len(resolved)):
                    next_scores = {}
                    for previous_state, (score, swap_penalty, path) in scores.items():
                        previous = self._pair_state(resolved[pair_index - 1], previous_state)
                        for state in (0, 1):
                            current = self._pair_state(resolved[pair_index], state)
                            transition_score = int(self._follows(previous[0], current[0], direction1))
                            transition_score += int(self._follows(previous[1], current[1], direction2))
                            candidate = (
                                score + transition_score,
                                swap_penalty - state,
                                path + (state,),
                            )
                            current_best = next_scores.get(state)
                            if current_best is None or candidate[:2] > current_best[:2]:
                                next_scores[state] = candidate
                    scores = next_scores

                path_score = max(scores.values(), key=lambda item: item[:2])
                candidate = (
                    path_score[0],
                    path_score[1],
                    direction1 == 1,
                    direction2 == -1,
                )
                if best is None or candidate > best[0]:
                    best = (candidate, direction1, direction2, path_score[2])

        assert best is not None
        expected = 2 * (len(resolved) - 1)
        if best[0][0] != expected:
            raise DSSRTopologyError(
                f"DSSR {unit.selector} is not two continuous backbone rails "
                f"({best[0][0]}/{expected} adjacent rail transitions). "
                "Select one of its constituent stems instead."
            )
        _, direction1, direction2, path = best
        states = [self._pair_state(pair, state) for pair, state in zip(resolved, path)]
        return OrientedPairs(
            resolved=resolved,
            row1=tuple(state[0] for state in states),
            row2=tuple(state[1] for state in states),
            direction1=direction1,
            direction2=direction2,
        )


__all__ = [
    "DSSRBuildResult",
    "DSSRSelectionError",
    "DSSRTopologyBuilder",
    "DSSRTopologyError",
]
