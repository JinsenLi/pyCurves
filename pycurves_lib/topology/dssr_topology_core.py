from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pycurves_lib.core.curves_dataclasses import MolecularStructure
from pycurves_lib.data.modified_bases import parent_base_name
from pycurves_lib.io.dssr_json import (
    DSSRDocument,
    DSSRPair,
    DSSRResidueKey,
    DSSRUnit,
    matched_pair_identities,
)
from pycurves_lib.topology.topology_inferrer import InferredTopology


_VALID_LW_TAG = re.compile(r"^[ct][WHS][WHS]$", re.IGNORECASE)
_O3_NAMES = {"O3'", "O3*"}


class DSSRTopologyError(ValueError):
    """Raised when DSSR data cannot define a safe Curves topology."""


class DSSRSelectionError(DSSRTopologyError):
    """Raised when a DSSR document requires an explicit unit selection."""


@dataclass(frozen=True)
class MoleculeResidue:
    subunit: int
    key: DSSRResidueKey
    atom_start: int
    atom_end: int


@dataclass(frozen=True)
class ResolvedPair:
    pair: DSSRPair
    first: int
    second: int


@dataclass(frozen=True)
class OrientedPairs:
    resolved: Tuple[ResolvedPair, ...]
    row1: Tuple[int, ...]
    row2: Tuple[int, ...]
    direction1: int
    direction2: int


@dataclass(frozen=True)
class DSSRBuildResult:
    topology: InferredTopology
    unit: DSSRUnit
    source_base_pairs: Tuple[dict, ...]
    provenance: dict


class DSSRTopologyBuilder:
    """Resolve a DSSR report against coordinates and build Curves strand maps."""

    def __init__(
        self,
        molecule: MolecularStructure,
        document: DSSRDocument,
        *,
        pdbfile: str,
    ):
        self.molecule = molecule
        self.document = document
        self.pdbfile = str(pdbfile)
        self.residues = self._collect_molecule_residues()
        self._by_location = self._location_index()
        self._resolved_nt_ids: Dict[str, int] = {}
        self._next_links = self._build_directed_backbone_links()
        self._synthetic_cache: Optional[Tuple[Tuple[DSSRUnit, ...], Tuple[str, ...]]] = None

    def build(self, unit_selector: Optional[str] = None) -> DSSRBuildResult:
        unit = self.select_unit(unit_selector)
        oriented = self._orient_unit(unit)
        ni_map = np.asarray([oriented.row1, oriented.row2], dtype=int)
        pair_geometry_markers = {}
        selected_edges = []
        for level, resolved in enumerate(oriented.resolved, start=1):
            selected_edges.append((resolved.first, resolved.second))
            tag = self._oriented_lw_tag(resolved, oriented.row1[level - 1])
            if tag:
                pair_geometry_markers[(1, level)] = tag

        topology = InferredTopology(
            pdbfile=Path(self.pdbfile).name,
            output_prefix=Path(self.pdbfile).stem,
            strands=[list(oriented.row1), list(oriented.row2)],
            nu_raw=[
                oriented.direction1 * len(oriented.row1),
                oriented.direction2 * len(oriented.row2),
            ],
            ni_map=ni_map,
            pair_edges=selected_edges,
            chain_ids=[
                self.residues[oriented.row1[0]].key.chain,
                self.residues[oriented.row2[0]].key.chain,
            ],
            comb=True,
            fit=True,
            grv=len(oriented.resolved) >= 4,
            pair_geometry_markers=pair_geometry_markers,
        )

        source_rows, mapping_warnings = self._source_base_pair_rows(unit)
        warnings = list(self._selection_warnings()) + mapping_warnings
        provenance = self._provenance(unit, oriented, warnings)
        return DSSRBuildResult(
            topology=topology,
            unit=unit,
            source_base_pairs=tuple(source_rows),
            provenance=provenance,
        )

    def select_unit(self, selector: Optional[str]) -> DSSRUnit:
        if selector:
            kind, index = self._parse_selector(selector)
            candidates = self._units_for_kind(kind)
            for unit in candidates:
                if unit.index == index:
                    return unit
            available = ", ".join(unit.selector for unit in candidates) or "none"
            raise DSSRSelectionError(
                f"DSSR unit {kind}:{index} is not present; available {kind} units: {available}."
            )

        if self.document.stems:
            if len(self.document.stems) == 1:
                return self.document.stems[0]
            raise DSSRSelectionError(self._selection_message("stem", self.document.stems))

        if self.document.helices:
            if len(self.document.helices) == 1:
                return self.document.helices[0]
            raise DSSRSelectionError(self._selection_message("helix", self.document.helices))

        pair_units = self._synthetic_pair_units()[0]
        if len(pair_units) == 1:
            return pair_units[0]
        if len(pair_units) > 1:
            raise DSSRSelectionError(self._selection_message("pairs", pair_units))
        if not self.document.pairs:
            raise DSSRTopologyError(
                "The DSSR report contains no base pairs, stems, or helices that can define a Curves topology."
            )
        warnings = self._synthetic_pair_units()[1]
        detail = f" ({'; '.join(warnings)})" if warnings else ""
        raise DSSRTopologyError(
            "The DSSR pair list contains no unambiguous two-pair backbone segment" + detail + "."
        )

    def unit_summaries(self) -> List[dict]:
        units: Tuple[DSSRUnit, ...]
        if self.document.stems or self.document.helices:
            units = self.document.stems + self.document.helices
        else:
            units = self._synthetic_pair_units()[0]

        summaries = []
        for unit in units:
            summary = {
                "selector": unit.selector,
                "kind": unit.kind,
                "index": unit.index,
                "pair_count": len(unit.pairs),
                "helix_index": unit.helix_index,
                "num_stems": unit.num_stems,
                "helix_form": unit.helix_form,
            }
            try:
                oriented = self._orient_unit(unit)
                chain_pairs = {
                    (
                        self.residues[left].key.chain,
                        self.residues[right].key.chain,
                    )
                    for left, right in zip(oriented.row1, oriented.row2)
                }
                summary.update({
                    "representable": True,
                    "chain_pairs": [f"{left}:{right}" for left, right in sorted(chain_pairs)],
                    "directions": [oriented.direction1, oriented.direction2],
                })
            except DSSRTopologyError as exc:
                summary.update({
                    "representable": False,
                    "chain_pairs": self._unit_chain_pairs(unit),
                    "reason": str(exc),
                })
            summaries.append(summary)
        return summaries

    def _collect_molecule_residues(self) -> Dict[int, MoleculeResidue]:
        boundaries = self.molecule.subunit_boundaries
        if boundaries is None:
            raise DSSRTopologyError(
                "MolecularStructure.subunit_boundaries is missing; load the coordinates with MolecularLoader first."
            )
        residues = {}
        for subunit in range(1, len(boundaries)):
            start = int(boundaries[subunit - 1])
            stop = int(boundaries[subunit])
            if start >= stop:
                continue
            chain = str(self.molecule.chain_ids[start]).strip() if self.molecule.chain_ids is not None else ""
            insertion = (
                str(self.molecule.insertion_codes[start]).strip()
                if self.molecule.insertion_codes is not None else ""
            )
            model = str(self.molecule.model_ids[start]).strip() if self.molecule.model_ids is not None else ""
            residues[subunit] = MoleculeResidue(
                subunit=subunit,
                key=DSSRResidueKey(
                    chain=chain,
                    residue_number=int(self.molecule.residue_ids[start]),
                    insertion_code=insertion,
                    residue_name=str(self.molecule.residue_names[start]).strip(),
                    model=model,
                ),
                atom_start=start,
                atom_end=stop,
            )
        return residues

    def _location_index(self) -> Dict[Tuple[str, int, str], List[int]]:
        index: Dict[Tuple[str, int, str], List[int]] = {}
        for subunit, residue in self.residues.items():
            location = (
                residue.key.chain,
                residue.key.residue_number,
                residue.key.insertion_code,
            )
            index.setdefault(location, []).append(subunit)
        return index

    def _resolve_nt_id(self, nt_id: str) -> int:
        cached = self._resolved_nt_ids.get(nt_id)
        if cached is not None:
            return cached
        key = self.document.residue_key(nt_id)
        matches = list(self._by_location.get((key.chain, key.residue_number, key.insertion_code), []))
        if key.model:
            expected_model = self._normalize_model(key.model)
            matches = [
                subunit for subunit in matches
                if self._normalize_model(self.residues[subunit].key.model) == expected_model
            ]
        if key.residue_name:
            exact = [
                subunit for subunit in matches
                if self.residues[subunit].key.residue_name.upper() == key.residue_name.upper()
            ]
            if exact:
                matches = exact
            else:
                expected_parent = parent_base_name(key.residue_name)
                parent_matches = [
                    subunit for subunit in matches
                    if parent_base_name(self.residues[subunit].key.residue_name) == expected_parent
                ]
                if parent_matches:
                    matches = parent_matches
                elif matches:
                    actual = ", ".join(self.residues[subunit].key.residue_name for subunit in matches)
                    raise DSSRTopologyError(
                        f"DSSR nucleotide {nt_id!r} resolves by location but its residue name "
                        f"does not match the coordinate residue(s): {actual}."
                    )

        if not matches:
            raise DSSRTopologyError(
                f"DSSR nucleotide {nt_id!r} does not map to the coordinate structure "
                f"at chain {key.chain!r}, residue {key.residue_number}{key.insertion_code}."
            )
        if len(matches) > 1:
            raise DSSRTopologyError(
                f"DSSR nucleotide {nt_id!r} maps ambiguously to coordinate subunits {matches}."
            )
        self._resolved_nt_ids[nt_id] = matches[0]
        return matches[0]

    @staticmethod
    def _normalize_model(value: str) -> str:
        text = str(value or "").strip().lower()
        return text[1:] if text.startswith("m") and text[1:].isdigit() else text

    def _build_directed_backbone_links(self) -> set[Tuple[int, int]]:
        links: set[Tuple[int, int]] = set()

        # The full DSSR nts table is authoritative about polymer predecessor and
        # successor identities. Resolve only entries that exist in the loaded
        # coordinate structure; selected-unit mapping remains strict later.
        for residue in self.document.residues:
            if not residue.next_nt:
                continue
            try:
                links.add((self._resolve_nt_id(residue.nt_id), self._resolve_nt_id(residue.next_nt)))
            except DSSRTopologyError:
                continue

        boundaries = self.molecule.subunit_boundaries
        atom_to_subunit = np.full(self.molecule.kam, -1, dtype=int)
        if boundaries is not None:
            for subunit in range(1, len(boundaries)):
                atom_to_subunit[int(boundaries[subunit - 1]):int(boundaries[subunit])] = subunit

        # Coordinate connectivity gives the directed chemical link O3'(i)-P(i+1).
        for first, second in (getattr(self.molecule, "connectivity_sources", None) or {}):
            name_1 = str(self.molecule.atom_names[first]).strip().upper()
            name_2 = str(self.molecule.atom_names[second]).strip().upper()
            subunit_1 = int(atom_to_subunit[first])
            subunit_2 = int(atom_to_subunit[second])
            if subunit_1 <= 0 or subunit_2 <= 0 or subunit_1 == subunit_2:
                continue
            if name_1 in _O3_NAMES and name_2 == "P":
                links.add((subunit_1, subunit_2))
            elif name_2 in _O3_NAMES and name_1 == "P":
                links.add((subunit_2, subunit_1))

        # Pair-only reports lack linked_nts. Use conservative same-chain file
        # order only when the neighboring residue centers remain close.
        by_chain: Dict[str, List[MoleculeResidue]] = {}
        for residue in self.residues.values():
            by_chain.setdefault(residue.key.chain, []).append(residue)
        for chain_residues in by_chain.values():
            chain_residues.sort(key=lambda item: item.subunit)
            for previous, current in zip(chain_residues, chain_residues[1:]):
                if (previous.subunit, current.subunit) in links:
                    continue
                prev_center = np.mean(
                    self.molecule.coordinates[previous.atom_start:previous.atom_end], axis=0
                )
                curr_center = np.mean(
                    self.molecule.coordinates[current.atom_start:current.atom_end], axis=0
                )
                if float(np.linalg.norm(curr_center - prev_center)) <= 12.0:
                    links.add((previous.subunit, current.subunit))
        return links

    def _resolve_pair(self, pair: DSSRPair) -> ResolvedPair:
        return ResolvedPair(
            pair=pair,
            first=self._resolve_nt_id(pair.nt1),
            second=self._resolve_nt_id(pair.nt2),
        )

    def _orient_unit(self, unit: DSSRUnit) -> OrientedPairs:
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
                # state 0 keeps nt1/nt2; state 1 swaps them at this level.
                scores = {(0,): (0, 0), (1,): (0, -1)}
                for pair_index in range(1, len(resolved)):
                    next_scores = {}
                    for path, (score, swap_penalty) in scores.items():
                        previous_state = path[-1]
                        previous = self._pair_state(resolved[pair_index - 1], previous_state)
                        for state in (0, 1):
                            current = self._pair_state(resolved[pair_index], state)
                            transition_score = int(self._follows(previous[0], current[0], direction1))
                            transition_score += int(self._follows(previous[1], current[1], direction2))
                            candidate_path = path + (state,)
                            candidate = (score + transition_score, swap_penalty - state)
                            if candidate > next_scores.get(candidate_path, (-1, -10**9)):
                                next_scores[candidate_path] = candidate
                    scores = next_scores

                path, path_score = max(scores.items(), key=lambda item: item[1])
                candidate = (path_score[0], path_score[1], direction1 == 1, direction2 == -1)
                if best is None or candidate > best[0]:
                    best = (candidate, direction1, direction2, path)

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

    @staticmethod
    def _pair_state(pair: ResolvedPair, state: int) -> Tuple[int, int]:
        return (pair.first, pair.second) if state == 0 else (pair.second, pair.first)

    def _follows(self, previous: int, current: int, direction: int) -> bool:
        return (previous, current) in self._next_links if direction > 0 else (current, previous) in self._next_links

    def _adjacent(self, first: int, second: int) -> bool:
        return (first, second) in self._next_links or (second, first) in self._next_links

    def _synthetic_pair_units(self) -> Tuple[Tuple[DSSRUnit, ...], Tuple[str, ...]]:
        if self._synthetic_cache is not None:
            return self._synthetic_cache
        resolved = [self._resolve_pair(pair) for pair in self.document.pairs]
        neighbors = {index: set() for index in range(len(resolved))}
        for left_index in range(len(resolved)):
            left = resolved[left_index]
            left_residues = {left.first, left.second}
            for right_index in range(left_index + 1, len(resolved)):
                right = resolved[right_index]
                if left_residues & {right.first, right.second}:
                    continue
                parallel = self._adjacent(left.first, right.first) and self._adjacent(left.second, right.second)
                crossed = self._adjacent(left.first, right.second) and self._adjacent(left.second, right.first)
                if parallel or crossed:
                    neighbors[left_index].add(right_index)
                    neighbors[right_index].add(left_index)

        components = []
        unseen = set(neighbors)
        while unseen:
            start = min(unseen)
            stack = [start]
            component = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(neighbors[current] - component)
            unseen -= component
            components.append(component)

        ordered_components = []
        warnings = []
        for component in components:
            if len(component) < 2:
                warnings.append(
                    f"ignored isolated pair {resolved[next(iter(component))].pair.index}"
                )
                continue
            degrees = {node: len(neighbors[node] & component) for node in component}
            endpoints = sorted(node for node, degree in degrees.items() if degree == 1)
            if any(degree > 2 for degree in degrees.values()) or len(endpoints) != 2:
                pair_ids = ",".join(str(resolved[node].pair.index) for node in sorted(component))
                warnings.append(f"ignored branched/cyclic pair component {pair_ids}")
                continue
            order = []
            previous = None
            current = endpoints[0]
            while current is not None:
                order.append(current)
                following = sorted((neighbors[current] & component) - ({previous} if previous is not None else set()))
                previous, current = current, (following[0] if following else None)
            ordered_components.append(order)

        ordered_components.sort(
            key=lambda component: min(resolved[node].pair.index for node in component)
        )
        units = []
        for unit_index, component in enumerate(ordered_components, start=1):
            unit = DSSRUnit(
                kind="pairs",
                index=unit_index,
                pairs=tuple(resolved[node].pair for node in component),
                raw={"inferred_from": "DSSR root pairs"},
            )
            try:
                self._orient_unit(unit)
            except DSSRTopologyError as exc:
                warnings.append(f"ignored pair component {unit_index}: {exc}")
                continue
            units.append(unit)
        self._synthetic_cache = (tuple(units), tuple(warnings))
        return self._synthetic_cache

    def _units_for_kind(self, kind: str) -> Tuple[DSSRUnit, ...]:
        if kind == "stem":
            return self.document.stems
        if kind == "helix":
            return self.document.helices
        if kind == "pairs":
            return self._synthetic_pair_units()[0]
        raise DSSRSelectionError(
            f"Unknown DSSR unit kind {kind!r}; use stem:N, helix:N, or pairs:N."
        )

    @staticmethod
    def _parse_selector(selector: str) -> Tuple[str, int]:
        text = str(selector or "").strip().lower()
        if ":" not in text:
            raise DSSRSelectionError(
                f"Invalid DSSR unit selector {selector!r}; use stem:N, helix:N, or pairs:N."
            )
        kind, raw_index = text.split(":", 1)
        kind = kind.rstrip("s") if kind != "pairs" else kind
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise DSSRSelectionError(f"Invalid DSSR unit index in {selector!r}.") from exc
        return kind, index

    def _selection_message(self, kind: str, units: Sequence[DSSRUnit]) -> str:
        summaries = []
        for unit in units:
            chain_pairs = ",".join(self._unit_chain_pairs(unit)) or "unknown chains"
            summaries.append(f"{unit.selector} ({len(unit.pairs)} pairs; {chain_pairs})")
        return (
            f"The DSSR report contains multiple {kind} units. Select one with --dssr-unit: "
            + "; ".join(summaries)
        )

    def _unit_chain_pairs(self, unit: DSSRUnit) -> List[str]:
        pairs = set()
        for pair in unit.pairs:
            try:
                first = self.document.residue_key(pair.nt1).chain
                second = self.document.residue_key(pair.nt2).chain
            except Exception:
                continue
            pairs.add(f"{first}:{second}")
        return sorted(pairs)

    def _oriented_lw_tag(self, resolved: ResolvedPair, row1_subunit: int) -> str:
        tag = str(resolved.pair.lw or "").strip()
        if not _VALID_LW_TAG.fullmatch(tag):
            return ""
        tag = tag[0].lower() + tag[1:].upper()
        if row1_subunit == resolved.second:
            tag = tag[0] + tag[2] + tag[1]
        return tag

    def _source_base_pair_rows(self, selected_unit: DSSRUnit) -> Tuple[List[dict], List[str]]:
        selected = matched_pair_identities(selected_unit.pairs)
        rows = []
        warnings = []
        for pair in self.document.pairs:
            try:
                first = self.residues[self._resolve_nt_id(pair.nt1)].key
                second = self.residues[self._resolve_nt_id(pair.nt2)].key
            except DSSRTopologyError as exc:
                warnings.append(str(exc))
                continue
            pair_name = str(pair.name or "").lower()
            rows.append({
                "source": "DSSR JSON",
                "pair_number": pair.index,
                "pair_name": f"{pair.nt1}:{pair.nt2}",
                "i_nt_id": pair.nt1,
                "i_chain_id": first.chain,
                "i_residue_id": first.residue_number,
                "i_insertion_code": first.insertion_code,
                "i_residue_name": first.residue_name,
                "j_nt_id": pair.nt2,
                "j_chain_id": second.chain,
                "j_residue_id": second.residue_number,
                "j_insertion_code": second.insertion_code,
                "j_residue_name": second.residue_name,
                "is_hoogsteen": "hoogsteen" in pair_name,
                "dssr_bp": pair.bp,
                "dssr_name": pair.name,
                "dssr_saenger": pair.saenger,
                "dssr_lw": pair.lw,
                "dssr_code": pair.dssr,
                "dssr_selected": pair.identity in selected,
                "dssr_unit": selected_unit.selector,
            })
        return rows, warnings

    def _selection_warnings(self) -> Tuple[str, ...]:
        warnings = []
        metadata_input = str(self.document.metadata.get("input_file") or "").strip()
        if metadata_input and Path(metadata_input).name.lower() != Path(self.pdbfile).name.lower():
            warnings.append(
                f"DSSR metadata input_file {metadata_input!r} differs from coordinate file {self.pdbfile!r}."
            )
        if self.document.kind == "pair_only":
            warnings.append(
                "Pair-only DSSR JSON lacks authoritative stem/helix grouping; pyCurves reconstructed continuous pair segments."
            )
        warnings.extend(self._synthetic_pair_units()[1] if not (self.document.stems or self.document.helices) else ())
        return tuple(warnings)

    def _provenance(
        self,
        unit: DSSRUnit,
        oriented: OrientedPairs,
        warnings: Sequence[str],
    ) -> dict:
        return {
            "source": "dssr_json",
            "json_file": self.document.path,
            "document_kind": self.document.kind,
            "program": self.document.program,
            "version": self.document.version,
            "metadata_input_file": self.document.metadata.get("input_file"),
            "structure_id": self.document.metadata.get("str_id"),
            "root_pair_count": len(self.document.pairs),
            "unit": {
                "selector": unit.selector,
                "kind": unit.kind,
                "index": unit.index,
                "pair_count": len(unit.pairs),
                "helix_index": unit.helix_index,
                "num_stems": unit.num_stems,
                "chain_ids": [
                    self.residues[oriented.row1[0]].key.chain,
                    self.residues[oriented.row2[0]].key.chain,
                ],
                "strand_directions": [oriented.direction1, oriented.direction2],
            },
            "warnings": list(dict.fromkeys(warnings)),
        }
