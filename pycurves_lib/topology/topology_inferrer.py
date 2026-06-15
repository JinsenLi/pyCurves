from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from pycurves_lib.core.curves_dataclasses import MolecularStructure
from pycurves_lib.data.modified_bases import parent_base_name


DNA_BASES = {"A", "C", "G", "T", "U", "I", "P", "Y", "R"}
PAIR_TYPES = {frozenset(("A", "T")), frozenset(("A", "U")), frozenset(("G", "C"))}
BASE_ATOMS = {
    "A": {"N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"},
    "G": {"N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"},
    "C": {"N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"},
    "T": {"N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"},
    "U": {"N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"},
}
HBOND_ATOMS = {
    "A": {"N1", "N6", "N7"},
    "G": {"N1", "N2", "O6", "N7"},
    "C": {"N3", "N4", "O2"},
    "T": {"N3", "O4", "O2"},
    "U": {"N3", "O4", "O2"},
    "I": {"N1", "O6", "N7"},
}

HBOND_DISTANCE_CUTOFF = 3.8
HBOND_PREFILTER_DISTANCE = 11.0
SUSPICIOUS_BACKBONE_GAP_DISTANCE = 8.5
REGISTER_GAP_CENTER_CUTOFF = 7.8
REGISTER_GAP_HBOND_CENTER_CUTOFF = 6.2
REGISTER_GAP_RELAXED_HBOND_CUTOFF = 4.1
REGISTER_GAP_NONCANONICAL_SINGLE_CONTACT_CENTER_CUTOFF = 6.6
REGISTER_GAP_NONCANONICAL_SINGLE_CONTACT_HBOND_CENTER_CUTOFF = 5.2
REGISTER_STACKING_BRIDGE_CENTER_CUTOFF = 8.2
REGISTER_STACKING_BRIDGE_HBOND_CENTER_CUTOFF = 8.2
REGISTER_STACKING_BRIDGE_SCORE_CUTOFF = 5.5
OPPOSING_GAP_PAIR_CENTER_CUTOFF = 8.4
OPPOSING_GAP_PAIR_HBOND_CENTER_CUTOFF = 8.4
OPPOSING_GAP_PAIR_STACKING_CUTOFF = 7.2
OPPOSING_GAP_MISMATCH_CENTER_CUTOFF = 7.0
OPPOSING_GAP_MISMATCH_HBOND_CENTER_CUTOFF = 6.2
OPPOSING_GAP_MISMATCH_STACKING_CUTOFF = 4.0
MIN_PAIR_COUNT_FOR_TOPOLOGY = 2

# Heavy-atom proxies for common nucleobase H-bonds. Hydrogens are normally
# absent from PDB/mmCIF files, so topology inference uses these donor/acceptor
# atom distances rather than center-of-mass proximity.
BASE_PAIR_HBONDS = {
    ("A", "T"): (("N1", "N3"), ("N6", "O4")),
    ("A", "U"): (("N1", "N3"), ("N6", "O4")),
    ("G", "C"): (("N1", "N3"), ("N2", "O2"), ("O6", "N4")),
    ("G", "U"): (("N1", "O2"), ("O6", "N3")),
    ("I", "C"): (("N1", "N3"), ("O6", "N4")),
    # Hoogsteen-like contacts are accepted for topology construction and are
    # marked in generated inp files so Hoogsteen-aware fitted frames are used.
    ("A", "T", "hoogsteen"): (("N7", "N3"), ("N6", "O4")),
    ("A", "U", "hoogsteen"): (("N7", "N3"), ("N6", "O4")),
    ("G", "C", "hoogsteen"): (("N7", "N3"), ("O6", "N4")),
}


@dataclass
class ResidueNode:
    subunit: int
    chain: str
    res_id: int
    res_name: str
    base: str
    atom_start: int
    atom_end: int
    center: np.ndarray
    sugar_center: np.ndarray
    hbond_center: np.ndarray


@dataclass(frozen=True)
class BasePairCandidate:
    first: int
    second: int
    first_strand: int
    second_strand: int
    hbond_count: int
    mean_distance: float
    score: float
    pair_family: str
    atom_pairs: Tuple[Tuple[str, str, float], ...]
    is_hoogsteen: bool = False


@dataclass
class InferredTopology:
    pdbfile: str
    output_prefix: str
    strands: List[List[int]]
    nu_raw: List[int]
    ni_map: np.ndarray
    pair_edges: List[Tuple[int, int]]
    chain_ids: List[str]
    comb: bool = True
    fit: bool = True
    grv: bool = True
    ends: bool = False
    hoogsteen_markers: set = field(default_factory=set)

    @property
    def n_strands(self) -> int:
        return len(self.strands)

    @property
    def n_levels(self) -> int:
        return int(self.ni_map.shape[1])

    def to_inp_text(self) -> str:
        def bool_token(value: bool) -> str:
            return ".t." if value else ".f."

        lines = [
            (
                f"&inp file={self.pdbfile}, comb={bool_token(self.comb)}, "
                f"fit={bool_token(self.fit)}, grv={bool_token(self.grv)}, "
                f"ends={bool_token(self.ends)}, "
                f"lis={self.output_prefix}, pdb={self.output_prefix}_grp, &end"
            )
        ]
        padded_nu = self.nu_raw + [0] * (4 - len(self.nu_raw))
        lines.append(" ".join([str(self.n_strands)] + [str(v) for v in padded_nu]))
        for strand, row in enumerate(self.ni_map, start=1):
            tokens = []
            for level, value in enumerate(row, start=1):
                token = str(int(value))
                if int(value) != 0 and (
                    (strand, level) in self.hoogsteen_markers
                    or level in self.hoogsteen_markers
                ):
                    token = f"{token}[Hoog]"
                tokens.append(token)
            lines.append(" " + " ".join(tokens))
        lines.append("0.0 0.0 0.0 0.0")
        if self.ends:
            lines.append("0.0 0.0 3.4 0.0 0.0 0.0")
            lines.append("0.0 0.0 3.4 0.0 0.0 0.0")
        return "\n".join(lines) + "\n"


class RobustTopologyInferrer:
    """
    Infer Curves-style strand maps from an already loaded MolecularStructure.

    The generated maps use Curves subunit numbers, because the rest of this port
    follows the original Fortran indexing model.
    """

    def __init__(self, mol: MolecularStructure, pdbfile: Optional[str] = None):
        self.mol = mol
        self.pdbfile = pdbfile or ""
        self.residues: Dict[int, ResidueNode] = {}
        self.strands: List[List[int]] = []
        self.complexes: List[List[int]] = []
        self.pair_edges: List[Tuple[int, int]] = []

    def infer(self, continuous_strands: bool = False) -> List[InferredTopology]:
        self._collect_residues()
        self._trace_strands()

        if continuous_strands:
            pairing_graph = self._build_pairing_graph()
            self._partition_complexes(pairing_graph)
            topologies = []
            for complex_strands in self.complexes:
                topologies.append(self._generate_context_data(complex_strands))
            if topologies:
                return topologies
            return [self._single_strand_topology(idx) for idx in range(len(self.strands))]

        pair_candidates = self._find_base_pair_candidates()
        topologies = self._generate_pair_topologies(pair_candidates)
        if topologies:
            return topologies
        return [self._single_strand_topology(idx) for idx in range(len(self.strands))]

    def write_inp_files(
        self,
        output_dir: str = ".",
        prefix: Optional[str] = None,
        continuous_strands: bool = False,
        fit_override: Optional[bool] = None,
        grv_override: Optional[bool] = None,
        comb_override: Optional[bool] = None,
        ends_override: Optional[bool] = None,
    ) -> List[str]:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_paths = []
        if comb_override is False:
            # In comb=.f. mode Curves analyzes each strand independently.  Avoid
            # complement-pair inference and emit one editable input per strand.
            self._collect_residues()
            self._trace_strands()
            topologies = [self._single_strand_topology(idx) for idx in range(len(self.strands))]
        else:
            topologies = self.infer(continuous_strands=continuous_strands)
        for idx, topology in enumerate(topologies, start=1):
            if fit_override is not None:
                topology.fit = fit_override
            if grv_override is not None:
                topology.grv = grv_override
            if comb_override is not None:
                topology.comb = comb_override
            if ends_override is not None:
                topology.ends = ends_override

            stem = prefix or Path(self.pdbfile).stem
            suffix = "" if len(topologies) == 1 else (f"_strand{idx}" if comb_override is False else f"_{idx}")
            path = output_root / f"{stem}{suffix}.inp"
            topology.output_prefix = path.stem
            path.write_text(topology.to_inp_text(), encoding="utf-8")
            output_paths.append(str(path))
        return output_paths

    def _collect_residues(self) -> None:
        self.residues = {}
        boundaries = self.mol.subunit_boundaries
        if boundaries is None:
            raise ValueError("MolecularStructure.subunit_boundaries is missing. Load with MolecularLoader first.")

        for subunit in range(1, len(boundaries)):
            start = int(boundaries[subunit - 1])
            end = int(boundaries[subunit])
            atom_names = [str(v).strip().upper() for v in self.mol.atom_names[start:end]]
            base = self._base_symbol(str(self.mol.residue_names[start]))
            if base not in DNA_BASES:
                continue
            if "C1'" not in atom_names and "C1*" not in atom_names:
                continue

            base_indices = [
                start + i for i, name in enumerate(atom_names)
                if name in BASE_ATOMS.get(base, set()) or name in {"N1", "N9", "C2", "C4", "C5", "C6", "C8"}
            ]
            sugar_indices = [
                start + i for i, name in enumerate(atom_names)
                if name in {"C1'", "C1*", "C2'", "C2*", "C3'", "C3*", "C4'", "C4*", "O4'", "O4*"}
            ]
            if len(base_indices) < 3:
                continue

            center = np.mean(self.mol.coordinates[base_indices], axis=0)
            sugar_center = np.mean(self.mol.coordinates[sugar_indices], axis=0)
            
            hbond_indices = [
                start + i for i, name in enumerate(atom_names)
                if name in HBOND_ATOMS.get(base, set())
            ]
            if not hbond_indices:
                hbond_indices = base_indices
            hbond_center = np.mean(self.mol.coordinates[hbond_indices], axis=0)

            chain = ""
            if self.mol.chain_ids is not None:
                chain = str(self.mol.chain_ids[start]).strip()
            self.residues[subunit] = ResidueNode(
                subunit=subunit,
                chain=chain,
                res_id=int(self.mol.residue_ids[start]),
                res_name=str(self.mol.residue_names[start]).strip(),
                base=base,
                atom_start=start,
                atom_end=end,
                center=center,
                sugar_center=sugar_center,
                hbond_center=hbond_center,
            )

    def _trace_strands(self) -> None:
        ordered = sorted(self.residues.values(), key=lambda r: r.subunit)
        strands = []
        current = []
        previous = None
        for residue in ordered:
            if previous is not None:
                gap = residue.subunit - previous.subunit
                distance = np.linalg.norm(residue.sugar_center - previous.sugar_center)
                chain_changed = bool(residue.chain and previous.chain and residue.chain != previous.chain)
                backbone_break = self._has_suspicious_backbone_break(previous, residue, distance)
                if chain_changed or gap > 4 or distance > 12.0 or backbone_break:
                    if current:
                        strands.append([r.subunit for r in current])
                    current = []
            current.append(residue)
            previous = residue
        if current:
            strands.append([r.subunit for r in current])

        self.strands = [s for s in strands if len(s) >= 2]

    def _has_suspicious_backbone_break(
        self,
        previous: ResidueNode,
        current: ResidueNode,
        sugar_center_distance: float,
    ) -> bool:
        """Detect chain-internal polymer breaks that are encoded with one chain ID."""
        if sugar_center_distance <= SUSPICIOUS_BACKBONE_GAP_DISTANCE:
            return False

        link_distance = self._phosphodiester_link_distance(previous, current)
        if link_distance is None:
            return True
        return False

    def _phosphodiester_link_distance(
        self,
        previous: ResidueNode,
        current: ResidueNode,
    ) -> Optional[float]:
        prev_o3 = self._atom_coordinate(previous, {"O3'", "O3*"})
        curr_p = self._atom_coordinate(current, {"P"})
        if prev_o3 is None or curr_p is None:
            return None
        return float(np.linalg.norm(prev_o3 - curr_p))

    def _atom_coordinate(self, residue: ResidueNode, atom_names: Iterable[str]) -> Optional[np.ndarray]:
        wanted = {name.strip().upper() for name in atom_names}
        for atom_idx in range(residue.atom_start, residue.atom_end):
            atom_name = str(self.mol.atom_names[atom_idx]).strip().upper()
            if atom_name in wanted:
                return self.mol.coordinates[atom_idx]
        return None

    def _find_base_pair_candidates(self) -> List[BasePairCandidate]:
        """Return one-to-one H-bonded base pairs for each strand pair."""
        strand_of = {
            subunit: strand_idx
            for strand_idx, strand in enumerate(self.strands)
            for subunit in strand
        }
        grouped: Dict[Tuple[int, int], List[BasePairCandidate]] = {}

        for candidate in self._source_base_pair_candidates(strand_of):
            key = tuple(sorted((candidate.first_strand, candidate.second_strand)))
            grouped.setdefault(key, []).append(candidate)

        for i, first_strand in enumerate(self.strands):
            for j in range(i + 1, len(self.strands)):
                second_strand = self.strands[j]
                candidates = []
                for first in first_strand:
                    for second in second_strand:
                        residue_1 = self.residues[first]
                        residue_2 = self.residues[second]
                        if np.linalg.norm(residue_1.hbond_center - residue_2.hbond_center) > HBOND_PREFILTER_DISTANCE:
                            continue
                        candidate = self._score_base_pair(first, second, i, j)
                        if candidate is not None:
                            candidates.append(candidate)
                if candidates:
                    grouped.setdefault((i, j), []).extend(candidates)

        selected_groups = [
            self._select_one_to_one_pairs(candidates)
            for candidates in grouped.values()
        ]
        return [candidate for group in selected_groups for candidate in group]

    def _source_base_pair_candidates(self, strand_of: Dict[int, int]) -> List[BasePairCandidate]:
        source_rows = list(getattr(self.mol, "source_base_pairs", None) or [])
        if not source_rows:
            return []

        by_location: Dict[Tuple[str, int], List[int]] = {}
        for subunit, residue in self.residues.items():
            by_location.setdefault((residue.chain, residue.res_id), []).append(subunit)

        candidates = []
        seen = set()
        for row in source_rows:
            left_chain = str(row.get("i_generated_chain_id") or row.get("i_chain_id", "")).strip()
            right_chain = str(row.get("j_generated_chain_id") or row.get("j_chain_id", "")).strip()
            try:
                left_resid = int(row.get("i_residue_id") or 0)
                right_resid = int(row.get("j_residue_id") or 0)
            except (TypeError, ValueError):
                continue

            for left in by_location.get((left_chain, left_resid), []):
                for right in by_location.get((right_chain, right_resid), []):
                    left_strand = strand_of.get(left)
                    right_strand = strand_of.get(right)
                    if left_strand is None or right_strand is None or left_strand == right_strand:
                        continue
                    key = tuple(sorted((left, right)))
                    if key in seen:
                        continue
                    seen.add(key)
                    residue_1 = self.residues[left]
                    residue_2 = self.residues[right]
                    center_distance = float(np.linalg.norm(residue_1.center - residue_2.center))
                    candidates.append(BasePairCandidate(
                        first=left,
                        second=right,
                        first_strand=left_strand,
                        second_strand=right_strand,
                        hbond_count=2,
                        mean_distance=center_distance,
                        score=-100.0 + center_distance,
                        pair_family="source_annotated",
                        atom_pairs=(),
                        is_hoogsteen=bool(row.get("is_hoogsteen")),
                    ))
        return candidates

    def _score_base_pair(
        self,
        first: int,
        second: int,
        first_strand: int,
        second_strand: int,
    ) -> Optional[BasePairCandidate]:
        residue_1 = self.residues[first]
        residue_2 = self.residues[second]
        atom_map_1 = self._atom_map(residue_1)
        atom_map_2 = self._atom_map(residue_2)

        pattern_matches, pattern_family = self._pattern_hbond_matches(residue_1.base, residue_2.base, atom_map_1, atom_map_2)
        generic_matches = self._generic_hbond_matches(residue_1, residue_2, atom_map_1, atom_map_2)
        matches = pattern_matches if len(pattern_matches) >= len(generic_matches) else generic_matches

        if len(pattern_matches) >= 2:
            pair_family = pattern_family
        elif len(generic_matches) >= 2:
            pair_family = "hbonded_noncanonical"
        else:
            return None

        center_distance = float(np.linalg.norm(residue_1.center - residue_2.center))
        if center_distance > 9.0:
            return None

        distances = [distance for _, _, distance in matches]
        mean_distance = float(np.mean(distances))
        # More H-bonds are better; shorter mean distance breaks ties.
        score = -10.0 * len(matches) + mean_distance + 0.05 * center_distance
        return BasePairCandidate(
            first=first,
            second=second,
            first_strand=first_strand,
            second_strand=second_strand,
            hbond_count=len(matches),
            mean_distance=mean_distance,
            score=score,
            pair_family=pair_family,
            atom_pairs=tuple(matches),
            is_hoogsteen=pair_family == "hoogsteen_like",
        )

    def _pattern_hbond_matches(
        self,
        base_1: str,
        base_2: str,
        atom_map_1: Dict[str, np.ndarray],
        atom_map_2: Dict[str, np.ndarray],
    ) -> Tuple[List[Tuple[str, str, float]], str]:
        possible_patterns = [
            ((base_1, base_2), "watson_crick_or_wobble"),
            ((base_1, base_2, "hoogsteen"), "hoogsteen_like"),
        ]
        reversed_patterns = [
            ((base_2, base_1), "watson_crick_or_wobble"),
            ((base_2, base_1, "hoogsteen"), "hoogsteen_like"),
        ]

        best_matches: List[Tuple[str, str, float]] = []
        best_family = "unknown"
        for key, family in possible_patterns:
            matches = self._matches_for_pattern(BASE_PAIR_HBONDS.get(key, ()), atom_map_1, atom_map_2)
            if len(matches) > len(best_matches):
                best_matches = matches
                best_family = family
        for key, family in reversed_patterns:
            reverse_pattern = tuple((right, left) for left, right in BASE_PAIR_HBONDS.get(key, ()))
            matches = self._matches_for_pattern(reverse_pattern, atom_map_1, atom_map_2)
            if len(matches) > len(best_matches):
                best_matches = matches
                best_family = family
        return best_matches, best_family

    @staticmethod
    def _matches_for_pattern(
        pattern: Iterable[Tuple[str, str]],
        atom_map_1: Dict[str, np.ndarray],
        atom_map_2: Dict[str, np.ndarray],
        cutoff: float = HBOND_DISTANCE_CUTOFF,
    ) -> List[Tuple[str, str, float]]:
        matches = []
        for atom_1, atom_2 in pattern:
            if atom_1 not in atom_map_1 or atom_2 not in atom_map_2:
                continue
            distance = float(np.linalg.norm(atom_map_1[atom_1] - atom_map_2[atom_2]))
            if distance <= cutoff:
                matches.append((atom_1, atom_2, distance))
        return matches

    def _generic_hbond_matches(
        self,
        residue_1: ResidueNode,
        residue_2: ResidueNode,
        atom_map_1: Dict[str, np.ndarray],
        atom_map_2: Dict[str, np.ndarray],
    ) -> List[Tuple[str, str, float]]:
        close_contacts = []
        for atom_1 in HBOND_ATOMS.get(residue_1.base, set()):
            if atom_1 not in atom_map_1:
                continue
            for atom_2 in HBOND_ATOMS.get(residue_2.base, set()):
                if atom_2 not in atom_map_2:
                    continue
                distance = float(np.linalg.norm(atom_map_1[atom_1] - atom_map_2[atom_2]))
                if distance <= HBOND_DISTANCE_CUTOFF:
                    close_contacts.append((atom_1, atom_2, distance))

        close_contacts.sort(key=lambda item: item[2])
        used_1 = set()
        used_2 = set()
        matches = []
        for atom_1, atom_2, distance in close_contacts:
            if atom_1 in used_1 or atom_2 in used_2:
                continue
            used_1.add(atom_1)
            used_2.add(atom_2)
            matches.append((atom_1, atom_2, distance))
        return matches

    def _atom_map(self, residue: ResidueNode) -> Dict[str, np.ndarray]:
        atom_map = {}
        for atom_idx in range(residue.atom_start, residue.atom_end):
            atom_name = str(self.mol.atom_names[atom_idx]).strip().upper()
            atom_map.setdefault(atom_name, self.mol.coordinates[atom_idx])
        return atom_map

    def _base_normal(self, residue: ResidueNode) -> np.ndarray:
        atom_map = self._atom_map(residue)
        coords = np.array(list(atom_map.values()))
        center = np.mean(coords, axis=0)
        cov = np.cov(coords - center, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        return eigvecs[:, 0]

    @staticmethod
    def _select_one_to_one_pairs(candidates: Sequence[BasePairCandidate]) -> List[BasePairCandidate]:
        selected = []
        used_first = set()
        used_second = set()
        for candidate in sorted(candidates, key=lambda item: item.score):
            if candidate.first in used_first or candidate.second in used_second:
                continue
            used_first.add(candidate.first)
            used_second.add(candidate.second)
            selected.append(candidate)
        return selected

    def _generate_pair_topologies(self, pair_candidates: Sequence[BasePairCandidate]) -> List[InferredTopology]:
        groups: Dict[Tuple[int, int], List[BasePairCandidate]] = {}
        for candidate in pair_candidates:
            key = tuple(sorted((candidate.first_strand, candidate.second_strand)))
            groups.setdefault(key, []).append(candidate)

        groups = {
            key: value
            for key, value in groups.items()
            if len(value) >= MIN_PAIR_COUNT_FOR_TOPOLOGY
        }
        topologies = []
        while groups:
            incident_counts = {idx: 0 for idx in range(len(self.strands))}
            for (first_idx, second_idx), candidates in groups.items():
                incident_counts[first_idx] += len(candidates)
                incident_counts[second_idx] += len(candidates)

            primary_idx = max(
                incident_counts,
                key=lambda idx: (incident_counts[idx], len(self.strands[idx]), -idx),
            )
            partner_keys = [
                key for key in groups
                if primary_idx in key
            ]
            if not partner_keys:
                break
            selected_key = max(
                partner_keys,
                key=lambda key: (len(groups[key]), len(self.strands[key[0] if key[1] == primary_idx else key[1]]), -key[0], -key[1]),
            )
            partner_idx = selected_key[1] if selected_key[0] == primary_idx else selected_key[0]
            selected_pairs = groups.pop(selected_key)
            topologies.append(self._generate_duplex_topology(primary_idx, partner_idx, selected_pairs))

        return topologies

    def _generate_duplex_topology(
        self,
        primary_idx: int,
        partner_idx: int,
        selected_pairs: Sequence[BasePairCandidate],
    ) -> InferredTopology:
        primary_strand = self.strands[primary_idx]
        partner_strand = self.strands[partner_idx]
        primary_position = {subunit: idx for idx, subunit in enumerate(primary_strand)}
        partner_position = {subunit: idx for idx, subunit in enumerate(partner_strand)}

        pair_by_primary: Dict[int, Tuple[int, BasePairCandidate]] = {}
        for candidate in selected_pairs:
            if candidate.first_strand == primary_idx:
                primary_subunit, partner_subunit = candidate.first, candidate.second
            else:
                primary_subunit, partner_subunit = candidate.second, candidate.first
            current = pair_by_primary.get(primary_subunit)
            if current is None or candidate.score < current[1].score:
                pair_by_primary[primary_subunit] = (partner_subunit, candidate)

        anchors = [
            (primary_subunit, partner_subunit, candidate)
            for primary_subunit, (partner_subunit, candidate) in pair_by_primary.items()
        ]
        anchors.sort(key=lambda item: primary_position[item[0]])

        partner_positions = [partner_position[partner_subunit] for _, partner_subunit, _ in anchors]
        partner_direction = -1
        if len(partner_positions) >= 2:
            partner_direction = 1 if partner_positions[-1] > partner_positions[0] else -1

        anchors = self._drop_isolated_register_outliers(
            anchors,
            primary_strand,
            partner_strand,
            primary_position,
            partner_position,
            partner_direction,
        )

        row_1, row_2 = self._rows_with_internal_gaps(
            primary_strand,
            partner_strand,
            anchors,
            primary_position,
            partner_position,
            partner_direction,
        )
        self._extend_terminal_register_pairs(
            row_1,
            row_2,
            primary_strand,
            partner_strand,
            primary_position,
            partner_position,
            partner_direction,
        )
        self._collapse_opposing_internal_gap_pairs(row_1, row_2)

        paired_count = sum(1 for left, right in zip(row_1, row_2) if left > 0 and right > 0)
        nu_raw = [sum(1 for subunit in row_1 if subunit > 0), partner_direction * sum(1 for subunit in row_2 if subunit > 0)]
        ni_map = np.array([row_1, row_2], dtype=int)
        pair_edges = [(candidate.first, candidate.second) for candidate in selected_pairs]
        level_by_subunit = {}
        strand_by_subunit = {}
        for level, subunit in enumerate(row_1, start=1):
            if subunit > 0:
                level_by_subunit[subunit] = level
                strand_by_subunit[subunit] = 1
        for level, subunit in enumerate(row_2, start=1):
            if subunit > 0:
                level_by_subunit[subunit] = level
                strand_by_subunit[subunit] = 2
        hoogsteen_markers = set()
        for candidate in selected_pairs:
            if not candidate.is_hoogsteen:
                continue
            if candidate.first not in level_by_subunit or candidate.second not in level_by_subunit:
                continue
            if level_by_subunit[candidate.first] != level_by_subunit[candidate.second]:
                continue
            marker_subunit = self._hoogsteen_marker_subunit(candidate)
            if marker_subunit not in level_by_subunit:
                continue
            hoogsteen_markers.add((strand_by_subunit[marker_subunit], level_by_subunit[marker_subunit]))

        return InferredTopology(
            pdbfile=self.pdbfile,
            output_prefix=Path(self.pdbfile).stem,
            strands=[row_1, [subunit for subunit in row_2 if subunit > 0]],
            nu_raw=nu_raw,
            ni_map=ni_map,
            pair_edges=pair_edges,
            chain_ids=[self.residues[primary_strand[0]].chain, self.residues[partner_strand[0]].chain],
            comb=True,
            fit=True,
            grv=paired_count >= 4,
            hoogsteen_markers=hoogsteen_markers,
        )

    def _hoogsteen_marker_subunit(self, candidate: BasePairCandidate) -> int:
        """Return the base that uses the Hoogsteen edge in a marked pair."""
        purines = {"A", "G", "I"}
        first_is_purine = self.residues[candidate.first].base in purines
        second_is_purine = self.residues[candidate.second].base in purines
        if first_is_purine and not second_is_purine:
            return candidate.first
        if second_is_purine and not first_is_purine:
            return candidate.second
        return candidate.first

    def _drop_isolated_register_outliers(
        self,
        anchors: Sequence[Tuple[int, int, BasePairCandidate]],
        primary_strand: Sequence[int],
        partner_strand: Sequence[int],
        primary_position: Dict[int, int],
        partner_position: Dict[int, int],
        partner_direction: int,
    ) -> List[Tuple[int, int, BasePairCandidate]]:
        """Remove one-off anchors that shift an otherwise continuous helix register."""
        filtered = list(anchors)
        if len(filtered) < 3:
            return filtered

        partner_at_position = {position: subunit for subunit, position in partner_position.items()}
        changed = True
        while changed and len(filtered) >= 3:
            changed = False
            for idx in range(1, len(filtered) - 1):
                prev_primary, prev_partner, _ = filtered[idx - 1]
                curr_primary, curr_partner, curr_candidate = filtered[idx]
                next_primary, next_partner, _ = filtered[idx + 1]

                prev_p = primary_position[prev_primary]
                curr_p = primary_position[curr_primary]
                next_p = primary_position[next_primary]
                prev_q = partner_position[prev_partner]
                curr_q = partner_position[curr_partner]
                next_q = partner_position[next_partner]

                through_mismatch = abs((next_q - prev_q) - partner_direction * (next_p - prev_p))
                split_mismatch = (
                    abs((curr_q - prev_q) - partner_direction * (curr_p - prev_p))
                    + abs((next_q - curr_q) - partner_direction * (next_p - curr_p))
                )

                geometry_score = self._pair_geometry_score(curr_primary, curr_partner)
                expected_partner_pos = prev_q + partner_direction * (curr_p - prev_p)
                expected_partner = partner_at_position.get(expected_partner_pos)
                current_is_register_shift = expected_partner is not None and expected_partner != curr_partner
                register_pair_supported = (
                    current_is_register_shift
                    and (
                        self._register_gap_pair_is_plausible(curr_primary, expected_partner)
                        or self._stacking_bridge_pair_is_plausible(
                            curr_primary,
                            expected_partner,
                            prev_primary,
                            next_primary,
                            prev_partner,
                            next_partner,
                        )
                    )
                )
                current_is_weak_noncanonical = (
                    curr_candidate.pair_family == "hbonded_noncanonical"
                    or not self._is_complementary(
                        self.residues[curr_primary].base,
                        self.residues[curr_partner].base,
                    )
                )

                if (
                    through_mismatch == 0
                    and split_mismatch >= 2
                    and (
                        geometry_score >= 7.0
                        or curr_candidate.pair_family == "source_annotated"
                        or (current_is_weak_noncanonical and register_pair_supported)
                    )
                ):
                    filtered.pop(idx)
                    changed = True
                    break
        filtered = self._drop_terminal_register_outliers(
            filtered,
            primary_strand,
            partner_strand,
            primary_position,
            partner_position,
            partner_direction,
        )
        return filtered

    def _drop_terminal_register_outliers(
        self,
        anchors: Sequence[Tuple[int, int, BasePairCandidate]],
        primary_strand: Sequence[int],
        partner_strand: Sequence[int],
        primary_position: Dict[int, int],
        partner_position: Dict[int, int],
        partner_direction: int,
    ) -> List[Tuple[int, int, BasePairCandidate]]:
        """Prefer a clean terminal register over a dangling mismatch-like endpoint."""
        filtered = list(anchors)
        primary_at_position = {idx: subunit for idx, subunit in enumerate(primary_strand)}
        partner_at_position = {idx: subunit for idx, subunit in enumerate(partner_strand)}

        while len(filtered) >= 2:
            first_primary, first_partner, first_candidate = filtered[0]
            next_primary, next_partner, _ = filtered[1]
            first_p = primary_position[first_primary]
            next_p = primary_position[next_primary]
            first_q = partner_position[first_partner]
            next_q = partner_position[next_partner]
            register_mismatch = abs((next_q - first_q) - partner_direction * (next_p - first_p))
            alternative_pairs = self._count_terminal_register_pairs(
                primary_at_position,
                partner_at_position,
                next_p - 1,
                next_q - partner_direction,
                -1,
                -partner_direction,
            )
            if (
                register_mismatch > 0
                and alternative_pairs >= 2
                and self._is_weak_terminal_anchor(first_primary, first_partner, first_candidate)
            ):
                filtered.pop(0)
                continue
            break

        while len(filtered) >= 2:
            prev_primary, prev_partner, _ = filtered[-2]
            last_primary, last_partner, last_candidate = filtered[-1]
            prev_p = primary_position[prev_primary]
            last_p = primary_position[last_primary]
            prev_q = partner_position[prev_partner]
            last_q = partner_position[last_partner]
            register_mismatch = abs((last_q - prev_q) - partner_direction * (last_p - prev_p))
            alternative_pairs = self._count_terminal_register_pairs(
                primary_at_position,
                partner_at_position,
                prev_p + 1,
                prev_q + partner_direction,
                1,
                partner_direction,
            )
            if (
                register_mismatch > 0
                and alternative_pairs >= 2
                and self._is_weak_terminal_anchor(last_primary, last_partner, last_candidate)
            ):
                filtered.pop()
                continue
            break

        return filtered

    def _count_terminal_register_pairs(
        self,
        primary_at_position: Dict[int, int],
        partner_at_position: Dict[int, int],
        primary_pos: int,
        partner_pos: int,
        primary_step: int,
        partner_step: int,
    ) -> int:
        count = 0
        while True:
            primary_subunit = primary_at_position.get(primary_pos)
            partner_subunit = partner_at_position.get(partner_pos)
            if primary_subunit is None or partner_subunit is None:
                return count
            if not self._terminal_subunit_pair_is_plausible(primary_subunit, partner_subunit):
                return count
            count += 1
            primary_pos += primary_step
            partner_pos += partner_step

    def _is_weak_terminal_anchor(
        self,
        primary_subunit: int,
        partner_subunit: int,
        candidate: BasePairCandidate,
    ) -> bool:
        residue_1 = self.residues[primary_subunit]
        residue_2 = self.residues[partner_subunit]
        if candidate.pair_family == "source_annotated":
            return False
        if candidate.pair_family == "hbonded_noncanonical":
            return True
        return not self._is_complementary(residue_1.base, residue_2.base)

    def _rows_with_internal_gaps(
        self,
        primary_strand: Sequence[int],
        partner_strand: Sequence[int],
        anchors: Sequence[Tuple[int, int, BasePairCandidate]],
        primary_position: Dict[int, int],
        partner_position: Dict[int, int],
        partner_direction: int,
    ) -> Tuple[List[int], List[int]]:
        row_1: List[int] = []
        row_2: List[int] = []

        for idx, (primary_subunit, partner_subunit, _) in enumerate(anchors[:-1]):
            next_primary, next_partner, _ = anchors[idx + 1]
            row_1.append(primary_subunit)
            row_2.append(partner_subunit)

            primary_between = list(primary_strand[primary_position[primary_subunit] + 1:primary_position[next_primary]])
            partner_between_positions = range(
                partner_position[partner_subunit] + partner_direction,
                partner_position[next_partner],
                partner_direction,
            )
            partner_between = [partner_strand[pos] for pos in partner_between_positions]
            self._append_unpaired_between(
                row_1,
                row_2,
                primary_between,
                partner_between,
                primary_subunit,
                next_primary,
                partner_subunit,
                next_partner,
            )

        last_primary, last_partner, _ = anchors[-1]
        row_1.append(last_primary)
        row_2.append(last_partner)
        return row_1, row_2

    def _collapse_opposing_internal_gap_pairs(self, row_1: List[int], row_2: List[int]) -> None:
        """Collapse adjacent opposing gaps into a paired level when geometry supports it."""
        idx = 0
        while idx < len(row_1) - 1:
            pair = self._opposing_gap_pair(row_1, row_2, idx)
            if pair is None:
                idx += 1
                continue

            primary_subunit, partner_subunit = pair
            left_idx = self._nearest_paired_column(row_1, row_2, idx - 1, -1)
            right_idx = self._nearest_paired_column(row_1, row_2, idx + 2, 1)
            if left_idx is None or right_idx is None:
                idx += 1
                continue

            if self._opposing_gap_pair_is_plausible(
                primary_subunit,
                partner_subunit,
                row_1[left_idx],
                row_1[right_idx],
                row_2[left_idx],
                row_2[right_idx],
            ):
                row_1[idx] = primary_subunit
                row_2[idx] = partner_subunit
                del row_1[idx + 1]
                del row_2[idx + 1]
                continue
            idx += 1

    @staticmethod
    def _opposing_gap_pair(row_1: Sequence[int], row_2: Sequence[int], idx: int) -> Optional[Tuple[int, int]]:
        first_primary = int(row_1[idx])
        first_partner = int(row_2[idx])
        second_primary = int(row_1[idx + 1])
        second_partner = int(row_2[idx + 1])
        if first_primary > 0 and first_partner == 0 and second_primary == 0 and second_partner > 0:
            return first_primary, second_partner
        if first_primary == 0 and first_partner > 0 and second_primary > 0 and second_partner == 0:
            return second_primary, first_partner
        return None

    @staticmethod
    def _nearest_paired_column(
        row_1: Sequence[int],
        row_2: Sequence[int],
        start: int,
        step: int,
    ) -> Optional[int]:
        idx = start
        while 0 <= idx < len(row_1):
            if row_1[idx] > 0 and row_2[idx] > 0:
                return idx
            idx += step
        return None

    def _opposing_gap_pair_is_plausible(
        self,
        primary_subunit: int,
        partner_subunit: int,
        left_primary: int,
        right_primary: int,
        left_partner: int,
        right_partner: int,
    ) -> bool:
        residue_1 = self.residues[primary_subunit]
        residue_2 = self.residues[partner_subunit]
        center_distance = float(np.linalg.norm(residue_1.center - residue_2.center))
        hbond_center_distance = float(np.linalg.norm(residue_1.hbond_center - residue_2.hbond_center))
        if center_distance > OPPOSING_GAP_PAIR_CENTER_CUTOFF:
            return False
        if hbond_center_distance > OPPOSING_GAP_PAIR_HBOND_CENTER_CUTOFF:
            return False

        atom_map_1 = self._atom_map(residue_1)
        atom_map_2 = self._atom_map(residue_2)
        pattern_matches, _ = self._pattern_hbond_matches(residue_1.base, residue_2.base, atom_map_1, atom_map_2)
        generic_matches = self._generic_hbond_matches(residue_1, residue_2, atom_map_1, atom_map_2)
        stacking_score = self._stacking_register_score(
            primary_subunit,
            partner_subunit,
            left_primary,
            right_primary,
            left_partner,
            right_partner,
        )

        if self._is_complementary(residue_1.base, residue_2.base):
            if self._register_gap_pair_is_plausible(primary_subunit, partner_subunit):
                return True
            if pattern_matches or generic_matches:
                return stacking_score <= OPPOSING_GAP_PAIR_STACKING_CUTOFF
            return stacking_score <= OPPOSING_GAP_PAIR_STACKING_CUTOFF

        if center_distance > OPPOSING_GAP_MISMATCH_CENTER_CUTOFF:
            return False
        if hbond_center_distance > OPPOSING_GAP_MISMATCH_HBOND_CENTER_CUTOFF:
            return False
        if not generic_matches and not pattern_matches:
            return False
        return stacking_score <= OPPOSING_GAP_MISMATCH_STACKING_CUTOFF

    def _append_unpaired_between(
        self,
        row_1: List[int],
        row_2: List[int],
        primary_between: Sequence[int],
        partner_between: Sequence[int],
        left_primary: int,
        right_primary: int,
        left_partner: int,
        right_partner: int,
    ) -> None:
        if not primary_between and not partner_between:
            return
        if primary_between and not partner_between:
            return
        if partner_between and not primary_between:
            return
        if len(primary_between) == len(partner_between):
            paired_geometry = [
                self._register_gap_pair_is_plausible(primary_subunit, partner_subunit)
                or self._bracketed_mismatch_pair_is_plausible(
                    primary_subunit,
                    partner_subunit,
                    left_primary,
                    right_primary,
                    left_partner,
                    right_partner,
                )
                for primary_subunit, partner_subunit in zip(primary_between, partner_between)
            ]
            if all(paired_geometry):
                for primary_subunit, partner_subunit in zip(primary_between, partner_between):
                    row_1.append(primary_subunit)
                    row_2.append(partner_subunit)
                return

        aligned_gap_rows = self._align_internal_register_segment(
            primary_between,
            partner_between,
            left_primary,
            right_primary,
            left_partner,
            right_partner,
        )
        if aligned_gap_rows is not None:
            for primary_subunit, partner_subunit in aligned_gap_rows:
                row_1.append(primary_subunit)
                row_2.append(partner_subunit)
            return

        primary_score = self._stacking_gap_score(primary_between, left_primary, right_primary)
        partner_score = self._stacking_gap_score(partner_between, left_partner, right_partner)
        if partner_score < primary_score:
            for subunit in partner_between:
                row_1.append(0)
                row_2.append(subunit)
        else:
            for subunit in primary_between:
                row_1.append(subunit)
                row_2.append(0)

    def _register_gap_pair_is_plausible(self, primary_subunit: int, partner_subunit: int) -> bool:
        residue_1 = self.residues[primary_subunit]
        residue_2 = self.residues[partner_subunit]
        center_distance = float(np.linalg.norm(residue_1.center - residue_2.center))
        hbond_center_distance = float(np.linalg.norm(residue_1.hbond_center - residue_2.hbond_center))
        if center_distance > REGISTER_GAP_CENTER_CUTOFF:
            return False
        if hbond_center_distance > REGISTER_GAP_HBOND_CENTER_CUTOFF:
            return False

        atom_map_1 = self._atom_map(residue_1)
        atom_map_2 = self._atom_map(residue_2)
        pattern_matches, _ = self._pattern_hbond_matches(
            residue_1.base,
            residue_2.base,
            atom_map_1,
            atom_map_2,
        )
        if not self._is_complementary(residue_1.base, residue_2.base):
            generic_matches = self._generic_hbond_matches(residue_1, residue_2, atom_map_1, atom_map_2)
            if len(pattern_matches) >= 1 or len(generic_matches) >= 2:
                return True
            return (
                len(generic_matches) >= 1
                and center_distance <= REGISTER_GAP_NONCANONICAL_SINGLE_CONTACT_CENTER_CUTOFF
                and hbond_center_distance <= REGISTER_GAP_NONCANONICAL_SINGLE_CONTACT_HBOND_CENTER_CUTOFF
            )

        if self._pair_geometry_score(primary_subunit, partner_subunit) < 7.0:
            return True
        if len(pattern_matches) >= 1:
            return True
        return self._has_relaxed_template_contact(residue_1, residue_2)

    def _align_internal_register_segment(
        self,
        primary_between: Sequence[int],
        partner_between: Sequence[int],
        left_primary: int,
        right_primary: int,
        left_partner: int,
        right_partner: int,
    ) -> Optional[List[Tuple[int, int]]]:
        if not primary_between or not partner_between:
            return None

        n = len(primary_between)
        m = len(partner_between)
        pair_cost = np.full((n, m), np.inf)
        has_pair = False
        for i, primary_subunit in enumerate(primary_between):
            for j, partner_subunit in enumerate(partner_between):
                strict_pair = self._register_gap_pair_is_plausible(primary_subunit, partner_subunit)
                bridge_pair = False
                if not strict_pair:
                    bridge_pair = self._stacking_bridge_pair_is_plausible(
                        primary_subunit,
                        partner_subunit,
                        left_primary,
                        right_primary,
                        left_partner,
                        right_partner,
                    )
                mismatch_pair = False
                if not strict_pair and not bridge_pair:
                    mismatch_pair = self._bracketed_mismatch_pair_is_plausible(
                        primary_subunit,
                        partner_subunit,
                        left_primary,
                        right_primary,
                        left_partner,
                        right_partner,
                    )
                if strict_pair or bridge_pair or mismatch_pair:
                    center_distance = float(np.linalg.norm(
                        self.residues[primary_subunit].center - self.residues[partner_subunit].center
                    ))
                    stacking_score = self._stacking_register_score(
                        primary_subunit,
                        partner_subunit,
                        left_primary,
                        right_primary,
                        left_partner,
                        right_partner,
                    )
                    pair_cost[i, j] = center_distance / 12.0 + stacking_score / 4.0
                    if bridge_pair:
                        pair_cost[i, j] += 0.35
                    if mismatch_pair:
                        pair_cost[i, j] += 0.25
                    has_pair = True

        if not has_pair:
            return None

        gap_cost = 2.0
        dp = np.full((n + 1, m + 1), np.inf)
        move = np.zeros((n + 1, m + 1), dtype=np.int8)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            dp[i, 0] = i * gap_cost
            move[i, 0] = 1
        for j in range(1, m + 1):
            dp[0, j] = j * gap_cost
            move[0, j] = 2

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                diag = dp[i - 1, j - 1] + pair_cost[i - 1, j - 1]
                up = dp[i - 1, j] + gap_cost
                left = dp[i, j - 1] + gap_cost
                if diag <= up and diag <= left:
                    dp[i, j] = diag
                    move[i, j] = 0
                elif left <= up:
                    dp[i, j] = left
                    move[i, j] = 2
                else:
                    dp[i, j] = up
                    move[i, j] = 1

        aligned: List[Tuple[int, int]] = []
        used_pairs = 0
        i, j = n, m
        while i > 0 or j > 0:
            step = int(move[i, j])
            if step == 0 and i > 0 and j > 0 and np.isfinite(pair_cost[i - 1, j - 1]):
                aligned.append((primary_between[i - 1], partner_between[j - 1]))
                used_pairs += 1
                i -= 1
                j -= 1
            elif i > 0 and (step == 1 or j == 0):
                aligned.append((primary_between[i - 1], 0))
                i -= 1
            elif j > 0:
                aligned.append((0, partner_between[j - 1]))
                j -= 1
            else:
                return None

        if used_pairs == 0:
            return None
        aligned.reverse()
        return aligned

    def _stacking_bridge_pair_is_plausible(
        self,
        primary_subunit: int,
        partner_subunit: int,
        left_primary: int,
        right_primary: int,
        left_partner: int,
        right_partner: int,
    ) -> bool:
        """Allow one distorted canonical pair only when the local helix register supports it."""
        residue_1 = self.residues[primary_subunit]
        residue_2 = self.residues[partner_subunit]
        if not self._is_complementary(residue_1.base, residue_2.base):
            return False

        center_distance = float(np.linalg.norm(residue_1.center - residue_2.center))
        hbond_center_distance = float(np.linalg.norm(residue_1.hbond_center - residue_2.hbond_center))
        if center_distance > REGISTER_STACKING_BRIDGE_CENTER_CUTOFF:
            return False
        if hbond_center_distance > REGISTER_STACKING_BRIDGE_HBOND_CENTER_CUTOFF:
            return False

        stacking_score = self._stacking_register_score(
            primary_subunit,
            partner_subunit,
            left_primary,
            right_primary,
            left_partner,
            right_partner,
        )
        return stacking_score <= REGISTER_STACKING_BRIDGE_SCORE_CUTOFF

    def _bracketed_mismatch_pair_is_plausible(
        self,
        primary_subunit: int,
        partner_subunit: int,
        left_primary: int,
        right_primary: int,
        left_partner: int,
        right_partner: int,
    ) -> bool:
        """Accept a single-contact mismatch only when neighboring pairs define the register."""
        residue_1 = self.residues[primary_subunit]
        residue_2 = self.residues[partner_subunit]
        if self._is_complementary(residue_1.base, residue_2.base):
            return False

        center_distance = float(np.linalg.norm(residue_1.center - residue_2.center))
        hbond_center_distance = float(np.linalg.norm(residue_1.hbond_center - residue_2.hbond_center))
        if center_distance > OPPOSING_GAP_MISMATCH_CENTER_CUTOFF:
            return False
        if hbond_center_distance > OPPOSING_GAP_MISMATCH_HBOND_CENTER_CUTOFF:
            return False

        atom_map_1 = self._atom_map(residue_1)
        atom_map_2 = self._atom_map(residue_2)
        pattern_matches, _ = self._pattern_hbond_matches(residue_1.base, residue_2.base, atom_map_1, atom_map_2)
        generic_matches = self._generic_hbond_matches(residue_1, residue_2, atom_map_1, atom_map_2)
        if not pattern_matches and not generic_matches:
            return False

        stacking_score = self._stacking_register_score(
            primary_subunit,
            partner_subunit,
            left_primary,
            right_primary,
            left_partner,
            right_partner,
        )
        return stacking_score <= OPPOSING_GAP_MISMATCH_STACKING_CUTOFF

    def _stacking_register_score(
        self,
        primary_subunit: int,
        partner_subunit: int,
        left_primary: int,
        right_primary: int,
        left_partner: int,
        right_partner: int,
    ) -> float:
        primary = self.residues[primary_subunit]
        partner = self.residues[partner_subunit]
        left_primary_residue = self.residues[left_primary]
        right_primary_residue = self.residues[right_primary]
        left_partner_residue = self.residues[left_partner]
        right_partner_residue = self.residues[right_partner]

        pair_center = 0.5 * (primary.center + partner.center)
        left_pair_center = 0.5 * (left_primary_residue.center + left_partner_residue.center)
        right_pair_center = 0.5 * (right_primary_residue.center + right_partner_residue.center)

        pair_line_distance, pair_outside = self._point_segment_distance(
            pair_center,
            left_pair_center,
            right_pair_center,
        )
        primary_line_distance, primary_outside = self._point_segment_distance(
            primary.center,
            left_primary_residue.center,
            right_primary_residue.center,
        )
        partner_line_distance, partner_outside = self._point_segment_distance(
            partner.center,
            left_partner_residue.center,
            right_partner_residue.center,
        )

        primary_normal_penalty = self._stacking_normal_penalty(
            primary,
            left_primary_residue,
            right_primary_residue,
        )
        partner_normal_penalty = self._stacking_normal_penalty(
            partner,
            left_partner_residue,
            right_partner_residue,
        )

        return float(
            pair_line_distance
            + 0.35 * (primary_line_distance + partner_line_distance)
            + 3.0 * (pair_outside + primary_outside + partner_outside)
            + 1.5 * (primary_normal_penalty + partner_normal_penalty)
        )

    def _stacking_normal_penalty(
        self,
        residue: ResidueNode,
        left_residue: ResidueNode,
        right_residue: ResidueNode,
    ) -> float:
        normal = self._base_normal(residue)
        left_normal = self._base_normal(left_residue)
        right_normal = self._base_normal(right_residue)
        best_alignment = max(
            abs(float(np.dot(normal, left_normal))),
            abs(float(np.dot(normal, right_normal))),
        )
        return float(max(0.0, 1.0 - best_alignment))

    @staticmethod
    def _point_segment_distance(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> Tuple[float, float]:
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1e-8:
            return float(np.linalg.norm(point - start)), 0.0
        t = float(np.dot(point - start, segment) / denominator)
        closest = start + np.clip(t, 0.0, 1.0) * segment
        outside = max(0.0, -t, t - 1.0)
        return float(np.linalg.norm(point - closest)), outside

    def _has_relaxed_template_contact(self, residue_1: ResidueNode, residue_2: ResidueNode) -> bool:
        atom_map_1 = self._atom_map(residue_1)
        atom_map_2 = self._atom_map(residue_2)
        patterns = [
            BASE_PAIR_HBONDS.get((residue_1.base, residue_2.base), ()),
            BASE_PAIR_HBONDS.get((residue_1.base, residue_2.base, "hoogsteen"), ()),
        ]
        reversed_patterns = [
            tuple((right, left) for left, right in BASE_PAIR_HBONDS.get((residue_2.base, residue_1.base), ())),
            tuple((right, left) for left, right in BASE_PAIR_HBONDS.get((residue_2.base, residue_1.base, "hoogsteen"), ())),
        ]
        for pattern in patterns + reversed_patterns:
            if self._matches_for_pattern(pattern, atom_map_1, atom_map_2, cutoff=REGISTER_GAP_RELAXED_HBOND_CUTOFF):
                return True
        return False

    def _extend_terminal_register_pairs(
        self,
        row_1: List[int],
        row_2: List[int],
        primary_strand: Sequence[int],
        partner_strand: Sequence[int],
        primary_position: Dict[int, int],
        partner_position: Dict[int, int],
        partner_direction: int,
    ) -> None:
        """Add plausible paired terminal residues omitted only because annotation stopped early."""
        while row_1 and row_2 and row_1[0] > 0 and row_2[0] > 0:
            next_primary_pos = primary_position[row_1[0]] - 1
            next_partner_pos = partner_position[row_2[0]] - partner_direction
            if not self._terminal_register_pair_is_plausible(
                primary_strand,
                partner_strand,
                next_primary_pos,
                next_partner_pos,
            ):
                break
            row_1.insert(0, primary_strand[next_primary_pos])
            row_2.insert(0, partner_strand[next_partner_pos])

        while row_1 and row_2 and row_1[-1] > 0 and row_2[-1] > 0:
            next_primary_pos = primary_position[row_1[-1]] + 1
            next_partner_pos = partner_position[row_2[-1]] + partner_direction
            if not self._terminal_register_pair_is_plausible(
                primary_strand,
                partner_strand,
                next_primary_pos,
                next_partner_pos,
            ):
                break
            row_1.append(primary_strand[next_primary_pos])
            row_2.append(partner_strand[next_partner_pos])

    def _terminal_register_pair_is_plausible(
        self,
        primary_strand: Sequence[int],
        partner_strand: Sequence[int],
        primary_pos: int,
        partner_pos: int,
    ) -> bool:
        if primary_pos < 0 or partner_pos < 0:
            return False
        if primary_pos >= len(primary_strand) or partner_pos >= len(partner_strand):
            return False
        primary_subunit = primary_strand[primary_pos]
        partner_subunit = partner_strand[partner_pos]
        return self._terminal_subunit_pair_is_plausible(primary_subunit, partner_subunit)

    def _terminal_subunit_pair_is_plausible(self, primary_subunit: int, partner_subunit: int) -> bool:
        return self._register_gap_pair_is_plausible(primary_subunit, partner_subunit)

    def _stacking_gap_score(self, subunits: Sequence[int], left_anchor: int, right_anchor: int) -> float:
        if not subunits:
            return float("inf")
        left = self.residues[left_anchor].center
        right = self.residues[right_anchor].center
        return float(np.mean([
            np.linalg.norm(self.residues[subunit].center - left)
            + np.linalg.norm(self.residues[subunit].center - right)
            for subunit in subunits
        ]))

    def _single_strand_topology(self, strand_idx: int) -> InferredTopology:
        selected = self.strands[strand_idx]
        row = np.array(selected, dtype=int)
        return InferredTopology(
            pdbfile=self.pdbfile,
            output_prefix=Path(self.pdbfile).stem,
            strands=[selected],
            nu_raw=[len(selected)],
            ni_map=row.reshape(1, -1),
            pair_edges=[],
            chain_ids=[self.residues[selected[0]].chain],
            comb=False,
            grv=False,
        )

    def _build_pairing_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for idx in range(len(self.strands)):
            graph.add_node(idx)

        candidate_edges = []
        for i in range(len(self.strands)):
            for j in range(i + 1, len(self.strands)):
                pairs = self._candidate_pairs(self.strands[i], self.strands[j])
                weight = len(pairs)
                if weight >= 2:
                    candidate_edges.append((i, j, weight, pairs))

        best_partner = {}
        for i, j, w, _ in candidate_edges:
            if i not in best_partner or w > best_partner[i][1]:
                best_partner[i] = (j, w)
            if j not in best_partner or w > best_partner[j][1]:
                best_partner[j] = (i, w)

        final_pair_edges = []
        for i, j, w, pairs in candidate_edges:
            shorter_len = min(len(self.strands[i]), len(self.strands[j]))
            density = w / shorter_len
            
            is_primary = (best_partner.get(i, (None,0))[0] == j or 
                          best_partner.get(j, (None,0))[0] == i)

            if (density > 0.45) or (is_primary and w >= 4):
                graph.add_edge(i, j, pairs=pairs, weight=w)
                final_pair_edges.extend((a, b) for a, b, _ in pairs)
        
        self.pair_edges = final_pair_edges
        return graph

    def _partition_complexes(self, pairing_graph: nx.Graph) -> None:
        if pairing_graph.number_of_nodes() == 0:
            self.complexes = []
            return

        paired_components = [sorted(c) for c in nx.connected_components(pairing_graph) if len(c) > 1]
        if paired_components:
            self.complexes = paired_components
        else:
            self.complexes = [[i] for i in range(len(self.strands))]

    def _generate_context_data(self, complex_strands: Sequence[int]) -> InferredTopology:
        selected = [self.strands[i] for i in complex_strands]
        if len(selected) == 1:
            row = np.array(selected[0], dtype=int)
            return InferredTopology(
                pdbfile=self.pdbfile,
                output_prefix=Path(self.pdbfile).stem,
                strands=selected,
                nu_raw=[len(selected[0])],
                ni_map=row.reshape(1, -1),
                pair_edges=[],
                chain_ids=[self.residues[selected[0][0]].chain],
                comb=False,
                grv=False,
            )

        ref = selected[0]
        rows = [ref]
        nu_raw = [len(ref)]

        for strand in selected[1:]:
            forward_aligned, forward_score = self._align_to_reference(ref, strand)
            reverse_aligned, reverse_score = self._align_to_reference(ref, list(reversed(strand)))
            if forward_score <= reverse_score:
                rows.append(forward_aligned)
                nu_raw.append(len(strand))
            else:
                rows.append(reverse_aligned)
                nu_raw.append(-len(strand))

        if len(rows) == 2:
            rows, nu_raw = self._trim_to_paired_core(rows, nu_raw)

        merged_rows = []
        merged_nu_raw = []
        merged_chains = []
        for r, row in enumerate(rows):
            merged = False
            for mr_idx, m_row in enumerate(merged_rows):
                padded_len = max(len(row), len(m_row))
                row_padded = np.pad(row, (0, padded_len - len(row)))
                m_row_padded = np.pad(m_row, (0, padded_len - len(m_row)))
                if not np.any((row_padded > 0) & (m_row_padded > 0)):
                    merged_rows[mr_idx] = np.where(row_padded > 0, row_padded, m_row_padded).tolist()
                    # Keep the nu_raw sign of the first one, adjust magnitude to max length
                    sgn = 1 if merged_nu_raw[mr_idx] > 0 else -1
                    merged_nu_raw[mr_idx] = sgn * len(merged_rows[mr_idx])
                    merged = True
                    break
            if not merged:
                merged_rows.append(row)
                merged_nu_raw.append(nu_raw[r])
                merged_chains.append(self.residues[selected[r][0]].chain)
                
        rows = merged_rows
        nu_raw = merged_nu_raw
        
        nux = max(len(row) for row in rows)
        ni_map = np.zeros((len(rows), nux), dtype=int)
        for r, row in enumerate(rows):
            ni_map[r, :len(row)] = row

        paired_count = sum(1 for col in range(nux) if np.count_nonzero(ni_map[:, col]) >= 2)

        return InferredTopology(
            pdbfile=self.pdbfile,
            output_prefix=Path(self.pdbfile).stem,
            strands=[[s for s in row if s > 0] for row in rows],
            nu_raw=nu_raw,
            ni_map=ni_map,
            pair_edges=self.pair_edges,
            chain_ids=merged_chains,
            comb=True,
            fit=True,
            grv=paired_count >= 4,
        )

    def _trim_to_paired_core(self, rows: List[List[int]], nu_raw: List[int]) -> Tuple[List[List[int]], List[int]]:
        good = []
        for a, b in zip(rows[0], rows[1]):
            if a == 0 or b == 0:
                good.append(False)
                continue
            ra = self.residues[a]
            rb = self.residues[b]
            distance = np.linalg.norm(ra.hbond_center - rb.hbond_center)
            good.append(distance <= 8.0 and self._is_complementary(ra.base, rb.base))

        if not any(good):
            return rows, nu_raw

        start = next(i for i, ok in enumerate(good) if ok)
        end = len(good) - 1 - next(i for i, ok in enumerate(reversed(good)) if ok)
        trimmed = [row[start:end + 1] for row in rows]
        nu_raw = [len(trimmed[0]), -sum(1 for v in trimmed[1] if v)]
        return trimmed, nu_raw

    def _candidate_pairs(self, first: Sequence[int], second: Sequence[int]) -> List[Tuple[int, int, float]]:
        pairs = []
        for a in first:
            best = None
            for b in second:
                score = self._pair_geometry_score(a, b)
                if best is None or score < best[2]:
                    best = (a, b, score)
            if best is not None and best[2] < 15.0:
                pairs.append(best)
        return pairs

    def _pair_geometry_score(self, first: int, second: int) -> float:
        residue_1 = self.residues[first]
        residue_2 = self.residues[second]
        distance = float(np.linalg.norm(residue_1.hbond_center - residue_2.hbond_center))
        if distance > 12.0:
            return float("inf")
        normal = self._base_normal(residue_1)
        plane_dist = abs(float(np.dot(residue_2.hbond_center - residue_1.hbond_center, normal)))
        plane_penalty = 10.0 if plane_dist > 2.0 else 0.0
        complement_bonus = -2.0 if self._is_complementary(residue_1.base, residue_2.base) else 0.0
        return distance + complement_bonus + plane_penalty

    def _orientation_score(self, ref: Sequence[int], strand: Sequence[int]) -> bool:
        forward = self._alignment_distance(ref, strand)
        reverse = self._alignment_distance(ref, list(reversed(strand)))
        return forward <= reverse

    def _alignment_distance(self, ref: Sequence[int], strand: Sequence[int]) -> float:
        n = min(len(ref), len(strand))
        if n == 0:
            return float("inf")
        return float(sum(np.linalg.norm(self.residues[ref[i]].hbond_center - self.residues[strand[i]].hbond_center) for i in range(n)) / n)

    def _align_to_reference(self, ref: Sequence[int], strand: Sequence[int]) -> Tuple[List[int], float]:
        n, m = len(ref), len(strand)
        dp = np.full((n + 1, m + 1), np.inf)
        move = np.zeros((n + 1, m + 1), dtype=np.int8)
        dp[0, 0] = 0.0
        gap = 5.0
        for i in range(1, n + 1):
            dp[i, 0] = i * gap
            move[i, 0] = 1
        for j in range(1, m + 1):
            dp[0, j] = j * gap
            move[0, j] = 2

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                r_a = ref[i - 1]
                r_b = strand[j - 1]
                ra = self.residues[r_a]
                rb = self.residues[r_b]
                
                candidate = self._score_base_pair(r_a, r_b, -1, -1)
                
                if candidate is not None:
                    cost = candidate.mean_distance
                    if not self._is_complementary(ra.base, rb.base):
                        cost += 2.0
                else:
                    cost = float(np.linalg.norm(ra.hbond_center - rb.hbond_center))
                    if self._is_complementary(ra.base, rb.base):
                        cost -= 2.0
                    else:
                        cost += 2.0

                choices = (
                    dp[i - 1, j - 1] + cost,
                    dp[i - 1, j] + gap,
                    dp[i, j - 1] + gap,
                )
                best = int(np.argmin(choices))
                dp[i, j] = choices[best]
                move[i, j] = best

        aligned = []
        i, j = n, m
        while i > 0 or j > 0:
            step = move[i, j]
            if step == 0:
                aligned.append(strand[j - 1])
                i -= 1
                j -= 1
            elif step == 1:
                aligned.append(0)
                i -= 1
            else:
                # Extra residue in this strand relative to the reference.
                j -= 1
        aligned.reverse()

        if len(aligned) < n:
            aligned.extend([0] * (n - len(aligned)))
        return aligned[:n], float(dp[n, m])

    @staticmethod
    def _base_symbol(res_name: str) -> str:
        name = parent_base_name(res_name)
        if name == "unknown":
            return "X"
        if len(name) >= 2 and name[0] in {"D", "R"} and name[1] in DNA_BASES:
            return name[1]
        for char in name:
            if char in DNA_BASES:
                return char
        return name[:1]

    @staticmethod
    def _is_complementary(first: str, second: str) -> bool:
        return frozenset((first, second)) in PAIR_TYPES
