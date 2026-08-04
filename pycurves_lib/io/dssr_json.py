from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_NT_ID_RE = re.compile(
    r"^(?:(?P<model>[^:]+):)?(?P<chain>.*)\."
    r"(?P<residue_name>.+?)(?P<residue_number>-?\d+)"
    r"(?:\^(?P<insertion_code>.+))?$"
)


class DSSRJSONError(ValueError):
    """Raised when a DSSR JSON document cannot be normalized safely."""


@dataclass(frozen=True)
class DSSRResidueKey:
    chain: str
    residue_number: int
    insertion_code: str = ""
    residue_name: str = ""
    model: str = ""

    def label(self) -> str:
        insertion = f"^{self.insertion_code}" if self.insertion_code else ""
        model = f"{self.model}:" if self.model else ""
        return f"{model}{self.chain}.{self.residue_name}{self.residue_number}{insertion}"


@dataclass(frozen=True)
class DSSRPair:
    index: int
    nt1: str
    nt2: str
    bp: str = ""
    name: str = ""
    saenger: str = ""
    lw: str = ""
    dssr: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_mapping(cls, value: Dict[str, Any], *, context: str) -> "DSSRPair":
        if not isinstance(value, dict):
            raise DSSRJSONError(f"{context} must contain JSON objects, not {type(value).__name__}.")
        nt1 = str(value.get("nt1") or "").strip()
        nt2 = str(value.get("nt2") or "").strip()
        if not nt1 or not nt2:
            raise DSSRJSONError(f"{context} pair is missing nt1 or nt2.")
        try:
            index = int(value.get("index", 0))
        except (TypeError, ValueError) as exc:
            raise DSSRJSONError(f"{context} pair index must be an integer.") from exc
        return cls(
            index=index,
            nt1=nt1,
            nt2=nt2,
            bp=str(value.get("bp") or "").strip(),
            name=str(value.get("name") or "").strip(),
            saenger=str(value.get("Saenger") or "").strip(),
            lw=str(value.get("LW") or "").strip(),
            dssr=str(value.get("DSSR") or "").strip(),
            raw=dict(value),
        )

    @property
    def identity(self) -> frozenset[str]:
        return frozenset((self.nt1, self.nt2))


@dataclass(frozen=True)
class DSSRResidue:
    nt_id: str
    key: DSSRResidueKey
    nt_code: str = ""
    previous_nt: str = ""
    next_nt: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_mapping(cls, value: Dict[str, Any], *, context: str) -> "DSSRResidue":
        if not isinstance(value, dict):
            raise DSSRJSONError(f"{context} must contain JSON objects, not {type(value).__name__}.")
        nt_id = str(value.get("nt_id") or "").strip()
        if not nt_id:
            raise DSSRJSONError(f"{context} nucleotide is missing nt_id.")

        parsed = parse_dssr_nt_id(nt_id)
        chain = str(value.get("chain_name") or parsed.chain).strip()
        residue_name = str(value.get("nt_name") or parsed.residue_name).strip()
        try:
            residue_number = int(value.get("nt_resnum", parsed.residue_number))
        except (TypeError, ValueError) as exc:
            raise DSSRJSONError(f"{context} nucleotide {nt_id!r} has an invalid nt_resnum.") from exc

        linked = value.get("linked_nts") or []
        if not isinstance(linked, list):
            linked = []
        previous_nt = str(linked[0] or "").strip() if len(linked) >= 1 else ""
        next_nt = str(linked[1] or "").strip() if len(linked) >= 2 else ""
        return cls(
            nt_id=nt_id,
            key=DSSRResidueKey(
                chain=chain,
                residue_number=residue_number,
                insertion_code=parsed.insertion_code,
                residue_name=residue_name,
                model=parsed.model,
            ),
            nt_code=str(value.get("nt_code") or "").strip(),
            previous_nt=previous_nt,
            next_nt=next_nt,
            raw=dict(value),
        )


@dataclass(frozen=True)
class DSSRUnit:
    kind: str
    index: int
    pairs: Tuple[DSSRPair, ...]
    helix_index: Optional[int] = None
    strand1: str = ""
    strand2: str = ""
    bp_type: str = ""
    helix_form: str = ""
    num_stems: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_mapping(cls, kind: str, value: Dict[str, Any], *, context: str) -> "DSSRUnit":
        if not isinstance(value, dict):
            raise DSSRJSONError(f"{context} must contain JSON objects, not {type(value).__name__}.")
        try:
            index = int(value.get("index", 0))
        except (TypeError, ValueError) as exc:
            raise DSSRJSONError(f"{context} index must be an integer.") from exc
        raw_pairs = value.get("pairs") or []
        if not isinstance(raw_pairs, list):
            raise DSSRJSONError(f"{context} pairs must be a JSON list.")
        pairs = tuple(
            DSSRPair.from_mapping(pair, context=f"{context} {index}")
            for pair in raw_pairs
        )
        helix_index = value.get("helix_index")
        num_stems = value.get("num_stems")
        return cls(
            kind=kind,
            index=index,
            pairs=pairs,
            helix_index=int(helix_index) if helix_index is not None else None,
            strand1=str(value.get("strand1") or ""),
            strand2=str(value.get("strand2") or ""),
            bp_type=str(value.get("bp_type") or ""),
            helix_form=str(value.get("helix_form") or ""),
            num_stems=int(num_stems) if num_stems is not None else None,
            raw=dict(value),
        )

    @property
    def selector(self) -> str:
        return f"{self.kind}:{self.index}"


@dataclass
class DSSRDocument:
    path: str
    program: str
    pairs: Tuple[DSSRPair, ...]
    residues: Tuple[DSSRResidue, ...]
    stems: Tuple[DSSRUnit, ...]
    helices: Tuple[DSSRUnit, ...]
    metadata: Dict[str, Any]
    raw_keys: Tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "DSSRDocument":
        source = Path(path)
        try:
            payload = source.read_bytes()
        except OSError as exc:
            raise DSSRJSONError(f"Could not read DSSR JSON file {str(source)!r}: {exc}") from exc

        # A few Windows redirection workflows leave a BOM or trailing NULs.
        payload = payload.lstrip(b"\xef\xbb\xbf").rstrip(b"\x00 \t\r\n")
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DSSRJSONError(f"Invalid DSSR JSON file {str(source)!r}: {exc}") from exc
        if not isinstance(data, dict):
            raise DSSRJSONError("A DSSR JSON document must have a JSON object at its root.")

        raw_pairs = data.get("pairs") or []
        raw_residues = data.get("nts") or []
        raw_stems = data.get("stems") or []
        raw_helices = data.get("helices") or []
        for key, value in (
            ("pairs", raw_pairs),
            ("nts", raw_residues),
            ("stems", raw_stems),
            ("helices", raw_helices),
        ):
            if not isinstance(value, list):
                raise DSSRJSONError(f"DSSR root field {key!r} must be a JSON list when present.")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise DSSRJSONError("DSSR root field 'metadata' must be a JSON object when present.")
        program = str(data.get("program") or metadata.get("version") or "").strip()

        document = cls(
            path=str(source),
            program=program,
            pairs=tuple(
                DSSRPair.from_mapping(pair, context="DSSR root pairs")
                for pair in raw_pairs
            ),
            residues=tuple(
                DSSRResidue.from_mapping(residue, context="DSSR nts")
                for residue in raw_residues
            ),
            stems=tuple(
                DSSRUnit.from_mapping("stem", unit, context="DSSR stems")
                for unit in raw_stems
            ),
            helices=tuple(
                DSSRUnit.from_mapping("helix", unit, context="DSSR helices")
                for unit in raw_helices
            ),
            metadata=dict(metadata),
            raw_keys=tuple(sorted(str(key) for key in data)),
        )
        document._validate_unique_residue_ids()
        return document

    @property
    def kind(self) -> str:
        return "full" if (self.residues or self.stems or self.helices or self.metadata) else "pair_only"

    @property
    def residues_by_id(self) -> Dict[str, DSSRResidue]:
        return {residue.nt_id: residue for residue in self.residues}

    @property
    def version(self) -> str:
        return str(self.metadata.get("version") or self.program or "").strip()

    def units(self, kind: Optional[str] = None) -> Tuple[DSSRUnit, ...]:
        if kind is None:
            return self.stems + self.helices
        normalized = str(kind).strip().lower().rstrip("s")
        if normalized == "stem":
            return self.stems
        if normalized == "helix":
            return self.helices
        raise DSSRJSONError(f"Unknown DSSR unit kind {kind!r}; use 'stem' or 'helix'.")

    def residue_key(self, nt_id: str) -> DSSRResidueKey:
        residue = self.residues_by_id.get(nt_id)
        return residue.key if residue is not None else parse_dssr_nt_id(nt_id)

    def _validate_unique_residue_ids(self) -> None:
        seen = set()
        duplicates = []
        for residue in self.residues:
            if residue.nt_id in seen:
                duplicates.append(residue.nt_id)
            seen.add(residue.nt_id)
        if duplicates:
            values = ", ".join(sorted(set(duplicates))[:5])
            raise DSSRJSONError(f"DSSR nts contains duplicate nt_id values: {values}.")


def parse_dssr_nt_id(value: str) -> DSSRResidueKey:
    """Parse the default DSSR model:chain.residue identifier form.

    Full DSSR reports remain authoritative because their ``nts`` records carry
    explicit chain/name/number fields. This parser is also used for pair-only
    reports, including common modified-residue IDs such as ``B.OMC6``.
    """

    text = str(value or "").strip()
    match = _NT_ID_RE.fullmatch(text)
    if match is None:
        raise DSSRJSONError(
            f"Cannot parse DSSR nucleotide identifier {text!r}; use a full DSSR JSON "
            "report with an nts table for nonstandard identifiers."
        )
    return DSSRResidueKey(
        chain=match.group("chain"),
        residue_number=int(match.group("residue_number")),
        insertion_code=str(match.group("insertion_code") or "").strip(),
        residue_name=str(match.group("residue_name") or "").strip(),
        model=str(match.group("model") or "").strip(),
    )


def matched_pair_identities(pairs: Iterable[DSSRPair]) -> set[frozenset[str]]:
    return {pair.identity for pair in pairs}
