from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


CANONICAL_BASE_PARENT = {
    "A": "A", "DA": "A", "ADE": "A",
    "G": "G", "DG": "G", "GUA": "G",
    "C": "C", "DC": "C", "CYT": "C",
    "T": "T", "DT": "T", "THY": "T",
    "U": "U", "DU": "U", "URA": "U",
}

SPECIAL_BASES = {"I", "Y", "R", "P"}

FALLBACK_MODIFIED_BASE_PARENT = {
    "5IU": "U", "5CM": "C", "5MC": "C", "1MA": "A", "2MG": "G",
    "M2G": "G", "PSU": "U", "5HU": "U", "OMC": "C", "OMG": "G",
    "A2M": "A", "G2M": "G", "YG": "G", "1MG": "G", "7MG": "G",
    "5MU": "T", "5HB": "U", "BRU": "U", "5BU": "U", "6MA": "A",
    "CBR": "C",
}


def _mapping_path() -> Path:
    package_path = Path(__file__).with_name("modified_to_change_data.json")
    if package_path.exists():
        return package_path
    # Fallback for old source trees before the data file was packaged.
    return Path(__file__).resolve().parents[2] / "modified_to_change_data.json"


def _standard_to_parent(standard_base) -> Optional[str]:
    if isinstance(standard_base, list):
        if not standard_base:
            return None
        standard_base = standard_base[0]
    if not standard_base:
        return None
    standard = str(standard_base).strip().upper()
    return CANONICAL_BASE_PARENT.get(standard)


@lru_cache(maxsize=1)
def nakb_modified_base_parent() -> Dict[str, str]:
    """Return residue-code to parent-base mappings from NAKB, if available."""
    path = _mapping_path()
    if not path.exists():
        return {}

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    mapping: Dict[str, str] = {}
    for residue_name, row in data.items():
        if not isinstance(row, dict):
            continue
        parent = _standard_to_parent(row.get("standard_base"))
        if parent is not None:
            mapping[str(residue_name).strip().upper()] = parent
    return mapping


@lru_cache(maxsize=1)
def modified_base_parent() -> Dict[str, str]:
    """Return the complete modified residue parent map used by pyCurves."""
    mapping = dict(FALLBACK_MODIFIED_BASE_PARENT)
    mapping.update(nakb_modified_base_parent())
    return mapping


def parent_base_name(residue_name: str) -> str:
    """Return the canonical parent base used for classification and fitting."""
    name = str(residue_name).strip().upper()
    if name in CANONICAL_BASE_PARENT:
        return CANONICAL_BASE_PARENT[name]
    if name in SPECIAL_BASES:
        return name
    parent = modified_base_parent().get(name)
    if parent is not None:
        return parent
    if len(name) == 2 and name.startswith(("D", "R")) and name[1:] in CANONICAL_BASE_PARENT:
        return CANONICAL_BASE_PARENT[name[1:]]
    return name if len(name) == 1 and name in {"A", "C", "G", "T", "U", "I"} else "unknown"


def is_modified_base(residue_name: str) -> bool:
    """True when the residue is not one of the standard DNA/RNA residue names."""
    name = str(residue_name).strip().upper()
    return name not in CANONICAL_BASE_PARENT


def is_known_modified_base(residue_name: str) -> bool:
    """True when NAKB/fallback maps this residue to a supported parent base."""
    return str(residue_name).strip().upper() in modified_base_parent()
