"""Backward-compatible import for the public linear DSSR topology builder."""

from pycurves_lib.topology.dssr_topology import (
    DSSRBuildResult,
    DSSRSelectionError,
    DSSRTopologyBuilder,
    DSSRTopologyError,
)


__all__ = [
    "DSSRBuildResult",
    "DSSRSelectionError",
    "DSSRTopologyBuilder",
    "DSSRTopologyError",
]
