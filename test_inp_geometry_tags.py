import numpy as np
import pytest

from pycurves_lib.io.curves_config_loader import ConfigLoader
from pycurves_lib.topology.topology_inferrer import InferredTopology


@pytest.mark.parametrize("tag", ["H", "Hoog", "Hoogsteen"])
def test_legacy_hoogsteen_tokens_are_rejected(tag):
    with pytest.raises(ValueError, match="Leontis-Westhof notation"):
        ConfigLoader._split_mapping_token(f"2[{tag}]")


def test_lw_tag_is_parsed_and_recorded():
    core, tags = ConfigLoader._split_mapping_token("2[tWH]")
    pair_geometry_markers = {}

    ConfigLoader._record_mapping_tags(
        tags,
        strand=2,
        level=1,
        mapped_unit=int(core),
        pair_geometry_markers=pair_geometry_markers,
    )

    assert pair_geometry_markers[(2, 1)]["tag"] == "tWH"
    assert pair_geometry_markers[(2, 1)]["edge_1"] == "W"
    assert pair_geometry_markers[(2, 1)]["edge_2"] == "H"


def test_inferred_input_serializes_only_explicit_lw_geometry_tags():
    topology = InferredTopology(
        pdbfile="example.pdb",
        output_prefix="example",
        strands=[[1], [2]],
        nu_raw=[1, -1],
        ni_map=np.array([[1], [2]], dtype=int),
        pair_edges=[(1, 2)],
        chain_ids=["A", "B"],
        pair_geometry_markers={(2, 1): "tWH"},
    )

    text = topology.to_inp_text()

    assert "2[tWH]" in text
    assert "[Hoog]" not in text


def test_inferred_input_does_not_invent_a_vague_pair_geometry_tag():
    topology = InferredTopology(
        pdbfile="example.pdb",
        output_prefix="example",
        strands=[[1], [2]],
        nu_raw=[1, -1],
        ni_map=np.array([[1], [2]], dtype=int),
        pair_edges=[(1, 2)],
        chain_ids=["A", "B"],
    )

    assert "[" not in topology.to_inp_text()
