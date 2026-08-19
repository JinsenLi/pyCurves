from types import SimpleNamespace

from pycurves_lib.core.parameter_conventions import _annotate_pair_normal_branches
from pycurves_lib.io.curves_output_core import CurvesOutputFormatter
from pycurves_lib.md.trajectory_cli import MDTrajectoryAnalyzer
from pycurves_lib.topology.base_annotations import render_section_m


def test_left_handed_cww_uses_helical_context_in_reports():
    row = {
        "strand_1": 1,
        "strand_2": 2,
        "level": 4,
        "pair_id": "10:20",
        "is_canonical": True,
        "pair_status": "present",
    }
    ctx = SimpleNamespace(annotations={"base_pair_annotations": [row]})
    key = (0, 1, 4)
    _annotate_pair_normal_branches(ctx, {key: -1}, {key: "left_handed_cww"}, {})

    assert row["helical_context"] == "left_handed_cww"
    formatter = CurvesOutputFormatter.__new__(CurvesOutputFormatter)
    observation = formatter._base_pair_observation_records(ctx.annotations)[0]
    assert observation["helical_context"] == "left_handed_cww"
    profile = MDTrajectoryAnalyzer._pairing_profile_rows(
        [observation], total_frames=1
    )
    assert profile[0]["helical_context"] == "left_handed_cww"

    text = render_section_m(ctx.annotations)
    assert "left-handed cWW context" in text
    assert "normal_branch" not in text
    assert "normal_sign" not in text
