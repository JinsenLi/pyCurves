import json
from pathlib import Path
import sys

import numpy as np
import pytest

import pycurves
from pycurves_md import analyze_trajectory
from pycurves_lib.curves_wrapper import CurvesWrapper


ROOT = Path(__file__).resolve().parents[1]


def _generated_input(tmp_path, structure_name, *, mini, xytp=(0.0, 0.0, 0.0, 0.0)):
    structure = ROOT / "test_data" / structure_name
    generator = CurvesWrapper(pdbfile=str(structure), output_dir=str(tmp_path))
    inpfile = Path(generator.inpfile)
    text = inpfile.read_text(encoding="utf-8")
    mini_token = ".t." if mini else ".f."
    text = text.replace("&inp ", f"&inp mini={mini_token}, ", 1)
    lines = text.splitlines()
    lines[-1] = " ".join(str(value) for value in xytp)
    inpfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return structure, inpfile


def test_mini_false_builds_finite_hoogsteen_axis_from_input(tmp_path):
    expected = np.array([1.25, -2.5, 8.0, -4.0])
    structure, inpfile = _generated_input(
        tmp_path,
        "3KZ8.cif.gz",
        mini=False,
        xytp=expected,
    )

    runner = CurvesWrapper(pdbfile=str(structure), inpfile=str(inpfile)).analyze()

    assert runner.ctx.cfg.mini is False
    assert type(runner.opt).__name__ == "HelicalOptimizerJAX"
    assert "MINIMISATION" not in runner.analysis_log

    active_helical = runner.ctx.params.helical[0, 1 : runner.ctx.nux + 1, :][:, [0, 1, 3, 4]]
    np.testing.assert_allclose(active_helical, np.broadcast_to(expected, active_helical.shape), atol=1e-7)

    for values in (
        runner.ctx.params.helical,
        runner.ctx.params.ux,
        runner.ctx.params.ox,
        runner.opt.uho,
        runner.opt.hho,
        runner.calc.vkin,
        runner.calc.bend,
    ):
        assert np.isfinite(values).all()

    axis_directions = runner.opt.uho[1 : runner.ctx.nux + 1, :, 0]
    np.testing.assert_allclose(np.linalg.norm(axis_directions, axis=1), 1.0, atol=1e-7)
    assert not np.allclose(runner.opt.hho[1 : runner.ctx.nux + 1, :, 0], 0.0)

    payload = json.loads(runner.output(fmt="json", file=tmp_path / "result.json"))
    assert payload["analysis_options"]["mini"] is False


def test_explicit_mini_override_wins_over_input_file(tmp_path):
    structure, inpfile = _generated_input(tmp_path, "1A1F_b_c.pdb", mini=False)
    runner = CurvesWrapper(pdbfile=str(structure), inpfile=str(inpfile))

    assert runner._load_config()["config"].mini is False
    assert runner._load_config(mini_override=True)["config"].mini is True


def test_analyze_molecule_applies_warm_start_after_optimizer_initialization(tmp_path):
    structure, inpfile = _generated_input(tmp_path, "1A1F_b_c.pdb", mini=False)
    initial = CurvesWrapper(pdbfile=str(structure), inpfile=str(inpfile)).analyze()
    warm_helical = initial.ctx.params.helical.copy()
    expected = np.array([2.0, -1.0, 6.0, -3.0])
    warm_helical[:, 1 : initial.ctx.nux + 1, 0] = expected[0]
    warm_helical[:, 1 : initial.ctx.nux + 1, 1] = expected[1]
    warm_helical[:, 1 : initial.ctx.nux + 1, 3] = expected[2]
    warm_helical[:, 1 : initial.ctx.nux + 1, 4] = expected[3]

    runner = CurvesWrapper(pdbfile=str(structure), inpfile=str(inpfile)).analyze_molecule(
        initial.ctx.molecule,
        prev_opt_helical=warm_helical,
    )

    active_helical = runner.ctx.params.helical[0, 1 : runner.ctx.nux + 1, :][:, [0, 1, 3, 4]]
    np.testing.assert_allclose(active_helical, np.broadcast_to(expected, active_helical.shape), atol=1e-7)


def test_z_axis_mode_preserves_its_prebuilt_axis(tmp_path):
    structure, inpfile = _generated_input(tmp_path, "1A1F_b_c.pdb", mini=False)
    text = inpfile.read_text(encoding="utf-8").replace("&inp ", "&inp zaxe=.t., ", 1)
    inpfile.write_text(text, encoding="utf-8")

    runner = CurvesWrapper(pdbfile=str(structure), inpfile=str(inpfile)).analyze()

    assert runner.ctx.cfg.mini is False
    assert runner.ctx.cfg.zaxe is True
    assert np.isfinite(runner.calc.vkin).all()
    np.testing.assert_allclose(
        runner.opt.uho[1 : runner.ctx.nux + 1, 2, 0],
        runner.ctx.idr[0],
    )
    assert not np.allclose(runner.opt.hho[1 : runner.ctx.nux + 1, :, 0], 0.0)


def test_md_workflow_honors_input_file_mini_false(tmp_path):
    structure, inpfile = _generated_input(tmp_path, "1A1F_b_c.pdb", mini=False)
    trajectory = tmp_path / "single-model.pdb"
    structure_text = structure.read_text(encoding="utf-8")
    trajectory.write_text(f"MODEL        1\n{structure_text}ENDMDL\n", encoding="utf-8")

    payload = analyze_trajectory(
        topology_file=str(trajectory),
        inpfile=str(inpfile),
        output_dir=str(tmp_path),
        mode="per-frame",
    )

    assert payload["selection"]["processed_frames"] == 1
    assert payload["analysis_options"]["mini"] is False
    assert payload["selection"]["warm_start"] is False


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], None),
        (["--mini"], True),
        (["--no-mini"], False),
    ],
)
def test_cli_forwards_only_explicit_mini_override(monkeypatch, extra_args, expected):
    captured = {}

    class DummyRunner:
        generated_inpfiles = []

        def __init__(self, **kwargs):
            captured["constructor"] = kwargs["mini_override"]

        def run(self, **kwargs):
            captured["run"] = kwargs["mini"]

    monkeypatch.setattr(pycurves, "CurvesWrapper", DummyRunner)
    monkeypatch.setattr(sys, "argv", ["pycurves", "fixture.pdb", "--no-output", *extra_args])

    pycurves.main()

    assert captured == {"constructor": expected, "run": expected}
