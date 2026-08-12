"""Trajectory entry point with static DSSR topology support."""

from __future__ import annotations

from typing import Dict, Optional

from pycurves_lib.curves_wrapper import CurvesWrapper
from pycurves_lib.md import trajectory_cli as _core


make_frame_selector = _core.make_frame_selector
collect_available_frame_indices = _core.collect_available_frame_indices
TrajectoryLoader = _core.TrajectoryLoader


class MDTrajectoryAnalyzer(_core.MDTrajectoryAnalyzer):
    """Reuse one DSSR-selected reference topology over all trajectory frames."""

    def __init__(
        self,
        topology_file: str,
        trajectory_file: Optional[str] = None,
        inpfile: Optional[str] = None,
        output_dir: str = ".",
        frame_convention: str = "standard",
        axis_convention: str = "legacy",
        axis_weighting: Optional[bool] = None,
        continuous_strands: bool = False,
        altloc: Optional[str] = None,
        fit_override: Optional[bool] = None,
        grv_override: Optional[bool] = None,
        mini_override: Optional[bool] = None,
        comb_override: Optional[bool] = None,
        ends_override: Optional[bool] = None,
        topology_mode: str = "reference",
        dssr_json: Optional[str] = None,
        dssr_unit: Optional[str] = None,
    ):
        if dssr_json and inpfile:
            raise ValueError("Use either --inp or --dssr-json for trajectory topology, not both.")
        if not dssr_json:
            super().__init__(
                topology_file=topology_file,
                trajectory_file=trajectory_file,
                inpfile=inpfile,
                output_dir=output_dir,
                frame_convention=frame_convention,
                axis_convention=axis_convention,
                axis_weighting=axis_weighting,
                continuous_strands=continuous_strands,
                altloc=altloc,
                fit_override=fit_override,
                grv_override=grv_override,
                mini_override=mini_override,
                comb_override=comb_override,
                ends_override=ends_override,
                topology_mode=topology_mode,
            )
            self.dssr_json = None
            self.dssr_unit = None
            self.dssr_provenance = None
            return

        reference_topology = self._reference_topology(
            topology_file, trajectory_file, output_dir
        )
        dssr_runner = CurvesWrapper(
            pdbfile=reference_topology,
            output_dir=output_dir,
            continuous_strands=continuous_strands,
            altloc=altloc,
            frame_convention=frame_convention,
            axis_convention=axis_convention,
            axis_weighting=axis_weighting,
            fit_override=fit_override,
            grv_override=grv_override,
            mini_override=mini_override,
            comb_override=comb_override,
            ends_override=ends_override,
            dssr_json=dssr_json,
            dssr_unit=dssr_unit,
        )
        super().__init__(
            topology_file=topology_file,
            trajectory_file=trajectory_file,
            inpfile=dssr_runner.inpfile,
            output_dir=output_dir,
            frame_convention=frame_convention,
            axis_convention=axis_convention,
            axis_weighting=axis_weighting,
            continuous_strands=False,
            altloc=altloc,
            fit_override=fit_override,
            grv_override=grv_override,
            mini_override=mini_override,
            comb_override=comb_override,
            ends_override=ends_override,
            topology_mode=topology_mode,
        )
        self.reference_runner = dssr_runner
        self.inpfile = dssr_runner.inpfile
        self.generated_inpfiles = list(dssr_runner.generated_inpfiles)
        dssr_runner.attach_dssr_annotations(self.template_molecule)
        self.dssr_json = str(dssr_json)
        self.dssr_unit = dssr_runner.dssr_unit
        self.dssr_provenance = dssr_runner.dssr_provenance

    def run(self, *args, **kwargs):
        payload = super().run(*args, **kwargs)
        if self.dssr_json:
            payload["inputs"]["dssr_json"] = self.dssr_json
            payload["topology_provenance"] = self.dssr_provenance
        return payload


def analyze_trajectory(
    topology_file: str,
    trajectory_file: Optional[str] = None,
    inpfile: Optional[str] = None,
    output_dir: str = ".",
    frames: Optional[str] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: int = 1,
    mode: str = "per-frame",
    frame_convention: str = "standard",
    axis_convention: str = "legacy",
    axis_weighting: Optional[bool] = None,
    continuous_strands: bool = False,
    altloc: Optional[str] = None,
    fit: Optional[bool] = None,
    grooves: Optional[bool] = None,
    mini: Optional[bool] = None,
    comb: Optional[bool] = None,
    ends: Optional[bool] = None,
    topology_mode: str = "reference",
    verbose: bool = False,
    warm_start: bool = True,
    axis_continuity: bool = True,
    dssr_json: Optional[str] = None,
    dssr_unit: Optional[str] = None,
) -> Dict:
    if mode not in {"per-frame", "summary", "both"}:
        raise ValueError("mode must be one of: per-frame, summary, both")
    if step <= 0:
        raise ValueError("step must be positive")
    frame_selector, selection = make_frame_selector(frames, start, stop, step)
    analyzer = MDTrajectoryAnalyzer(
        topology_file=topology_file,
        trajectory_file=trajectory_file,
        inpfile=inpfile,
        output_dir=output_dir,
        frame_convention=frame_convention,
        axis_convention=axis_convention,
        axis_weighting=axis_weighting,
        continuous_strands=continuous_strands,
        altloc=altloc,
        fit_override=fit,
        grv_override=grooves,
        mini_override=mini,
        comb_override=comb,
        ends_override=ends,
        topology_mode=topology_mode,
        dssr_json=dssr_json,
        dssr_unit=dssr_unit,
    )
    return analyzer.run(
        frame_selector=frame_selector,
        selection=selection,
        mode=mode,
        mini=mini,
        verbose=verbose,
        warm_start=warm_start,
        axis_continuity=axis_continuity,
    )


# The original CLI and helpers resolve this global at call time.
_core.MDTrajectoryAnalyzer = MDTrajectoryAnalyzer


def main() -> None:
    return _core.main()


if __name__ == "__main__":
    main()


__all__ = [
    "MDTrajectoryAnalyzer",
    "TrajectoryLoader",
    "analyze_trajectory",
    "collect_available_frame_indices",
    "make_frame_selector",
    "main",
]
