from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from pycurves_lib.core.curves_dataclasses import MolecularStructure
from pycurves_lib.io.curves_mol_loader import MolecularLoader
from pycurves_lib.io.curves_output import CurvesOutputFormatter, _to_jsonable
from pycurves_lib.curves_wrapper import CurvesWrapper
from pycurves_lib.cli.pycurves_cli_options import (
    add_pycurves_analysis_options,
    pycurves_runner_kwargs,
    resolved_mini,
)
from pycurves_lib.md.trajectory_loader import TrajectoryLoader


class MDTrajectoryAnalyzer:
    """Run pyCurves over selected trajectory frames and summarize named records."""

    def __init__(
        self,
        topology_file: str,
        trajectory_file: Optional[str] = None,
        inpfile: Optional[str] = None,
        output_dir: str = ".",
        annotations: bool = True,
        frame_convention: str = "legacy",
        axis_convention: str = "legacy",
        continuous_strands: bool = False,
        fit_override: Optional[bool] = None,
        grv_override: Optional[bool] = None,
        mini_override: Optional[bool] = None,
        comb_override: Optional[bool] = None,
        ends_override: Optional[bool] = None,
    ):
        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        self.output_dir = output_dir
        self.include_annotations = annotations
        self.frame_convention, self.axis_convention = CurvesWrapper.normalize_conventions(
            frame_convention,
            axis_convention,
        )
        self.continuous_strands = continuous_strands
        self.fit_override = fit_override
        self.grv_override = grv_override
        self.mini_override = mini_override
        self.comb_override = comb_override
        self.ends_override = ends_override
        self.reference_topology_file = self._reference_topology(topology_file, trajectory_file, output_dir)
        self.template_molecule = self._load_template_molecule(self.reference_topology_file)
        self.runner_kwargs = {
            "continuous_strands": continuous_strands,
            "frame_convention": self.frame_convention,
            "axis_convention": self.axis_convention,
            "fit_override": fit_override,
            "grv_override": grv_override,
            "mini_override": mini_override,
            "comb_override": comb_override,
            "ends_override": ends_override,
        }

        self.reference_runner = CurvesWrapper(
            pdbfile=self.reference_topology_file,
            inpfile=inpfile,
            output_dir=output_dir,
            **self.runner_kwargs,
        )
        self.inpfile = self.reference_runner.inpfile
        self.generated_inpfiles = self.reference_runner.generated_inpfiles

    def run(
        self,
        frame_selector,
        selection: Dict,
        mode: str = "summary",
        mini: bool = True,
        verbose: bool = False,
        warm_start: bool = True,
        axis_continuity: bool = True,
    ) -> Dict:
        if self.axis_convention == "curvesplus":
            mini = False
            warm_start = False
        frame_payloads = []
        table_records: Dict[str, List[Dict]] = {}
        processed = 0

        prev_helical = None
        axis_sign_reference = None
        runner = self.reference_runner

        for frame in tqdm(TrajectoryLoader.iter_frames(self.topology_file, self.trajectory_file, frame_selector), desc="Processing frames"):
            molecule = self._molecule_for_frame(frame.coordinates)
            runner.analyze_molecule(
                molecule,
                mini=mini,
                verbose=verbose,
                prev_opt_helical=prev_helical if warm_start else None,
                axis_sign_reference=axis_sign_reference if axis_continuity else None,
            )

            if warm_start and mini and hasattr(runner, 'ctx') and hasattr(runner.ctx, 'params') and hasattr(runner.ctx.params, 'helical'):
                prev_helical = runner.ctx.params.helical.copy()
            axis_direction_signs = []
            if hasattr(runner, "calc") and hasattr(runner.calc, "axis_direction_sign"):
                axis_direction_signs = [int(v) for v in np.asarray(runner.calc.axis_direction_sign[:runner.ctx.nst]).tolist()]
                if axis_continuity and axis_sign_reference is None:
                    axis_sign_reference = np.asarray(axis_direction_signs, dtype=int)

            formatter = CurvesOutputFormatter(runner, annotations=self.include_annotations)
            dataframes = self._normalize_frame_dataframes(formatter._build_dataframes())

            for table_name, rows in dataframes.items():
                if isinstance(rows, list):
                    for row in rows:
                        row = dict(row)
                        row["frame"] = frame.index
                        row["time"] = frame.time
                        table_records.setdefault(table_name, []).append(row)

            if mode in {"per-frame", "both"}:
                frame_payloads.append({
                    "frame": frame.index,
                    "time": frame.time,
                    "dataframes": dataframes,
                })

            processed += 1

        if processed == 0:
            raise ValueError("No trajectory frames matched the requested frame selection.")

        payload = {
            "program": "pyCurves",
            "format": "pycurves-trajectory-slim-v1",
            "inputs": {
                "topology_file": self.topology_file,
                "reference_topology_file": self.reference_topology_file,
                "trajectory_file": self.trajectory_file,
                "inpfile": self.inpfile,
                "generated_inpfiles": self.generated_inpfiles,
            },
            "analysis_options": {
                "continuous_strands": self.continuous_strands,
                "fit": self.fit_override,
                "grooves": self.grv_override,
                "mini": False if self.axis_convention == "curvesplus" else (self.mini_override if self.mini_override is not None else mini),
                "comb": self.comb_override,
                "ends": self.ends_override,
                "axis_convention": self.axis_convention,
            },
            "selection": {
                **selection,
                "processed_frames": processed,
                "warm_start": bool(warm_start),
                "axis_continuity": bool(axis_continuity),
            },
            "frame_convention": {
                "name": self.frame_convention,
                "compatible_with": ["Curves+", "3DNA", "x3dna"] if self.frame_convention == "standard" else ["Curves 5.3"],
                "axis_convention": self.axis_convention,
            },
            "annotations_enabled": self.include_annotations,
        }

        if mode in {"per-frame", "both"}:
            payload["frames"] = frame_payloads
        if mode in {"summary", "both"}:
            payload["summary"] = self._summarize_tables(table_records)
        return payload

    def write_csv(self, payload: Dict, prefix: str) -> None:
        prefix_path = Path(prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)

        if "summary" in payload:
            for name, rows in payload["summary"].items():
                self._write_rows_csv(f"{prefix}_{name}_summary.csv", rows)

        if "frames" in payload:
            per_table: Dict[str, List[Dict]] = {}
            for frame in payload["frames"]:
                for name, rows in frame["dataframes"].items():
                    if isinstance(rows, list):
                        for row in rows:
                            row = dict(row)
                            row["frame"] = frame["frame"]
                            row["time"] = frame["time"]
                            per_table.setdefault(name, []).append(row)
            for name, rows in per_table.items():
                self._write_rows_csv(f"{prefix}_{name}_frames.csv", rows)

    @staticmethod
    def _load_template_molecule(topology_file: str) -> MolecularStructure:
        holder = type("MoleculeHolder", (), {"molecule": MolecularStructure()})()
        MolecularLoader.load(topology_file, holder)
        return holder.molecule

    @staticmethod
    def _reference_topology(topology_file: str, trajectory_file: Optional[str], output_dir: str) -> str:
        if trajectory_file is not None:
            return topology_file
        path = Path(topology_file)
        if path.suffix.lower() not in {".pdb", ".brk"}:
            return topology_file

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        ref_path = output_root / f"{path.stem}_first_model.pdb"
        if ref_path.exists():
            return str(ref_path)

        lines = []
        in_model = False
        saw_model = False
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                record = line[:6]
                if record.startswith("MODEL"):
                    saw_model = True
                    in_model = True
                    continue
                if record.startswith("ENDMDL") and in_model:
                    break
                if saw_model and not in_model:
                    continue
                if record in {"ATOM  ", "HETATM", "TITLE ", "CRYST1", "TER   "}:
                    lines.append(line)

        if saw_model and lines:
            ref_path.write_text("".join(lines) + "END\n", encoding="utf-8")
            return str(ref_path)
        return topology_file

    def _molecule_for_frame(self, coordinates: np.ndarray) -> MolecularStructure:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != self.template_molecule.coordinates.shape:
            raise ValueError(
                "Trajectory frame atom count does not match the topology molecule: "
                f"{coordinates.shape[0]} vs {self.template_molecule.coordinates.shape[0]}"
            )

        molecule = copy.copy(self.template_molecule)
        molecule.coordinates = coordinates.copy()
        return molecule

    @staticmethod
    def _normalize_frame_dataframes(dataframes: Dict) -> Dict:
        normalized = {}
        for name, rows in dataframes.items():
            if name == "groove" and isinstance(rows, dict) and "data" in rows:
                normalized[name] = MDTrajectoryAnalyzer._flatten_groove_records(rows)
            else:
                normalized[name] = rows
        return normalized

    @staticmethod
    def _flatten_groove_records(groove: Dict) -> List[Dict]:
        flat_rows: List[Dict] = []
        for level, level_data in groove.get("data", {}).items():
            for sub_level, sub_data in level_data.get("sub_levels", {}).items():
                row = {
                    "atom_defining_backbone": groove.get("atom_defining_backbone", ""),
                    "total_levels": groove.get("levels", 0),
                    "total_sub_levels": groove.get("sub_levels", 0),
                    "level": int(level),
                    "base_pair": level_data.get("base_pair", ""),
                    "sub_level": int(sub_level),
                }
                for key, value in sub_data.items():
                    if key != "geometry":
                        row[key] = value
                flat_rows.append(row)
        return flat_rows

    @staticmethod
    def _summarize_tables(tables: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        summaries = {}
        for name, rows in tables.items():
            if not rows:
                summaries[name] = []
                continue

            all_cols = list(dict.fromkeys(col for row in rows for col in row))
            identifier_cols = {
                "strand", "level", "residue_id", "subunit", "sub_level",
                "total_levels", "total_sub_levels", "partner_strand",
                "sequence_index", "next_level", "fit_atom_count",
            }
            numeric_cols = []
            for col in all_cols:
                if col in {"frame", "time"} or col in identifier_cols:
                    continue
                values = [row.get(col) for row in rows if row.get(col) is not None]
                if values and all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                    numeric_cols.append(col)
            key_cols = [col for col in all_cols if col not in numeric_cols and col not in {"frame", "time"}]
            if not numeric_cols:
                summaries[name] = []
                continue

            groups = {}
            if key_cols:
                for row in rows:
                    key = tuple(MDTrajectoryAnalyzer._summary_key_value(row.get(col)) for col in key_cols)
                    groups.setdefault(key, []).append(row)
            else:
                groups[()] = rows

            table_summary = []
            for key, group_rows in groups.items():
                out = {col: key[idx] for idx, col in enumerate(key_cols)}
                out["count"] = len(group_rows)
                for col in numeric_cols:
                    vals = np.asarray(
                        [row.get(col) for row in group_rows if row.get(col) is not None],
                        dtype=float,
                    )
                    if vals.size == 0:
                        out[f"{col}_mean"] = None
                        out[f"{col}_variance"] = None
                    else:
                        out[f"{col}_mean"] = float(np.mean(vals))
                        out[f"{col}_variance"] = float(np.var(vals))
                table_summary.append(out)
            summaries[name] = table_summary
        return summaries

    @staticmethod
    def _summary_key_value(value):
        """Return a hashable grouping key for JSON-like table identifier values."""
        if isinstance(value, (list, dict)):
            return json.dumps(_to_jsonable(value), sort_keys=True)
        if isinstance(value, np.ndarray):
            return json.dumps(_to_jsonable(value), sort_keys=True)
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _write_rows_csv(path: str, rows: List[Dict]) -> None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            if not fieldnames:
                handle.write("")
                return
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def make_frame_selector(spec: Optional[str], start: Optional[int], stop: Optional[int], step: int):
    if spec:
        exact = set()
        ranges = []
        explicit_indices = set()
        all_ranges_finite = True
        range_specs = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                values = [int(v) if v else None for v in part.split(":")]
                while len(values) < 3:
                    values.append(None)
                s, e, st = values
                st = st or 1
                range_start = 0 if s is None else s
                ranges.append((range_start, e, st))
                range_specs.append((range_start, e, st))
                if e is None:
                    all_ranges_finite = False
                else:
                    explicit_indices.update(range(range_start, e, st))
            else:
                value = int(part)
                exact.add(value)
                explicit_indices.add(value)

        def selected(index: int) -> bool:
            if index in exact:
                return True
            for s, e, st in ranges:
                if index >= s and (e is None or index < e) and (index - s) % st == 0:
                    return True
            return False

        if all_ranges_finite:
            selected.explicit_indices = sorted(explicit_indices)
        if not exact and len(range_specs) == 1:
            selected.mdtraj_range = range_specs[0]
            if hasattr(selected, "explicit_indices"):
                delattr(selected, "explicit_indices")

        return selected, {"frames": spec}

    s = 0 if start is None else start
    e = stop

    def selected(index: int) -> bool:
        return index >= s and (e is None or index < e) and (index - s) % step == 0

    if e is not None:
        selected.mdtraj_range = (s, e, step)

    return selected, {"start": s, "stop": e, "step": step}


def collect_available_frame_indices(topology_file: str, trajectory_file: Optional[str]) -> List[int]:
    return [frame.index for frame in TrajectoryLoader.iter_frames(topology_file, trajectory_file)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pyCurves over an MD trajectory.")
    parser.add_argument("topology", help="Topology/reference structure file, usually PDB or CIF.")
    parser.add_argument("trajectory", nargs="?", help="Trajectory file. Omit for multi-model PDB input.")
    parser.add_argument("--inp", help="Existing Curves .inp file. If omitted, inferred from the topology.")
    parser.add_argument("--output-dir", default=".", help="Directory for auto-generated .inp files.")
    parser.add_argument("--output-file", default="pycurves_trajectory.json", help="JSON file or CSV prefix.")
    parser.add_argument("--mode", choices=["per-frame", "summary", "both"], default="summary")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--frames", help="Frame list/ranges, e.g. '0,10,20:100:5'. Overrides start/stop/step.")
    parser.add_argument("--start", type=int, help="First frame index to include.")
    parser.add_argument("--stop", type=int, help="Stop before this frame index.")
    parser.add_argument("--step", type=int, default=1, help="Frame stride.")
    parser.add_argument("--no-annotations", action="store_true", help="Suppress pyCurves annotation records.")
    add_pycurves_analysis_options(parser)
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Do not seed each frame's optimizer from the previous frame. Slower, but useful for diagnosing axis branch flips.",
    )
    parser.add_argument(
        "--no-axis-continuity",
        action="store_true",
        help="Do not keep the Curves global-axis direction signs aligned to the first processed frame.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-frame pyCurves logs.")
    args = parser.parse_args()
    mini = resolved_mini(args, default=True)
    if args.axis_convention != "curvesplus" and not mini:
        raise SystemExit(
            "--no-mini is not supported for trajectory analysis yet; downstream axis, bend, and groove "
            "parameters require the fitted helical axis."
        )

    frame_selector, selection = make_frame_selector(args.frames, args.start, args.stop, args.step)
    analyzer = MDTrajectoryAnalyzer(
        topology_file=args.topology,
        trajectory_file=args.trajectory,
        inpfile=args.inp,
        output_dir=args.output_dir,
        annotations=not args.no_annotations,
        **pycurves_runner_kwargs(args),
    )
    try:
        payload = analyzer.run(
            frame_selector=frame_selector,
            selection=selection,
            mode=args.mode,
            mini=mini,
            verbose=args.verbose,
            warm_start=not args.no_warm_start,
            axis_continuity=not args.no_axis_continuity,
        )
    except (ValueError, ImportError, NotImplementedError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.format == "json":
        Path(args.output_file).write_text(json.dumps(_to_jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        analyzer.write_csv(payload, args.output_file.removesuffix(".csv"))

    generated = analyzer.generated_inpfiles
    if generated:
        print("Generated input file(s):", file=sys.stderr)
        for path in generated:
            print(f"  {path}", file=sys.stderr)
    print(f"Processed {payload['selection']['processed_frames']} frame(s).", file=sys.stderr)


if __name__ == "__main__":
    main()



