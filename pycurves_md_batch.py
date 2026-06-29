from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from pycurves_lib.io.curves_output import _to_jsonable
from pycurves_lib.md.trajectory_loader import TrajectoryLoader
from pycurves_lib.md.trajectory_statistics import (
    circular_degree_summary,
    is_circular_degree_column,
)
from pycurves_md import MDTrajectoryAnalyzer, make_frame_selector


class BatchSummaryAccumulator:
    """Accumulate trajectory summary statistics without materializing frames."""

    def __init__(self) -> None:
        self._tables: Dict[str, Dict[tuple, Dict]] = {}
        self._population_tables: Dict[str, Dict[tuple, Dict]] = {}

    def ensure_table(self, table_name: str) -> None:
        self._tables.setdefault(table_name, {})

    def add_population_counts(
        self,
        table_name: str,
        metadata: Dict,
        category_metadata: Dict,
        count: int,
        total_count: int,
    ) -> None:
        if total_count <= 0:
            return
        table = self._population_tables.setdefault(table_name, {})
        row_metadata = dict(metadata)
        row_metadata.update(category_metadata)
        key = tuple(row_metadata.items())
        group = table.get(key)
        if group is None:
            group = {"metadata": row_metadata, "count": 0, "total_count": 0}
            table[key] = group
        group["count"] += int(count)
        group["total_count"] += int(total_count)

    @staticmethod
    def _new_stats(name: str) -> Dict[str, object]:
        circular = is_circular_degree_column(name)
        return {
            "valid": 0,
            "sum": 0.0,
            "sumsq": 0.0,
            "values": [] if circular else None,
            "circular": circular,
        }

    def add_values(self, table_name: str, metadata: Dict, parameter_names, values) -> None:
        self.ensure_table(table_name)
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0:
            return
        row_count = int(arr.shape[0])
        metadata = dict(metadata)
        key = tuple(metadata.items())
        group = self._tables[table_name].get(key)
        if group is None:
            group = {
                "metadata": metadata,
                "count": 0,
                "stats": {
                    name: self._new_stats(name)
                    for name in parameter_names
                },
            }
            self._tables[table_name][key] = group
        group["count"] += row_count
        for col_index, name in enumerate(parameter_names):
            column = arr[:, col_index]
            finite = np.isfinite(column)
            if not np.any(finite):
                continue
            vals = column[finite]
            stats = group["stats"][name]
            stats["valid"] += int(vals.size)
            if stats["circular"]:
                stats["values"].append(vals.astype(float, copy=True))
            else:
                stats["sum"] += float(np.sum(vals))
                stats["sumsq"] += float(np.sum(vals * vals))

    def add_rows(self, table_name: str, rows: List[Dict], numeric_names) -> None:
        self.ensure_table(table_name)
        numeric_names = tuple(numeric_names)
        numeric_set = set(numeric_names)
        for row in rows:
            metadata = {
                key: value
                for key, value in row.items()
                if key not in numeric_set and key not in {"frame", "time"}
            }
            key = tuple(metadata.items())
            group = self._tables[table_name].get(key)
            if group is None:
                group = {
                    "metadata": metadata,
                    "count": 0,
                    "stats": {
                        name: self._new_stats(name)
                        for name in numeric_names
                    },
                }
                self._tables[table_name][key] = group
            group["count"] += 1
            for name in numeric_names:
                value = row.get(name)
                if value is None:
                    continue
                number = float(value)
                if not np.isfinite(number):
                    continue
                stats = group["stats"][name]
                stats["valid"] += 1
                if stats["circular"]:
                    stats["values"].append(float(number))
                else:
                    stats["sum"] += number
                    stats["sumsq"] += number * number

    def to_summary(self) -> Dict[str, List[Dict]]:
        output: Dict[str, List[Dict]] = {}
        for table_name, groups in self._tables.items():
            rows = []
            for group in groups.values():
                out = dict(group["metadata"])
                out["count"] = int(group["count"])
                for name, stats in group["stats"].items():
                    valid = int(stats["valid"])
                    if valid == 0:
                        out[f"{name}_mean"] = None
                        out[f"{name}_stddev"] = None
                    elif stats["circular"]:
                        parts = stats["values"]
                        if parts and isinstance(parts[0], np.ndarray):
                            values = np.concatenate(parts)
                        else:
                            values = np.asarray(parts, dtype=float)
                        summary = circular_degree_summary(values)
                        out[f"{name}_mean"] = summary.mean
                        out[f"{name}_stddev"] = summary.stddev
                    else:
                        mean = stats["sum"] / valid
                        variance = stats["sumsq"] / valid - mean * mean
                        if abs(variance) < 1e-15:
                            variance = 0.0
                        out[f"{name}_mean"] = float(mean)
                        out[f"{name}_stddev"] = float(np.sqrt(max(float(variance), 0.0)))
                rows.append(out)
            output[table_name] = rows

        for table_name, groups in self._population_tables.items():
            rows = []
            for group in groups.values():
                total_count = int(group["total_count"])
                count = int(group["count"])
                fraction = count / float(total_count) if total_count > 0 else 0.0
                row = dict(group["metadata"])
                row["count"] = count
                row["total_count"] = total_count
                row["fraction"] = float(fraction)
                row["percent"] = float(100.0 * fraction)
                rows.append(row)
            output[table_name] = rows
        return output


def _flush_batch(
    analyzer: BatchCurvesPlusMDAnalyzer,
    coordinates: List[np.ndarray],
    frame_indices: List[int],
    times: List[Optional[float]],
    mode: str,
    frame_payloads: List[Dict],
    table_records: Dict[str, List[Dict]],
    summary_accumulator: Optional[BatchSummaryAccumulator] = None,
) -> int:
    if not coordinates:
        return 0
    batch_coordinates = np.asarray(coordinates, dtype=float)
    if mode == "summary" and summary_accumulator is not None:
        return analyzer.accumulate_batch_summary(batch_coordinates, frame_indices, times, summary_accumulator)

    batch_frames, batch_tables = analyzer.analyze_batch(batch_coordinates, frame_indices, times)
    if mode in {"per-frame", "both"}:
        frame_payloads.extend(batch_frames)
    if mode in {"summary", "both"}:
        for name, rows in batch_tables.items():
            table_records.setdefault(name, []).extend(rows)
    return len(coordinates)


def _write_csv_payload(payload: Dict, prefix: str) -> None:
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    if "summary" in payload:
        for name, rows in payload["summary"].items():
            MDTrajectoryAnalyzer._write_rows_csv(f"{prefix}_{name}_summary.csv", rows)

    if "frames" in payload:
        per_table: Dict[str, List[Dict]] = {}
        for frame in payload["frames"]:
            for name, rows in frame["dataframes"].items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    out = dict(row)
                    out["frame"] = frame["frame"]
                    out["time"] = frame["time"]
                    per_table.setdefault(name, []).append(out)
        for name, rows in per_table.items():
            MDTrajectoryAnalyzer._write_rows_csv(f"{prefix}_{name}_frames.csv", rows)


def run_batch(args) -> Dict:
    from pycurves_lib.md.batch_curvesplus import BatchCurvesPlusMDAnalyzer

    if args.axis_convention.lower().replace("-", "_") not in {"curvesplus", "curves_plus", "curves+", "canal"}:
        raise SystemExit("pycurves-md-batch currently supports only --axis-convention curvesplus.")
    if args.frame_convention.lower().replace("-", "_") not in {"standard", "curvesplus", "curves_plus", "curves+", "x3dna", "3dna"}:
        raise SystemExit("pycurves-md-batch currently supports only --frame-convention standard.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    frame_selector, selection = make_frame_selector(args.frames, args.start, args.stop, args.step)
    reference_topology = MDTrajectoryAnalyzer._reference_topology(args.topology, args.trajectory, args.output_dir)
    analyzer = BatchCurvesPlusMDAnalyzer(
        topology_file=reference_topology,
        inpfile=args.inp,
        output_dir=args.output_dir,
        continuous_strands=args.continuous_strands,
        fit_override=args.fit,
        comb_override=args.comb,
        ends_override=args.ends,
        include_grooves=args.grooves,
        include_curvesplus_axis_steps=args.curvesplus_axis_steps,
        include_fit_quality=args.fit_quality,
    )

    frame_payloads: List[Dict] = []
    table_records: Dict[str, List[Dict]] = {}
    summary_accumulator = BatchSummaryAccumulator() if args.mode == "summary" else None
    coordinates: List[np.ndarray] = []
    frame_indices: List[int] = []
    times: List[Optional[float]] = []
    processed = 0

    iterator = TrajectoryLoader.iter_frames(args.topology, args.trajectory, frame_selector)
    for frame in tqdm(iterator, desc="Processing frames in batches"):
        coordinates.append(np.asarray(frame.coordinates, dtype=float))
        frame_indices.append(int(frame.index))
        times.append(None if frame.time is None else float(frame.time))
        if len(coordinates) >= args.batch_size:
            processed += _flush_batch(
                analyzer,
                coordinates,
                frame_indices,
                times,
                args.mode,
                frame_payloads,
                table_records,
                summary_accumulator,
            )
            coordinates.clear()
            frame_indices.clear()
            times.clear()

    processed += _flush_batch(
        analyzer,
        coordinates,
        frame_indices,
        times,
        args.mode,
        frame_payloads,
        table_records,
        summary_accumulator,
    )
    if processed == 0:
        raise SystemExit("No trajectory frames matched the requested frame selection.")

    payload = {
        "program": "pyCurves",
        "format": "pycurves-trajectory-batch-curvesplus-v1",
        "inputs": {
            "topology_file": args.topology,
            "reference_topology_file": reference_topology,
            "trajectory_file": args.trajectory,
            "inpfile": analyzer.inpfile,
            "generated_inpfiles": analyzer.generated_inpfiles,
        },
        "analysis_options": {
            "batch_size": args.batch_size,
            "continuous_strands": args.continuous_strands,
            "fit": True if args.fit is None else args.fit,
            "comb": True if args.comb is None else args.comb,
            "ends": False if args.ends is None else args.ends,
            "mini": False,
            "axis_convention": "curvesplus",
            "grooves": analyzer.include_grooves,
            "curvesplus_axis_steps": args.curvesplus_axis_steps,
            "fit_quality": args.fit_quality,
        },
        "selection": {
            **selection,
            "processed_frames": processed,
            "engine": "experimental_batch_curvesplus",
        },
        "frame_convention": {
            "name": "standard",
            "compatible_with": ["Curves+", "3DNA", "x3dna"],
            "axis_convention": "curvesplus",
        },
        "annotations_enabled": False,
    }
    if args.mode in {"per-frame", "both"}:
        payload["frames"] = frame_payloads
    if args.mode == "summary":
        payload["summary"] = summary_accumulator.to_summary() if summary_accumulator is not None else {}
    elif args.mode == "both":
        payload["summary"] = MDTrajectoryAnalyzer._summarize_tables(table_records)
    return payload


def analyze_trajectory_batch(
    topology_file: str,
    trajectory_file: Optional[str] = None,
    inpfile: Optional[str] = None,
    output_dir: str = ".",
    frames: Optional[str] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: int = 1,
    batch_size: int = 128,
    mode: str = "per-frame",
    continuous_strands: bool = False,
    fit: Optional[bool] = None,
    grooves: Optional[bool] = None,
    comb: Optional[bool] = None,
    ends: Optional[bool] = None,
    frame_convention: str = "standard",
    axis_convention: str = "curvesplus",
    curvesplus_axis_steps: bool = False,
    fit_quality: bool = False,
) -> Dict:
    """Run the vectorized Curves+/standard-frame MD path from Python.

    This is the notebook-friendly equivalent of ``pycurves-md-batch``. It is
    intended for canonical, standard-frame analyses where the experimental
    batch engine is applicable.
    """
    if mode not in {"per-frame", "summary", "both"}:
        raise ValueError("mode must be one of: per-frame, summary, both")
    args = SimpleNamespace(
        topology=topology_file,
        trajectory=trajectory_file,
        inp=inpfile,
        output_dir=output_dir,
        output_file=None,
        mode=mode,
        format="json",
        frames=frames,
        start=start,
        stop=stop,
        step=step,
        batch_size=batch_size,
        continuous_strands=continuous_strands,
        fit=fit,
        grooves=grooves,
        comb=comb,
        ends=ends,
        frame_convention=frame_convention,
        axis_convention=axis_convention,
        curvesplus_axis_steps=curvesplus_axis_steps,
        fit_quality=fit_quality,
    )
    try:
        return run_batch(args)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental vectorized pyCurves MD runner for standard-frame "
            "Curves+ axis analyses."
        )
    )
    parser.add_argument("topology", help="Topology/reference structure file, usually PDB or CIF.")
    parser.add_argument("trajectory", nargs="?", help="Trajectory file. Omit for multi-model PDB input.")
    parser.add_argument("--inp", help="Existing Curves .inp file. If omitted, inferred from the topology.")
    parser.add_argument("--output-dir", default=".", help="Directory for auto-generated .inp files.")
    parser.add_argument("--output-file", default="pycurves_trajectory_batch.json", help="JSON file or CSV prefix.")
    parser.add_argument("--mode", choices=["per-frame", "summary", "both"], default="summary")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--frames", help="Frame list/ranges, e.g. '0,10,20:100:5'. Overrides start/stop/step.")
    parser.add_argument("--start", type=int, help="First frame index to include.")
    parser.add_argument("--stop", type=int, help="Stop before this frame index.")
    parser.add_argument("--step", type=int, default=1, help="Frame stride.")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of selected frames processed per vectorized batch.")
    parser.add_argument("--continuous-strands", action="store_true", help="Treat connected helical components as continuous during .inp inference.")
    parser.add_argument("--fit", action=argparse.BooleanOptionalAction, default=None, help="Override least-squares base fitting; batch mode currently requires true.")
    parser.add_argument("--comb", action=argparse.BooleanOptionalAction, default=None, help="Override combined strand analysis; batch mode currently requires true.")
    parser.add_argument("--ends", action=argparse.BooleanOptionalAction, default=None, help="Override terminal virtual end levels; batch mode currently requires false.")
    parser.add_argument("--grooves", action=argparse.BooleanOptionalAction, default=None, help="Override groove analysis; defaults to the .inp grv setting.")
    parser.add_argument("--frame-convention", default="standard", help="Currently only standard is supported.")
    parser.add_argument("--axis-convention", default="curvesplus", help="Currently only curvesplus is supported.")
    parser.add_argument("--curvesplus-axis-steps", action="store_true", help="Include the Curves+ smooth-axis inter-base-pair step table.")
    parser.add_argument("--fit-quality", action="store_true", help="Include vectorized base-fitting RMSD diagnostics.")
    args = parser.parse_args()

    try:
        payload = run_batch(args)
    except (ValueError, ImportError, NotImplementedError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.format == "json":
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_to_jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        _write_csv_payload(payload, args.output_file.removesuffix(".csv"))

    generated = payload["inputs"].get("generated_inpfiles") or []
    if generated:
        print("Generated input file(s):", file=sys.stderr)
        for path in generated:
            print(f"  {path}", file=sys.stderr)
    print(f"Processed {payload['selection']['processed_frames']} frame(s).", file=sys.stderr)


if __name__ == "__main__":
    main()






