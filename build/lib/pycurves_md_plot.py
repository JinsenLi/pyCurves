from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "pycurves-md-plot needs the full optional dependencies. Install from the source tree with:\n"
        "  pip install \".[all]\""
    ) from exc

PARAMETER_BLOCKS = {
    "step": {
        "label": "Global Inter-Base-Pair Step Parameters",
        "table": "global_inter_base_pair",
        "parameters": ["shift", "slide", "rise", "tilt", "roll", "twist"],
        "angle_parameters": ["tilt", "roll", "twist"],
        "id_columns": ["level", "partner_strand", "step", "duplex"],
    },
    "local_step": {
        "label": "Local Inter-Base-Pair Step Parameters",
        "table": "local_inter_base_pair",
        "parameters": ["shift", "slide", "rise", "tilt", "roll", "twist"],
        "angle_parameters": ["tilt", "roll", "twist"],
        "id_columns": ["level", "partner_strand", "step", "duplex"],
    },
    "strand_step": {
        "label": "Global Strand Inter-Base Step Parameters",
        "table": "global_inter_base",
        "parameters": ["shift", "slide", "rise", "tilt", "roll", "twist"],
        "angle_parameters": ["tilt", "roll", "twist"],
        "id_columns": ["level", "strand", "step"],
    },
    "local_strand_step": {
        "label": "Local Strand Inter-Base Step Parameters",
        "table": "local_inter_base",
        "parameters": ["shift", "slide", "rise", "tilt", "roll", "twist"],
        "angle_parameters": ["tilt", "roll", "twist"],
        "id_columns": ["level", "strand", "step"],
    },
    "base_pair": {
        "label": "Helical Base-Pair Parameters",
        "table": "global_base_base",
        "parameters": ["shear", "stretch", "stagger", "buckle", "propel", "opening"],
        "angle_parameters": ["buckle", "propel", "opening"],
        "id_columns": ["level", "duplex"],
    },
    "local_base_pair": {
        "label": "Local Intra-Base-Pair Parameters",
        "table": "local_base_base",
        "parameters": ["shear", "stretch", "stagger", "buckle", "propel", "opening"],
        "angle_parameters": ["buckle", "propel", "opening"],
        "id_columns": ["level", "partner_strand", "duplex"],
    },
    "axis": {
        "label": "Axis Base-Pair Parameters",
        "table": "global_base_pair_axis",
        "parameters": ["xdisp", "ydisp", "inclin", "tip"],
        "angle_parameters": ["inclin", "tip"],
        "id_columns": ["level", "duplex"],
    },
    "axis_curvature": {
        "label": "Global Axis Curvature",
        "table": "global_axis_curvature",
        "parameters": ["ax", "ay", "ainc", "atip", "adis", "angle", "path"],
        "angle_parameters": ["ainc", "atip", "angle"],
        "id_columns": ["strand", "level"],
    },
    "axis_bending": {
        "label": "Global Axis Bending",
        "table": "global_axis_bending",
        "parameters": ["offset", "local_direction"],
        "angle_parameters": ["local_direction"],
        "id_columns": ["strand", "level", "residue_name", "residue_id"],
    },
    "axis_bending_summary": {
        "label": "Global Axis Bending Summary",
        "table": "global_axis_bending_summary",
        "parameters": ["path_length", "end_to_end", "shortening_percent", "overall_bend_uu", "overall_bend_pp"],
        "angle_parameters": ["overall_bend_uu", "overall_bend_pp"],
        "id_columns": ["strand"],
    },
    "groove": {
        "label": "Groove Parameters",
        "table": "groove",
        "parameters": ["minor_width", "minor_depth", "major_width", "major_depth", "diameter"],
        "id_columns": ["level", "base_pair"],
    },
    "backbone": {
        "label": "Backbone Torsions and Sugar Pucker",
        "table": "backbone",
        "parameters": [
            "c1_c2",
            "c2_c3",
            "phase",
            "amplitude",
            "c1_prime",
            "c2_prime",
            "c3_prime",
            "chi",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "alpha",
            "beta",
        ],
        "angle_parameters": [
            "c1_c2",
            "c2_c3",
            "phase",
            "c1_prime",
            "c2_prime",
            "c3_prime",
            "chi",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "alpha",
            "beta",
        ],
        "id_columns": ["strand", "level", "residue_name", "residue_id"],
    },
}

DEFAULT_BLOCKS = [
    "step",
    "local_step",
    "strand_step",
    "local_strand_step",
    "base_pair",
    "local_base_pair",
    "axis",
    "axis_curvature",
    "axis_bending",
    "axis_bending_summary",
    "backbone",
    "groove",
]

BLOCK_PRESETS = {
    "default": DEFAULT_BLOCKS,
    "global": ["step", "strand_step", "base_pair", "axis"],
    "local": ["local_base_pair", "local_step", "local_strand_step"],
    "intra_pair": ["local_base_pair"],
    "curvature": ["axis_curvature"],
    "axis_analysis": ["axis", "axis_curvature", "axis_bending", "axis_bending_summary"],
    "torsions": ["backbone"],
    "all": DEFAULT_BLOCKS,
}

BLOCK_ALIASES = {
    "basepair": "base_pair",
    "bp": "base_pair",
    "local_base_base": "local_base_pair",
    "local_basepair": "local_base_pair",
    "local_bp": "local_base_pair",
    "intra_base_pair": "local_base_pair",
    "local_intra_base_pair": "local_base_pair",
    "bps": "step",
    "base_pair_step": "step",
    "global_step": "step",
    "inter_base_pair": "step",
    "global_inter_base_pair": "step",
    "global_inter_base": "strand_step",
    "inter_base": "strand_step",
    "local_inter_base": "local_strand_step",
    "strand_inter_base": "strand_step",
    "local_strand_inter_base": "local_strand_step",
    "local_inter_base_pair": "local_step",
    "pair_step_raw": "step",
    "helical": "step",
    "global_axis_curvature": "axis_curvature",
    "curvature_analysis": "axis_curvature",
    "global_axis_bending": "axis_bending",
    "bending": "axis_bending",
    "axis_bending": "axis_bending",
    "global_axis_bending_summary": "axis_bending_summary",
    "bending_summary": "axis_bending_summary",
    "backbone_torsions": "backbone",
    "torsion": "backbone",
}


def _block_key(block: str) -> str:
    key = block.lower().replace("-", "_")
    return BLOCK_ALIASES.get(key, key)


def available_block_names() -> str:
    names = sorted(set(PARAMETER_BLOCKS) | set(BLOCK_PRESETS) | set(BLOCK_ALIASES))
    return ", ".join(names)


def expand_block_selection(blocks: Optional[Sequence[str]]) -> List[str]:
    """Expand block names and presets into canonical plotting blocks."""
    selected = list(blocks or ["default"])
    out: List[str] = []
    for block in selected:
        key = _block_key(block)
        if key in BLOCK_PRESETS:
            candidates = BLOCK_PRESETS[key]
        elif key in PARAMETER_BLOCKS:
            candidates = [key]
        else:
            raise ValueError(f"Unknown block {block!r}. Use one of: {available_block_names()}.")
        for candidate in candidates:
            if candidate not in out:
                out.append(candidate)
    return out


def load_trajectory_payload(source: str | Path | Dict) -> Dict:
    """Load a pyCurves MD payload from a dict or JSON file path."""
    if isinstance(source, dict):
        return source
    return json.loads(Path(source).read_text(encoding="utf-8"))


def iter_pycurves_frames(source: str | Path | Dict | List[Dict]) -> Iterator[Dict]:
    """Yield per-frame payloads from a pyCurves trajectory payload or JSON file.

    Passing a dict is convenient in notebooks after ``analyze_trajectory``. Passing
    a path streams frames with ``ijson`` when available, which is better for large
    JSON files written by the CLI.
    """
    if isinstance(source, dict):
        for frame in source.get("frames", []):
            yield frame
        return
    if isinstance(source, list):
        for frame in source:
            yield frame
        return

    path = Path(source)
    try:
        import ijson
    except ImportError:
        data = json.loads(path.read_text(encoding="utf-8"))
        for frame in data.get("frames", []):
            yield frame
        return

    with path.open("rb") as handle:
        for frame in ijson.items(handle, "frames.item"):
            yield frame


def flatten_groove_table(groove_payload: Dict) -> List[Dict]:
    """Flatten nested pyCurves groove records into long-form rows."""
    if not isinstance(groove_payload, dict):
        return []

    rows: List[Dict] = []
    atom_defining_backbone = groove_payload.get("atom_defining_backbone")
    for level_text, level_data in (groove_payload.get("data") or {}).items():
        try:
            level = int(level_text)
        except (TypeError, ValueError):
            level = level_text
        base_pair = level_data.get("base_pair", "")
        for sub_level_text, values in (level_data.get("sub_levels") or {}).items():
            if not isinstance(values, dict):
                continue
            try:
                sub_level = int(sub_level_text)
            except (TypeError, ValueError):
                sub_level = sub_level_text
            row = {
                "level": level,
                "sub_level": sub_level,
                "base_pair": base_pair,
                "atom_defining_backbone": atom_defining_backbone,
            }
            for key, value in values.items():
                if key != "geometry":
                    row[key] = value
            rows.append(row)
    return rows


def frame_table_rows(frame: Dict, table_name: str) -> List[Dict]:
    """Return one frame's table rows with `frame` and `time` columns attached."""
    dataframes = frame.get("dataframes", {})
    payload = dataframes.get(table_name)
    if isinstance(payload, list):
        rows = payload
    elif table_name == "groove" and isinstance(payload, dict):
        rows = flatten_groove_table(payload)
    else:
        rows = []

    frame_index = frame.get("frame")
    time_value = frame.get("time")
    out = []
    for row in rows:
        clean = dict(row)
        clean["frame"] = frame_index
        clean["time"] = time_value if time_value is not None else frame_index
        out.append(clean)
    return out


def extract_table(source: str | Path | Dict | List[Dict], table_name: str) -> pd.DataFrame:
    """Extract a per-frame pyCurves trajectory table into a long-form DataFrame."""
    rows: List[Dict] = []
    for frame in iter_pycurves_frames(source):
        rows.extend(frame_table_rows(frame, table_name))
    return pd.DataFrame(rows)


def extract_block(source: str | Path | Dict | List[Dict], block: str) -> pd.DataFrame:
    """Extract one named analysis block such as step, base_pair, axis, groove, or backbone."""
    spec = block_spec(block)
    return extract_table(source, spec["table"])


def extract_summary_table(source: str | Path | Dict, table_name: str) -> pd.DataFrame:
    """Extract a summary table from a summary/both-mode MD payload or JSON file."""
    payload = load_trajectory_payload(source)
    return pd.DataFrame((payload.get("summary") or {}).get(table_name, []))


def extract_summary_block(source: str | Path | Dict, block: str) -> pd.DataFrame:
    """Extract a named block's summary table from a summary/both-mode payload."""
    spec = block_spec(block)
    return extract_summary_table(source, spec["table"])


def block_spec(block: str) -> Dict:
    key = _block_key(block)
    if key not in PARAMETER_BLOCKS:
        allowed = available_block_names()
        raise ValueError(f"Unknown block {block!r}. Use one of: {allowed}.")
    return PARAMETER_BLOCKS[key]

def filter_rows(
    df: pd.DataFrame,
    level: Optional[Sequence[int]] = None,
    strand: Optional[Sequence[int]] = None,
    duplex_contains: Optional[str] = None,
    drop_terminal: int = 0,
) -> pd.DataFrame:
    """Convenience filter for selected levels, strands, or duplex labels."""
    out = df
    if level is not None and "level" in out:
        out = out[out["level"].isin([int(v) for v in level])]
    if drop_terminal > 0 and "level" in out and not out.empty:
        levels = sorted(int(v) for v in out["level"].dropna().unique())
        if len(levels) > 2 * drop_terminal:
            keep = set(levels[drop_terminal:-drop_terminal])
            out = out[out["level"].isin(keep)]
    if strand is not None and "strand" in out:
        out = out[out["strand"].isin([int(v) for v in strand])]
    if duplex_contains and "duplex" in out:
        needle = str(duplex_contains)
        out = out[out["duplex"].fillna("").astype(str).str.contains(needle, regex=False)]
    return out.copy()


def wrap_degrees(values) -> pd.Series:
    """Wrap angular values to the conventional [-180, 180) degree interval."""
    numeric = pd.to_numeric(values, errors="coerce")
    return ((numeric + 180.0) % 360.0) - 180.0


def parameter_timeseries(
    df: pd.DataFrame,
    parameter: str,
    time_column: str = "time",
    aggregate: bool = True,
    statistic: str = "mean",
) -> pd.DataFrame:
    """Return time series for one parameter.

    With aggregate=True, all levels/strands at each time are summarized as
    mean/std/min/max. With aggregate=False, the original long-form rows are
    returned with only finite values retained.
    """
    if parameter not in df.columns:
        raise ValueError(f"Parameter {parameter!r} is not present. Columns: {list(df.columns)}")
    out = df[[time_column, "frame", parameter] + [c for c in ("level", "sub_level", "strand", "duplex", "base_pair") if c in df.columns]].copy()
    out[parameter] = pd.to_numeric(out[parameter], errors="coerce")
    out = out[np.isfinite(out[parameter])]
    if not aggregate:
        return out

    grouped = out.groupby(time_column, dropna=False)[parameter]
    summary = grouped.agg(["mean", "median", "std", "min", "max", "count"]).reset_index()
    if statistic == "median":
        quantiles = grouped.quantile([0.16, 0.84]).unstack()
        quantiles.columns = ["q16", "q84"]
        summary = summary.merge(quantiles.reset_index(), on=time_column, how="left")
    else:
        summary["q16"] = summary["mean"] - summary["std"]
        summary["q84"] = summary["mean"] + summary["std"]
    summary["std"] = summary["std"].fillna(0.0)
    summary["q16"] = summary["q16"].fillna(summary[statistic])
    summary["q84"] = summary["q84"].fillna(summary[statistic])
    return summary


def outlier_rows(
    df: pd.DataFrame,
    parameter: str,
    abs_limit: Optional[float] = None,
    robust_z: float = 8.0,
) -> pd.DataFrame:
    """Return rows with unusually large parameter values for diagnostics."""
    if parameter not in df.columns:
        return pd.DataFrame()
    id_cols = [
        c for c in ("frame", "time", "plot_time", "level", "sub_level", "strand", "partner_strand", "duplex", "base_pair", "step")
        if c in df.columns
    ]
    work = df[id_cols + [parameter]].copy()
    work[parameter] = pd.to_numeric(work[parameter], errors="coerce")
    work = work[np.isfinite(work[parameter])]
    if work.empty:
        return work

    mask = np.zeros(len(work), dtype=bool)
    if abs_limit is not None:
        mask |= work[parameter].abs().to_numpy(dtype=float) >= float(abs_limit)

    values = work[parameter].to_numpy(dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-12 and robust_z is not None and robust_z > 0:
        modified_z = 0.6745 * np.abs(values - median) / mad
        mask |= modified_z >= float(robust_z)

    out = work.loc[mask].copy()
    if out.empty:
        return out
    out["abs_value"] = out[parameter].abs()
    return out.sort_values("abs_value", ascending=False)


def pivot_parameter_matrix(
    df: pd.DataFrame,
    parameter: str,
    index_column: str = "time",
    column: str = "level",
) -> pd.DataFrame:
    """Build a time x level matrix for heatmaps."""
    if parameter not in df.columns:
        raise ValueError(f"Parameter {parameter!r} is not present.")
    if column not in df.columns:
        raise ValueError(f"Column {column!r} is not present.")
    work = df[[index_column, column, parameter]].copy()
    work[parameter] = pd.to_numeric(work[parameter], errors="coerce")
    work = work[np.isfinite(work[parameter])]
    matrix = work.pivot_table(index=index_column, columns=column, values=parameter, aggfunc="mean")
    return matrix.sort_index().sort_index(axis=1)


def add_time_axis(df: pd.DataFrame, time_scale: float = 1.0, time_label: str = "time") -> pd.DataFrame:
    """Add `plot_time` to a table, preserving original frame/time columns."""
    out = df.copy()
    if "time" in out and out["time"].notna().any():
        out["plot_time"] = pd.to_numeric(out["time"], errors="coerce") * time_scale
    else:
        out["plot_time"] = pd.to_numeric(out["frame"], errors="coerce") * time_scale
    out.attrs["time_label"] = time_label
    return out


def plot_parameter_summary(
    df: pd.DataFrame,
    parameter: str,
    output_path: str | Path,
    title: Optional[str] = None,
    time_column: str = "plot_time",
    time_label: str = "time",
    statistic: str = "mean",
) -> None:
    """Plot mean parameter value over time with a side density distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Plotting requires matplotlib. Install the full optional dependencies with: pip install \".[all]\"") from exc

    series = parameter_timeseries(df, parameter, time_column=time_column, aggregate=True, statistic=statistic)
    if series.empty:
        return
    values = pd.to_numeric(df[parameter], errors="coerce")
    values = values[np.isfinite(values)].to_numpy(dtype=float)
    x = series[time_column].to_numpy(dtype=float)
    center = series[statistic].to_numpy(dtype=float)
    lower = series["q16"].to_numpy(dtype=float)
    upper = series["q84"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(10.5, 4.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.1], wspace=0.05)
    ax = fig.add_subplot(grid[0, 0])
    density_ax = fig.add_subplot(grid[0, 1], sharey=ax)

    ax.plot(x, center, color="#1f5a99", lw=1.8)
    ax.fill_between(x, lower, upper, color="#72a7d8", alpha=0.28, linewidth=0)
    ax.set_xlabel(time_label)
    ax.set_ylabel(parameter)
    ax.set_title(title or parameter)
    ax.grid(True, color="#d0d7de", alpha=0.7, linewidth=0.8)

    if values.size >= 2 and np.nanmax(values) > np.nanmin(values):
        y_min = float(np.nanmin(values))
        y_max = float(np.nanmax(values))
        try:
            from scipy.stats import gaussian_kde

            y_grid = np.linspace(y_min, y_max, 240)
            density = gaussian_kde(values)(y_grid)
            density_ax.fill_betweenx(y_grid, 0.0, density, color="#1f5a99", alpha=0.22, linewidth=0)
            density_ax.plot(density, y_grid, color="#1f5a99", lw=1.25)
        except Exception:
            density, edges = np.histogram(values, bins="auto", density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            density_ax.fill_betweenx(centers, 0.0, density, color="#1f5a99", alpha=0.22, step="mid")
            density_ax.plot(density, centers, color="#1f5a99", lw=1.25, drawstyle="steps-mid")

    density_ax.set_xlabel("density")
    density_ax.tick_params(axis="y", labelleft=False, left=False)
    density_ax.grid(False)
    for spine in ("top", "right"):
        density_ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_parameter_heatmap(
    df: pd.DataFrame,
    parameter: str,
    output_path: str | Path,
    title: Optional[str] = None,
    time_column: str = "plot_time",
    time_label: str = "time",
    level_column: str = "level",
) -> None:
    """Plot parameter values as a time x level heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Plotting requires matplotlib. Install the full optional dependencies with: pip install \".[all]\"") from exc

    matrix = pivot_parameter_matrix(df, parameter, index_column=time_column, column=level_column)
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(matrix.T, aspect="auto", origin="lower", interpolation="nearest", cmap="coolwarm")
    tick_count = min(8, len(matrix.index))
    if tick_count > 0:
        tick_positions = np.linspace(0, len(matrix.index) - 1, tick_count, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{matrix.index[i]:.3g}" for i in tick_positions], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(matrix.columns)))
    ax.set_yticklabels([str(v) for v in matrix.columns])
    ax.set_xlabel(time_label)
    ax.set_ylabel(level_column)
    ax.set_title(title or f"{parameter} by {level_column} and time")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(parameter)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_block_overview(
    df: pd.DataFrame,
    block: str,
    output_dir: str | Path,
    parameters: Optional[Sequence[str]] = None,
    time_scale: float = 1.0,
    time_label: str = "time",
    export_csv: bool = False,
    make_plots: bool = True,
    statistic: str = "mean",
    diagnose_outliers: bool = False,
    outlier_abs: Optional[float] = None,
    outlier_z: float = 8.0,
) -> None:
    """Create summary time-series and heatmap plots for one helical block."""
    spec = block_spec(block)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    df = add_time_axis(df, time_scale=time_scale, time_label=time_label)
    params = list(parameters or spec["parameters"])
    angle_parameters = set(spec.get("angle_parameters", []))
    for parameter in params:
        if parameter in angle_parameters and parameter in df.columns:
            df[parameter] = wrap_degrees(df[parameter])

    if export_csv:
        df.to_csv(output_root / f"{block}_long.csv", index=False)
    if diagnose_outliers:
        for parameter in params:
            if parameter not in df.columns:
                continue
            outliers = outlier_rows(df, parameter, abs_limit=outlier_abs, robust_z=outlier_z)
            if not outliers.empty:
                outliers.to_csv(output_root / f"{block}_{parameter}_outliers.csv", index=False)
    if not make_plots:
        return

    for parameter in params:
        if parameter not in df.columns:
            continue
        plot_parameter_summary(
            df,
            parameter,
            output_root / f"{block}_{parameter}_timeseries.png",
            title=f"{spec['label']}: {parameter}",
            time_label=time_label,
            statistic=statistic,
        )
        if "level" in df.columns:
            plot_parameter_heatmap(
                df,
                parameter,
                output_root / f"{block}_{parameter}_heatmap.png",
                title=f"{spec['label']}: {parameter}",
                time_label=time_label,
            )


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, end_text = part.split(":", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and plot pyCurves MD trajectory helical parameters."
    )
    parser.add_argument("json_file", help="pyCurves trajectory JSON generated by pycurves_md.py --mode per-frame or both.")
    parser.add_argument("--outdir", default="pycurves_md_plots", help="Directory for extracted CSV files and PNG plots.")
    parser.add_argument(
        "--block",
        action="append",
        metavar="NAME",
        help=(
            "Analysis block or preset to plot. Repeat for multiple blocks. "
            "Presets include global, local, curvature, torsions, axis_analysis, all. "
            "Default: global and local shape features, axis curvature/bending, backbone, and groove."
        ),
    )
    parser.add_argument("--parameter", action="append", help="Specific parameter to plot. Repeat for multiple parameters.")
    parser.add_argument("--levels", help="Comma/range levels to keep, e.g. '4,5,8:12'.")
    parser.add_argument("--drop-terminal", type=int, default=0, help="Drop this many terminal levels from both ends before plotting.")
    parser.add_argument("--strands", help="Comma/range strands to keep for backbone/global strand tables.")
    parser.add_argument("--duplex-contains", help="Keep only rows whose duplex label contains this literal text.")
    parser.add_argument("--time-scale", type=float, default=0.001, help="Scale applied to JSON time values. Default converts ps to ns.")
    parser.add_argument("--time-label", default="time (ns)", help="X-axis label after applying --time-scale.")
    parser.add_argument("--export-csv", action="store_true", help="Write extracted long-form CSV tables.")
    parser.add_argument("--no-plots", action="store_true", help="Only extract CSV tables; do not create PNG plots.")
    parser.add_argument("--stat", choices=["mean", "median"], default="mean", help="Center statistic for time-series plots across selected rows. Default: mean.")
    parser.add_argument("--diagnose-outliers", action="store_true", help="Write per-parameter outlier CSV files with frame/level/duplex identifiers.")
    parser.add_argument("--outlier-abs", type=float, help="Also flag outliers whose absolute value is at least this threshold.")
    parser.add_argument("--outlier-z", type=float, default=8.0, help="Robust modified-z threshold for outlier diagnostics. Default: 8.")
    args = parser.parse_args()

    try:
        blocks = expand_block_selection(args.block)
    except ValueError as exc:
        parser.error(str(exc))
    levels = parse_int_list(args.levels)
    strands = parse_int_list(args.strands)

    for block in blocks:
        spec = block_spec(block)
        df = extract_block(args.json_file, block)
        if df.empty:
            print(f"[skip] {block}: no per-frame rows found. Was pycurves_md.py run with --mode per-frame or both?")
            continue
        df = filter_rows(
            df,
            level=levels,
            strand=strands,
            duplex_contains=args.duplex_contains,
            drop_terminal=args.drop_terminal,
        )
        if df.empty:
            print(f"[skip] {block}: all rows were removed by filters.")
            continue

        parameters = args.parameter or spec["parameters"]
        plot_block_overview(
            df,
            block=block,
            output_dir=args.outdir,
            parameters=parameters,
            time_scale=args.time_scale,
            time_label=args.time_label,
            export_csv=args.export_csv,
            make_plots=not args.no_plots,
            statistic=args.stat,
            diagnose_outliers=args.diagnose_outliers,
            outlier_abs=args.outlier_abs,
            outlier_z=args.outlier_z,
        )
        print(f"[ok] {block}: {len(df)} rows -> {args.outdir}")


if __name__ == "__main__":
    main()
