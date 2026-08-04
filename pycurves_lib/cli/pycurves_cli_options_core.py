import argparse


def add_pycurves_analysis_options(parser: argparse.ArgumentParser) -> None:
    """Add pyCurves analysis options shared by structure and trajectory CLIs."""
    parser.add_argument(
        "--continuous-strands",
        action="store_true",
        help="Treat separated helices in connected components as a single continuous structure.",
    )
    parser.add_argument(
        "--altloc",
        help=("Alternate-conformation identifier to analyze, such as A or B. "
              "By default Gemmi keeps the first-listed conformer, not the highest-occupancy conformer."),
    )
    parser.add_argument(
        "--fit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override least-squares base fitting (default: inferred).",
    )
    parser.add_argument(
        "--grooves",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override groove analysis (default: inferred).",
    )
    parser.add_argument(
        "--mini",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override axis minimization (default: read from the .inp file).",
    )
    parser.add_argument(
        "--comb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override combined strand analysis (default: inferred).",
    )
    parser.add_argument(
        "--ends",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Curves terminal virtual end levels (default: inferred).",
    )
    parser.add_argument(
        "--frame-convention",
        default="standard",
        help="Base reference-frame convention (default: standard; use 'legacy' for Curves 5.3-compatible frames).",
    )
    parser.add_argument(
        "--axis-convention",
        default="legacy",
        help="Global-axis convention. Default keeps the pyCurves legacy minimization axis; 'curvesplus' reproduces Curves+ axis/smooth output.",
    )

def pycurves_runner_kwargs(args) -> dict:
    """Return CurvesWrapper keyword arguments from shared CLI options."""
    return {
        "continuous_strands": getattr(args, "continuous_strands", False),
        "altloc": getattr(args, "altloc", None),
        "frame_convention": getattr(args, "frame_convention", "standard"),
        "axis_convention": getattr(args, "axis_convention", "legacy"),
        "fit_override": getattr(args, "fit", None),
        "grv_override": getattr(args, "grooves", None),
        "mini_override": getattr(args, "mini", None),
        "comb_override": getattr(args, "comb", None),
        "ends_override": getattr(args, "ends", None),
    }
