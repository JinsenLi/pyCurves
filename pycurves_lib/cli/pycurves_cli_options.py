"""Analysis options shared by the structure and trajectory CLIs."""

from __future__ import annotations

import argparse

from pycurves_lib.cli.pycurves_cli_options_core import (
    add_pycurves_analysis_options as _add_core_options,
    pycurves_runner_kwargs as _core_runner_kwargs,
)


def add_pycurves_analysis_options(parser: argparse.ArgumentParser) -> None:
    _add_core_options(parser)
    parser.add_argument(
        "--dssr-json",
        help=(
            "DSSR JSON report used as the topology source. Coordinates are still "
            "read from the PDB/mmCIF input."
        ),
    )
    parser.add_argument(
        "--dssr-unit",
        help=(
            "DSSR unit selector such as stem:1, helix:2, or pairs:1. Required "
            "when the report contains multiple candidate units."
        ),
    )


def pycurves_runner_kwargs(args) -> dict:
    values = _core_runner_kwargs(args)
    values.update({
        "dssr_json": getattr(args, "dssr_json", None),
        "dssr_unit": getattr(args, "dssr_unit", None),
    })
    return values


__all__ = ["add_pycurves_analysis_options", "pycurves_runner_kwargs"]
