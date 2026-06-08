from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_NAME = "pycurves-dna"


def _run_pip(args: list[str]) -> int:
    command = [sys.executable, "-m", "pip", *args]
    print(" ".join(command))
    return subprocess.call(command)


def _install_target(extras: list[str]) -> str:
    root = Path(__file__).resolve().parent
    if not extras:
        return str(root)
    return f"{root}[{','.join(extras)}]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install or uninstall pyCurves in the current Python environment."
    )
    parser.add_argument(
        "action",
        choices=["install", "uninstall"],
        help="Install pyCurves from this source tree or uninstall the package.",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        help="Use pip install -e for development installs.",
    )
    parser.add_argument(
        "--extras",
        default="",
        help="Comma-separated extras to install, for example 'md,plot' or 'all'.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Pass --upgrade to pip install.",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Do not install dependencies.",
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help="Use the current environment's build tools instead of creating an isolated build environment.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Do not prompt when uninstalling.",
    )
    args = parser.parse_args()

    if args.action == "uninstall":
        pip_args = ["uninstall"]
        if args.yes:
            pip_args.append("-y")
        pip_args.append(PACKAGE_NAME)
        return _run_pip(pip_args)

    extras = [item.strip() for item in args.extras.split(",") if item.strip()]
    pip_args = ["install"]
    if args.editable:
        pip_args.append("-e")
    if args.upgrade:
        pip_args.append("--upgrade")
    if args.no_deps:
        pip_args.append("--no-deps")
    if args.no_build_isolation:
        pip_args.append("--no-build-isolation")
    pip_args.append(_install_target(extras))
    return _run_pip(pip_args)


if __name__ == "__main__":
    raise SystemExit(main())
