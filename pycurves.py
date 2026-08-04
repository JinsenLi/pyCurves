"""Console entry point for static-structure pyCurves analysis."""

from pycurves_lib.cli.pycurves_main import main as _main


def main():
    try:
        return _main()
    except (ValueError, ImportError, NotImplementedError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
