import argparse
from pathlib import Path
import sys

from pycurves_lib.curves_wrapper import CurvesWrapper
from pycurves_lib.cli.pycurves_cli_options import (
    add_pycurves_analysis_options,
    pycurves_runner_kwargs,
    resolved_mini,
)


def main():
    parser = argparse.ArgumentParser(description="Run pyCurves from a PDB or Curves .inp file.")
    parser.add_argument("input", nargs="?", help="PDB or .inp file. If omitted, you will be prompted.")
    parser.add_argument("--pdb", help="PDB file to use with an explicit .inp file.")
    parser.add_argument("--output-dir", default=".", help="Directory for auto-generated .inp files.")
    parser.add_argument("--no-output", action="store_true", help="Run calculations without printing the Curves report.")
    parser.add_argument("--format", choices=["curves", "json", "csv"], default="curves", help="Output format (csv requires Pandas).")
    parser.add_argument("--output-file", help="Write the selected output format to a file instead of stdout (for csv, writes multiple files based on prefix).")
    parser.add_argument("--no-annotations", action="store_true", help="Suppress the pyCurves |M| annotation report and annotation records.")
    parser.add_argument("--visualization", "--visualize", action="store_true", help="Include HTML-viewer geometry in JSON output.")
    parser.add_argument("--verbose-opt", action="store_true", help="Echo fitting and minimization progress while analyzing.")
    parser.add_argument("--quiet-opt", action="store_true", help=argparse.SUPPRESS)
    add_pycurves_analysis_options(parser)
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        input_path = input("PDB or .inp file to analyze: ").strip()

    if not input_path:
        raise SystemExit("No input file provided.")

    path = Path(input_path)
    
    runner_kwargs = pycurves_runner_kwargs(args)
    mini_override = resolved_mini(args, default=True)
    
    if path.suffix.lower() == ".inp":
        runner = CurvesWrapper(
            pdbfile=args.pdb,
            inpfile=str(path),
            output_dir=args.output_dir,
            **runner_kwargs,
        )
    else:
        runner = CurvesWrapper(
            pdbfile=str(path),
            output_dir=args.output_dir,
            **runner_kwargs,
        )
        if runner.generated_inpfiles:
            stream = sys.stderr if args.format in ["json", "csv"] and not args.output_file else sys.stdout
            print("Generated input file(s):", file=stream)
            for inpfile in runner.generated_inpfiles:
                print(f"  {inpfile}", file=stream)

    runner.run(output=False, mini=mini_override, verbose=args.verbose_opt and not args.quiet_opt)
    include_annotations = not args.no_annotations
    
    if not args.no_output:
        if args.format == "csv" and args.output_file:
            # For CSV with an output file specified, use it as a prefix
            prefix = args.output_file.removesuffix('.csv')
            from pycurves_lib.io.curves_output import CurvesOutputFormatter
            dfs = CurvesOutputFormatter(runner, annotations=include_annotations).get_dataframes()
                
            for name, df in dfs.items():
                df.to_csv(f"{prefix}_{name}.csv", index=False)
            print(f"Saved CSV files with prefix '{prefix}_'")
        else:
            runner.output(
                fmt=args.format,
                file=args.output_file,
                annotations=include_annotations,
                visualization=args.visualization,
            )


if __name__ == "__main__":
    main()
