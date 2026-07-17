import argparse
import glob
from pathlib import Path
import sys

from pycurves_lib.curves_wrapper import CurvesWrapper
from pycurves_lib.cli.pycurves_cli_options import (
    add_pycurves_analysis_options,
    pycurves_runner_kwargs,
    resolved_mini,
)


STRUCTURE_EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz")


def _is_structure_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in STRUCTURE_EXTENSIONS)


def _expand_generation_inputs(inputs):
    paths = []
    for raw in inputs:
        matches = glob.glob(raw, recursive=True) if any(ch in raw for ch in "*?[]") else [raw]
        if not matches:
            raise SystemExit(f"No files matched {raw!r}.")
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.extend(sorted(child for child in path.iterdir() if child.is_file() and _is_structure_file(child)))
            else:
                paths.append(path)

    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _unique_inp_prefix(path: Path, used_prefixes: dict[str, int]) -> str:
    prefix = f"{path.stem}_auto"
    count = used_prefixes.get(prefix, 0) + 1
    used_prefixes[prefix] = count
    if count == 1:
        return prefix
    return f"{prefix}_{count}"


def _generate_inp_only(args, runner_kwargs) -> None:
    structure_paths = _expand_generation_inputs(args.input)
    if not structure_paths:
        raise SystemExit("No structure files found for .inp generation.")

    generated_count = 0
    failed = []
    used_prefixes = {}
    for path in structure_paths:
        if path.suffix.lower() == ".inp":
            failed.append((path, "input-generation mode expects PDB/mmCIF structure files, not existing .inp files"))
            continue
        try:
            runner = CurvesWrapper(
                pdbfile=str(path),
                output_dir=args.output_dir,
                auto_generate_inp=False,
                **runner_kwargs,
            )
            generated_inpfiles = runner.generate_inp(
                pdbfile=str(path),
                output_dir=args.output_dir,
                prefix=_unique_inp_prefix(path, used_prefixes),
                continuous_strands=args.continuous_strands,
            )
            runner.generated_inpfiles = generated_inpfiles
        except Exception as exc:
            failed.append((path, str(exc)))
            continue

        print(f"{path}:")
        for inpfile in generated_inpfiles:
            generated_count += 1
            print(f"  {inpfile}")

    if failed:
        print("\nFailed input file(s):", file=sys.stderr)
        for path, reason in failed:
            print(f"  {path}: {reason}", file=sys.stderr)
        print(
            f"Generated {generated_count} .inp file(s) from {len(structure_paths) - len(failed)}/{len(structure_paths)} structure file(s).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(
        f"Generated {generated_count} .inp file(s) from {len(structure_paths)} structure file(s).",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description="Run pyCurves from a PDB or Curves .inp file.")
    parser.add_argument("input", nargs="*", help="PDB/mmCIF or .inp file. Use --generate-inp-only for multiple structures, globs, or directories.")
    parser.add_argument("--pdb", help="PDB file to use with an explicit .inp file.")
    parser.add_argument("--output-dir", default=".", help="Directory for auto-generated .inp files.")
    parser.add_argument("--generate-inp-only", "--inp-only", action="store_true", help="Generate inferred Curves .inp files and exit without running analysis.")
    parser.add_argument("--no-output", action="store_true", help="Run calculations without printing the Curves report.")
    parser.add_argument("--format", choices=["curves", "json", "csv"], default="curves", help="Output format (csv requires Pandas).")
    parser.add_argument("--output-file", help="Write the selected output format to a file instead of stdout (for csv, writes multiple files based on prefix).")
    parser.add_argument("--no-annotations", action="store_true", help="Suppress the pyCurves |M| annotation report and annotation records.")
    parser.add_argument("--visualization", "--visualize", action="store_true", help="Include HTML-viewer geometry in JSON output.")
    parser.add_argument("--verbose-opt", action="store_true", help="Echo fitting and minimization progress while analyzing.")
    parser.add_argument("--quiet-opt", action="store_true", help=argparse.SUPPRESS)
    add_pycurves_analysis_options(parser)
    args = parser.parse_args()

    if not args.input:
        prompt = "PDB/mmCIF file to generate .inp for: " if args.generate_inp_only else "PDB or .inp file to analyze: "
        input_path = input(prompt).strip()
        if input_path:
            args.input = [input_path]

    if not args.input:
        raise SystemExit("No input file provided.")
    
    runner_kwargs = pycurves_runner_kwargs(args)
    mini_override = resolved_mini(args, default=True)

    if args.generate_inp_only:
        _generate_inp_only(args, runner_kwargs)
        return

    if len(args.input) != 1:
        raise SystemExit("Multiple inputs are only supported with --generate-inp-only.")

    path = Path(args.input[0])
    
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
