# CLI Script Reference

## `pycurves_main.py`

Implements the static command-line workflow:

- parses coordinate/`.inp` inputs and output options;
- expands files, directories, and glob patterns in `--generate-inp-only` mode;
- generates collision-free `.inp` prefixes for multi-file runs;
- builds `CurvesWrapper`, runs the requested calculation, and sends diagnostic
  text to the appropriate stream so JSON/CSV output stays clean;
- writes Curves text or JSON directly, or one CSV per result table.

Only input-generation mode accepts multiple inputs. Normal analysis requires
one coordinate or `.inp` file.

## `pycurves_cli_options_core.py`

Defines options shared by static and trajectory analysis: continuous strands,
alternate conformations, fitting, grooves, minimization, combined-strand and
terminal-level behavior, plus frame and axis conventions.
`pycurves_runner_kwargs()` translates an argparse namespace into wrapper keyword
arguments.

## `pycurves_cli_options.py`

Public extension of the core option set. It adds `--dssr-json` and
`--dssr-unit`, then includes those values in the wrapper keyword dictionary.

## `__init__.py`

Package marker only; callers import the option helpers or `main` from their
defining modules.

## Dispatch flow

```text
pycurves.py
  -> pycurves_main.main()
     -> add_pycurves_analysis_options()
     -> CurvesWrapper(...)
     -> runner.run(output=False)
     -> runner.output(...) or dataframe CSV files
```
