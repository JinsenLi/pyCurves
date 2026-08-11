# Command-Line Layer

This package defines the `pycurves` static-structure command. It keeps shared
analysis options separate from command dispatch so the trajectory CLI can reuse
the same meanings and defaults.

See [`SCRIPT_REFERENCE.md`](SCRIPT_REFERENCE.md) for argument ownership and the
dispatch flow.
