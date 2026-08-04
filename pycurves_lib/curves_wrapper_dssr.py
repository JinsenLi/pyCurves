"""Public pyCurves runner with optional DSSR topology input."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from pycurves_lib.core.curves_dataclasses import MolecularStructure
from pycurves_lib.curves_wrapper_core import CurvesWrapper as _CoreCurvesWrapper
from pycurves_lib.io.curves_mol_loader import MolecularLoader
from pycurves_lib.io.dssr_json import DSSRDocument
from pycurves_lib.topology.dssr_topology_linear import DSSRTopologyBuilder


class CurvesWrapper(_CoreCurvesWrapper):
    """High-level runner supporting coordinates plus an optional DSSR report.

    DSSR supplies residue pairing and the selected stem/helix topology. Curves
    analysis options and the coordinate structure remain pyCurves inputs.
    """

    def __init__(
        self,
        *args,
        dssr_json: Optional[str] = None,
        dssr_unit: Optional[str] = None,
        **kwargs,
    ):
        positional_inp = args[1] if len(args) >= 2 else None
        explicit_inp = kwargs.get("inpfile", positional_inp)
        if dssr_json and explicit_inp:
            raise ValueError("Use either an explicit Curves .inp file or --dssr-json, not both.")
        continuous = kwargs.get("continuous_strands", args[3] if len(args) >= 4 else False)
        if dssr_json and continuous:
            raise ValueError(
                "--continuous-strands cannot be combined with DSSR topology; select a DSSR stem or helix explicitly."
            )

        self.dssr_json = str(dssr_json) if dssr_json else None
        self.dssr_unit = str(dssr_unit) if dssr_unit else None
        self.dssr_document = DSSRDocument.load(self.dssr_json) if self.dssr_json else None
        self.dssr_provenance = None
        self.dssr_source_base_pairs = []
        super().__init__(*args, **kwargs)

    @classmethod
    def from_file(
        cls,
        path: str,
        output_dir: str = ".",
        continuous_strands: bool = False,
        altloc: Optional[str] = None,
        frame_convention: str = "standard",
        axis_convention: str = "legacy",
        dssr_json: Optional[str] = None,
        dssr_unit: Optional[str] = None,
        **kwargs,
    ):
        suffix = Path(path).suffix.lower()
        if suffix == ".inp":
            if dssr_json:
                raise ValueError("Use either a Curves .inp file or dssr_json, not both.")
            pdbfile = cls._pdbfile_from_inp(path)
            return cls(
                pdbfile=pdbfile,
                inpfile=path,
                output_dir=output_dir,
                continuous_strands=continuous_strands,
                altloc=altloc,
                frame_convention=frame_convention,
                axis_convention=axis_convention,
                **kwargs,
            )
        return cls(
            pdbfile=path,
            output_dir=output_dir,
            continuous_strands=continuous_strands,
            altloc=altloc,
            frame_convention=frame_convention,
            axis_convention=axis_convention,
            dssr_json=dssr_json,
            dssr_unit=dssr_unit,
            **kwargs,
        )

    def generate_inp(
        self,
        pdbfile: Optional[str] = None,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        continuous_strands: bool = False,
        altloc: Optional[str] = None,
    ):
        if self.dssr_document is None:
            return super().generate_inp(
                pdbfile=pdbfile,
                output_dir=output_dir,
                prefix=prefix,
                continuous_strands=continuous_strands,
                altloc=altloc,
            )
        if continuous_strands:
            raise ValueError(
                "continuous_strands cannot be combined with a DSSR-selected topology."
            )

        pdbfile = pdbfile or self.pdbfile
        if pdbfile is None:
            raise ValueError("DSSR topology generation requires a coordinate structure file.")
        output_root = Path(output_dir or self.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        selected_altloc = self.altloc if altloc is None else MolecularLoader.normalize_altloc(altloc)
        holder = SimpleNamespace(molecule=MolecularStructure())
        MolecularLoader.load(pdbfile, holder, altloc=selected_altloc)
        builder = DSSRTopologyBuilder(
            holder.molecule,
            self.dssr_document,
            pdbfile=pdbfile,
        )
        result = builder.build(self.dssr_unit)
        topology = result.topology
        for attribute, override in (
            ("fit", getattr(self, "fit_override", None)),
            ("grv", getattr(self, "grv_override", None)),
            ("comb", getattr(self, "comb_override", None)),
            ("ends", getattr(self, "ends_override", None)),
        ):
            if override is not None:
                setattr(topology, attribute, override)

        stem = prefix or f"{Path(pdbfile).stem}_auto"
        output_path = output_root / f"{stem}_dssr_{result.unit.kind}{result.unit.index}.inp"
        topology.output_prefix = output_path.stem
        output_path.write_text(topology.to_inp_text(), encoding="utf-8")
        self.dssr_unit = result.unit.selector
        self.dssr_provenance = result.provenance
        self.dssr_source_base_pairs = list(result.source_base_pairs)
        return [str(output_path)]

    def list_dssr_units(self, pdbfile: Optional[str] = None):
        if self.dssr_document is None:
            raise ValueError("list_dssr_units requires dssr_json=... or --dssr-json.")
        pdbfile = pdbfile or self.pdbfile
        if pdbfile is None:
            raise ValueError("Listing DSSR units requires a coordinate structure file.")
        holder = SimpleNamespace(molecule=MolecularStructure())
        MolecularLoader.load(pdbfile, holder, altloc=self.altloc)
        return DSSRTopologyBuilder(
            holder.molecule,
            self.dssr_document,
            pdbfile=pdbfile,
        ).unit_summaries()

    def attach_dssr_annotations(self, molecule: MolecularStructure) -> None:
        if not self.dssr_source_base_pairs:
            return
        existing = [
            row for row in list(getattr(molecule, "source_base_pairs", None) or [])
            if row.get("source") != "DSSR JSON"
        ]
        molecule.source_base_pairs = existing + [dict(row) for row in self.dssr_source_base_pairs]

    def analyze(self, *args, **kwargs):
        result = super().analyze(*args, **kwargs)
        if self.ctx is not None:
            self.attach_dssr_annotations(self.ctx.molecule)
        return result

    def analyze_molecule(self, molecule: MolecularStructure, *args, **kwargs):
        self.attach_dssr_annotations(molecule)
        return super().analyze_molecule(molecule, *args, **kwargs)


__all__ = ["CurvesWrapper"]
