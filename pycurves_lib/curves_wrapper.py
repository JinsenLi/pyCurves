import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
import numpy as np
from pycurves_lib.io.base_reference import BaseReferenceLibrary
from pycurves_lib.io.curves_mol_loader import MolecularLoader
from pycurves_lib.core.curves_dataclasses import (
    BaseGeometryConstants,
    BaseLocator,
    CurvesContext,
    MolecularStructure,
)
from pycurves_lib.io.curves_config_loader import ConfigLoader
from pycurves_lib.core.curves_analyzer import BackboneAnalyzer, HelicalOptimizer
from pycurves_lib.core.curves_calculator import HelicalCalculator
from pycurves_lib.io.curves_output import CurvesOutputFormatter
from pycurves_lib.topology.topology_inferrer import RobustTopologyInferrer
from pycurves_lib.topology.base_annotations import annotate_context


class CurvesWrapper:
    """High-level pyCurves runner.

    It can run from an existing Curves `.inp` file or infer one from a PDB.
    """

    def __init__(
        self,
        pdbfile: Optional[str] = None,
        inpfile: Optional[str] = None,
        output_dir: str = ".",
        continuous_strands: bool = False,
        frame_convention: str = "legacy",
        axis_convention: str = "legacy",
        fit_override: Optional[bool] = None,
        grv_override: Optional[bool] = None,
        mini_override: Optional[bool] = None,
        comb_override: Optional[bool] = None,
        ends_override: Optional[bool] = None,
    ):
        if pdbfile is None and inpfile is None:
            raise ValueError("Provide at least a PDB file or an input file.")

        self.pdbfile = pdbfile
        self.inpfile = inpfile
        self.output_dir = output_dir
        self.continuous_strands = continuous_strands
        self.frame_convention, self.axis_convention = self.normalize_conventions(frame_convention, axis_convention)
        self.fit_override = fit_override
        self.grv_override = grv_override
        self.mini_override = mini_override
        self.comb_override = comb_override
        self.ends_override = ends_override
        self.generated_inpfiles: List[str] = []

        self.cfg = None
        self.ctx = None
        self.bak = None
        self.opt = None
        self.calc = None
        self.analysis_log = ""
        self._parsed_config_cache = None
        self._parsed_config_cache_key = None
        self._reference_library_cache = {}

        if self.inpfile is None:
            self.generated_inpfiles = self.generate_inp(pdbfile=self.pdbfile, output_dir=output_dir, continuous_strands=continuous_strands)
            if not self.generated_inpfiles:
                raise ValueError(f"Could not infer a Curves input file from {self.pdbfile!r}.")
            self.inpfile = self.generated_inpfiles[0]

    @classmethod
    def from_file(cls, path: str, output_dir: str = ".", continuous_strands: bool = False, frame_convention: str = "legacy", axis_convention: str = "legacy"):
        suffix = Path(path).suffix.lower()
        if suffix == ".inp":
            pdbfile = cls._pdbfile_from_inp(path)
            return cls(pdbfile=pdbfile, inpfile=path, output_dir=output_dir, continuous_strands=continuous_strands, frame_convention=frame_convention, axis_convention=axis_convention)
        return cls(pdbfile=path, output_dir=output_dir, continuous_strands=continuous_strands, frame_convention=frame_convention, axis_convention=axis_convention)

    def analyze(
        self,
        inpfile: Optional[str] = None,
        pdbfile: Optional[str] = None,
        mini: bool = True,
        verbose: bool = False,
        continuous_strands: Optional[bool] = None,
        frame_convention: Optional[str] = None,
        axis_convention: Optional[str] = None,
    ):
        if inpfile is not None:
            self.inpfile = inpfile
        if pdbfile is not None:
            self.pdbfile = pdbfile
        if continuous_strands is not None:
            self.continuous_strands = continuous_strands
        next_frame_convention = self.frame_convention if frame_convention is None else frame_convention
        next_axis_convention = self.axis_convention if axis_convention is None else axis_convention
        self.frame_convention, self.axis_convention = self.normalize_conventions(
            next_frame_convention,
            next_axis_convention,
        )
        if self.mini_override is not None:
            mini = self.mini_override
            
        if self.inpfile is None:
            self.generated_inpfiles = self.generate_inp(pdbfile=self.pdbfile, output_dir=self.output_dir, continuous_strands=self.continuous_strands)
            self.inpfile = self.generated_inpfiles[0]

        self.cfg = self._load_config()

        if self.pdbfile is None:
            self.pdbfile = self._pdbfile_from_config(self.cfg) or self._pdbfile_from_inp(self.inpfile)
        if self.pdbfile is None:
            raise ValueError("No PDB file is available. Pass pdbfile=... or include file=... in the .inp.")

        self.ctx = CurvesContext(self.cfg)
        MolecularLoader.load(self.pdbfile, self.ctx)

        return self._analyze_loaded_context(mini=mini, verbose=verbose)

    def analyze_molecule(
        self,
        molecule: MolecularStructure,
        inpfile: Optional[str] = None,
        mini: bool = True,
        verbose: bool = False,
        prev_opt_helical: Optional[np.ndarray] = None,
        axis_sign_reference: Optional[np.ndarray] = None,
    ):
        """Run pyCurves on an already populated MolecularStructure.

        This is intended for trajectory workflows where atom metadata and
        topology stay fixed while coordinates change frame by frame.
        """
        if inpfile is not None:
            self.inpfile = inpfile
        if self.mini_override is not None:
            mini = self.mini_override
        if self.inpfile is None:
            raise ValueError("analyze_molecule requires an existing Curves .inp file.")

        self.cfg = self._load_config()
        self.ctx = CurvesContext(self.cfg)
        self.ctx.molecule = molecule
        return self._analyze_loaded_context(
            mini=mini,
            verbose=verbose,
            prev_opt_helical=prev_opt_helical,
            axis_sign_reference=axis_sign_reference,
        )

    def _analyze_loaded_context(
        self,
        mini: bool = True,
        verbose: bool = False,
        prev_opt_helical: Optional[np.ndarray] = None,
        axis_sign_reference: Optional[np.ndarray] = None,
    ):
        self._validate_supported_legacy_options(mini=mini)
        log_parts = []
        curvesplus_axis = str(getattr(self.ctx.cfg, "axis_convention", self.axis_convention)).lower() == "curvesplus"
        if curvesplus_axis:
            # Curves+ axis mode is derived from standard base-pair frames and
            # smooth.f-style axis construction, not the legacy minimizer.
            mini = False
            self.ctx.cfg.mini = False

        reference_library = self._reference_library(getattr(self.ctx.cfg, "frame_convention", self.frame_convention))
        self.ctx.base_reference_library = reference_library
        locator = BaseLocator(BaseGeometryConstants(), reference_library=reference_library)
        log_parts.append(self._capture_call(lambda: locator.locate_all(self.ctx), echo=verbose))
        annotate_context(self.ctx)

        self.bak = BackboneAnalyzer()
        log_parts.append(self._capture_call(lambda: self.bak.analyze(self.ctx), echo=verbose))

        if prev_opt_helical is not None:
            self.ctx.params.helical = prev_opt_helical.copy()
        if axis_sign_reference is not None:
            self.ctx.axis_direction_sign_reference = np.asarray(axis_sign_reference, dtype=int).copy()

        if mini:
            from pycurves_lib.core.curves_optimizer_jax import HelicalOptimizerJAX
            self.opt = HelicalOptimizerJAX(self.ctx)
        else:
            self.opt = HelicalOptimizer(self.ctx)
        log_parts.append(self._capture_call(lambda: self.opt.print_fortran_setup_report(self.ctx), echo=verbose))
        log_parts.append(self._capture_call(lambda: self.opt.run(mini=mini), echo=verbose))
        if mini:
            log_parts.append(self._capture_call(lambda: self.opt.print_final_report(), echo=verbose))
        self.analysis_log = "".join(log_parts)

        self.calc = HelicalCalculator(self.ctx, self.opt)
        self.calc.calculate_all()
        return self

    def _validate_supported_legacy_options(self, mini: bool = True) -> None:
        """Reject legacy modes that would currently produce misleading results."""
        cfg = self.ctx.cfg
        if cfg.line:
            raise NotImplementedError(
                "line=.t. is not implemented in the current JAX optimizer. "
                "Curves 5.3 treats it as a distinct straight-axis minimization mode."
            )
        if cfg.dinu:
            raise NotImplementedError(
                "dinu=.t. is not implemented in the current JAX optimizer. "
                "Curves 5.3 changes the objective from adjacent-base to dinucleotide terms."
            )

    def _load_config(self) -> dict:
        """Return a fresh runtime config from a cached parse of the .inp file."""
        if self.inpfile is None:
            raise ValueError("No Curves input file is available.")
        cache_key = (
            str(Path(self.inpfile).resolve()),
            tuple(sorted(self._config_overrides().items())),
        )
        if cache_key != self._parsed_config_cache_key:
            self._parsed_config_cache = ConfigLoader.parse_inp(
                self.inpfile,
                config_overrides=self._config_overrides(),
            )
            self._parsed_config_cache_key = cache_key
        return copy.deepcopy(self._parsed_config_cache)

    def _reference_library(self, convention: str):
        convention = self._normalize_frame_convention(convention)
        if convention not in self._reference_library_cache:
            self._reference_library_cache[convention] = BaseReferenceLibrary.load(convention)
        return self._reference_library_cache[convention]

    def _config_overrides(self) -> dict:
        return {
            "fit": self.fit_override,
            "grv": self.grv_override,
            "mini": self.mini_override,
            "comb": self.comb_override,
            "ends": self.ends_override,
            "frame_convention": self.frame_convention,
            "axis_convention": self.axis_convention,
        }

    def run(self, output: bool = True, mini: bool = True, verbose: bool = False, output_format: str = "curves", annotations: bool = True, visualization: bool = False):
        if self.mini_override is not None:
            mini = self.mini_override
        self.analyze(mini=mini, verbose=verbose)
        if output:
            self.output(fmt=output_format, annotations=annotations, visualization=visualization)
        return self

    def output(self, fmt: str = "curves", file=None, annotations: bool = True, visualization: bool = False) -> str:
        if self.ctx is None or self.opt is None or self.calc is None:
            raise RuntimeError("Call analyze() before output().")
        text = CurvesOutputFormatter(self, annotations=annotations, visualization=visualization).render(fmt)
        if file is None:
            print(text, end="")
        else:
            Path(file).write_text(text, encoding="utf-8")
        return text

    def write_output(self, path: str, fmt: str = "curves", annotations: bool = True, visualization: bool = False) -> str:
        return self.output(fmt=fmt, file=path, annotations=annotations, visualization=visualization)

    def getFeatures(self):
        if self.calc is None:
            raise RuntimeError("Call analyze() before getFeatures().")
        return {
            "helical": self.ctx.params.helical,
            "inter_base": self.ctx.params.inter_base,
            "global_axis_steps": self.calc.global_axis_steps,
            "vkin": self.calc.vkin,  # Backward-compatible Fortran name.
            "local_inter_base": self.calc.local_inter_base,
            "local_inter_base_pair": self.calc.local_inter_base_pair,
            "bend": self.calc.bend,
            "annotations": CurvesOutputFormatter(self, annotations=True)._annotations(),
        }

    @staticmethod
    def _capture_call(func, echo: bool = False) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            func()
        text = buf.getvalue()
        if echo and text:
            print(text, end="")
        return text

    def generate_inp(
        self,
        pdbfile: Optional[str] = None,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        continuous_strands: bool = False,
    ) -> List[str]:
        pdbfile = pdbfile or self.pdbfile
        if pdbfile is None:
            raise ValueError("generate_inp requires a PDB file.")

        output_dir = output_dir or self.output_dir
        mol_holder = SimpleNamespace(molecule=MolecularStructure())
        MolecularLoader.load(pdbfile, mol_holder)
        inferrer = RobustTopologyInferrer(mol_holder.molecule, pdbfile=Path(pdbfile).name)
        stem = prefix or f"{Path(pdbfile).stem}_auto"
        return inferrer.write_inp_files(
            output_dir=output_dir,
            prefix=stem,
            continuous_strands=continuous_strands,
            fit_override=getattr(self, "fit_override", None),
            grv_override=getattr(self, "grv_override", None),
            comb_override=getattr(self, "comb_override", None),
            ends_override=getattr(self, "ends_override", None),
        )

    @staticmethod
    def _normalize_frame_convention(value: str) -> str:
        normalized = str(value or "legacy").strip().lower().replace("-", "_")
        if normalized in {"legacy"}:
            return "legacy"
        if normalized in {"standard", "curves_plus", "curves+", "curvesplus", "x3dna", "3dna"}:
            return "standard"
        raise ValueError(f"Unknown frame convention {value!r}; use 'legacy' or 'standard'.")

    @staticmethod
    def _normalize_axis_convention(value: str) -> str:
        normalized = str(value or "legacy").strip().lower().replace("-", "_")
        if normalized in {"legacy", "pycurves"}:
            return "legacy"
        if normalized in {"curves_plus", "curves+", "curvesplus", "canal"}:
            return "curvesplus"
        raise ValueError(f"Unknown axis convention {value!r}; use 'legacy' or 'curvesplus'.")

    @classmethod
    def normalize_conventions(cls, frame_convention: str, axis_convention: str):
        frame = cls._normalize_frame_convention(frame_convention)
        axis = cls._normalize_axis_convention(axis_convention)
        if axis == "curvesplus":
            # Curves+ axis/smooth construction is defined on standard base-pair
            # reference frames.  Treat axis_convention=curvesplus as a complete
            # Curves+ mode request so callers cannot accidentally disable the
            # legacy minimizer while still using legacy base frames.
            frame = "standard"
        return frame, axis

    @staticmethod
    def _pdbfile_from_config(cfg) -> Optional[str]:
        # ConfigLoader currently does not preserve namelist file=..., so this is
        # a placeholder for future structured namelist support.
        return None

    @staticmethod
    def _pdbfile_from_inp(inpfile: str) -> Optional[str]:
        inp_path = Path(inpfile)
        text = inp_path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        key = "file="
        pos = lowered.find(key)
        if pos < 0:
            return None
        rest = text[pos + len(key):]
        value = rest.split(",", 1)[0].split()[0].strip().strip("'\"")
        if not value:
            return None
        pdb_path = Path(value)
        if pdb_path.is_absolute():
            return str(pdb_path)
        sibling = inp_path.parent / pdb_path
        if sibling.exists():
            return str(sibling)
        return value
