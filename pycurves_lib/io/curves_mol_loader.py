import gzip
from typing import TYPE_CHECKING
import warnings

import gemmi
import numpy as np
from pycurves_lib.data.modified_bases import is_known_modified_base, parent_base_name

if TYPE_CHECKING:
    from pycurves_lib.core.curves_dataclasses import CurvesContext


class MolecularLoader:
    """
    Replaces the functionality of input.f.
    Responsible for parsing coordinate files and populating MolecularStructure.
    """

    # Element lookup table simplified from Fortran 'nam' and 'ind' [cite: 2, 12]
    ELEMENT_MAP = {
        'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
        # ... Add other elements as needed.
        # Curves specific: 'M' maps to Carbon (6), 'W' maps to Oxygen (8).
        'M': 6, 'W': 8
    }
    DNA_BASES = {"A", "C", "G", "T", "U", "I", "P", "Y", "R"}
    PAIR_TYPES = {frozenset(("A", "T")), frozenset(("A", "U")), frozenset(("G", "C"))}
    BASE_ATOMS = {
        "A": {"N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"},
        "G": {"N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"},
        "C": {"N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"},
        "T": {"N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"},
        "U": {"N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"},
    }
    @staticmethod
    def _standardize_and_map(context: 'CurvesContext'):
        mol = context.molecule
        standardized_names = []
        element_indices = []

        for name in mol.atom_names:
            while len(name) > 0:
                first_char = name[0]
                if first_char == ' ' or ('0' <= first_char <= '9'):
                    name = name[1:]
                    continue
                break
            clean_name = name.strip()
            standardized_names.append(clean_name)

            first_char = "".join(filter(str.isalpha, clean_name))[:1].upper()
            element_indices.append(MolecularLoader.ELEMENT_MAP.get(first_char, 0))

        mol.atom_names = np.array(standardized_names)
        mol.atom_types = np.array(element_indices)

    @staticmethod
    def _open_file(file_path: str):
        if file_path.lower().endswith('.gz'):
            return gzip.open(file_path, 'rt')
        return open(file_path, 'r')

    @staticmethod
    def load(file_path: str, context: 'CurvesContext', altloc=None):
        """
        Main entry point for loading molecular data.
        """
        altloc = MolecularLoader.normalize_altloc(altloc)
        file_lower = file_path.lower()

        if file_lower.endswith('.pdb') or file_lower.endswith('.pdb.gz') or file_lower.endswith('.brk'):
            MolecularLoader._read_pdb(file_path, context, altloc=altloc)
        elif file_lower.endswith('.cif') or file_lower.endswith('.cif.gz'):
            MolecularLoader._read_cif(file_path, context, altloc=altloc)
        else:
            raise ValueError(
                f"Unsupported geometry file type: {file_path}. "
                "Use PDB, PDB.GZ, BRK, CIF, or CIF.GZ."
            )

        # Post-processing equivalent to Fortran loops 10 and 12
        MolecularLoader._standardize_and_map(context)
        # Identify subunit boundaries (ncen)
        MolecularLoader._find_subunits(context)
        MolecularLoader._build_connectivity(context)

    @staticmethod
    def normalize_altloc(altloc):
        """Normalize an optional explicit alternate-conformation identifier."""
        if altloc is None:
            return None
        value = str(altloc).strip()
        if value.lower() in {"", "first", "default"}:
            return None
        if len(value) != 1 or value in {".", "?", "*"}:
            raise ValueError(
                f"Invalid alternate-conformation identifier {altloc!r}; "
                "use one character such as 'A' or 'B', or omit it for Gemmi's first conformer."
            )
        return value

    @staticmethod
    def _read_pdb(file_path: str, context: 'CurvesContext', altloc=None):
        """Read PDB with Gemmi, falling back to the legacy fixed-column parser."""
        try:
            MolecularLoader._read_pdb_gemmi(file_path, context, altloc=altloc)
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.warn(
                f"Gemmi could not read PDB file {file_path!r}; using the legacy "
                f"fixed-column parser ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            MolecularLoader._read_pdb_legacy(file_path, context, altloc=altloc)

    @staticmethod
    def _read_pdb_gemmi(file_path: str, context: 'CurvesContext', altloc=None):
        """Parse the first PDB model through Gemmi."""
        structure = gemmi.read_structure(file_path)
        if len(structure) == 0:
            raise ValueError("the file contains no structural models")

        atoms_data, available_altlocs = MolecularLoader._gemmi_first_model_atoms(
            structure,
            altloc=altloc,
            return_altlocs=True,
        )
        atoms_data = MolecularLoader._filter_unfit_modified_residues(atoms_data)
        if not atoms_data:
            raise ValueError("the first model contains no supported atom records")

        crystal_cell = None
        spacegroup_hm = ""
        if structure.cell.is_crystal():
            crystal_cell = MolecularLoader._gemmi_cell_tuple(structure)
            spacegroup_hm = structure.spacegroup_hm

        source_base_pairs = []
        atoms_data, source_base_pairs = MolecularLoader._append_detected_crystal_mates(
            atoms_data,
            source_base_pairs,
            crystal_cell,
            spacegroup_hm,
        )

        info = dict(structure.info)
        title = MolecularLoader._clean_title(info.get("_struct.title"))
        MolecularLoader._populate_molecule(
            context,
            atoms_data,
            title=title,
            crystal_cell=crystal_cell,
            spacegroup_hm=spacegroup_hm,
            source_base_pairs=source_base_pairs,
            altloc_selection=altloc,
            available_altlocs=available_altlocs,
        )

    @staticmethod
    def _read_pdb_legacy(file_path: str, context: 'CurvesContext', altloc=None):
        """
        Parse PDB ATOM records using the legacy fixed-column layout.

        Curves reads the coordinate file as a flat atom stream and reports all
        ATOM records, including protein atoms that are not part of the DNA
        analysis map. Keeping that behavior also preserves the original residue
        numbering used by the .inp file.
        """
        atoms_data = []
        title = "No Title"
        crystal_cell = None
        spacegroup_hm = ""
        source_atom_index = 0
        available_altlocs = set()

        with MolecularLoader._open_file(file_path) as handle:
            for line in handle:
                record = line[:6]
                if record.startswith("TITLE") and title == "No Title":
                    title = line[10:].strip() or title
                if record == "CRYST1":
                    try:
                        crystal_cell = (
                            float(line[6:15]),
                            float(line[15:24]),
                            float(line[24:33]),
                            float(line[33:40]),
                            float(line[40:47]),
                            float(line[47:54]),
                        )
                        spacegroup_hm = line[55:66].strip()
                    except ValueError:
                        crystal_cell = None
                        spacegroup_hm = ""

                is_atom = record == "ATOM  "
                is_hetatm = record == "HETATM"
                res_name = line[17:20].strip()

                if not is_atom:
                    if not (is_hetatm and is_known_modified_base(res_name)):
                        continue
                record_atom_index = source_atom_index
                source_atom_index += 1

                try:
                    res_id = int(line[22:26])
                    pos = [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                except ValueError:
                    continue
                atom_altloc = line[16].strip()
                if atom_altloc:
                    available_altlocs.add(atom_altloc)

                atoms_data.append({
                    'name': line[12:16],
                    'res_name': line[17:20].strip(),
                    'chain_id': line[21].strip(),
                    'res_id': res_id,
                    'pos': pos,
                    'het_flag': "H" if is_hetatm else "A",
                    'insertion_code': line[26].strip(),
                    'altloc': atom_altloc,
                    'occupancy': MolecularLoader._pdb_float(line[54:60], 1.0),
                    'model_id': "1",
                    'source_atom_index': record_atom_index,
                })
        atoms_data = MolecularLoader._select_atom_records_altloc(atoms_data, altloc)

        atoms_data = MolecularLoader._filter_unfit_modified_residues(atoms_data)
        source_base_pairs = []
        atoms_data, source_base_pairs = MolecularLoader._append_detected_crystal_mates(
            atoms_data,
            source_base_pairs,
            crystal_cell,
            spacegroup_hm,
        )

        MolecularLoader._populate_molecule(
            context,
            atoms_data,
            title=title,
            crystal_cell=crystal_cell,
            spacegroup_hm=spacegroup_hm,
            source_base_pairs=source_base_pairs,
            altloc_selection=altloc,
            available_altlocs=available_altlocs,
        )

    @staticmethod
    def _read_cif(file_path: str, context: 'CurvesContext', altloc=None):
        """
        Parse mmCIF files for atomic coordinates using gemmi.
        """
        doc = gemmi.cif.read_file(file_path)
        if not doc:
            raise ValueError(f"Empty CIF file: {file_path}")

        block = doc[-1]
        title = MolecularLoader._clean_title(block.find_value('_struct.title'))

        st = gemmi.make_structure_from_block(block)

        atoms_data, available_altlocs = MolecularLoader._gemmi_first_model_atoms(
            st,
            altloc=altloc,
            return_altlocs=True,
        )

        atoms_data = MolecularLoader._filter_unfit_modified_residues(atoms_data)
        source_base_pairs = MolecularLoader._read_cif_base_pair_annotations(block)
        atoms_data, source_base_pairs = MolecularLoader._append_cif_symmetry_mates(
            atoms_data,
            source_base_pairs,
            st,
            block,
        )

        MolecularLoader._populate_molecule(
            context,
            atoms_data,
            title=title,
            crystal_cell=MolecularLoader._gemmi_cell_tuple(st) if len(st) > 0 else None,
            spacegroup_hm=st.spacegroup_hm if len(st) > 0 else "",
            source_base_pairs=source_base_pairs,
            altloc_selection=altloc,
            available_altlocs=available_altlocs,
        )

    @staticmethod
    def _gemmi_first_model_atoms(structure, altloc=None, return_altlocs=False):
        """Return one coherent conformer from the first Gemmi model.

        With no explicit altloc, Gemmi keeps the first-listed conformer. An
        explicit identifier keeps blank/shared atoms plus that conformer.
        """
        atoms_data = []
        if len(structure) == 0:
            return (atoms_data, ()) if return_altlocs else atoms_data

        available_altlocs = tuple(sorted({
            str(atom.altloc)
            for chain in structure[0]
            for residue in chain
            for atom in residue
            if str(atom.altloc) not in {"", "\x00", " "}
        }))
        altloc = MolecularLoader.normalize_altloc(altloc)
        if altloc is not None and available_altlocs and altloc not in available_altlocs:
            raise ValueError(
                f"Alternate conformation {altloc!r} is not present in the first model; "
                f"available identifiers: {', '.join(available_altlocs)}."
            )
        first_source_indices = {}
        if altloc is None:
            original_indices = {}
            source_index = 0
            for chain in structure[0]:
                for residue in chain:
                    residue_key = (
                        chain.name,
                        int(residue.seqid.num),
                        str(residue.seqid.icode).strip(),
                        residue.name,
                    )
                    for atom in residue:
                        atom_altloc = str(atom.altloc).replace("\x00", "")
                        original_indices[(*residue_key, atom.name, atom_altloc)] = source_index
                        source_index += 1
                    for atom in residue.first_conformer():
                        atom_altloc = str(atom.altloc).replace("\x00", "")
                        first_source_indices[(*residue_key, atom.name)] = original_indices[
                            (*residue_key, atom.name, atom_altloc)
                        ]
            structure.remove_alternative_conformations()

        source_atom_index = 0
        model_id = str(getattr(structure[0], "name", getattr(structure[0], "num", 1)))
        for chain in structure[0]:
            for residue in chain:
                for atom in residue:
                    record_atom_index = source_atom_index
                    source_atom_index += 1
                    if altloc is None:
                        record_atom_index = first_source_indices.get(
                            (
                                chain.name,
                                int(residue.seqid.num),
                                str(residue.seqid.icode).strip(),
                                residue.name,
                                atom.name,
                            ),
                            record_atom_index,
                        )
                    atom_altloc = str(atom.altloc)
                    if atom_altloc == "\x00":
                        atom_altloc = ""
                    if altloc is not None and atom_altloc not in {"", altloc}:
                        continue
                    if residue.het_flag != "A" and not is_known_modified_base(residue.name):
                        continue
                    atoms_data.append({
                        "name": atom.name,
                        "res_name": residue.name,
                        "chain_id": chain.name,
                        "res_id": residue.seqid.num,
                        "pos": [atom.pos.x, atom.pos.y, atom.pos.z],
                        "het_flag": residue.het_flag,
                        "insertion_code": str(residue.seqid.icode).strip(),
                        "altloc": atom_altloc,
                        "occupancy": float(atom.occ),
                        "model_id": model_id,
                        "source_atom_index": record_atom_index,
                    })
        if return_altlocs:
            return atoms_data, available_altlocs
        return atoms_data

    @staticmethod
    def _gemmi_cell_tuple(structure):
        cell = structure.cell
        return cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma

    @staticmethod
    def _clean_title(value):
        title = str(value or "").strip(' \'"\n\r')
        return title or "No Title"

    @staticmethod
    def _populate_molecule(
        context,
        atoms_data,
        *,
        title,
        crystal_cell,
        spacegroup_hm,
        source_base_pairs,
        altloc_selection=None,
        available_altlocs=(),
    ):
        molecule = context.molecule
        molecule.mcode = title
        molecule.crystal_cell = crystal_cell
        molecule.spacegroup_hm = spacegroup_hm
        molecule.kam = len(atoms_data)
        molecule.atom_names = np.array([atom["name"] for atom in atoms_data])
        molecule.coordinates = np.array([atom["pos"] for atom in atoms_data])
        molecule.residue_names = np.array([atom["res_name"] for atom in atoms_data])
        molecule.residue_ids = np.array([atom["res_id"] for atom in atoms_data])
        molecule.chain_ids = np.array([atom["chain_id"] for atom in atoms_data])
        molecule.insertion_codes = np.array([atom.get("insertion_code", "") for atom in atoms_data])
        molecule.altlocs = np.array([atom.get("altloc", "") for atom in atoms_data])
        molecule.occupancies = np.array([atom.get("occupancy", 1.0) for atom in atoms_data], dtype=float)
        molecule.model_ids = np.array([atom.get("model_id", "1") for atom in atoms_data])
        molecule.source_atom_indices = np.array(
            [atom.get("source_atom_index", index) for index, atom in enumerate(atoms_data)],
            dtype=int,
        )
        molecule.altloc_selection = altloc_selection or "first"
        molecule.available_altlocs = tuple(available_altlocs)
        molecule.source_base_pairs = source_base_pairs

    @staticmethod
    def _pdb_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _select_atom_records_altloc(atoms_data, altloc):
        """Apply first-conformer or explicit-altloc selection to legacy PDB records."""
        altloc = MolecularLoader.normalize_altloc(altloc)
        if altloc is not None:
            available = {
                str(atom.get("altloc", "")).strip()
                for atom in atoms_data
                if str(atom.get("altloc", "")).strip()
            }
            if available and altloc not in available:
                raise ValueError(
                    f"Alternate conformation {altloc!r} is not present; "
                    f"available identifiers: {', '.join(sorted(available))}."
                )
            return [
                atom for atom in atoms_data
                if str(atom.get("altloc", "")).strip() in {"", altloc}
            ]

        first_by_residue = {}
        for atom in atoms_data:
            atom_altloc = str(atom.get("altloc", "")).strip()
            if not atom_altloc:
                continue
            key = (
                str(atom.get("chain_id", "")).strip(),
                int(atom.get("res_id", 0)),
                str(atom.get("insertion_code", "")).strip(),
                str(atom.get("res_name", "")).strip().upper(),
            )
            first_by_residue.setdefault(key, atom_altloc)

        selected = []
        for atom in atoms_data:
            atom_altloc = str(atom.get("altloc", "")).strip()
            key = (
                str(atom.get("chain_id", "")).strip(),
                int(atom.get("res_id", 0)),
                str(atom.get("insertion_code", "")).strip(),
                str(atom.get("res_name", "")).strip().upper(),
            )
            if not atom_altloc or atom_altloc == first_by_residue.get(key, atom_altloc):
                selected.append({**atom, "altloc": ""})
        return selected

    @staticmethod
    def _filter_unfit_modified_residues(atoms_data):
        """Drop HETATM linkers that NAKB maps to bases but cannot define a base frame.

        Some nucleotide linkers, for example YRR in 1BNK, are listed in the
        modified-base mapping because they replace a parent nucleotide.  They
        do not contain the base ring atoms needed for least-squares fitting,
        so counting them as Curves units shifts legacy .inp unit numbers and
        creates wrong pairings downstream.
        """
        grouped = {}
        order = []
        for atom in atoms_data:
            key = (
                str(atom.get("chain_id", "")).strip(),
                int(atom.get("res_id", 0)),
                str(atom.get("insertion_code", "")).strip(),
                str(atom.get("res_name", "")).strip().upper(),
                str(atom.get("het_flag", "")),
            )
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(atom)

        filtered = []
        for key in order:
            atoms = grouped[key]
            _, _, _, residue_name, het_flag = key
            if het_flag == "H" and is_known_modified_base(residue_name):
                atom_names = {MolecularLoader._clean_atom_name(atom["name"]) for atom in atoms}
                if not MolecularLoader._has_minimal_base_frame_atoms(residue_name, atom_names):
                    continue
            filtered.extend(atoms)
        return filtered

    @staticmethod
    def _has_minimal_base_frame_atoms(residue_name, atom_names):
        parent = parent_base_name(residue_name)
        if parent == "unknown":
            return False
        base = MolecularLoader._base_symbol(parent)
        if base not in MolecularLoader.DNA_BASES:
            return False

        atom_names = set(atom_names)
        ring_atoms = MolecularLoader.BASE_ATOMS.get(base, set())
        if not ring_atoms:
            return len(atom_names & {"N1", "N9", "C2", "C4", "C5", "C6"}) >= 3

        glycosidic_atom = "N9" if base in {"A", "G", "I", "R"} else "N1"
        has_glycosidic_anchor = glycosidic_atom in atom_names
        return has_glycosidic_anchor and len(atom_names & ring_atoms) >= 3

    @staticmethod
    def _read_cif_base_pair_annotations(block):
        """Read NDB base-pair annotations, if present in the mmCIF file."""
        table = block.find_mmcif_category('_ndb_struct_na_base_pair.')
        if not table:
            return []

        def value(row, tag, default=""):
            full_tag = f"_ndb_struct_na_base_pair.{tag}"
            try:
                raw = row[full_tag]
            except Exception:
                return default
            if raw in {None, "?", "."}:
                return default
            return str(raw).strip()

        def int_value(row, tag):
            raw = value(row, tag)
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None

        def float_value(row, tag):
            raw = value(row, tag)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        rows = []
        for row in table:
            hbond_type_12 = value(row, "hbond_type_12")
            hbond_type_28 = value(row, "hbond_type_28")
            is_hoogsteen = hbond_type_12 == "3" or hbond_type_28 in {"23", "24"}
            rows.append({
                "source": "mmcif_ndb_struct_na_base_pair",
                "pair_number": int_value(row, "pair_number"),
                "pair_name": value(row, "pair_name"),
                "i_chain_id": value(row, "i_auth_asym_id") or value(row, "i_label_asym_id"),
                "i_residue_id": int_value(row, "i_auth_seq_id") or int_value(row, "i_label_seq_id"),
                "i_residue_name": value(row, "i_label_comp_id"),
                "i_symmetry": value(row, "i_symmetry"),
                "j_chain_id": value(row, "j_auth_asym_id") or value(row, "j_label_asym_id"),
                "j_residue_id": int_value(row, "j_auth_seq_id") or int_value(row, "j_label_seq_id"),
                "j_residue_name": value(row, "j_label_comp_id"),
                "j_symmetry": value(row, "j_symmetry"),
                "hbond_type_28": hbond_type_28,
                "hbond_type_12": hbond_type_12,
                "shear": float_value(row, "shear"),
                "stretch": float_value(row, "stretch"),
                "stagger": float_value(row, "stagger"),
                "buckle": float_value(row, "buckle"),
                "propeller": float_value(row, "propeller"),
                "opening": float_value(row, "opening"),
                "pair_family": "hoogsteen" if is_hoogsteen else "watson_crick_or_other",
                "is_hoogsteen": is_hoogsteen,
                "shape_parameters_supported": True,
                "shape_skip_reason": "",
            })
        return rows

    @staticmethod
    def _append_cif_symmetry_mates(atoms_data, source_base_pairs, structure, block=None):
        """Append coordinate copies needed by NDB base-pair symmetry annotations."""
        if not source_base_pairs or len(structure) == 0:
            return atoms_data, source_base_pairs

        try:
            spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
            operations = list(spacegroup.operations())
        except Exception:
            return atoms_data, source_base_pairs
        if not operations:
            return atoms_data, source_base_pairs

        operator_transforms = MolecularLoader._cif_operator_transforms(block) if block is not None else {}

        residues = {}
        for atom in atoms_data:
            key = (str(atom["chain_id"]).strip(), int(atom["res_id"]), str(atom["res_name"]).strip().upper())
            residues.setdefault(key, []).append(atom)

        generated = {}
        expanded_atoms = list(atoms_data)
        for pair in source_base_pairs:
            for side in ("i", "j"):
                symmetry = pair.get(f"{side}_symmetry", "")
                if MolecularLoader._is_identity_symmetry(symmetry):
                    continue
                chain_id = str(pair.get(f"{side}_chain_id", "")).strip()
                residue_id = pair.get(f"{side}_residue_id")
                residue_name = str(pair.get(f"{side}_residue_name", "")).strip().upper()
                if residue_id is None:
                    continue

                source_key = (chain_id, int(residue_id), residue_name)
                source_atoms = residues.get(source_key)
                if not source_atoms:
                    continue

                generated_key = (chain_id, symmetry)
                generated_chain = generated.get(generated_key)
                if generated_chain is None:
                    generated_chain = MolecularLoader._generated_symmetry_chain(chain_id, symmetry)
                    transform = MolecularLoader._cif_symmetry_transform(
                        symmetry,
                        operations,
                        structure.cell,
                        operator_transforms,
                    )
                    if transform is None:
                        continue
                    c1_distance = MolecularLoader._annotated_pair_c1_distance(
                        pair,
                        side,
                        source_atoms,
                        residues,
                        transform,
                    )
                    if c1_distance is not None and not 4.0 <= c1_distance <= 25.0:
                        warnings.warn(
                            f"Rejected implausible symmetry mate {symmetry!r} for "
                            f"{pair.get('pair_name', 'annotated base pair')}: "
                            f"C1'-C1' distance is {c1_distance:.2f} A.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    generated[generated_key] = generated_chain
                    source_chain_atoms = [
                        atom for atom in atoms_data
                        if str(atom["chain_id"]).strip() == chain_id
                        and MolecularLoader._base_symbol(str(atom["res_name"])) in MolecularLoader.DNA_BASES
                    ]
                    for atom in source_chain_atoms:
                        pos = transform(atom["pos"])
                        expanded_atoms.append({
                            **atom,
                            "chain_id": generated_chain,
                            "pos": pos,
                        })

                pair[f"{side}_generated_chain_id"] = generated_chain

        return expanded_atoms, source_base_pairs

    @staticmethod
    def _cif_operator_transforms(block):
        """Read explicit Cartesian assembly/crystal operator matrices from mmCIF."""
        if block is None:
            return {}
        table = block.find_mmcif_category('_pdbx_struct_oper_list.')
        if not table:
            return {}

        def value(row, tag, default=""):
            full_tag = f"_pdbx_struct_oper_list.{tag}"
            try:
                raw = row[full_tag]
            except Exception:
                return default
            if raw in {None, "?", "."}:
                return default
            return str(raw).strip().strip("'\"")

        transforms = {"exact": {}, "by_operation_number": {}}
        for row in table:
            try:
                matrix = np.array([
                    [float(value(row, "matrix[1][1]")), float(value(row, "matrix[1][2]")), float(value(row, "matrix[1][3]"))],
                    [float(value(row, "matrix[2][1]")), float(value(row, "matrix[2][2]")), float(value(row, "matrix[2][3]"))],
                    [float(value(row, "matrix[3][1]")), float(value(row, "matrix[3][2]")), float(value(row, "matrix[3][3]"))],
                ], dtype=float)
                vector = np.array([
                    float(value(row, "vector[1]")),
                    float(value(row, "vector[2]")),
                    float(value(row, "vector[3]")),
                ], dtype=float)
            except (TypeError, ValueError):
                continue

            def make_transform(op_matrix, op_vector):
                def transform(position):
                    return (op_matrix @ np.asarray(position, dtype=float) + op_vector).tolist()
                return transform

            transform = make_transform(matrix, vector)
            operator_name = value(row, "name")
            for key in (value(row, "id"), operator_name, value(row, "symmetry_operation")):
                if key:
                    transforms["exact"][key] = transform
            operation_number = operator_name.split("_", 1)[0]
            if operation_number.isdigit():
                transforms["by_operation_number"].setdefault(operation_number, transform)
        return transforms

    @staticmethod
    def _annotated_pair_c1_distance(pair, transformed_side, source_atoms, residues, transform):
        """Return C1'-C1' distance to an identity-symmetry annotated partner."""
        other_side = "j" if transformed_side == "i" else "i"
        if not MolecularLoader._is_identity_symmetry(pair.get(f"{other_side}_symmetry", "")):
            return None
        other_residue_id = pair.get(f"{other_side}_residue_id")
        if other_residue_id is None:
            return None
        other_key = (
            str(pair.get(f"{other_side}_chain_id", "")).strip(),
            int(other_residue_id),
            str(pair.get(f"{other_side}_residue_name", "")).strip().upper(),
        )
        other_atoms = residues.get(other_key)
        if not other_atoms:
            return None

        def c1_position(atoms):
            for atom in atoms:
                if MolecularLoader._clean_atom_name(atom["name"]) in {"C1'", "C1*"}:
                    return np.asarray(atom["pos"], dtype=float)
            return None

        source_c1 = c1_position(source_atoms)
        other_c1 = c1_position(other_atoms)
        if source_c1 is None or other_c1 is None:
            return None
        transformed_c1 = np.asarray(transform(source_c1), dtype=float)
        return float(np.linalg.norm(transformed_c1 - other_c1))

    @staticmethod
    def _is_identity_symmetry(symmetry: str) -> bool:
        return str(symmetry).strip() in {"", ".", "?", "1_555"}

    @staticmethod
    def _generated_symmetry_chain(chain_id: str, symmetry: str) -> str:
        clean = "".join(ch if ch.isalnum() else "" for ch in str(symmetry))
        return f"{chain_id}_sym{clean}"

    @staticmethod
    def _cif_symmetry_transform(symmetry: str, operations, cell, operator_transforms=None):
        transform_maps = operator_transforms or {}
        exact_transforms = transform_maps.get("exact", transform_maps)
        symmetry_text = str(symmetry).strip()
        explicit_transform = exact_transforms.get(symmetry_text)
        if explicit_transform is not None:
            return explicit_transform

        try:
            op_text, translation_text = symmetry_text.split("_", 1)
            explicit_transform = transform_maps.get("by_operation_number", {}).get(op_text)
            if explicit_transform is not None:
                return explicit_transform
            op_index = int(op_text) - 1
            translation = [int(ch) - 5 for ch in translation_text[:3]]
            operation = operations[op_index]
        except Exception:
            return None

        def transform(position):
            frac = cell.fractionalize(gemmi.Position(*position))
            xyz = operation.apply_to_xyz([frac.x, frac.y, frac.z])
            out = cell.orthogonalize(gemmi.Fractional(
                xyz[0] + translation[0],
                xyz[1] + translation[1],
                xyz[2] + translation[2],
            ))
            return [out.x, out.y, out.z]

        return transform

    @staticmethod
    def _append_detected_crystal_mates(atoms_data, source_base_pairs, crystal_cell, spacegroup_hm):
        """Detect and append crystallographic DNA partners when no annotation table exists."""
        if not atoms_data or crystal_cell is None or not spacegroup_hm:
            return atoms_data, source_base_pairs

        explicit_strands = MolecularLoader._dna_strands_from_atoms(atoms_data)
        if not explicit_strands:
            return atoms_data, source_base_pairs
        if MolecularLoader._has_explicit_duplex(explicit_strands):
            return atoms_data, source_base_pairs

        try:
            cell = gemmi.UnitCell(*crystal_cell)
            spacegroup = gemmi.find_spacegroup_by_name(spacegroup_hm)
            operations = list(spacegroup.operations())
        except Exception:
            return atoms_data, source_base_pairs
        if len(operations) <= 1:
            return atoms_data, source_base_pairs

        expanded_atoms = list(atoms_data)
        generated_pairs = list(source_base_pairs)
        used_generated_chains = set()

        for strand in explicit_strands:
            best = MolecularLoader._best_symmetry_mate_for_strand(strand, operations, cell)
            if best is None:
                continue
            op_index, translation, mate_residues, pairs = best
            code = MolecularLoader._symmetry_code_from_operation(op_index, translation)
            generated_chain = MolecularLoader._generated_symmetry_chain(strand[0]["chain_id"], code)
            if generated_chain in used_generated_chains:
                continue
            used_generated_chains.add(generated_chain)

            for residue in mate_residues:
                for atom in residue["atoms"]:
                    expanded_atoms.append({
                        **atom,
                        "chain_id": generated_chain,
                        "pos": atom["sym_pos"],
                    })

            for pair_number, (left, right, distance) in enumerate(pairs, start=len(generated_pairs) + 1):
                hinfo = MolecularLoader._hoogsteen_geometry(left, right)
                generated_pairs.append({
                    "source": "geometry_inferred_crystal_symmetry",
                    "pair_number": pair_number,
                    "pair_name": f"{left['chain_id']}_{left['res_name']}{left['res_id']}:{right['res_name']}{right['res_id']}_{generated_chain}",
                    "i_chain_id": left["chain_id"],
                    "i_residue_id": left["res_id"],
                    "i_residue_name": left["res_name"],
                    "i_symmetry": "1_555",
                    "j_chain_id": right["chain_id"],
                    "j_generated_chain_id": generated_chain,
                    "j_residue_id": right["res_id"],
                    "j_residue_name": right["res_name"],
                    "j_symmetry": code,
                    "center_distance": float(distance),
                    "hbond_distances": hinfo["distances"],
                    "pair_family": "hoogsteen" if hinfo["is_hoogsteen"] else "geometry_inferred",
                    "is_hoogsteen": hinfo["is_hoogsteen"],
                    "shape_parameters_supported": True,
                    "shape_skip_reason": "",
                })

        return expanded_atoms, generated_pairs

    @staticmethod
    def _dna_strands_from_atoms(atoms_data):
        residues = []
        current_key = None
        current_atoms = []
        for atom in atoms_data:
            key = (str(atom["chain_id"]).strip(), int(atom["res_id"]), str(atom["res_name"]).strip().upper())
            if current_key is not None and key != current_key:
                residue = MolecularLoader._residue_record(current_key, current_atoms)
                if residue is not None:
                    residues.append(residue)
                current_atoms = []
            current_key = key
            current_atoms.append(atom)
        if current_key is not None:
            residue = MolecularLoader._residue_record(current_key, current_atoms)
            if residue is not None:
                residues.append(residue)

        by_chain = {}
        for residue in residues:
            by_chain.setdefault(residue["chain_id"], []).append(residue)

        strands = []
        for chain_residues in by_chain.values():
            ordered = sorted(chain_residues, key=lambda r: r["order"])
            current = []
            previous = None
            for residue in ordered:
                if previous is not None:
                    gap = residue["order"] - previous["order"]
                    distance = np.linalg.norm(residue["sugar_center"] - previous["sugar_center"])
                    if gap > 4 or distance > 12.0:
                        if len(current) >= 2:
                            strands.append(current)
                        current = []
                current.append(residue)
                previous = residue
            if len(current) >= 2:
                strands.append(current)
        return strands

    @staticmethod
    def _residue_record(key, atoms):
        chain_id, res_id, res_name = key
        base = MolecularLoader._base_symbol(res_name)
        if base not in MolecularLoader.DNA_BASES:
            return None

        atom_map = {MolecularLoader._clean_atom_name(atom["name"]): atom for atom in atoms}
        if "C1'" not in atom_map and "C1*" not in atom_map:
            return None

        base_atoms = [
            atom for name, atom in atom_map.items()
            if name in MolecularLoader.BASE_ATOMS.get(base, set()) or name in {"N1", "N9", "C2", "C4", "C5", "C6", "C8"}
        ]
        sugar_atoms = [
            atom for name, atom in atom_map.items()
            if name in {"C1'", "C1*", "C2'", "C2*", "C3'", "C3*", "C4'", "C4*", "O4'", "O4*"}
        ]
        if len(base_atoms) < 3 or not sugar_atoms:
            return None

        return {
            "chain_id": chain_id,
            "res_id": int(res_id),
            "res_name": res_name,
            "base": base,
            "order": int(res_id),
            "atoms": list(atoms),
            "atom_map": atom_map,
            "center": np.mean([atom["pos"] for atom in base_atoms], axis=0),
            "sugar_center": np.mean([atom["pos"] for atom in sugar_atoms], axis=0),
        }

    @staticmethod
    def _has_explicit_duplex(strands):
        for i in range(len(strands)):
            for j in range(i + 1, len(strands)):
                pairs = MolecularLoader._candidate_residue_pairs(strands[i], strands[j])
                if MolecularLoader._pair_density(pairs, strands[i], strands[j]) >= 0.45 or len(pairs) >= 4:
                    return True
        return False

    @staticmethod
    def _best_symmetry_mate_for_strand(strand, operations, cell):
        best = None
        translations = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        for op_index, operation in enumerate(operations, start=1):
            for translation in translations:
                if op_index == 1 and translation == (0, 0, 0):
                    continue
                mate = MolecularLoader._transform_residue_strand(strand, operation, translation, cell)
                pairs = MolecularLoader._candidate_residue_pairs(strand, mate)
                density = MolecularLoader._pair_density(pairs, strand, mate)
                if density < 0.45 and len(pairs) < 4:
                    continue
                avg_distance = float(np.mean([p[2] for p in pairs])) if pairs else float("inf")
                hoogsteen_count = sum(1 for left, right, _ in pairs if MolecularLoader._hoogsteen_geometry(left, right)["is_hoogsteen"])
                score = (len(pairs), hoogsteen_count, -avg_distance)
                if best is None or score > best[0]:
                    best = (score, op_index, translation, mate, pairs)
        if best is None:
            return None
        _, op_index, translation, mate, pairs = best
        return op_index, translation, mate, pairs

    @staticmethod
    def _transform_residue_strand(strand, operation, translation, cell):
        mate = []
        for residue in strand:
            atoms = []
            for atom in residue["atoms"]:
                sym_pos = MolecularLoader._apply_crystal_operation(atom["pos"], operation, translation, cell)
                atoms.append({**atom, "sym_pos": sym_pos, "pos": sym_pos})
            transformed = MolecularLoader._residue_record(
                (residue["chain_id"], residue["res_id"], residue["res_name"]),
                atoms,
            )
            if transformed is not None:
                mate.append(transformed)
        return mate

    @staticmethod
    def _apply_crystal_operation(position, operation, translation, cell):
        frac = cell.fractionalize(gemmi.Position(*position))
        xyz = operation.apply_to_xyz([frac.x, frac.y, frac.z])
        out = cell.orthogonalize(gemmi.Fractional(
            xyz[0] + translation[0],
            xyz[1] + translation[1],
            xyz[2] + translation[2],
        ))
        return [out.x, out.y, out.z]

    @staticmethod
    def _candidate_residue_pairs(first, second):
        pairs = []
        used_second = set()
        for left in first:
            best = None
            for idx, right in enumerate(second):
                if idx in used_second:
                    continue
                distance = float(np.linalg.norm(left["center"] - right["center"]))
                if distance > 9.5:
                    continue
                if not MolecularLoader._is_complementary(left["base"], right["base"]):
                    continue
                if best is None or distance < best[2]:
                    best = (left, right, distance, idx)
            if best is not None:
                used_second.add(best[3])
                pairs.append((best[0], best[1], best[2]))
        return pairs

    @staticmethod
    def _pair_density(pairs, first, second):
        denom = max(1, min(len(first), len(second)))
        return len(pairs) / denom

    @staticmethod
    def _hoogsteen_geometry(first, second):
        bases = {first["base"], second["base"]}
        distances = {}
        if bases <= {"A", "T", "U"} and "A" in bases:
            adenine = first if first["base"] == "A" else second
            pyrimidine = second if adenine is first else first
            n7_n3 = MolecularLoader._atom_distance(adenine, "N7", pyrimidine, "N3")
            n6_o4 = MolecularLoader._atom_distance(adenine, "N6", pyrimidine, "O4")
            if n6_o4 is None and pyrimidine["base"] == "U":
                n6_o4 = MolecularLoader._atom_distance(adenine, "N6", pyrimidine, "O4")
            n1_n3 = MolecularLoader._atom_distance(adenine, "N1", pyrimidine, "N3")
            distances = {"A_N7_to_T_N3": n7_n3, "A_N6_to_T_O4": n6_o4, "A_N1_to_T_N3": n1_n3}
            is_hoogsteen = (
                n7_n3 is not None and n6_o4 is not None
                and n7_n3 <= 3.7 and n6_o4 <= 3.7
                and (n1_n3 is None or n1_n3 >= 3.3 or n7_n3 + 0.6 < n1_n3)
            )
            return {"is_hoogsteen": bool(is_hoogsteen), "distances": distances}
        return {"is_hoogsteen": False, "distances": distances}

    @staticmethod
    def _atom_distance(first, first_atom, second, second_atom):
        left = first["atom_map"].get(first_atom)
        right = second["atom_map"].get(second_atom)
        if left is None or right is None:
            return None
        return float(np.linalg.norm(np.asarray(left["pos"], dtype=float) - np.asarray(right["pos"], dtype=float)))

    @staticmethod
    def _symmetry_code_from_operation(op_index, translation):
        return f"{op_index}_{translation[0] + 5}{translation[1] + 5}{translation[2] + 5}"

    @staticmethod
    def _clean_atom_name(name):
        clean = str(name)
        while clean and (clean[0] == " " or clean[0].isdigit()):
            clean = clean[1:]
        return clean.strip().upper()

    @staticmethod
    def _base_symbol(res_name: str) -> str:
        name = parent_base_name(res_name)
        if name == "unknown":
            return "X"
        if len(name) >= 2 and name[0] in {"D", "R"} and name[1] in MolecularLoader.DNA_BASES:
            return name[1]
        for char in name:
            if char in MolecularLoader.DNA_BASES:
                return char
        return name[:1]

    @staticmethod
    def _is_complementary(first: str, second: str) -> bool:
        return frozenset((first, second)) in MolecularLoader.PAIR_TYPES

    @staticmethod
    def _find_subunits(context: 'CurvesContext'):
        """
        Detects subunit boundaries based on residue name/ID changes [cite: 9-10].
        Replaces the 'ncen' discovery loop.
        """
        mol = context.molecule
        res_names = mol.residue_names
        res_ids = mol.residue_ids
        chain_ids = mol.chain_ids if mol.chain_ids is not None else np.full(mol.kam, "")
        insertion_codes = mol.insertion_codes if mol.insertion_codes is not None else np.full(mol.kam, "")

        boundaries = [0]
        if mol.kam > 0:
            current_name = res_names[0]
            current_id = res_ids[0]
            current_chain = chain_ids[0]
            current_insertion = insertion_codes[0]

            for i in range(1, mol.kam):
                if (res_names[i] != current_name or
                        res_ids[i] != current_id or
                        chain_ids[i] != current_chain or
                        insertion_codes[i] != current_insertion):
                    boundaries.append(i) # Store index before new subunit starts
                    current_name = res_names[i]
                    current_id = res_ids[i]
                    current_chain = chain_ids[i]
                    current_insertion = insertion_codes[i]
            boundaries.append(mol.kam)

        mol.subunit_boundaries = np.array(boundaries)
        mol.kcen = len(boundaries) - 1

    @staticmethod
    def _build_connectivity(context: 'CurvesContext', threshold=1.8):
        """
        """
        mol = context.molecule
        from scipy.spatial import KDTree
        coords = mol.coordinates
        tree = KDTree(coords)

        pairs = tree.query_pairs(threshold)

        connectivity = np.zeros((len(coords), 7), dtype=int)
        subunit_by_atom = np.full(len(coords), -1, dtype=int)
        for subunit in range(mol.kcen):
            start = int(mol.subunit_boundaries[subunit])
            stop = int(mol.subunit_boundaries[subunit + 1])
            subunit_by_atom[start:stop] = subunit

        counts = np.zeros(len(coords), dtype=int)

        for i, j in pairs:
            same_subunit = subunit_by_atom[i] == subunit_by_atom[j]
            if not same_subunit:
                same_chain = str(mol.chain_ids[i]).strip() == str(mol.chain_ids[j]).strip()
                adjacent_subunits = abs(int(subunit_by_atom[i]) - int(subunit_by_atom[j])) == 1
                names = {
                    MolecularLoader._clean_atom_name(mol.atom_names[i]),
                    MolecularLoader._clean_atom_name(mol.atom_names[j]),
                }
                is_phosphodiester = names in ({"O3'", "P"}, {"O3*", "P"})
                if not (same_chain and adjacent_subunits and is_phosphodiester):
                    continue
            if mol.model_ids is not None and mol.model_ids[i] != mol.model_ids[j]:
                continue

            if counts[i] < 6:
                connectivity[i, counts[i]] = j + 1
                counts[i] += 1
            if counts[j] < 6:
                connectivity[j, counts[j]] = i + 1
                counts[j] += 1

        connectivity[:, 6] = counts

        mol.connectivity = connectivity

    def _identify_base_atoms(self, strand: int, level: int, ctx: 'CurvesContext'):
        mol = ctx.molecule
        start_idx = mol.subunit_boundaries[level]
        end_idx = mol.subunit_boundaries[level + 1]

        res_name = mol.residue_names[start_idx].strip().upper()
        is_purine = any(p in res_name for p in ['A', 'G', 'ADE', 'GUA'])

        indices = {}
        for i in range(start_idx, end_idx):
            name = mol.atom_names[i]

            if name in ["C1'", "C1*"]:
                indices['v1_ref'] = i

            if is_purine:
                if name == 'N9':
                    indices['base_origin_ref'] = i
                if name == 'C4':
                    indices['v2_ref'] = i
                if name == 'C8':
                    indices['v3_ref'] = i
            else:
                if name == 'N1':
                    indices['base_origin_ref'] = i
                if name == 'C2':
                    indices['v2_ref'] = i
                if name == 'C6':
                    indices['v3_ref'] = i

        if len(indices) < 4:
            return None
        return indices
