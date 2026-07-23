import re

import numpy as np

from pycurves_lib.core.curves_dataclasses import HelicalConfig

LW_EDGES = {"W", "H", "S"}


class ConfigLoader:
    @staticmethod
    def parse_inp(file_path: str, config_overrides=None):
        with open(file_path, "r") as f:
            raw_content = f.read()

        cfg = HelicalConfig()
        content = raw_content.replace("\n", " ")
        lines = [line.strip() for line in raw_content.splitlines() if line.strip()]

        ConfigLoader._parse_namelist_values(content, cfg)
        ConfigLoader._apply_config_overrides(cfg, config_overrides)
        ConfigLoader._resolve_convention_pair(cfg)

        data_lines = ConfigLoader._topology_lines(lines)
        if not data_lines:
            raise ValueError(f"No strand topology rows found in {file_path!r}")

        strand_info = list(map(int, data_lines[0].split()))
        strand_count = strand_info[0]  # Fortran nst
        signed_strand_lengths = strand_info[1:1 + strand_count]  # Fortran nu input, sign encodes direction
        if strand_count <= 0:
            raise ValueError(f"Invalid strand count {strand_count} in {file_path!r}")
        if len(signed_strand_lengths) != strand_count:
            raise ValueError(f"Expected {strand_count} strand descriptors in {file_path!r}")

        strand_directions = [1 if value >= 0 else -1 for value in signed_strand_lengths]  # Fortran idr

        (
            expanded_maps,
            current_idx,
            hoogsteen_markers,
            pair_geometry_markers,
            glycosidic_conformation_markers,
        ) = ConfigLoader._parse_strand_maps(
            data_lines,
            strand_count,
            signed_strand_lengths,
            file_path,
        )
        level_count = max(len(mapping) for mapping in expanded_maps)  # Fortran nux
        strand_lengths = [sum(1 for unit in mapping if unit != 0) for mapping in expanded_maps]  # Fortran nu
        total_nucleotides = sum(strand_lengths)  # Fortran nt

        subunit_map = np.zeros((strand_count, level_count), dtype=int)  # Fortran ni
        initial_level_status = np.zeros((strand_count, level_count), dtype=int)  # Fortran li
        active_start_levels = np.zeros(strand_count, dtype=int)  # Fortran ng, 1-based
        active_end_levels = np.zeros(strand_count, dtype=int)  # Fortran nr, 1-based

        for strand, mapping in enumerate(expanded_maps):
            for level in range(level_count):
                mapped_unit = mapping[level] if level < len(mapping) else 0
                subunit_map[strand, level] = abs(mapped_unit)
                if mapped_unit == 0:
                    continue
                initial_level_status[strand, level] = 1 if mapped_unit > 0 else -1
                active_end_levels[strand] = level + 1
                if active_start_levels[strand] == 0:
                    active_start_levels[strand] = level + 1

        ConfigLoader._apply_fortran_option_rules(cfg, strand_count, strand_lengths, active_start_levels, active_end_levels, subunit_map)

        ConfigLoader._initialize_helical_input_defaults(cfg, strand_count, level_count, total_nucleotides)
        helical_rows = data_lines[current_idx:]
        consumed_rows = ConfigLoader._parse_helical_input_rows(cfg, helical_rows)
        if cfg.ends:
            ConfigLoader._parse_end_input_rows(cfg, helical_rows[consumed_rows:])

        return {
            "n_strands": strand_count,
            "n_levels": level_count,
            "nu_raw": signed_strand_lengths,
            "idr": strand_directions,
            "nu": strand_lengths,
            "nt": total_nucleotides,
            "ng": active_start_levels,
            "nr": active_end_levels,
            "li_map": initial_level_status,
            "ni_map": subunit_map,
            "hoogsteen_markers": hoogsteen_markers,
            "pair_geometry_markers": pair_geometry_markers,
            "glycosidic_conformation_markers": glycosidic_conformation_markers,
            "config": cfg,
        }

    @staticmethod
    def _parse_namelist_values(content: str, cfg: HelicalConfig):
        """Parse the Curves namelist while leaving unknown/string keys alone."""
        for field_name, value in cfg.__dict__.items():
            if isinstance(value, bool):
                match = re.search(fr"\b{field_name}\s*=\s*\.(t|f)\.", content, re.I)
                if match:
                    setattr(cfg, field_name, match.group(1).lower() == "t")

        # Curves 5.3 accepts axonly in the namelist but prints it as axonl in
        # the .lis header.  Accept both spellings so legacy inputs and reports
        # round-trip cleanly.
        match = re.search(r"\baxonl\s*=\s*\.(t|f)\.", content, re.I)
        if match:
            cfg.axonly = match.group(1).lower() == "t"

        int_fields = {
            "break": "break_lvl",
            "nlevel": "nlevel",
            "nbac": "nbac",
            "spline": "spline",
            "splin": "spline",
            "ior": "ior",
            "ibond": "ibond",
            "maxn": "maxn",
        }
        for key, attr in int_fields.items():
            if not hasattr(cfg, attr):
                continue
            match = re.search(fr"\b{key}\s*=\s*([-+]?\d+)", content, re.I)
            if match:
                setattr(cfg, attr, int(match.group(1)))

        for attr in ("acc", "wid"):
            match = re.search(
                fr"\b{attr}\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
                content,
                re.I,
            )
            if match:
                setattr(cfg, attr, float(match.group(1)))

        for key in ("frame_convention", "frames", "convention"):
            match = re.search(fr"\b{key}\s*=\s*['\"]?([A-Za-z0-9_+\-]+)['\"]?", content, re.I)
            if match:
                value = match.group(1).lower().replace("-", "_")
                if value in {"legacy"}:
                    cfg.frame_convention = "legacy"
                elif value in {"standard", "curves_plus", "curves+", "curvesplus", "x3dna", "3dna"}:
                    cfg.frame_convention = "standard"
                break

        for key in ("axis_convention", "global_axis_convention", "axis_frames"):
            match = re.search(fr"\b{key}\s*=\s*['\"]?([A-Za-z0-9_+\-]+)['\"]?", content, re.I)
            if match:
                value = match.group(1).lower().replace("-", "_")
                if value in {"legacy", "pycurves"}:
                    cfg.axis_convention = "legacy"
                elif value in {"curves_plus", "curves+", "curvesplus", "canal"}:
                    cfg.axis_convention = "curvesplus"
                else:
                    raise ValueError(f"Unknown axis convention {match.group(1)!r}; use legacy or curvesplus.")
                break

    @staticmethod
    def _apply_config_overrides(cfg: HelicalConfig, overrides):
        """Apply CLI/API overrides before derived Fortran settings are computed."""
        if not overrides:
            return
        for attr, value in overrides.items():
            if value is None:
                continue
            if not hasattr(cfg, attr):
                raise ValueError(f"Unknown Curves config override {attr!r}.")
            setattr(cfg, attr, value)

    @staticmethod
    def _resolve_convention_pair(cfg: HelicalConfig):
        """Keep frame and axis conventions in a physically valid combination."""
        if str(getattr(cfg, "axis_convention", "legacy")).lower() == "curvesplus":
            cfg.frame_convention = "standard"

    @staticmethod
    def _topology_lines(lines):
        """Return only numeric/range topology rows, skipping all namelist rows."""
        data_lines = []
        for line in lines:
            lower = line.lower()
            if lower.startswith("&") or lower.startswith("/") or lower == "&end":
                continue
            if "=" in line:
                continue
            data_lines.append(line)
        return data_lines

    @staticmethod
    def _expand_mapping_token(token: str):
        core, _ = ConfigLoader._split_mapping_token(token)
        if ":" not in core:
            return [int(core)]
        start_text, stop_text = core.split(":", 1)
        start = int(start_text)
        stop = int(stop_text)
        step = 1 if stop >= start else -1
        return list(range(start, stop + step, step))

    @staticmethod
    def _split_mapping_token(token: str):
        match = re.fullmatch(r"\s*([+-]?\d+(?::[+-]?\d+)?)((?:\[[^\]]+\])*)\s*", token)
        if not match:
            raise ValueError(f"Invalid Curves topology token {token!r}.")
        raw_tags = re.findall(r"\[([^\]]+)\]", match.group(2) or "")
        tag_infos = []
        seen_kinds = set()
        for tag in raw_tags:
            normalized = tag.strip().lower().replace("_", "").replace("-", "")
            if normalized in {"h", "hoog", "hoogsteen"}:
                tag_info = {"kind": "hoogsteen", "tag": "Hoog"}
            elif normalized in {"syn", "anti"}:
                tag_info = {
                    "kind": "glycosidic_conformation",
                    "glycosidic_conformation": normalized,
                    "tag": normalized,
                }
            else:
                tag_info = ConfigLoader._parse_lw_geometry_tag(normalized)
            if tag_info is None:
                raise ValueError(
                    f"Unknown Curves topology tag [{tag}] in token {token!r}; "
                    "supported tags are [syn], [anti], [Hoog], and Leontis-Westhof-style "
                    "geometry tags like [cWW], [tWH], and [cSS]."
                )
            kind_group = "pair_geometry" if tag_info["kind"] in {"lw", "hoogsteen"} else tag_info["kind"]
            if kind_group in seen_kinds:
                raise ValueError(f"Duplicate {kind_group.replace('_', ' ')} tag in token {token!r}.")
            seen_kinds.add(kind_group)
            tag_infos.append(tag_info)
        return match.group(1), tag_infos

    @staticmethod
    def _parse_lw_geometry_tag(normalized: str):
        parts = normalized.split(":", 1)
        lw_text = parts[0]
        direction_text = parts[1] if len(parts) == 2 else ""
        if len(lw_text) != 3:
            return None
        orientation = lw_text[0]
        if orientation not in {"c", "t"}:
            return None
        edge_1 = lw_text[1].upper()
        edge_2 = lw_text[2].upper()
        if edge_1 not in LW_EDGES or edge_2 not in LW_EDGES:
            return None
        # Older pyCurves builds emitted :p/:ap suffixes. Accept but ignore
        # them: the standard LW tag itself determines local strand orientation.
        if direction_text and ConfigLoader._parse_lw_strand_direction(direction_text) is None:
            return None
        tag = f"{orientation}{edge_1}{edge_2}"
        strand_direction = ConfigLoader._lw_strand_direction(orientation, edge_1, edge_2)
        return {
            "kind": "lw",
            "tag": tag,
            "orientation": orientation,
            "glycosidic_orientation": "cis" if orientation == "c" else "trans",
            "edge_1": edge_1,
            "edge_2": edge_2,
            "lw_strand_orientation": strand_direction,
            "strand_direction": strand_direction,
            "strand_direction_source": "inferred_from_lw_tag",
        }

    @staticmethod
    def _parse_lw_strand_direction(direction_text: str):
        if not direction_text:
            return None
        normalized = direction_text.strip().lower()
        if normalized in {"p", "par", "para", "parallel"}:
            return "parallel"
        if normalized in {"a", "ap", "anti", "antiparallel"}:
            return "antiparallel"
        return None

    @staticmethod
    def _lw_strand_direction(orientation: str, edge_1: str, edge_2: str) -> str:
        one_hoogsteen_edge = (edge_1 == "H") ^ (edge_2 == "H")
        cis_is_parallel = one_hoogsteen_edge
        is_parallel = cis_is_parallel if orientation == "c" else not cis_is_parallel
        return "parallel" if is_parallel else "antiparallel"

    @staticmethod
    def _parse_strand_maps(data_lines, strand_count, signed_strand_lengths, file_path):
        """Support both explicit unit maps and Curves shorthand ranges like 1:12."""
        maps = []
        hoogsteen_markers = set()
        pair_geometry_markers = {}
        glycosidic_conformation_markers = {}
        current_idx = 1
        range_style = any(":" in line for line in data_lines[1:1 + strand_count])

        if range_style:
            for strand in range(strand_count):
                if current_idx >= len(data_lines):
                    raise ValueError(f"Missing mapping row for strand {strand + 1} in {file_path!r}")
                mapping = []
                for token in data_lines[current_idx].split():
                    core, tag_info = ConfigLoader._split_mapping_token(token)
                    for mapped_unit in ConfigLoader._expand_mapping_token(core):
                        mapping.append(mapped_unit)
                        ConfigLoader._record_mapping_tags(
                            tag_info,
                            strand + 1,
                            len(mapping),
                            mapped_unit,
                            hoogsteen_markers,
                            pair_geometry_markers,
                            glycosidic_conformation_markers,
                        )
                maps.append(mapping)
                current_idx += 1
            return (
                maps,
                current_idx,
                hoogsteen_markers,
                pair_geometry_markers,
                glycosidic_conformation_markers,
            )

        level_count = max(abs(value) for value in signed_strand_lengths)
        for strand in range(strand_count):
            mapping = []
            while len(mapping) < level_count:
                if current_idx >= len(data_lines):
                    raise ValueError(f"Missing mapping values for strand {strand + 1} in {file_path!r}")
                for token in data_lines[current_idx].split():
                    core, tag_info = ConfigLoader._split_mapping_token(token)
                    for mapped_unit in ConfigLoader._expand_mapping_token(core):
                        if len(mapping) >= level_count:
                            break
                        mapping.append(mapped_unit)
                        ConfigLoader._record_mapping_tags(
                            tag_info,
                            strand + 1,
                            len(mapping),
                            mapped_unit,
                            hoogsteen_markers,
                            pair_geometry_markers,
                            glycosidic_conformation_markers,
                        )
                current_idx += 1
            maps.append(mapping[:level_count])
        return (
            maps,
            current_idx,
            hoogsteen_markers,
            pair_geometry_markers,
            glycosidic_conformation_markers,
        )

    @staticmethod
    def _record_mapping_tags(
        tag_infos,
        strand: int,
        level: int,
        mapped_unit: int,
        hoogsteen_markers: set,
        pair_geometry_markers: dict,
        glycosidic_conformation_markers: dict,
    ) -> None:
        if not tag_infos or mapped_unit == 0:
            return
        for tag_info in tag_infos:
            if tag_info.get("kind") == "hoogsteen":
                hoogsteen_markers.add((strand, level))
                continue
            if tag_info.get("kind") == "lw":
                marker = dict(tag_info)
                marker["annotated_strand"] = strand
                marker["level"] = level
                pair_geometry_markers[(strand, level)] = marker
                continue
            if tag_info.get("kind") == "glycosidic_conformation":
                glycosidic_conformation_markers[(strand, level)] = tag_info[
                    "glycosidic_conformation"
                ]

    @staticmethod
    def _initialize_helical_input_defaults(
        cfg: HelicalConfig,
        strand_count: int,
        level_count: int,
        total_nucleotides: int,
    ):
        if cfg.rest:
            cfg.inpv = level_count if cfg.comb else total_nucleotides
        elif strand_count > 1 and not cfg.comb:
            cfg.inpv = strand_count
        else:
            cfg.inpv = 1

        cfg.xdi = np.zeros(cfg.inpv, dtype=float)
        cfg.ydi = np.zeros(cfg.inpv, dtype=float)
        cfg.cln = np.zeros(cfg.inpv, dtype=float)
        cfg.tip = np.zeros(cfg.inpv, dtype=float)

    @staticmethod
    def _parse_helical_input_rows(cfg: HelicalConfig, rows):
        idx = 0
        consumed_rows = 0
        last_values = None
        for row in rows:
            if idx >= cfg.inpv:
                break
            consumed_rows += 1
            fields = row.split()
            if len(fields) < 4:
                continue
            last_values = tuple(map(float, fields[:4]))
            cfg.xdi[idx], cfg.ydi[idx], cfg.cln[idx], cfg.tip[idx] = last_values
            idx += 1
        if last_values is not None and idx < cfg.inpv:
            # A CLI override can expand the Fortran XYTP input count, for
            # example reading a comb=.t. file as --no-comb.  Reuse the last
            # supplied initial-axis row instead of leaving later strands at an
            # unrelated zero default.
            cfg.xdi[idx:] = last_values[0]
            cfg.ydi[idx:] = last_values[1]
            cfg.cln[idx:] = last_values[2]
            cfg.tip[idx:] = last_values[3]
        return consumed_rows

    @staticmethod
    def _parse_end_input_rows(cfg: HelicalConfig, rows):
        """Parse the two optional ENDS rows: Xdisp Ydisp Rise Inclin Tip Twist."""
        parsed = []
        for row in rows:
            fields = row.split()
            if len(fields) < 6:
                continue
            parsed.append(np.array(tuple(map(float, fields[:6])), dtype=float))
            if len(parsed) == 2:
                break

        # Curves 5.3 requires these rows when ends=.t.; pyCurves keeps a
        # conservative default so --ends can be used with auto-generated inputs.
        if len(parsed) >= 1:
            cfg.end_start = parsed[0]
        if len(parsed) >= 2:
            cfg.end_stop = parsed[1]

    @staticmethod
    def _apply_fortran_option_rules(cfg, strand_count, strand_lengths, active_start_levels, active_end_levels, subunit_map):
        """Apply Curves 5.3 option constraints that affect topology parsing."""
        if cfg.ends and cfg.line:
            raise ValueError("Curves option error: ends=.t. is not allowed with line=.t.")
        if cfg.line and not cfg.mini:
            cfg.mini = True
        if cfg.zaxe and cfg.mini:
            cfg.mini = False
        if cfg.ends and cfg.zaxe:
            raise ValueError("Curves option error: ends=.t. is not allowed with zaxe=.t.")
        if cfg.ends and strand_count > 2:
            raise ValueError("Curves option error: ends=.t. is not allowed with more than two strands.")
        if strand_count == 1 and cfg.comb:
            cfg.comb = False

        for strand in range(strand_count):
            ng = int(active_start_levels[strand])
            nr = int(active_end_levels[strand])
            if ng == 0 or nr == 0:
                continue
            has_internal_gap = any(subunit_map[strand, level - 1] == 0 for level in range(ng, nr + 1))
            if has_internal_gap and not cfg.comb:
                raise ValueError("Curves option error: internal gaps require comb=.t.")

        level_count = subunit_map.shape[1]
        if cfg.ends and any(length != level_count for length in strand_lengths):
            raise ValueError("Curves option error: ends=.t. requires strands of equal length.")
