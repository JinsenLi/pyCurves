from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


_SCHEMA_PREFIX = "pycurves-visualization-"


def _load_results(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sanitize_name(value: str, fallback: str = "pycurves") -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip())
    text = text.strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"_{text}"
    return text


def _python_joined_string(text: str, chunk_size: int = 4000) -> str:
    chunks = [text[index:index + chunk_size] for index in range(0, len(text), chunk_size)]
    if not chunks:
        chunks = [""]
    joined = ",\n    ".join(repr(chunk) for chunk in chunks)
    return f"''.join([\n    {joined}\n])"


def render_pymol_script(results: Dict[str, Any], scene_prefix: str = "pycurves") -> str:
    visualization = results.get("visualization")
    if not isinstance(visualization, dict):
        raise ValueError(
            "This JSON file does not contain viewer geometry. "
            "Regenerate it with: python pycurves.py <structure> --format json --visualization --output-file <file.json>"
        )
    schema = str(visualization.get("schema", ""))
    if schema and not schema.startswith(_SCHEMA_PREFIX):
        raise ValueError(f"Unsupported visualization schema: {schema}")

    scene_prefix = _sanitize_name(scene_prefix, "pycurves")
    vis_json = json.dumps(visualization, separators=(",", ":"))
    summary = {
        "program": results.get("program", "pyCurves"),
        "format": results.get("format", ""),
        "pdbfile": (results.get("inputs") or {}).get("pdbfile", ""),
        "inpfile": (results.get("inputs") or {}).get("inpfile", ""),
        "schema": visualization.get("schema", ""),
        "axis_points": len(visualization.get("axis", []) or []),
        "base_pairs": len(visualization.get("base_pairs", []) or []),
    }

    return PYMOL_TEMPLATE.replace("__VIS_JSON__", _python_joined_string(vis_json)).replace(
        "__SUMMARY_JSON__", _python_joined_string(json.dumps(summary, separators=(",", ":")))
    ).replace(
        "__SCENE_PREFIX__", repr(scene_prefix)
    )


def write_pymol_scene(
    json_file: str,
    output_file: Optional[str] = None,
    structure_file: Optional[str] = None,
    structure_object: str = "pycurves_structure",
    scene_prefix: str = "pycurves",
) -> Path:
    json_path = Path(json_file).resolve()
    results = _load_results(json_path)
    visualization = results.get("visualization")
    if not visualization:
        raise ValueError(
            "This JSON file does not contain viewer geometry. "
            "Regenerate it with: python pycurves.py <structure> --format json --visualization --output-file <file.json>"
        )

    # Backwards-compatible no-ops. The PyMOL writer emits a structure-free
    # overlay so users can load any matching coordinate file separately.
    _ = structure_file, structure_object

    output_path = Path(output_file) if output_file else json_path.with_suffix(".pml")
    script = render_pymol_script(results=results, scene_prefix=scene_prefix)
    output_path.write_text(script, encoding="utf-8")
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a lean PyMOL .pml overlay from pyCurves JSON visualization output."
    )
    parser.add_argument("json_file", help="pyCurves JSON output containing a visualization payload.")
    parser.add_argument("-o", "--output", help="PML file to write. Defaults to <json>.pml.")
    parser.add_argument("--scene-prefix", default="pycurves", help="Prefix used for generated PyMOL objects and groups.")
    parser.add_argument("--structure", help=argparse.SUPPRESS)
    parser.add_argument("--structure-object", default="pycurves_structure", help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_path = write_pymol_scene(
        args.json_file,
        output_file=args.output,
        structure_file=args.structure,
        structure_object=args.structure_object,
        scene_prefix=args.scene_prefix,
    )
    print(f"Wrote {output_path}")


PYMOL_TEMPLATE = """# pyCurves PyMOL overlay
# Load in PyMOL with: @this_file.pml
# This file intentionally contains pyCurves geometry only; load coordinates separately if desired.

python
import json
import math
import re
from pymol import cmd
from pymol.cgo import *

VIS = json.loads(__VIS_JSON__)
SUMMARY = json.loads(__SUMMARY_JSON__)
PREFIX = __SCENE_PREFIX__
ROOT_GROUP = PREFIX + "_scene"
AXIS_GROUP = PREFIX + "_axis"
BACKBONE_GROUP = PREFIX + "_backbone"
BLOCK_GROUP = PREFIX + "_base_blocks"
GROOVE_GROUP = PREFIX + "_grooves"
MINOR_GROOVE_GROUP = PREFIX + "_grooves_minor"
MAJOR_GROOVE_GROUP = PREFIX + "_grooves_major"

PAIR_OBJECTS_BY_LEVEL = {}
GROOVE_OBJECTS_BY_LEVEL = {}

COLORS = {
    "axis": (0.10, 0.35, 0.92),
    "axis_back": (0.02, 0.04, 0.09),
    "canonical": (0.10, 0.58, 0.42),
    "hoogsteen": (0.55, 0.25, 0.88),
    "mismatch": (0.93, 0.43, 0.05),
    "modified": (0.04, 0.56, 0.70),
    "unsupported": (0.84, 0.15, 0.18),
    "other": (0.40, 0.46, 0.55),
    "dark": (0.06, 0.08, 0.12),
    "minor": (0.88, 0.14, 0.36),
    "major": (0.18, 0.31, 0.92),
}
STRAND_COLORS = [
    (0.02, 0.48, 0.78),
    (0.94, 0.33, 0.12),
    (0.08, 0.58, 0.35),
    (0.50, 0.28, 0.78),
    (0.82, 0.50, 0.02),
    (0.00, 0.56, 0.60),
]
PAIR_CLASS_GROUPS = {
    "canonical": PREFIX + "_bp_canonical",
    "hoogsteen": PREFIX + "_bp_hoogsteen",
    "mismatch": PREFIX + "_bp_mismatch",
    "modified": PREFIX + "_bp_modified",
    "unsupported": PREFIX + "_bp_unsupported",
    "other": PREFIX + "_bp_other",
}


def pt(point):
    if point is None:
        return None
    try:
        if isinstance(point, (list, tuple)):
            if len(point) < 3:
                return None
            values = [point[0], point[1], point[2]]
        else:
            values = [point.get("x"), point.get("y"), point.get("z")]
        coords = [float(value) for value in values]
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coords):
        return None
    return coords


def vec(point):
    return pt(point) or [0.0, 0.0, 0.0]


def addv(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def subv(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def scale(v, s):
    return [v[0] * s, v[1] * s, v[2] * s]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def norm(v):
    return math.sqrt(max(dot(v, v), 0.0))


def unit(v, fallback=(1.0, 0.0, 0.0)):
    length = norm(v)
    if not math.isfinite(length) or length < 1.0e-8:
        return list(fallback)
    return scale(v, 1.0 / length)


def color(name):
    return COLORS.get(name, COLORS["other"])


def safe_name(value, fallback="item", limit=96):
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = "_" + text
    return text[:limit]


def level_tag(value):
    try:
        return f"L{int(float(value)):03d}"
    except (TypeError, ValueError):
        return safe_name(value, "Lunknown")


def numeric_level(value):
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value or "")


def residue_tag(base):
    name = base.get("residue_name") or base.get("parent_base") or "N"
    residue_id = base.get("residue_id") or base.get("level") or ""
    chain_id = base.get("chain_id") or base.get("subunit") or ""
    return safe_name(f"{chain_id}{name}{residue_id}", "base", 28)


def pair_notation(pair):
    edge = str(pair.get("edge_pair") or "").replace("/", "")
    orientation = str(pair.get("glycosidic_orientation") or "").lower()
    if edge:
        prefix = "c" if orientation.startswith("cis") else "t" if orientation.startswith("trans") else ""
        return safe_name(prefix + edge, edge, 12)
    family = pair.get("pair_family") or "pair"
    return safe_name(family, "pair", 16)


def pair_class(pair):
    if pair.get("is_hoogsteen"):
        return "hoogsteen"
    if pair.get("is_mismatch"):
        return "mismatch"
    if pair.get("has_modified_base"):
        return "modified"
    if not pair.get("shape_parameters_supported", True):
        return "unsupported"
    if pair.get("is_canonical"):
        return "canonical"
    return "other"


def pair_color(pair):
    return color(pair_class(pair))


def pair_object_name(pair, index):
    first = pair.get("first") or {}
    second = pair.get("second") or {}
    parts = [PREFIX, "bp", level_tag(pair.get("level")), residue_tag(first), residue_tag(second), pair_notation(pair)]
    if pair.get("is_mismatch"):
        parts.append("mismatch")
    name = safe_name("_".join(parts), f"{PREFIX}_bp_{index:03d}", 112)
    return f"{name}_{index:03d}"


def groove_object_name(row, side, row_index):
    width = row.get(f"{side}_width")
    width_tag = ""
    if width is not None:
        try:
            width_tag = f"_W{float(width):.1f}".replace(".", "p")
        except (TypeError, ValueError):
            width_tag = ""
    base_pair = safe_name(row.get("base_pair") or "bp", "bp", 12)
    level = level_tag(row.get("level"))
    sub_level = int(row.get("sub_level") or 0)
    return safe_name(f"{PREFIX}_{side}_groove_{level}_{sub_level:02d}_{base_pair}{width_tag}_{row_index:03d}", f"{PREFIX}_{side}_groove_{row_index:03d}", 112)


def add_cylinder(cgo, a, b, radius, rgb, rgb2=None):
    a = pt(a)
    b = pt(b)
    if a is None or b is None:
        return
    rgb2 = rgb if rgb2 is None else rgb2
    cgo.extend([
        CYLINDER,
        a[0], a[1], a[2],
        b[0], b[1], b[2],
        float(radius),
        rgb[0], rgb[1], rgb[2],
        rgb2[0], rgb2[1], rgb2[2],
    ])


def add_sphere(cgo, p, radius, rgb):
    p = pt(p)
    if p is None:
        return
    cgo.extend([COLOR, rgb[0], rgb[1], rgb[2], SPHERE, p[0], p[1], p[2], float(radius)])


def ensure_group(group, parent=ROOT_GROUP):
    if parent:
        cmd.group(parent, group)


def load_cgo(name, cgo, group, enabled=True, parent=ROOT_GROUP):
    if not cgo:
        return None
    cmd.load_cgo(cgo, name)
    cmd.group(group, name)
    ensure_group(group, parent)
    if not enabled:
        cmd.disable(name)
    return name


def base_display_point(base):
    return [float(base["x"]), float(base["y"]), float(base["z"])]


def actual_block_base(base, pair):
    if pair.get("is_hoogsteen") and base.get("hoogsteen_plate_x_axis"):
        copy = dict(base)
        copy["plate_x_axis"] = base.get("hoogsteen_plate_x_axis")
        copy["plate_y_axis"] = base.get("hoogsteen_plate_y_axis")
        copy["plate_z_axis"] = base.get("hoogsteen_plate_z_axis")
        copy["plate_length"] = base.get("hoogsteen_plate_length") or base.get("plate_length")
        copy["plate_width"] = base.get("hoogsteen_plate_width") or base.get("plate_width")
        return copy
    return base


def base_plate_corners(base, partner=None):
    center = base_display_point(base)
    long_axis = unit(vec(base.get("plate_x_axis") or base.get("x_axis") or {"x": 1, "y": 0, "z": 0}))
    short_axis = unit(vec(base.get("plate_y_axis") or base.get("y_axis") or {"x": 0, "y": 1, "z": 0}))
    short_axis = unit(subv(short_axis, scale(long_axis, dot(short_axis, long_axis))), short_axis)
    if partner is not None:
        to_partner = unit(subv(base_display_point(partner), center))
        if dot(long_axis, to_partner) < 0.0:
            long_axis = scale(long_axis, -1.0)
    half_long = scale(long_axis, float(base.get("plate_length") or 3.6) / 2.0)
    half_short = scale(short_axis, float(base.get("plate_width") or 1.45) / 2.0)
    return [
        addv(addv(center, half_long), half_short),
        addv(subv(center, half_long), half_short),
        subv(subv(center, half_long), half_short),
        addv(subv(center, half_short), half_long),
    ]


def add_plate_surface(cgo, corners, rgb):
    normal = unit(cross(subv(corners[1], corners[0]), subv(corners[2], corners[0])), (0.0, 0.0, 1.0))
    reverse = scale(normal, -1.0)
    cgo.extend([BEGIN, TRIANGLES, COLOR, rgb[0], rgb[1], rgb[2]])
    for normal_vec, indices in (
        (normal, (0, 1, 2, 0, 2, 3)),
        (reverse, (0, 2, 1, 0, 3, 2)),
    ):
        cgo.extend([NORMAL, normal_vec[0], normal_vec[1], normal_vec[2]])
        for index in indices:
            corner = corners[index]
            cgo.extend([VERTEX, corner[0], corner[1], corner[2]])
    cgo.extend([END])


def add_plate_block(cgo, base, partner, rgb):
    corners = base_plate_corners(base, partner)
    add_plate_surface(cgo, corners, rgb)
    for i in range(4):
        add_cylinder(cgo, corners[i], corners[(i + 1) % 4], 0.035, color("dark"))


def draw_axis():
    points = VIS.get("axis") or []
    groups = {}
    for point in points:
        key = f"{point.get('axis_scope', '')}:{point.get('strand', 0)}"
        groups.setdefault(key, []).append(point)
    any_strand = any(int(point.get("strand") or 0) > 0 for point in points)
    for index, group_points in enumerate(groups.values()):
        group_points.sort(key=lambda item: float(item.get("level") or 0))
        rgb = STRAND_COLORS[index % len(STRAND_COLORS)] if any_strand else color("axis")
        scope = safe_name(group_points[0].get("axis_scope") or "axis", "axis", 24)
        strand = int(group_points[0].get("strand") or 0)
        path_cgo = []
        for i in range(1, len(group_points)):
            add_cylinder(path_cgo, group_points[i - 1], group_points[i], 0.18, color("axis_back"))
            add_cylinder(path_cgo, group_points[i - 1], group_points[i], 0.115, rgb)
        load_cgo(f"{PREFIX}_axis_{scope}_S{strand:02d}_path", path_cgo, AXIS_GROUP, enabled=True)
        for point in group_points:
            point_cgo = []
            add_sphere(point_cgo, point, 0.25, color("axis_back"))
            add_sphere(point_cgo, point, 0.18, rgb)
            load_cgo(f"{PREFIX}_axis_{scope}_S{strand:02d}_{level_tag(point.get('level'))}", point_cgo, AXIS_GROUP, enabled=True)


def draw_backbones():
    for index, backbone in enumerate(VIS.get("backbones") or []):
        rgb = STRAND_COLORS[index % len(STRAND_COLORS)]
        strand = int(backbone.get("strand") or index + 1)
        points = backbone.get("spline_points") if len(backbone.get("spline_points") or []) > 1 else backbone.get("points") or []
        cgo = []
        for i in range(1, len(points)):
            add_cylinder(cgo, points[i - 1], points[i], 0.085, rgb)
        load_cgo(f"{PREFIX}_backbone_S{strand:02d}", cgo, BACKBONE_GROUP, enabled=True)


def draw_base_blocks():
    for group in PAIR_CLASS_GROUPS.values():
        ensure_group(group, BLOCK_GROUP)
    for index, pair in enumerate(VIS.get("base_pairs") or []):
        rgb = pair_color(pair)
        first = pair.get("first") or {}
        second = pair.get("second") or {}
        if pt(first) is None or pt(second) is None:
            continue
        cgo = []
        add_plate_block(cgo, actual_block_base(first, pair), actual_block_base(second, pair), rgb)
        add_plate_block(cgo, actual_block_base(second, pair), actual_block_base(first, pair), rgb)
        family_group = PAIR_CLASS_GROUPS.get(pair_class(pair), PAIR_CLASS_GROUPS["other"])
        name = pair_object_name(pair, index)
        obj = load_cgo(name, cgo, family_group, enabled=True, parent=BLOCK_GROUP)
        if obj:
            PAIR_OBJECTS_BY_LEVEL.setdefault(numeric_level(pair.get("level")), []).append(obj)


def draw_grooves():
    ensure_group(MINOR_GROOVE_GROUP, GROOVE_GROUP)
    ensure_group(MAJOR_GROOVE_GROUP, GROOVE_GROUP)
    for row_index, row in enumerate((VIS.get("parameters") or {}).get("groove") or []):
        geometry = row.get("geometry") or {}
        for side, rgb_name, group in (
            ("minor", "minor", MINOR_GROOVE_GROUP),
            ("major", "major", MAJOR_GROOVE_GROUP),
        ):
            item = geometry.get(side) or {}
            if item.get("width_endpoint_1") and item.get("width_endpoint_2"):
                cgo = []
                add_cylinder(cgo, item["width_endpoint_1"], item["width_endpoint_2"], 0.090, color(rgb_name))
                add_sphere(cgo, item["width_endpoint_1"], 0.115, color(rgb_name))
                add_sphere(cgo, item["width_endpoint_2"], 0.115, color(rgb_name))
                obj = load_cgo(groove_object_name(row, side, row_index), cgo, group, enabled=True, parent=GROOVE_GROUP)
                if obj:
                    key = (numeric_level(row.get("level")), side)
                    GROOVE_OBJECTS_BY_LEVEL.setdefault(key, []).append(obj)


def set_group(group_name, mode="on"):
    mode = str(mode or "on").lower()
    if mode in {"off", "hide", "0", "false"}:
        cmd.disable(group_name)
    else:
        cmd.enable(group_name)


def set_objects(objects, mode="on"):
    mode = str(mode or "on").lower()
    action = cmd.disable if mode in {"off", "hide", "0", "false"} else cmd.enable
    for obj in objects:
        action(obj)


def pyc_axis(mode="on"):
    set_group(AXIS_GROUP, mode)


def pyc_backbone(mode="on"):
    set_group(BACKBONE_GROUP, mode)


def pyc_blocks(mode="on"):
    set_group(BLOCK_GROUP, mode)


def pyc_grooves(mode="on"):
    set_group(GROOVE_GROUP, mode)


def pyc_pair(level="", mode="on"):
    key = numeric_level(level)
    objects = PAIR_OBJECTS_BY_LEVEL.get(key)
    if not objects:
        print("No pyCurves base-pair block for level", level)
        return
    set_objects(objects, mode)


def pyc_groove(level="", side="", mode="on"):
    level_key = numeric_level(level)
    side_key = str(side or "").lower()
    if side_key in {"minor", "major"}:
        objects = GROOVE_OBJECTS_BY_LEVEL.get((level_key, side_key), [])
    else:
        objects = []
        for candidate_side in ("minor", "major"):
            objects.extend(GROOVE_OBJECTS_BY_LEVEL.get((level_key, candidate_side), []))
    if not objects:
        print("No pyCurves groove line for level", level, side)
        return
    set_objects(objects, mode)


def register_commands():
    cmd.extend("pyc_axis", pyc_axis)
    cmd.extend("pyc_backbone", pyc_backbone)
    cmd.extend("pyc_blocks", pyc_blocks)
    cmd.extend("pyc_grooves", pyc_grooves)
    cmd.extend("pyc_pair", pyc_pair)
    cmd.extend("pyc_groove", pyc_groove)


def build_scene():
    draw_axis()
    draw_backbones()
    draw_base_blocks()
    draw_grooves()
    register_commands()
    try:
        cmd.set("two_sided_lighting", 1)
        cmd.set("ambient", 0.42)
        cmd.set("specular", 0.18)
    except Exception:
        pass
    cmd.orient(ROOT_GROUP)
    print("pyCurves PyMOL overlay loaded:")
    print("  object panel contains individually toggleable axis points, backbone strands, base pairs, and groove lines")
    print("  pyc_pair <level> on/off            show or hide one base-pair block")
    print("  pyc_groove <level> [minor|major] on/off  show or hide groove lines")
    print("  pyc_axis/pyc_backbone/pyc_blocks/pyc_grooves on/off toggle whole layers")


build_scene()
python end
"""


if __name__ == "__main__":
    main()
