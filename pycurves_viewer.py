from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _resolve_path(path_text: Optional[str], json_path: Path) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(json_path.parent / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def _structure_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return "mmcif"
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    return suffix.lstrip(".") or "pdb"


def _load_results(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_viewer_html(results: Dict[str, Any], structure_text: str, structure_format: str, title: str) -> str:
    visualization = results.get("visualization", {})
    inputs = results.get("inputs", {})
    summary = {
        "program": results.get("program", "pyCurves"),
        "format": results.get("format", ""),
        "pdbfile": inputs.get("pdbfile", ""),
        "inpfile": inputs.get("inpfile", ""),
        "axis_points": len(visualization.get("axis", [])),
        "base_pairs": len(visualization.get("base_pairs", [])),
        "backbones": len(visualization.get("backbones", [])),
    }

    return HTML_TEMPLATE.replace("__TITLE_JSON__", json.dumps(title)).replace(
        "__STRUCTURE_TEXT_JSON__", json.dumps(structure_text)
    ).replace(
        "__STRUCTURE_FORMAT_JSON__", json.dumps(structure_format)
    ).replace(
        "__VISUALIZATION_JSON__", json.dumps(visualization)
    ).replace(
        "__SUMMARY_JSON__", json.dumps(summary)
    )


def write_viewer(json_file: str, output_file: Optional[str] = None, structure_file: Optional[str] = None) -> Path:
    json_path = Path(json_file).resolve()
    results = _load_results(json_path)
    visualization = results.get("visualization")
    if not visualization:
        raise ValueError(
            "This JSON file does not contain viewer geometry. "
            "Regenerate it with: python pycurves.py <structure> --format json --visualization --output-file <file.json>"
        )
    inputs = results.get("inputs", {})
    structure_path = _resolve_path(structure_file or inputs.get("pdbfile"), json_path)
    if structure_path is None or not structure_path.exists():
        raise FileNotFoundError(
            "Could not find the source PDB/mmCIF file. Pass it explicitly with --structure."
        )

    structure_text = structure_path.read_text(encoding="utf-8", errors="ignore")
    output_path = Path(output_file) if output_file else json_path.with_suffix(".viewer.html")
    html = render_viewer_html(
        results=results,
        structure_text=structure_text,
        structure_format=_structure_format(structure_path),
        title=f"pyCurves viewer: {structure_path.name}",
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an interactive HTML viewer from pyCurves JSON output.")
    parser.add_argument("json_file", help="pyCurves JSON output file.")
    parser.add_argument("-o", "--output", help="HTML file to write. Defaults to <json>.viewer.html.")
    parser.add_argument("--structure", help="Override the structure file referenced in the pyCurves JSON.")
    args = parser.parse_args()

    output_path = write_viewer(args.json_file, output_file=args.output, structure_file=args.structure)
    print(f"Wrote {output_path}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title></title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --panel: #f7f8fb;
      --line: #d9dde8;
      --text: #162033;
      --muted: #627089;
      --accent: #1266b0;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      background: #ffffff;
    }
    .app {
      display: grid;
      grid-template-columns: 420px minmax(0, 1fr);
      height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px 16px;
      overflow: auto;
    }
    main {
      min-width: 0;
      position: relative;
    }
    h1 {
      margin: 0 0 14px;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }
    h2 {
      margin: 22px 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
    }
    .meta {
      display: grid;
      gap: 8px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      font-size: 12px;
    }
    .meta div {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 8px;
    }
    .meta span:first-child {
      color: var(--muted);
    }
    .meta span:last-child {
      overflow-wrap: anywhere;
    }
    label {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 30px;
      font-size: 14px;
      cursor: pointer;
    }
    input {
      accent-color: var(--accent);
    }
    select {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--text);
      font-size: 13px;
      padding: 0 8px;
    }
    button {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--text);
      font-size: 13px;
      cursor: pointer;
    }
    button:hover {
      border-color: var(--accent);
    }
    #viewer {
      position: absolute;
      inset: 0;
    }
    .load-error {
      margin-top: 14px;
      color: #9b1c1c;
      font-size: 13px;
      line-height: 1.4;
    }
    .tabs {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-bottom: 8px;
    }
    .tabs button {
      height: 30px;
      font-size: 12px;
      padding: 0 6px;
    }
    .tabs button.active {
      border-color: var(--accent);
      color: #ffffff;
      background: var(--accent);
    }
    .table-wrap {
      max-height: 260px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      scrollbar-gutter: stable both-edges;
      padding-right: 18px;
    }
    table {
      min-width: 720px;
      border-collapse: collapse;
      font-size: 12px;
      margin-right: 18px;
    }
    th,
    td {
      padding: 6px 7px;
      border-bottom: 1px solid #edf0f5;
      text-align: right;
      white-space: nowrap;
    }
    th:last-child,
    td:last-child {
      padding-right: 24px;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f7f8fb;
    }
    th:first-child,
    td:first-child {
      text-align: left;
      position: sticky;
      left: 0;
      z-index: 1;
      background: #ffffff;
      box-shadow: 1px 0 0 #edf0f5;
    }
    th:first-child {
      z-index: 3;
      background: #f7f8fb;
    }
    tr {
      cursor: pointer;
    }
    tr:hover,
    tr.selected {
      background: #e9f3ff;
    }
    .empty {
      padding: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 780px) {
      .app {
        grid-template-columns: 1fr;
        grid-template-rows: auto minmax(420px, 1fr);
      }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1 id="title"></h1>
      <div class="meta" id="meta"></div>

      <h2>Display</h2>
      <label><input type="checkbox" id="showOtherChains" checked> Show chains other than DNA</label>
      <label><input type="checkbox" id="showCartoon" checked> Cartoon</label>
      <label><input type="checkbox" id="showAllAtoms"> All atoms</label>
      <select id="colorMode" aria-label="Color mode">
        <option value="chain">Color by chain</option>
        <option value="residue">Color by residue</option>
        <option value="element">Color by element</option>
      </select>
      <label><input type="checkbox" id="showAxis" checked> Smoothed axis</label>
      <label><input type="checkbox" id="showBackbone" checked> Backbone spline curve</label>
      <label><input type="checkbox" id="showActualBlocks" checked> Actual base blocks</label>
      <label><input type="checkbox" id="showAnalyticalBases"> Analytical base frames</label>
      <label><input type="checkbox" id="showLabels"> Labels</label>
      <label><input type="checkbox" id="showOnlyUnusual"> Only unusual pairs</label>

      <h2>Parameters</h2>
      <div class="tabs">
        <button type="button" class="active" data-tab="base_pair">Base Pair</button>
        <button type="button" data-tab="base_pair_axis">BP Axis</button>
        <button type="button" data-tab="global_step">Global Step</button>
        <button type="button" data-tab="local_step">Local Step</button>
        <button type="button" data-tab="base_axis">Base Axis</button>
        <button type="button" data-tab="groove">Groove</button>
      </div>
      <div class="table-wrap" id="parameterTable"></div>

      <h2>View</h2>
      <button id="resetView" type="button">Reset View</button>
      <p class="load-error" id="loadError" hidden></p>
    </aside>
    <main>
      <div id="viewer"></div>
    </main>
  </div>

  <script>
    const PAGE_TITLE = __TITLE_JSON__;
    const STRUCTURE_TEXT = __STRUCTURE_TEXT_JSON__;
    const STRUCTURE_FORMAT = __STRUCTURE_FORMAT_JSON__;
    const VIS = __VISUALIZATION_JSON__;
    const SUMMARY = __SUMMARY_JSON__;

    document.title = PAGE_TITLE;
    document.getElementById("title").textContent = PAGE_TITLE;
    document.getElementById("meta").innerHTML = [
      ["Structure", SUMMARY.pdbfile || "(embedded)"],
      ["Input", SUMMARY.inpfile || ""],
      ["Axis points", SUMMARY.axis_points],
      ["Base pairs", SUMMARY.base_pairs],
      ["Backbone curves", SUMMARY.backbones]
    ].map(([key, value]) => `<div><span>${key}</span><span>${value}</span></div>`).join("");

    let viewer = null;
    let overlayShapes = [];
    let overlayLabels = [];
    function initialParameterTab() {
      const params = VIS.parameters || {};
      if ((params.base_pair || []).length) return "base_pair";
      if ((params.base_pair_axis || []).length) return "base_pair_axis";
      if ((params.global_step || []).length) return "global_step";
      if ((params.local_step || []).length) return "local_step";
      if ((params.base_axis || []).length) return "base_axis";
      if ((params.groove || []).length) return "groove";
      return "base_pair";
    }

    let activeTab = initialParameterTab();
    let selectedInspection = null;
    const strandColors = ["#1b74b7", "#c03a2b", "#2d8a4e", "#8b5cc7", "#b6801d", "#008c95"];
    const baseColors = {
      A: "#2e9d57",
      C: "#2369b3",
      G: "#f0b72f",
      T: "#d64b3f",
      U: "#8a5bb8",
      I: "#8a8f98"
    };
    const proteinResidueColors = {
      ALA: "#9aa1aa", VAL: "#9aa1aa", LEU: "#9aa1aa", ILE: "#9aa1aa", MET: "#9aa1aa", PRO: "#9aa1aa",
      PHE: "#b28b2c", TYR: "#b28b2c", TRP: "#b28b2c", HIS: "#b28b2c",
      SER: "#2c9a9a", THR: "#2c9a9a", ASN: "#2c9a9a", GLN: "#2c9a9a", CYS: "#2c9a9a",
      LYS: "#3b68b8", ARG: "#3b68b8",
      ASP: "#c54b45", GLU: "#c54b45",
      GLY: "#6f7784"
    };

    function colorForChain(chain, index) {
      return strandColors[index % strandColors.length];
    }

    function xyz(point) {
      if (Array.isArray(point)) return {x: Number(point[0]), y: Number(point[1]), z: Number(point[2])};
      return {x: Number(point.x), y: Number(point.y), z: Number(point.z)};
    }

    function vec(point) {
      if (Array.isArray(point)) return [Number(point[0]), Number(point[1]), Number(point[2])];
      return [Number(point.x), Number(point.y), Number(point.z)];
    }

    function pointFrom(values) {
      return {x: values[0], y: values[1], z: values[2]};
    }

    function addVec(left, right) {
      return [left[0] + right[0], left[1] + right[1], left[2] + right[2]];
    }

    function subVec(left, right) {
      return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
    }

    function scaleVec(values, scale) {
      return [values[0] * scale, values[1] * scale, values[2] * scale];
    }

    function crossVec(left, right) {
      return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]
      ];
    }

    function normVec(values) {
      return Math.hypot(values[0], values[1], values[2]);
    }

    function unitVec(values, fallback = [1, 0, 0]) {
      const length = normVec(values);
      if (!Number.isFinite(length) || length < 1e-6) return fallback;
      return scaleVec(values, 1 / length);
    }

    function addShape(shape) {
      overlayShapes.push(shape);
      return shape;
    }

    function addSegment(a, b, color, radius, opacity = 1.0) {
      if (!a || !b) return;
      addShape(viewer.addCylinder({
        start: xyz(a),
        end: xyz(b),
        radius,
        color,
        opacity,
        fromCap: 1,
        toCap: 1
      }));
    }

    function addPoint(point, color, radius) {
      if (!point) return;
      addShape(viewer.addSphere({center: xyz(point), radius, color}));
    }

    function addFilledQuad(corners, color, opacity = 0.7) {
      if (viewer.addCustom && window.$3Dmol && window.$3Dmol.Vector3) {
        try {
          return addShape(viewer.addCustom({
            vertexArr: corners.map(point => new $3Dmol.Vector3(point.x, point.y, point.z)),
            faceArr: [0, 1, 2, 0, 2, 3],
            normalArr: [],
            color,
            opacity
          }));
        } catch (error) {
          // Fall back to a solid-looking strip fill below.
        }
      }

      const strips = 11;
      for (let i = 0; i < strips; i += 1) {
        const t = strips === 1 ? 0.5 : i / (strips - 1);
        const left = pointFrom(addVec(scaleVec(vec(corners[0]), 1 - t), scaleVec(vec(corners[3]), t)));
        const right = pointFrom(addVec(scaleVec(vec(corners[1]), 1 - t), scaleVec(vec(corners[2]), t)));
        addSegment(left, right, color, 0.12, opacity);
      }
    }

    function baseColor(base) {
      const key = String(base || "").trim().toUpperCase().replace(/^D/, "").slice(0, 1);
      return baseColors[key] || "#8a8f98";
    }

    function residueColor(residueName) {
      const raw = String(residueName || "").trim().toUpperCase();
      const baseKey = raw.replace(/^D/, "").slice(0, 1);
      if (baseColors[baseKey] && ["A", "C", "G", "T", "U", "I"].includes(baseKey)) {
        return baseColors[baseKey];
      }
      return proteinResidueColors[raw] || "#8a8f98";
    }

    function pairColor(pair) {
      if (pair.is_hoogsteen) return "#bb3bbf";
      if (pair.is_mismatch) return "#d97904";
      if (pair.has_modified_base) return "#008c95";
      if (!pair.shape_parameters_supported) return "#b22b2b";
      return pair.is_canonical ? "#2d8a4e" : "#6e7687";
    }

    function basePlateCorners(base, partner) {
      const center = vec(base);
      let longAxis = base.plate_x_axis ? unitVec(vec(base.plate_x_axis)) : null;
      let shortAxis = base.plate_y_axis ? unitVec(vec(base.plate_y_axis)) : null;
      if (!longAxis || !shortAxis) {
        const toPartner = partner ? unitVec(subVec(vec(partner), center)) : [1, 0, 0];
        const zAxis = [0, 0, 1];
        shortAxis = unitVec(crossVec(toPartner, zAxis), [0, 1, 0]);
        longAxis = unitVec(crossVec(shortAxis, toPartner), [1, 0, 0]);
      }
      shortAxis = unitVec(subVec(shortAxis, scaleVec(longAxis, shortAxis[0] * longAxis[0] + shortAxis[1] * longAxis[1] + shortAxis[2] * longAxis[2])), shortAxis);
      if (partner) {
        const toPartner = unitVec(subVec(vec(partner), center));
        if (longAxis[0] * toPartner[0] + longAxis[1] * toPartner[1] + longAxis[2] * toPartner[2] < 0) {
          longAxis = scaleVec(longAxis, -1);
        }
      }
      const halfLong = scaleVec(longAxis, Number(base.plate_length || 3.6) / 2);
      const halfShort = scaleVec(shortAxis, Number(base.plate_width || 1.45) / 2);
      return [
        pointFrom(addVec(addVec(center, halfLong), halfShort)),
        pointFrom(addVec(subVec(center, halfLong), halfShort)),
        pointFrom(subVec(subVec(center, halfLong), halfShort)),
        pointFrom(addVec(subVec(center, halfShort), halfLong))
      ];
    }

    function analyticalBasePlate(base) {
      const origin = base.frame_origin || base;
      return Object.assign({}, base, {
        x: origin.x,
        y: origin.y,
        z: origin.z,
        plate_x_axis: base.plate_x_axis || base.y_axis,
        plate_y_axis: base.plate_y_axis || base.x_axis,
        plate_z_axis: base.plate_z_axis || base.z_axis,
        plate_length: Number(base.plate_length || 5.0) * 0.72,
        plate_width: Number(base.plate_width || 2.7) * 0.72
      });
    }

    function actualBlockForPair(base, pair) {
      if (!pair.is_hoogsteen || !base.hoogsteen_plate_x_axis) return base;
      return Object.assign({}, base, {
        plate_x_axis: base.hoogsteen_plate_x_axis,
        plate_y_axis: base.hoogsteen_plate_y_axis,
        plate_z_axis: base.hoogsteen_plate_z_axis,
        plate_length: base.hoogsteen_plate_length || base.plate_length,
        plate_width: base.hoogsteen_plate_width || base.plate_width,
        hbond_edge_center: base.hoogsteen_hbond_edge_center || base.hbond_edge_center,
        hbond_edge_atoms: base.hoogsteen_hbond_edge_atoms || base.hbond_edge_atoms
      });
    }

    function drawBasePlate(base, partner, outlineColor, options = {}) {
      if (!base) return;
      const color = baseColor(base.parent_base || base.residue_name);
      const corners = basePlateCorners(base, partner);
      const fillOpacity = options.fillOpacity ?? 0.68;
      const edgeRadius = options.edgeRadius ?? 0.075;
      const edgeOpacity = options.edgeOpacity ?? 0.95;
      const fillColor = options.fillColor || color;
      addFilledQuad(corners, fillColor, fillOpacity);
      addSegment(corners[0], corners[1], outlineColor, edgeRadius, edgeOpacity);
      addSegment(corners[1], corners[2], outlineColor, edgeRadius, edgeOpacity);
      addSegment(corners[2], corners[3], outlineColor, edgeRadius, edgeOpacity);
      addSegment(corners[3], corners[0], outlineColor, edgeRadius, edgeOpacity);
      if (partner && options.highlightFacingEdge !== false) {
        addSegment(corners[3], corners[0], outlineColor, edgeRadius * 1.9, 1.0);
      }
    }

    function fmt(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
      if (typeof value === "number") return value.toFixed(2);
      return String(value);
    }

    function pairAtLevel(level) {
      return (VIS.base_pairs || []).find(pair => Number(pair.level) === Number(level));
    }

    function baseAtLevel(strand, level) {
      return (VIS.base_origins || []).find(base => Number(base.level) === Number(level) && Number(base.strand || 0) === Number(strand));
    }

    function axisAtLevel(level, strand = null) {
      const points = VIS.axis || [];
      if (strand !== null && strand !== undefined) {
        const matched = points.find(point => Number(point.level) === Number(level) && Number(point.strand || 0) === Number(strand));
        if (matched) return matched;
      }
      return points.find(point => Number(point.level) === Number(level));
    }

    function addValueLabel(point, text, color = "#151a24") {
      if (!point || !text) return;
      overlayLabels.push(viewer.addLabel(text, {
        position: xyz(point),
        fontSize: 11,
        fontColor: color,
        backgroundColor: "#ffffff",
        backgroundOpacity: 0.82,
        borderThickness: 0.5,
        borderColor: color
      }));
    }

    function isUnusual(pair) {
      return pair.is_hoogsteen || pair.is_mismatch || pair.has_modified_base || !pair.shape_parameters_supported || !pair.is_canonical;
    }

    function clearOverlays() {
      overlayShapes.forEach(shape => viewer.removeShape(shape));
      overlayLabels.forEach(label => viewer.removeLabel(label));
      overlayShapes = [];
      overlayLabels = [];
    }

    function drawStructure() {
      viewer.setStyle({}, {});
      const colorMode = document.getElementById("colorMode").value;
      const baseStyle = {};
      if (document.getElementById("showCartoon").checked) {
        baseStyle.cartoon = {opacity: 0.58};
      }
      if (document.getElementById("showAllAtoms").checked) {
        baseStyle.stick = {radius: 0.11};
      }
      if (!baseStyle.cartoon && !baseStyle.stick) return;

      const analyzedNames = VIS.analyzed_residue_names || [];
      const useAnalyzedOnly = !document.getElementById("showOtherChains").checked && analyzedNames.length > 0;
      const baseSelection = useAnalyzedOnly ? {resn: analyzedNames} : {};

      function styleWith(color) {
        const style = JSON.parse(JSON.stringify(baseStyle));
        if (style.cartoon) style.cartoon.color = color;
        if (style.stick) {
          if (colorMode === "element") style.stick.colorscheme = "Jmol";
          else style.stick.color = color;
        }
        return style;
      }

      if (colorMode === "element") {
        const style = JSON.parse(JSON.stringify(baseStyle));
        if (style.cartoon) style.cartoon.colorscheme = "chainHetatm";
        if (style.stick) style.stick.colorscheme = "Jmol";
        viewer.setStyle(baseSelection, style);
        return;
      }

      if (colorMode === "chain") {
        const chains = VIS.structure_chains || [];
        if (chains.length === 0) {
          viewer.setStyle(baseSelection, styleWith("#8a8f98"));
          return;
        }
        chains.forEach((chain, index) => {
          const selection = Object.assign({}, baseSelection, {chain});
          viewer.setStyle(selection, styleWith(colorForChain(chain, index)));
        });
        return;
      }

      const residueNames = useAnalyzedOnly ? analyzedNames : (VIS.structure_residue_names || analyzedNames);
      if (residueNames.length === 0) {
        viewer.setStyle(baseSelection, styleWith("#8a8f98"));
        return;
      }
      residueNames.forEach(resn => {
        const selection = Object.assign({}, baseSelection, {resn});
        viewer.setStyle(selection, styleWith(residueColor(resn)));
      });
    }

    function drawAxis() {
      const points = VIS.axis || [];
      const groups = new Map();
      points.forEach(point => {
        const key = String(point.axis_scope || "") + ":" + String(point.strand || 0);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(point);
      });
      Array.from(groups.values()).forEach((group, index) => {
        const color = points.some(point => Number(point.strand || 0) > 0)
          ? strandColors[index % strandColors.length]
          : "#00a7c7";
        group.sort((a, b) => Number(a.level) - Number(b.level));
        for (let i = 1; i < group.length; i += 1) {
          addSegment(group[i - 1], group[i], "#07111f", 0.34, 0.28);
          addSegment(group[i - 1], group[i], color, 0.23, 1.0);
        }
        group.forEach(point => addPoint(point, color, 0.34));
      });
    }

    function drawBackbones() {
      (VIS.backbones || []).forEach((backbone, index) => {
        const color = strandColors[index % strandColors.length];
        const points = (backbone.spline_points && backbone.spline_points.length > 1) ? backbone.spline_points : (backbone.points || []);
        const usingSpline = backbone.spline_points && backbone.spline_points.length > 1;
        for (let i = 1; i < points.length; i += 1) {
          addSegment(points[i - 1], points[i], "#101820", usingSpline ? 0.24 : 0.32, 0.42);
          addSegment(points[i - 1], points[i], color, usingSpline ? 0.15 : 0.22, 1.0);
        }
        if (usingSpline) {
          (backbone.points || []).forEach(point => addPoint(point, color, 0.16));
        } else {
          points.forEach(point => addPoint(point, color, 0.28));
        }
      });
    }

    function tableRowsForTab(tab) {
      const params = VIS.parameters || {};
      if (tab === "base_pair") {
        return (params.base_pair || []).map(row => ({
          id: `base_pair:${row.level}`,
          type: "base_pair",
          level: Number(row.level),
          cells: [
            row.level,
            row.duplex || "",
            fmt(row.shear),
            fmt(row.stretch),
            fmt(row.stagger),
            fmt(row.buckle),
            fmt(row.propel),
            fmt(row.opening),
          ],
          row
        }));
      }
      if (tab === "base_pair_axis") {
        return (params.base_pair_axis || []).map(row => ({
          id: `base_pair_axis:${row.partner_strand || 0}:${row.level}`,
          type: "base_pair_axis",
          level: Number(row.level),
          cells: [
            row.level,
            row.duplex || "",
            fmt(row.xdisp),
            fmt(row.ydisp),
            fmt(row.inclin),
            fmt(row.tip),
          ],
          row
        }));
      }
      if (tab === "base_axis") {
        return (params.base_axis || []).map(row => ({
          id: `base_axis:${row.strand || 0}:${row.level}`,
          type: "base_axis",
          level: Number(row.level),
          strand: row.strand === undefined || row.strand === null ? null : Number(row.strand),
          cells: [
            row.level,
            row.strand || "",
            `${row.residue_name || ""} ${row.residue_id || ""}`.trim(),
            fmt(row.xdisp),
            fmt(row.ydisp),
            fmt(row.inclin),
            fmt(row.tip),
          ],
          row
        }));
      }
      if (tab === "global_step" || tab === "local_step") {
        const source = tab === "global_step" ? (params.global_step || []) : (params.local_step || []);
        return source.map(row => ({
          id: `${tab}:${row.strand || row.partner_strand || 0}:${row.level}`,
          type: "step",
          tab,
          level: Number(row.level),
          strand: row.strand === undefined || row.strand === null ? null : Number(row.strand),
          cells: [
            row.level,
            row.duplex || row.step || (row.strand ? `Strand ${row.strand}` : ""),
            fmt(row.shift),
            fmt(row.slide),
            fmt(row.rise),
            fmt(row.tilt),
            fmt(row.roll),
            fmt(row.twist),
          ],
          row
        }));
      }
      return (params.groove || []).map(row => ({
        id: `groove:${row.level}:${row.sub_level}`,
        type: "groove",
        level: Number(row.level),
        subLevel: Number(row.sub_level),
        cells: [
          row.level + "." + row.sub_level,
          row.base_pair || "",
          fmt(row.minor_width),
          fmt(row.minor_depth),
          fmt(row.minor_angle),
          fmt(row.major_width),
          fmt(row.major_depth),
          fmt(row.major_angle),
          fmt(row.diameter),
        ],
        row
      }));
    }

    function headersForTab(tab) {
      if (tab === "base_pair") return ["Level", "Pair", "Shear", "Stretch", "Stagger", "Buckle", "Propel", "Open"];
      if (tab === "base_pair_axis") return ["Level", "Pair", "Xdisp", "Ydisp", "Inclin", "Tip"];
      if (tab === "base_axis") return ["Level", "Strand", "Residue", "Xdisp", "Ydisp", "Inclin", "Tip"];
      if (tab === "global_step") return ["Step", "Residues", "Shift", "Slide", "Rise", "Tilt", "Roll", "Twist"];
      if (tab === "local_step") return ["Step", "Residues", "Shift", "Slide", "Rise", "Tilt", "Roll", "Twist"];
      return ["Level", "Pair", "Minor W", "Minor D", "Minor A", "Major W", "Major D", "Major A", "Diam"];
    }

    function renderParameterTable() {
      const rows = tableRowsForTab(activeTab);
      const container = document.getElementById("parameterTable");
      if (!rows.length) {
        container.innerHTML = "<div class=\"empty\">No records for this tab.</div>";
        return;
      }
      const headers = headersForTab(activeTab);
      container.innerHTML = `
        <table>
          <thead><tr>${headers.map(header => `<th>${header}</th>`).join("")}</tr></thead>
          <tbody>
            ${rows.map((entry, index) => `
              <tr data-index="${index}" class="${selectedInspection && selectedInspection.id === entry.id ? "selected" : ""}">
                ${entry.cells.map(cell => `<td>${cell}</td>`).join("")}
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
      container.querySelectorAll("tbody tr").forEach(row => {
        row.addEventListener("click", () => {
          selectedInspection = rows[Number(row.dataset.index)];
          renderParameterTable();
          redraw();
        });
      });
    }

    function setActiveTab(tab) {
      activeTab = tab;
      selectedInspection = null;
      document.querySelectorAll(".tabs button").forEach(button => {
        button.classList.toggle("active", button.dataset.tab === tab);
      });
      renderParameterTable();
      redraw();
    }

    function drawActualBlocks() {
      const onlyUnusual = document.getElementById("showOnlyUnusual").checked;
      const showLabels = document.getElementById("showLabels").checked;
      (VIS.base_pairs || []).forEach(pair => {
        if (onlyUnusual && !isUnusual(pair)) return;
        const color = pairColor(pair);
        const first = actualBlockForPair(pair.first, pair);
        const second = actualBlockForPair(pair.second, pair);
        drawBasePlate(first, second, color);
        drawBasePlate(second, first, color);
        addSegment(pair.first, pair.second, color, pair.is_canonical ? 0.05 : 0.08, pair.is_canonical ? 0.35 : 0.7);
        if (showLabels && pair.midpoint) {
          overlayLabels.push(viewer.addLabel(pair.label || `Level ${pair.level}`, {
            position: xyz(pair.midpoint),
            fontSize: 10,
            fontColor: "#162033",
            backgroundColor: "#ffffff",
            backgroundOpacity: 0.72,
            borderThickness: 0.5,
            borderColor: color
          }));
        }
      });
    }

    function drawAnalyticalBaseFrames() {
      (VIS.base_origins || []).forEach(base => {
        const analytical = analyticalBasePlate(base);
        const color = baseColor(base.parent_base || base.residue_name);
        drawBasePlate(analytical, null, "#151a24", {
          fillColor: color,
          fillOpacity: 0.22,
          edgeRadius: 0.045,
          edgeOpacity: 0.72,
          highlightFacingEdge: false
        });
        addSegment(
          analytical,
          pointFrom(addVec(vec(analytical), scaleVec(vec(base.z_axis || {x: 0, y: 0, z: 1}), 1.0))),
          "#151a24",
          0.035,
          0.65
        );
      });
    }

    function drawSelectedBasePair(entry) {
      const pair = pairAtLevel(entry.level);
      if (!pair) return;
      drawBasePlate(pair.first, pair.second, "#ff2f00", {fillOpacity: 0.42, edgeRadius: 0.13});
      drawBasePlate(pair.second, pair.first, "#ff2f00", {fillOpacity: 0.42, edgeRadius: 0.13});
      addSegment(pair.first, pair.second, "#ff2f00", 0.13, 1.0);
      addValueLabel(pair.midpoint, `Level ${entry.level}`, "#ff2f00");
    }

    function drawSelectedBaseAxis(entry) {
      const base = baseAtLevel(entry.strand, entry.level);
      if (!base) return;
      drawBasePlate(analyticalBasePlate(base), null, "#ff2f00", {fillOpacity: 0.36, edgeRadius: 0.12, highlightFacingEdge: false});
      const axis = axisAtLevel(entry.level, entry.strand);
      if (axis) {
        addPoint(axis, "#ff2f00", 0.44);
        addSegment(base.frame_origin || base, axis, "#ff2f00", 0.08, 0.85);
      }
      addValueLabel(base.frame_origin || base, `Strand ${entry.strand} level ${entry.level}`, "#ff2f00");
    }

    function drawSelectedStep(entry) {
      const first = pairAtLevel(entry.level);
      const second = pairAtLevel(entry.level + 1);
      if (first) drawSelectedBasePair({level: entry.level});
      if (second) drawSelectedBasePair({level: entry.level + 1});
      if (!first && entry.strand !== null && entry.strand !== undefined) {
        const firstBase = baseAtLevel(entry.strand, entry.level);
        const secondBase = baseAtLevel(entry.strand, entry.level + 1);
        if (firstBase) drawBasePlate(analyticalBasePlate(firstBase), null, "#ff2f00", {fillOpacity: 0.34, edgeRadius: 0.11, highlightFacingEdge: false});
        if (secondBase) drawBasePlate(analyticalBasePlate(secondBase), null, "#ff2f00", {fillOpacity: 0.34, edgeRadius: 0.11, highlightFacingEdge: false});
        if (firstBase && secondBase) addSegment(firstBase, secondBase, "#ff2f00", 0.12, 0.85);
      }
      const axis1 = axisAtLevel(entry.level, entry.strand);
      const axis2 = axisAtLevel(entry.level + 1, entry.strand);
      if (axis1 && axis2) {
        addSegment(axis1, axis2, "#ff2f00", 0.32, 1.0);
        const label = entry.strand ? `Strand ${entry.strand} step ${entry.level}` : `Step ${entry.level}`;
        addValueLabel(pointFrom(scaleVec(addVec(vec(axis1), vec(axis2)), 0.5)), label, "#ff2f00");
      }
    }

    function drawGrooveMeasurement(row, side, color) {
      const geometry = row.geometry && row.geometry[side];
      const value = side === "minor" ? row.minor_width : row.major_width;
      if (!geometry || value === null || value === undefined) return;
      addSegment(geometry.width_endpoint_1, geometry.width_endpoint_2, color, 0.15, 1.0);
      addSegment(geometry.depth_reference, geometry.depth_point, color, 0.08, 0.86);
      addPoint(geometry.width_endpoint_1, color, 0.24);
      addPoint(geometry.width_endpoint_2, color, 0.24);
      addPoint(geometry.depth_point, color, 0.2);
      const mid = pointFrom(scaleVec(addVec(vec(geometry.width_endpoint_1), vec(geometry.width_endpoint_2)), 0.5));
      addValueLabel(mid, `${side} ${fmt(value)} A`, color);
    }

    function drawSelectedGroove(entry) {
      drawGrooveMeasurement(entry.row, "minor", "#d12b72");
      drawGrooveMeasurement(entry.row, "major", "#6246ea");
      const axis = axisAtLevel(entry.level);
      if (axis) {
        addPoint(axis, "#111827", 0.42);
        addValueLabel(axis, `Groove ${entry.level}.${entry.subLevel}`, "#111827");
      }
    }

    function drawInspectionSelection() {
      if (!selectedInspection) return;
      if (selectedInspection.type === "base_pair") drawSelectedBasePair(selectedInspection);
      if (selectedInspection.type === "base_pair_axis") drawSelectedBasePair(selectedInspection);
      if (selectedInspection.type === "base_axis") drawSelectedBaseAxis(selectedInspection);
      if (selectedInspection.type === "step") drawSelectedStep(selectedInspection);
      if (selectedInspection.type === "groove") drawSelectedGroove(selectedInspection);
    }

    function redraw() {
      if (!viewer) return;
      clearOverlays();
      drawStructure();
      if (document.getElementById("showAxis").checked) drawAxis();
      if (document.getElementById("showBackbone").checked) drawBackbones();
      if (document.getElementById("showActualBlocks").checked) drawActualBlocks();
      if (document.getElementById("showAnalyticalBases").checked) drawAnalyticalBaseFrames();
      drawInspectionSelection();
      viewer.render();
    }

    function initialize() {
      if (!window.$3Dmol) {
        const error = document.getElementById("loadError");
        error.hidden = false;
        error.textContent = "3Dmol.js could not be loaded. Check the network connection or serve this page with a bundled viewer library.";
        return;
      }
      viewer = $3Dmol.createViewer("viewer", {backgroundColor: "white"});
      viewer.addModel(STRUCTURE_TEXT, STRUCTURE_FORMAT);
      redraw();
      viewer.zoomTo();
      viewer.render();
    }

    document.querySelectorAll("input[type='checkbox']").forEach(input => {
      input.addEventListener("change", redraw);
    });
    document.querySelectorAll(".tabs button").forEach(button => {
      button.classList.toggle("active", button.dataset.tab === activeTab);
      button.addEventListener("click", () => setActiveTab(button.dataset.tab));
    });
    document.getElementById("colorMode").addEventListener("change", redraw);
    document.getElementById("resetView").addEventListener("click", () => {
      if (!viewer) return;
      viewer.zoomTo();
      viewer.render();
    });
    renderParameterTable();
    window.addEventListener("load", initialize);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
