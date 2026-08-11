from __future__ import annotations

import argparse
import gzip
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
    suffix = path.with_suffix("").suffix.lower() if path.suffix.lower() == ".gz" else path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return "cif"
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    return suffix.lstrip(".") or "pdb"


def _read_structure_text(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_results(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_viewer_html(results: Dict[str, Any], structure_text: str, structure_format: str, title: str) -> str:
    visualization = dict(results.get("visualization", {}))
    parameters = dict(visualization.get("parameters", {}))
    for name in (
        "global_base_axis",
        "global_base_pair_axis",
        "curvesplus_base_pair_axis",
        "global_base_base",
        "global_inter_base",
        "global_inter_base_pair",
        "global_axis_curvature",
        "global_axis_bending",
        "global_axis_bending_summary",
        "local_base_base",
        "local_inter_base",
        "local_inter_base_pair",
        "backbone",
    ):
        rows = results.get("dataframes", {}).get(name)
        if isinstance(rows, list):
            parameters[name] = rows
    visualization["parameters"] = parameters
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

    structure_text = _read_structure_text(structure_path)
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
      grid-template-columns: 290px minmax(0, 1fr);
      grid-template-rows: minmax(260px, 1fr) auto minmax(220px, 32vh);
      height: 100vh;
    }
    aside {
      grid-row: 1 / -1;
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px 16px;
      overflow: auto;
    }
    main {
      grid-column: 2;
      grid-row: 1;
      min-width: 0;
      min-height: 0;
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
    .view-actions {
      display: flex;
      margin: 14px 0;
    }
    .view-actions button {
      width: auto;
      padding: 0 12px;
    }
    .display-options {
      margin-top: 14px;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 10px 0;
    }
    .display-options summary {
      color: var(--muted);
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
    }
    .option-grid {
      display: grid;
      gap: 2px;
      padding-top: 8px;
    }
    .select-label {
      display: block;
      min-height: 0;
      margin: 7px 0 4px;
      color: var(--muted);
      cursor: default;
      font-size: 12px;
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
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin: 0;
    }
    .tabs button {
      width: auto;
      height: 30px;
      font-size: 12px;
      padding: 0 10px;
    }
    .tabs button.active {
      border-color: #9fc4e5;
      color: var(--accent);
      background: #eaf3fb;
      font-weight: 700;
    }
    .feature-groups {
      display: flex;
      min-width: 0;
      flex: 1;
      flex-wrap: wrap;
      gap: 6px 18px;
    }
    .feature-group {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 5px 8px;
    }
    .feature-group > span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .sequence-panel {
      grid-column: 2;
      grid-row: 2;
      min-width: 0;
      padding: 8px 14px 9px;
      border-top: 1px solid var(--line);
      background: #ffffff;
    }
    .sequence-panel[hidden] {
      display: none;
    }
    .sequence-head {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 5px 14px;
      margin-bottom: 5px;
    }
    .sequence-head h2 {
      margin: 0;
    }
    .sequence-summary {
      flex: 1;
      min-width: 220px;
      color: var(--muted);
      font-size: 12px;
    }
    .sequence-legend {
      display: flex;
      gap: 10px;
      color: var(--muted);
      font-size: 11px;
    }
    .legend-mark {
      display: inline-block;
      width: 8px;
      height: 8px;
      margin-right: 4px;
      border-radius: 50%;
    }
    .legend-mark.critical {
      background: #c23b22;
    }
    .legend-mark.warning {
      background: #c58a10;
    }
    .sequence-scroll {
      overflow-x: auto;
      padding: 2px 0 4px;
      scrollbar-gutter: stable;
    }
    .sequence-row {
      display: flex;
      width: max-content;
      min-width: 100%;
      align-items: flex-end;
    }
    .sequence-strand {
      position: sticky;
      left: 0;
      z-index: 2;
      width: 70px;
      flex: 0 0 70px;
      padding: 0 7px 7px 0;
      color: var(--muted);
      background: #ffffff;
      font-size: 10px;
      text-align: right;
    }
    .sequence-base,
    .sequence-gap {
      width: 28px;
      height: 38px;
      flex: 0 0 28px;
    }
    .sequence-base {
      position: relative;
      border: 0;
      border-bottom: 3px solid transparent;
      border-radius: 4px;
      padding: 11px 0 0;
      color: var(--base-color);
      background: transparent;
      font-size: 17px;
      font-weight: 700;
      line-height: 22px;
    }
    .sequence-base::before {
      content: attr(data-residue);
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      color: #8a93a3;
      font-size: 9px;
      font-weight: 400;
      line-height: 11px;
    }
    .sequence-base:hover {
      background: #f1f5f9;
    }
    .sequence-base.critical {
      border-bottom-color: #c23b22;
      background: #fff2ee;
    }
    .sequence-base.warning {
      border-bottom-color: #c58a10;
      background: #fff8e7;
    }
    .sequence-base.selected {
      outline: 2px solid var(--accent);
      outline-offset: -2px;
      background: #eaf3fb;
    }
    .sequence-controls {
      display: flex;
      align-items: center;
      gap: 8px;
      padding-left: 70px;
    }
    .sequence-controls input {
      min-width: 160px;
      flex: 1;
    }
    .sequence-controls output {
      min-width: 58px;
      color: var(--muted);
      font-size: 11px;
      text-align: right;
    }
    .parameters-panel {
      grid-column: 2;
      grid-row: 3;
      display: flex;
      min-width: 0;
      min-height: 0;
      flex-direction: column;
      gap: 8px;
      padding: 10px 14px 12px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }
    .parameters-head {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px 16px;
    }
    .parameters-head h2 {
      margin: 0;
    }
    .table-wrap {
      flex: 1;
      min-height: 0;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      scrollbar-gutter: stable both-edges;
    }
    table {
      min-width: 720px;
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
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
    td.selected-feature {
      background: #ffd79a;
      color: #111827;
      font-weight: 700;
    }
    .annotation-badge {
      display: inline-block;
      min-width: 28px;
      margin-right: 6px;
      padding: 1px 4px;
      border-radius: 999px;
      color: #ffffff;
      font-size: 9px;
      font-weight: 700;
      line-height: 14px;
      text-align: center;
      vertical-align: 1px;
    }
    .annotation-badge.critical {
      background: #c23b22;
    }
    .annotation-badge.warning {
      background: #c58a10;
    }
    .empty {
      padding: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 820px) {
      .app {
        grid-template-columns: 1fr;
        grid-template-rows: auto minmax(420px, 55vh) auto minmax(240px, 35vh);
        height: auto;
        min-height: 100vh;
      }
      aside {
        grid-column: 1;
        grid-row: 1;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      main {
        grid-column: 1;
        grid-row: 2;
      }
      .sequence-panel {
        grid-column: 1;
        grid-row: 3;
      }
      .parameters-panel {
        grid-column: 1;
        grid-row: 4;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1 id="title"></h1>
      <div class="meta" id="meta"></div>
      <div class="view-actions">
        <button id="resetView" type="button">Reset view</button>
      </div>
      <details class="display-options">
        <summary>Display options</summary>
        <div class="option-grid">
          <label><input type="checkbox" id="showOtherChains" checked> Show chains other than DNA</label>
          <label><input type="checkbox" id="showCartoon" checked> Cartoon</label>
          <label><input type="checkbox" id="showAllAtoms"> All atoms</label>
          <label class="select-label" for="colorMode">Molecule color</label>
          <select id="colorMode">
            <option value="chain">By chain</option>
            <option value="residue">By residue</option>
            <option value="element">By element</option>
          </select>
          <label><input type="checkbox" id="showAxis" checked> Smoothed axis</label>
          <label><input type="checkbox" id="showBackbone" checked> Backbone spline curve</label>
          <label><input type="checkbox" id="showActualBlocks" checked> Actual base blocks</label>
          <label><input type="checkbox" id="showAnalyticalBases"> Analytical base frames</label>
          <label><input type="checkbox" id="showLabels"> Labels</label>
          <label><input type="checkbox" id="showOnlyUnusual"> Only unusual pairs</label>
        </div>
      </details>
      <p class="load-error" id="loadError" hidden></p>
    </aside>
    <main>
      <div id="viewer"></div>
    </main>
    <section class="sequence-panel" aria-labelledby="sequenceTitle">
      <div class="sequence-head">
        <h2 id="sequenceTitle">DNA sequence</h2>
        <div class="sequence-summary" id="sequenceSummary">Choose a base to inspect it in 3D.</div>
        <div class="sequence-legend" aria-label="Annotation legend">
          <span><i class="legend-mark critical"></i>Critical</span>
          <span><i class="legend-mark warning"></i>Review</span>
        </div>
      </div>
      <div class="sequence-scroll" id="sequenceViewer"></div>
      <div class="sequence-controls">
        <input id="sequenceSlider" type="range" aria-label="DNA sequence level">
        <output id="sequenceLevel" for="sequenceSlider"></output>
      </div>
    </section>
    <section class="parameters-panel" aria-labelledby="parametersTitle">
      <div class="parameters-head">
        <h2 id="parametersTitle">DNA shape</h2>
        <div class="feature-groups">
          <div class="feature-group" data-feature-group="global">
            <span>Global features</span>
            <div class="tabs">
              <button type="button" data-tab="global_base_axis">Base Axis</button>
              <button type="button" data-tab="global_base_pair_axis">BP Axis</button>
              <button type="button" data-tab="global_base_pair">Base Pair</button>
              <button type="button" data-tab="global_strand_step">Strand Step</button>
              <button type="button" data-tab="global_base_pair_step">BP Step</button>
              <button type="button" data-tab="global_axis_curvature">Axis Curvature</button>
              <button type="button" data-tab="global_axis_bending">Axis Bending</button>
              <button type="button" data-tab="global_axis_bending_summary">Bending Summary</button>
            </div>
          </div>
          <div class="feature-group" data-feature-group="local">
            <span>Local features</span>
            <div class="tabs">
              <button type="button" data-tab="local_base_pair">Base Pair</button>
              <button type="button" data-tab="local_strand_step">Strand Step</button>
              <button type="button" data-tab="local_base_pair_step">BP Step</button>
              <button type="button" data-tab="backbone">Backbone</button>
              <button type="button" data-tab="groove">Groove</button>
            </div>
          </div>
        </div>
      </div>
      <div class="table-wrap" id="parameterTable"></div>
    </section>
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
    let selectionShapes = [];
    let selectionLabels = [];
    let drawingSelection = false;
    const parameterTabOrder = [
      "global_base_axis", "global_base_pair_axis", "global_base_pair",
      "global_strand_step", "global_base_pair_step", "global_axis_curvature",
      "global_axis_bending", "global_axis_bending_summary", "local_base_pair",
      "local_strand_step", "local_base_pair_step", "backbone", "groove"
    ];

    function initialParameterTab() {
      return parameterTabOrder.find(tab => tableRowsForTab(tab).length) || "global_base_pair";
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

    function dotVec(left, right) {
      return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    }

    function alignVec(values, reference) {
      return dotVec(values, reference) < 0 ? scaleVec(values, -1) : values;
    }

    function addShape(shape) {
      (drawingSelection ? selectionShapes : overlayShapes).push(shape);
      return shape;
    }

    function addLabel(label) {
      (drawingSelection ? selectionLabels : overlayLabels).push(label);
      return label;
    }

    function addCurve(points, color, radius, opacity = 1.0) {
      if (!points || points.length < 2) return;
      addShape(viewer.addCurve({
        points: points.map(xyz),
        radius,
        color,
        opacity,
        smooth: 1
      }));
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

    function addDashedSegment(a, b, color, radius, segments = 9, opacity = 1.0) {
      if (!a || !b) return;
      const start = vec(a);
      const delta = subVec(vec(b), start);
      for (let i = 0; i < segments; i += 2) {
        const t1 = i / segments;
        const t2 = Math.min((i + 1) / segments, 1.0);
        addSegment(
          pointFrom(addVec(start, scaleVec(delta, t1))),
          pointFrom(addVec(start, scaleVec(delta, t2))),
          color,
          radius,
          opacity
        );
      }
    }

    function addPoint(point, color, radius) {
      if (!point) return;
      addShape(viewer.addSphere({center: xyz(point), radius, color}));
    }

    function addArrowLikeSegment(a, b, color, radius, opacity = 1.0) {
      if (!a || !b) return;
      addSegment(a, b, color, radius, opacity);
      const direction = unitVec(subVec(vec(b), vec(a)), [1, 0, 0]);
      const end = vec(b);
      const fallback = Math.abs(direction[2]) < 0.82 ? [0, 0, 1] : [0, 1, 0];
      const side1 = unitVec(crossVec(direction, fallback), [0, 1, 0]);
      const side2 = unitVec(crossVec(direction, side1), [0, 0, 1]);
      const headLength = Math.max(radius * 8.0, 0.22);
      const headWidth = Math.max(radius * 4.2, 0.12);
      const back = subVec(end, scaleVec(direction, headLength));
      addSegment(b, pointFrom(addVec(back, scaleVec(side1, headWidth))), color, radius, opacity);
      addSegment(b, pointFrom(subVec(back, scaleVec(side1, headWidth))), color, radius, opacity);
      addSegment(b, pointFrom(addVec(back, scaleVec(side2, headWidth))), color, radius, opacity);
      addSegment(b, pointFrom(subVec(back, scaleVec(side2, headWidth))), color, radius, opacity);
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
      const contactFrame = String(base.analysis_frame_source || "") === "contact_geometry";
      return Object.assign({}, base, {
        x: origin.x,
        y: origin.y,
        z: origin.z,
        plate_x_axis: base.x_axis || base.plate_x_axis || base.y_axis,
        plate_y_axis: base.y_axis || base.plate_y_axis || base.x_axis,
        plate_z_axis: base.z_axis || base.plate_z_axis,
        plate_length: contactFrame ? 2.2 : Number(base.plate_length || 5.0) * 0.58,
        plate_width: contactFrame ? 1.1 : Number(base.plate_width || 2.7) * 0.58
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
      if (options.outline !== false) {
        addSegment(corners[0], corners[1], outlineColor, edgeRadius, edgeOpacity);
        addSegment(corners[1], corners[2], outlineColor, edgeRadius, edgeOpacity);
        addSegment(corners[2], corners[3], outlineColor, edgeRadius, edgeOpacity);
        addSegment(corners[3], corners[0], outlineColor, edgeRadius, edgeOpacity);
        if (partner && options.highlightFacingEdge !== false) {
          addSegment(corners[3], corners[0], outlineColor, edgeRadius * 1.9, 1.0);
        }
      }
    }

    function drawFrameAxes(base, scale = 1.15) {
      if (!base) return;
      const origin = base.frame_origin || base;
      const xAxis = unitVec(vec(base.x_axis || {x: 1, y: 0, z: 0}), [1, 0, 0]);
      const yAxis = unitVec(vec(base.y_axis || {x: 0, y: 1, z: 0}), [0, 1, 0]);
      const zAxis = unitVec(vec(base.z_axis || {x: 0, y: 0, z: 1}), [0, 0, 1]);
      addPoint(origin, "#151a24", 0.09);
      addSegment(origin, pointFrom(addVec(vec(origin), scaleVec(xAxis, scale))), "#d22f2f", 0.035, 0.9);
      addSegment(origin, pointFrom(addVec(vec(origin), scaleVec(yAxis, scale))), "#268a3e", 0.035, 0.9);
      addSegment(origin, pointFrom(addVec(vec(origin), scaleVec(zAxis, scale))), "#2866d8", 0.035, 0.9);
    }

    function clampNumber(value, minimum, maximum) {
      return Math.max(minimum, Math.min(maximum, value));
    }

    function framePoint(frame, axisName, distance) {
      return pointFrom(addVec(vec(frame.origin), scaleVec(frame[axisName], distance)));
    }

    function baseAnalysisFrame(base) {
      if (!base) return null;
      const analytical = analyticalBasePlate(base);
      return {
        origin: analytical.frame_origin || analytical,
        x: unitVec(vec(analytical.x_axis || {x: 1, y: 0, z: 0}), [1, 0, 0]),
        y: unitVec(vec(analytical.y_axis || {x: 0, y: 1, z: 0}), [0, 1, 0]),
        z: unitVec(vec(analytical.z_axis || {x: 0, y: 0, z: 1}), [0, 0, 1])
      };
    }

    function averageFrames(first, second, origin = null) {
      if (!first) return second;
      if (!second) return first;
      const x2 = alignVec(second.x, first.x);
      const y2 = alignVec(second.y, first.y);
      const z2 = alignVec(second.z, first.z);
      const xAxis = unitVec(addVec(first.x, x2), first.x);
      let yAxis = unitVec(addVec(first.y, y2), first.y);
      yAxis = unitVec(subVec(yAxis, scaleVec(xAxis, dotVec(yAxis, xAxis))), first.y);
      const zPreferred = unitVec(addVec(first.z, z2), first.z);
      let zAxis = unitVec(crossVec(xAxis, yAxis), zPreferred);
      if (dotVec(zAxis, zPreferred) < 0) {
        yAxis = scaleVec(yAxis, -1);
        zAxis = scaleVec(zAxis, -1);
      }
      const frameOrigin = origin || pointFrom(scaleVec(addVec(vec(first.origin), vec(second.origin)), 0.5));
      return {origin: frameOrigin, x: xAxis, y: yAxis, z: zAxis};
    }

    function pairFrame(pair) {
      if (!pair) return null;
      const first = baseAnalysisFrame(pair.first);
      const second = baseAnalysisFrame(pair.second);
      const origin = pair.frame_midpoint || pair.midpoint || null;
      return averageFrames(first, second, origin);
    }

    function drawFrameGlyph(frame, scale = 1.35, color = "#111827") {
      if (!frame) return;
      addPoint(frame.origin, color, 0.11);
      addSegment(frame.origin, framePoint(frame, "x", scale), "#d22f2f", 0.045, 0.95);
      addSegment(frame.origin, framePoint(frame, "y", scale), "#268a3e", 0.045, 0.95);
      addSegment(frame.origin, framePoint(frame, "z", scale), "#2866d8", 0.045, 0.95);
    }

    function featureLabel(feature) {
      const labels = {
        shear: "Shear", stretch: "Stretch", stagger: "Stagger",
        buckle: "Buckle", propel: "Propel", opening: "Opening",
        xdisp: "Xdisp", ydisp: "Ydisp", inclin: "Inclin", tip: "Tip",
        shift: "Shift", slide: "Slide", rise: "Rise",
        tilt: "Tilt", roll: "Roll", twist: "Twist",
        minor_width: "Minor width", minor_depth: "Minor depth", minor_angle: "Minor angle",
        major_width: "Major width", major_depth: "Major depth", major_angle: "Major angle",
        diameter: "Diameter"
      };
      return labels[feature] || feature || "Feature";
    }

    function featureUnit(feature) {
      if (["buckle", "propel", "opening", "inclin", "tip", "tilt", "roll", "twist", "minor_angle", "major_angle"].includes(feature)) return "deg";
      if (["shear", "stretch", "stagger", "xdisp", "ydisp", "shift", "slide", "rise", "minor_width", "minor_depth", "major_width", "major_depth", "diameter"].includes(feature)) return "A";
      return "";
    }

    function translationAxisForFeature(feature) {
      return {
        shear: "x", stretch: "y", stagger: "z",
        xdisp: "x", ydisp: "y",
        shift: "x", slide: "y", rise: "z"
      }[feature] || null;
    }

    function rotationAxisForFeature(feature) {
      return {
        buckle: "x", propel: "y", opening: "z",
        inclin: "x", tip: "y",
        tilt: "x", roll: "y", twist: "z"
      }[feature] || null;
    }

    function drawAxisGuide(frame, axisName, color) {
      addDashedSegment(framePoint(frame, axisName, -1.75), framePoint(frame, axisName, 1.75), color, 0.035, 10, 0.55);
    }

    function drawTranslationFeature(frame, feature, value, color) {
      const axisName = translationAxisForFeature(feature);
      if (!frame || !axisName) return false;
      const numeric = Number(value);
      drawAxisGuide(frame, axisName, color);
      const sign = Number.isFinite(numeric) && numeric < 0 ? -1 : 1;
      const distance = sign * clampNumber(Number.isFinite(numeric) ? Math.abs(numeric) : 1.2, 0.75, 3.2);
      const end = framePoint(frame, axisName, distance);
      addArrowLikeSegment(frame.origin, end, color, 0.09, 0.96);
      const unit = featureUnit(feature);
      addValueLabel(end, `${featureLabel(feature)} ${fmt(value)}${unit ? " " + unit : ""}`, color);
      return true;
    }

    function drawRotationFeature(frame, feature, value, color) {
      const axisName = rotationAxisForFeature(feature);
      if (!frame || !axisName) return false;
      const numeric = Number(value);
      drawAxisGuide(frame, axisName, color);
      const axis = frame[axisName];
      const startAxisName = axisName === "x" ? "y" : "x";
      const startDirection = unitVec(subVec(frame[startAxisName], scaleVec(axis, dotVec(frame[startAxisName], axis))), [1, 0, 0]);
      const turnDirection = unitVec(crossVec(axis, startDirection), [0, 1, 0]);
      const sign = Number.isFinite(numeric) && numeric < 0 ? -1 : 1;
      const sweepDegrees = clampNumber(Number.isFinite(numeric) ? Math.abs(numeric) : 45.0, 18.0, 125.0);
      const sweep = sign * sweepDegrees * Math.PI / 180.0;
      const radius = 1.25;
      const steps = 14;
      let previous = null;
      let current = null;
      for (let i = 0; i <= steps; i += 1) {
        const theta = sweep * i / steps;
        const direction = addVec(scaleVec(startDirection, Math.cos(theta)), scaleVec(turnDirection, Math.sin(theta)));
        current = pointFrom(addVec(vec(frame.origin), scaleVec(direction, radius)));
        if (previous) {
          if (i === steps) addArrowLikeSegment(previous, current, color, 0.055, 0.96);
          else addSegment(previous, current, color, 0.055, 0.96);
        }
        previous = current;
      }
      const unit = featureUnit(feature);
      addValueLabel(current, `${featureLabel(feature)} ${fmt(value)}${unit ? " " + unit : ""}`, color);
      return true;
    }

    function drawFeatureMeasurement(frame, feature, value, color) {
      if (!feature) return false;
      return drawTranslationFeature(frame, feature, value, color) || drawRotationFeature(frame, feature, value, color);
    }

    function drawAnalyticalPairSource(pair, color) {
      if (!pair) return;
      const first = analyticalBasePlate(pair.first);
      const second = analyticalBasePlate(pair.second);
      drawFrameAxes(first, first.analysis_frame_source === "contact_geometry" ? 1.35 : 1.1);
      drawFrameAxes(second, second.analysis_frame_source === "contact_geometry" ? 1.35 : 1.1);
      const firstOrigin = first.frame_origin || first;
      const secondOrigin = second.frame_origin || second;
      addDashedSegment(firstOrigin, secondOrigin, color, 0.045, 9, 0.85);
    }

    function fmt(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
      if (typeof value === "number") return value.toFixed(2);
      return String(value);
    }

    function pairAtLevel(level, partnerStrand = null) {
      return (VIS.base_pairs || []).find(pair => {
        if (Number(pair.level) !== Number(level)) return false;
        if (partnerStrand === null || partnerStrand === undefined || Number(partnerStrand) === 0) return true;
        return Number(pair.first && pair.first.strand || 0) === Number(partnerStrand)
          || Number(pair.second && pair.second.strand || 0) === Number(partnerStrand);
      });
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
      addLabel(viewer.addLabel(text, {
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
      return Boolean(pairAnnotation(pair));
    }

    function pairAnnotation(pair) {
      if (!pair) return null;
      const observed = String(pair.observed_lw_family || "").trim();
      const family = String(pair.pair_family || "").trim().toLowerCase();
      const mode = String(pair.pairing_mode || "").trim().toLowerCase();
      const status = String(pair.pair_status || "present").trim().toLowerCase();
      if (pair.has_modified_base) {
        return {kind: "critical", label: "Modified", detail: "Modified base in pair"};
      }
      const namedModes = {
        reverse_watson_crick: ["Reverse WC", "Reverse Watson-Crick base pair"],
        hoogsteen: ["Hoogsteen", "Hoogsteen base pair"],
        reverse_hoogsteen: ["Reverse Hoogsteen", "Reverse Hoogsteen base pair"],
        wobble: ["Wobble", "Wobble base pair"]
      };
      if (namedModes[mode]) {
        return {kind: "critical", label: namedModes[mode][0], detail: namedModes[mode][1]};
      }
      if (mode === "other_noncanonical") {
        return {kind: "critical", label: observed || "Noncanonical", detail: `Other noncanonical base pair${observed ? ": " + observed : ""}`};
      }
      if (pair.is_hoogsteen || family.includes("hoogsteen")) {
        return {kind: "critical", label: "Hoogsteen", detail: "Hoogsteen base pair"};
      }
      if (family === "wobble") {
        return {kind: "critical", label: "Wobble", detail: "Wobble base pair"};
      }
      if (pair.is_mismatch || family === "mismatch") {
        return {kind: "critical", label: "Mismatch", detail: "Mismatched base pair"};
      }
      if ((observed && observed.toLowerCase() !== "cww") || family.includes("noncanonical")) {
        const label = observed && observed.toLowerCase() !== "cww" ? observed : "Non-WC";
        return {kind: "critical", label, detail: `Non-canonical contact${observed ? ": " + observed : ""}`};
      }
      if (!pair.shape_parameters_supported) {
        return {kind: "critical", label: "No shape", detail: "Shape parameters are unsupported for this pair"};
      }
      if (status !== "present") {
        return {kind: "warning", label: status || "Review", detail: `Pair status: ${status || "uncertain"}`};
      }
      const flags = (pair.diagnostic_flags || []).filter(Boolean);
      if (flags.length) {
        return {kind: "warning", label: "Review", detail: flags.map(flag => String(flag).replaceAll("_", " ")).join("; ")};
      }
      return null;
    }

    function highlightSequenceLevel(level) {
      const numericLevel = Number(level);
      document.querySelectorAll(".sequence-base.selected").forEach(base => base.classList.remove("selected"));
      const selectedBases = document.querySelectorAll(`.sequence-base[data-level="${numericLevel}"]`);
      selectedBases.forEach(base => base.classList.add("selected"));
      if (selectedBases.length) selectedBases[0].scrollIntoView({block: "nearest", inline: "center"});
      const slider = document.getElementById("sequenceSlider");
      slider.value = numericLevel;
      document.getElementById("sequenceLevel").value = `Level ${numericLevel}`;
      const pair = pairAtLevel(numericLevel);
      const annotation = pairAnnotation(pair);
      const bases = (VIS.base_origins || []).filter(base => Number(base.level) === numericLevel);
      const labels = bases.map(base => `${base.chain_id || "S" + base.strand}:${base.parent_base || base.residue_name}${base.residue_id}`).join(" / ");
      document.getElementById("sequenceSummary").textContent = pair
        ? `Level ${numericLevel} · ${pair.label || labels}${annotation ? " · " + annotation.detail : ""}`
        : `Level ${numericLevel} · ${labels || "no base"} · no inferred base pair`;
    }

    function selectSequenceLevel(level, focusViewer = false) {
      const numericLevel = Number(level);
      highlightSequenceLevel(numericLevel);
      const pair = pairAtLevel(numericLevel);
      activeTab = tableRowsForTab("global_base_pair").length ? "global_base_pair" : "local_base_pair";
      document.querySelectorAll(".tabs button").forEach(button => {
        button.classList.toggle("active", button.dataset.tab === activeTab);
      });
      const entry = tableRowsForTab(activeTab).find(row => Number(row.level) === numericLevel);
      selectedInspection = entry
        ? Object.assign({}, entry, {type: pair ? "base_pair" : "sequence", feature: null, featureLabel: null, featureValue: null})
        : {id: `sequence:${numericLevel}`, type: "sequence", level: numericLevel};
      renderParameterTable();
      redrawSelection();
      const tableRow = document.querySelector("#parameterTable tbody tr.selected");
      if (tableRow) tableRow.scrollIntoView({block: "nearest"});
      if (focusViewer && viewer) {
        const bases = pair
          ? [pair.first, pair.second]
          : (VIS.base_origins || []).filter(base => Number(base.level) === numericLevel);
        const selections = bases.filter(Boolean).map(base => {
          const selection = {resi: Number(base.residue_id)};
          if (base.chain_id) selection.chain = base.chain_id;
          return selection;
        });
        if (selections.length) {
          viewer.center({or: selections}, 250);
          viewer.render();
        }
      }
    }

    function renderSequenceNavigator() {
      const bases = VIS.base_origins || [];
      const panel = document.querySelector(".sequence-panel");
      if (!bases.length) {
        panel.hidden = true;
        return;
      }
      const groups = new Map();
      bases.forEach(base => {
        const strand = Number(base.strand || 0);
        if (!groups.has(strand)) groups.set(strand, new Map());
        groups.get(strand).set(Number(base.level), base);
      });
      const levels = bases.map(base => Number(base.level));
      const minLevel = Math.min(...levels);
      const maxLevel = Math.max(...levels);
      const pairByLevel = new Map((VIS.base_pairs || []).map(pair => [Number(pair.level), pair]));
      const container = document.getElementById("sequenceViewer");
      container.replaceChildren();
      Array.from(groups.entries()).sort((left, right) => left[0] - right[0]).forEach(([strand, basesByLevel], index) => {
        const row = document.createElement("div");
        row.className = "sequence-row";
        const strandLabel = document.createElement("span");
        strandLabel.className = "sequence-strand";
        strandLabel.textContent = `S${strand} ${index === 0 ? "5′→3′" : "3′→5′"}`;
        row.appendChild(strandLabel);
        for (let level = minLevel; level <= maxLevel; level += 1) {
          const base = basesByLevel.get(level);
          if (!base) {
            const gap = document.createElement("span");
            gap.className = "sequence-gap";
            row.appendChild(gap);
            continue;
          }
          const button = document.createElement("button");
          const pair = pairByLevel.get(level);
          const annotation = pair ? pairAnnotation(pair) : (groups.size > 1 ? {kind: "warning", detail: "No inferred base pair"} : null);
          const letter = String(base.parent_base || base.residue_name || "?").replace(/^D/i, "").slice(0, 1).toUpperCase();
          button.type = "button";
          button.className = `sequence-base${annotation ? " " + annotation.kind : ""}`;
          button.dataset.level = level;
          button.dataset.residue = base.residue_id;
          button.style.setProperty("--base-color", baseColor(letter));
          button.textContent = letter;
          button.title = `${base.chain_id || "Strand " + strand}:${base.residue_name}${base.residue_id} · level ${level}${annotation ? " · " + annotation.detail : ""}`;
          button.setAttribute("aria-label", button.title);
          button.addEventListener("click", () => selectSequenceLevel(level, true));
          row.appendChild(button);
        }
        container.appendChild(row);
      });
      const slider = document.getElementById("sequenceSlider");
      slider.min = minLevel;
      slider.max = maxLevel;
      slider.step = 1;
      slider.value = minLevel;
      slider.addEventListener("input", event => highlightSequenceLevel(event.target.value));
      slider.addEventListener("change", event => selectSequenceLevel(event.target.value, true));
      highlightSequenceLevel(minLevel);
    }

    function clearOverlays() {
      overlayShapes.forEach(shape => viewer.removeShape(shape));
      overlayLabels.forEach(label => viewer.removeLabel(label));
      selectionShapes.forEach(shape => viewer.removeShape(shape));
      selectionLabels.forEach(label => viewer.removeLabel(label));
      overlayShapes = [];
      overlayLabels = [];
      selectionShapes = [];
      selectionLabels = [];
    }

    function clearSelection() {
      selectionShapes.forEach(shape => viewer.removeShape(shape));
      selectionLabels.forEach(label => viewer.removeLabel(label));
      selectionShapes = [];
      selectionLabels = [];
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
        if (group.length === 1) addPoint(group[0], color, 0.34);
        addCurve(group, "#07111f", 0.34, 0.28);
        addCurve(group, color, 0.23, 1.0);
      });
    }

    function drawBackbones() {
      (VIS.backbones || []).forEach((backbone, index) => {
        const color = strandColors[index % strandColors.length];
        const points = (backbone.spline_points && backbone.spline_points.length > 1) ? backbone.spline_points : (backbone.points || []);
        const usingSpline = backbone.spline_points && backbone.spline_points.length > 1;
        if (points.length === 1) addPoint(points[0], color, usingSpline ? 0.16 : 0.28);
        addCurve(points, "#101820", usingSpline ? 0.24 : 0.32, 0.42);
        addCurve(points, color, usingSpline ? 0.15 : 0.22, 1.0);
      });
    }

    function tableRowsForTab(tab) {
      const params = VIS.parameters || {};
      const rowsFor = (name, fallback = null) => {
        if (Object.prototype.hasOwnProperty.call(params, name)) return params[name] || [];
        return fallback ? (params[fallback] || []) : [];
      };
      if (tab === "global_base_pair" || tab === "local_base_pair") {
        const source = tab === "global_base_pair"
          ? rowsFor("global_base_base", "base_pair")
          : rowsFor("local_base_base", "local_base_pair");
        return source.map(row => ({
          id: `${tab}:${row.partner_strand || 0}:${row.level}`,
          type: "base_pair",
          level: Number(row.level),
          partnerStrand: row.partner_strand === undefined || row.partner_strand === null ? null : Number(row.partner_strand),
          features: [null, null, "shear", "stretch", "stagger", "buckle", "propel", "opening"],
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
      if (tab === "global_base_pair_axis") {
        const source = rowsFor("global_base_pair_axis").length
          ? rowsFor("global_base_pair_axis")
          : (rowsFor("curvesplus_base_pair_axis").length ? rowsFor("curvesplus_base_pair_axis") : rowsFor("base_pair_axis"));
        return source.map(row => ({
          id: `${tab}:${row.partner_strand || 0}:${row.level}`,
          type: "base_pair_axis",
          level: Number(row.level),
          partnerStrand: row.partner_strand === undefined || row.partner_strand === null ? null : Number(row.partner_strand),
          features: [null, null, "xdisp", "ydisp", "inclin", "tip"],
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
      if (tab === "global_base_axis") {
        return rowsFor("global_base_axis", "base_axis").map(row => ({
          id: `${tab}:${row.strand || 0}:${row.level}`,
          type: "base_axis",
          level: Number(row.level),
          strand: row.strand === undefined || row.strand === null ? null : Number(row.strand),
          features: [null, null, null, "xdisp", "ydisp", "inclin", "tip"],
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
      if (["global_strand_step", "global_base_pair_step", "local_strand_step", "local_base_pair_step"].includes(tab)) {
        const sources = {
          global_strand_step: ["global_inter_base", "global_strand_step"],
          global_base_pair_step: ["global_inter_base_pair", "global_step"],
          local_strand_step: ["local_inter_base", "local_strand_step"],
          local_base_pair_step: ["local_inter_base_pair", "local_step"]
        };
        const [name, fallback] = sources[tab];
        const source = rowsFor(name, fallback);
        return source.map(row => ({
          id: `${tab}:${row.strand || row.partner_strand || 0}:${row.level}`,
          type: "step",
          tab,
          level: Number(row.level),
          nextLevel: row.next_level === undefined || row.next_level === null ? Number(row.level) + 1 : Number(row.next_level),
          strand: row.strand === undefined || row.strand === null ? null : Number(row.strand),
          partnerStrand: row.partner_strand === undefined || row.partner_strand === null ? null : Number(row.partner_strand),
          features: [null, null, "shift", "slide", "rise", "tilt", "roll", "twist"],
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
      if (tab === "global_axis_curvature") {
        return rowsFor(tab).map(row => ({
          id: `${tab}:${row.strand || 0}:${row.level}`,
          type: "data",
          level: Number(row.level),
          strand: row.strand === undefined || row.strand === null ? null : Number(row.strand),
          features: [null, null, null, null, "ax", "ay", "ainc", "atip", "adis", "angle", "path", null],
          cells: [
            row.level,
            row.next_level,
            row.strand,
            row.step || "",
            fmt(row.ax),
            fmt(row.ay),
            fmt(row.ainc),
            fmt(row.atip),
            fmt(row.adis),
            fmt(row.angle),
            fmt(row.path),
            fmt(row.dc),
          ],
          row
        }));
      }
      if (tab === "global_axis_bending") {
        return rowsFor(tab).map(row => ({
          id: `${tab}:${row.strand || 0}:${row.level}`,
          type: "data",
          level: Number(row.level),
          strand: row.strand === undefined || row.strand === null ? null : Number(row.strand),
          features: [null, null, null, "offset", "local_direction"],
          cells: [
            row.level,
            row.strand,
            `${row.residue_name || ""} ${row.residue_id || ""}`.trim(),
            fmt(row.offset),
            fmt(row.local_direction),
          ],
          row
        }));
      }
      if (tab === "global_axis_bending_summary") {
        return rowsFor(tab).map(row => ({
          id: `${tab}:${row.strand || 0}`,
          type: "data",
          level: null,
          strand: row.strand === undefined || row.strand === null ? null : Number(row.strand),
          features: [null, "path_length", "end_to_end", "shortening_percent", "overall_bend_uu", "overall_bend_pp"],
          cells: [
            row.strand,
            fmt(row.path_length),
            fmt(row.end_to_end),
            fmt(row.shortening_percent),
            fmt(row.overall_bend_uu),
            fmt(row.overall_bend_pp),
          ],
          row
        }));
      }
      if (tab === "backbone") {
        return rowsFor(tab).map(row => ({
          id: `${tab}:${row.strand || 0}:${row.level}`,
          type: "data",
          level: Number(row.level),
          strand: row.strand === undefined || row.strand === null ? null : Number(row.strand),
          features: [null, null, null, null, null, "c1_c2", "c2_c3", "phase", "amplitude", "c1_prime", "c2_prime", "c3_prime", "chi", "gamma", "delta", "epsilon", "zeta", "alpha", "beta"],
          cells: [
            row.level,
            row.strand,
            `${row.residue_name || ""} ${row.residue_id || ""}`.trim(),
            row.status || "",
            row.pucker || "",
            fmt(row.c1_c2),
            fmt(row.c2_c3),
            fmt(row.phase),
            fmt(row.amplitude),
            fmt(row.c1_prime),
            fmt(row.c2_prime),
            fmt(row.c3_prime),
            fmt(row.chi),
            fmt(row.gamma),
            fmt(row.delta),
            fmt(row.epsilon),
            fmt(row.zeta),
            fmt(row.alpha),
            fmt(row.beta),
          ],
          row
        }));
      }
      return (params.groove || []).map(row => ({
        id: `groove:${row.level}:${row.sub_level}`,
        type: "groove",
        level: Number(row.level),
        subLevel: Number(row.sub_level),
        features: [null, null, "minor_width", "minor_depth", "minor_angle", "major_width", "major_depth", "major_angle", "diameter"],
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
      if (tab === "global_base_pair" || tab === "local_base_pair") return ["Level", "Pair", "Shear", "Stretch", "Stagger", "Buckle", "Propel", "Open"];
      if (tab === "global_base_pair_axis") return ["Level", "Pair", "Xdisp", "Ydisp", "Inclin", "Tip"];
      if (tab === "global_base_axis") return ["Level", "Strand", "Residue", "Xdisp", "Ydisp", "Inclin", "Tip"];
      if (["global_strand_step", "global_base_pair_step", "local_strand_step", "local_base_pair_step"].includes(tab)) return ["Step", "Residues", "Shift", "Slide", "Rise", "Tilt", "Roll", "Twist"];
      if (tab === "global_axis_curvature") return ["Level", "Next", "Strand", "Step", "Ax", "Ay", "Ainc", "Atip", "Adis", "Angle", "Path", "DC"];
      if (tab === "global_axis_bending") return ["Level", "Strand", "Residue", "Offset", "Local Direction"];
      if (tab === "global_axis_bending_summary") return ["Strand", "Path Length", "End-to-End", "Shortening %", "Bend UU", "Bend PP"];
      if (tab === "backbone") return ["Level", "Strand", "Residue", "Status", "Pucker", "C1-C2", "C2-C3", "Phase", "Amplitude", "C1'", "C2'", "C3'", "Chi", "Gamma", "Delta", "Epsilon", "Zeta", "Alpha", "Beta"];
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
      const annotations = rows.map(entry => {
        if (!["global_base_pair", "local_base_pair"].includes(activeTab)) return null;
        const pair = pairAtLevel(entry.level, entry.partnerStrand);
        return pairAnnotation(pair) || (!pair && baseAtLevel(1, entry.level)
          ? {kind: "warning", label: "Unpaired", detail: "No inferred base pair"}
          : null);
      });
      container.innerHTML = `
        <table>
          <thead><tr>${headers.map(header => `<th>${header}</th>`).join("")}</tr></thead>
          <tbody>
            ${rows.map((entry, index) => {
              const annotation = annotations[index];
              return `
              <tr data-index="${index}" class="${selectedInspection && selectedInspection.id === entry.id ? "selected" : ""}${annotation ? " annotation-" + annotation.kind : ""}">
                ${entry.cells.map((cell, cellIndex) => {
                  const feature = (entry.features || [])[cellIndex] || "";
                  const selected = feature && selectedInspection && selectedInspection.id === entry.id && selectedInspection.feature === feature;
                  const badge = cellIndex === 0 && annotation ? `<span class="annotation-badge ${annotation.kind}">${annotation.label}</span>` : "";
                  return `<td data-feature="${feature}" class="${selected ? "selected-feature" : ""}">${badge}${cell}</td>`;
                }).join("")}
              </tr>
            `;
            }).join("")}
          </tbody>
        </table>
      `;
      container.querySelectorAll("tbody tr").forEach(row => {
        const annotation = annotations[Number(row.dataset.index)];
        if (annotation) row.title = annotation.detail;
        row.addEventListener("click", event => {
          const entry = rows[Number(row.dataset.index)];
          const cell = event.target.closest("td");
          const feature = cell ? (cell.dataset.feature || null) : null;
          selectedInspection = Object.assign({}, entry, {
            type: ["global_base_pair", "local_base_pair"].includes(activeTab) && !pairAtLevel(entry.level, entry.partnerStrand) ? "sequence" : entry.type,
            feature,
            featureLabel: feature ? featureLabel(feature) : null,
            featureValue: feature ? entry.row[feature] : null
          });
          if (Number.isFinite(Number(entry.level))) highlightSequenceLevel(entry.level);
          renderParameterTable();
          redrawSelection();
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
      redrawSelection();
    }

    function drawActualBlocks() {
      const onlyUnusual = document.getElementById("showOnlyUnusual").checked;
      const showLabels = document.getElementById("showLabels").checked;
      (VIS.base_pairs || []).forEach(pair => {
        if (onlyUnusual && !isUnusual(pair)) return;
        const color = pairColor(pair);
        const first = actualBlockForPair(pair.first, pair);
        const second = actualBlockForPair(pair.second, pair);
        drawBasePlate(first, second, color, {outline: false});
        drawBasePlate(second, first, color, {outline: false});
        addSegment(pair.first, pair.second, color, pair.is_canonical ? 0.05 : 0.08, pair.is_canonical ? 0.35 : 0.7);
        if (showLabels && pair.midpoint) {
          addLabel(viewer.addLabel(pair.label || `Level ${pair.level}`, {
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
        drawFrameAxes(analytical, analytical.analysis_frame_source === "contact_geometry" ? 1.25 : 1.0);
      });
      (VIS.base_pairs || []).forEach(pair => {
        if (pair.frame_mode !== "contact_geometry" && pair.is_canonical) return;
        const color = pairColor(pair);
        const first = analyticalBasePlate(pair.first);
        const second = analyticalBasePlate(pair.second);
        addDashedSegment(first.frame_origin || first, second.frame_origin || second, color, 0.035, 9, 0.72);
      });
    }

    function drawSelectedBasePair(entry) {
      const pair = pairAtLevel(entry.level, entry.partnerStrand);
      if (!pair) return;
      const color = "#ff2f00";
      const first = actualBlockForPair(pair.first, pair);
      const second = actualBlockForPair(pair.second, pair);
      drawBasePlate(first, second, color, {fillOpacity: 0.32, edgeRadius: 0.12});
      drawBasePlate(second, first, color, {fillOpacity: 0.32, edgeRadius: 0.12});
      drawAnalyticalPairSource(pair, color);
      const frame = pairFrame(pair);
      drawFrameGlyph(frame, 1.65, "#111827");
      if (entry.feature && drawFeatureMeasurement(frame, entry.feature, entry.featureValue, color)) return;
      addValueLabel(pair.frame_midpoint || pair.midpoint, `Pair frame level ${entry.level}`, color);
    }

    function drawSelectedSequence(entry) {
      const bases = (VIS.base_origins || []).filter(base => Number(base.level) === Number(entry.level));
      bases.forEach(base => drawBasePlate(base, null, "#ff2f00", {fillOpacity: 0.5, edgeRadius: 0.09}));
      if (bases.length) addValueLabel(bases[0], `Sequence level ${entry.level}`, "#ff2f00");
    }

    function drawSelectedBasePairAxis(entry) {
      const pair = pairAtLevel(entry.level, entry.partnerStrand);
      if (!pair) return;
      const color = "#ff2f00";
      drawSelectedBasePair(Object.assign({}, entry, {feature: null, featureValue: null}));
      const frame = pairFrame(pair);
      const axis = axisAtLevel(entry.level);
      if (axis) {
        addPoint(axis, color, 0.46);
        addArrowLikeSegment(frame.origin, axis, color, 0.075, 0.9);
      }
      if (entry.feature && drawFeatureMeasurement(frame, entry.feature, entry.featureValue, color)) return;
      if (axis) addValueLabel(axis, `Axis relation level ${entry.level}`, color);
    }

    function drawSelectedBaseAxis(entry) {
      const base = baseAtLevel(entry.strand, entry.level);
      if (!base) return;
      const color = "#ff2f00";
      const analytical = analyticalBasePlate(base);
      const origin = analytical.frame_origin || analytical;
      const frame = baseAnalysisFrame(base);
      drawFrameAxes(analytical, 1.45);
      const axis = axisAtLevel(entry.level, entry.strand);
      if (axis) {
        addPoint(axis, color, 0.44);
        addArrowLikeSegment(origin, axis, color, 0.075, 0.88);
      }
      if (entry.feature && drawFeatureMeasurement(frame, entry.feature, entry.featureValue, color)) return;
      addValueLabel(origin, `Strand ${entry.strand} level ${entry.level}`, color);
    }

    function stepEndpointFrame(entry, level) {
      if (entry.strand !== null && entry.strand !== undefined) {
        return baseAnalysisFrame(baseAtLevel(entry.strand, level));
      }
      const pair = pairAtLevel(level, entry.partnerStrand);
      return pairFrame(pair);
    }

    function drawStepEndpoint(entry, level) {
      if (entry.strand !== null && entry.strand !== undefined) {
        const base = baseAtLevel(entry.strand, level);
        if (base) drawFrameAxes(analyticalBasePlate(base), 1.25);
        return;
      }
      const pair = pairAtLevel(level, entry.partnerStrand);
      if (pair) {
        drawAnalyticalPairSource(pair, "#ff2f00");
        drawFrameGlyph(pairFrame(pair), 1.35, "#111827");
      }
    }

    function drawSelectedStep(entry) {
      const color = "#ff2f00";
      const nextLevel = entry.nextLevel || entry.level + 1;
      drawStepEndpoint(entry, entry.level);
      drawStepEndpoint(entry, nextLevel);
      const firstFrame = stepEndpointFrame(entry, entry.level);
      const secondFrame = stepEndpointFrame(entry, nextLevel);
      if (firstFrame && secondFrame) {
        addArrowLikeSegment(firstFrame.origin, secondFrame.origin, color, 0.12, 0.9);
        const midOrigin = pointFrom(scaleVec(addVec(vec(firstFrame.origin), vec(secondFrame.origin)), 0.5));
        const midFrame = averageFrames(firstFrame, secondFrame, midOrigin);
        drawFrameGlyph(midFrame, 1.45, "#111827");
        if (entry.feature && drawFeatureMeasurement(midFrame, entry.feature, entry.featureValue, color)) return;
        addValueLabel(midOrigin, entry.strand ? `Strand ${entry.strand} step ${entry.level}` : `Step ${entry.level}`, color);
      }
      const axis1 = axisAtLevel(entry.level, entry.strand);
      const axis2 = axisAtLevel(nextLevel, entry.strand);
      if (axis1 && axis2) {
        addSegment(axis1, axis2, color, 0.24, 0.72);
      }
    }

    function drawGrooveMeasurement(row, side, color, feature = null) {
      const geometry = row.geometry && row.geometry[side];
      if (!geometry) return;
      const widthFeature = `${side}_width`;
      const depthFeature = `${side}_depth`;
      const angleFeature = `${side}_angle`;
      const showWidth = !feature || feature === widthFeature || feature === "diameter";
      const showDepth = !feature || feature === depthFeature;
      const showAngle = feature === angleFeature;
      if (showWidth && row[widthFeature] !== null && row[widthFeature] !== undefined) {
        addSegment(geometry.width_endpoint_1, geometry.width_endpoint_2, color, 0.15, 1.0);
        addPoint(geometry.width_endpoint_1, color, 0.24);
        addPoint(geometry.width_endpoint_2, color, 0.24);
        const mid = pointFrom(scaleVec(addVec(vec(geometry.width_endpoint_1), vec(geometry.width_endpoint_2)), 0.5));
        addValueLabel(mid, `${featureLabel(widthFeature)} ${fmt(row[widthFeature])} A`, color);
      }
      if (showDepth && row[depthFeature] !== null && row[depthFeature] !== undefined) {
        addSegment(geometry.depth_reference, geometry.depth_point, color, 0.09, 0.9);
        addPoint(geometry.depth_point, color, 0.22);
        addValueLabel(geometry.depth_point, `${featureLabel(depthFeature)} ${fmt(row[depthFeature])} A`, color);
      }
      if (showAngle && row[angleFeature] !== null && row[angleFeature] !== undefined) {
        addSegment(geometry.width_endpoint_1, geometry.depth_point, color, 0.07, 0.78);
        addSegment(geometry.width_endpoint_2, geometry.depth_point, color, 0.07, 0.78);
        addValueLabel(geometry.depth_point, `${featureLabel(angleFeature)} ${fmt(row[angleFeature])} deg`, color);
      }
    }

    function drawSelectedGroove(entry) {
      const feature = entry.feature;
      const showMinor = !feature || feature.startsWith("minor_") || feature === "diameter";
      const showMajor = !feature || feature.startsWith("major_") || feature === "diameter";
      if (showMinor) drawGrooveMeasurement(entry.row, "minor", "#d12b72", feature);
      if (showMajor) drawGrooveMeasurement(entry.row, "major", "#6246ea", feature);
      const axis = axisAtLevel(entry.level);
      if (axis) {
        addPoint(axis, "#111827", 0.42);
        const label = feature ? `${featureLabel(feature)} ${fmt(entry.featureValue)}${featureUnit(feature) ? " " + featureUnit(feature) : ""}` : `Groove ${entry.level}.${entry.subLevel}`;
        addValueLabel(axis, label, "#111827");
      }
    }

    function drawInspectionSelection() {
      if (!selectedInspection) return;
      if (selectedInspection.type === "sequence") drawSelectedSequence(selectedInspection);
      if (selectedInspection.type === "base_pair") drawSelectedBasePair(selectedInspection);
      if (selectedInspection.type === "base_pair_axis") drawSelectedBasePairAxis(selectedInspection);
      if (selectedInspection.type === "base_axis") drawSelectedBaseAxis(selectedInspection);
      if (selectedInspection.type === "step") drawSelectedStep(selectedInspection);
      if (selectedInspection.type === "groove") drawSelectedGroove(selectedInspection);
    }

    function redrawSelection() {
      if (!viewer) return;
      clearSelection();
      drawingSelection = true;
      try {
        drawInspectionSelection();
      } finally {
        drawingSelection = false;
      }
      viewer.render();
    }

    function redraw() {
      if (!viewer) return;
      clearOverlays();
      drawStructure();
      if (document.getElementById("showAxis").checked) drawAxis();
      if (document.getElementById("showBackbone").checked) drawBackbones();
      if (document.getElementById("showActualBlocks").checked) drawActualBlocks();
      if (document.getElementById("showAnalyticalBases").checked) drawAnalyticalBaseFrames();
      drawingSelection = true;
      try {
        drawInspectionSelection();
      } finally {
        drawingSelection = false;
      }
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
      button.hidden = !tableRowsForTab(button.dataset.tab).length;
      button.classList.toggle("active", button.dataset.tab === activeTab);
      button.addEventListener("click", () => setActiveTab(button.dataset.tab));
    });
    document.querySelectorAll(".feature-group").forEach(group => {
      group.hidden = !Array.from(group.querySelectorAll("button")).some(button => !button.hidden);
    });
    document.getElementById("colorMode").addEventListener("change", redraw);
    document.getElementById("resetView").addEventListener("click", () => {
      if (!viewer) return;
      viewer.zoomTo();
      viewer.render();
    });
    window.addEventListener("resize", () => {
      if (viewer) viewer.resize();
    });
    renderSequenceNavigator();
    renderParameterTable();
    window.addEventListener("load", initialize);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
