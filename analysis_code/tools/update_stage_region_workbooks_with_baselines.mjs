import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const projectDir = ".";
const sourceDir = path.join(projectDir, "outputs/grape_bern_stage_region_20260831");
const outputDir = path.join(projectDir, "outputs/grape_bern_stage_region_baselines_20260901");
const baselinePath = process.env.BASELINE_METRICS_PATH ?? path.join(
  projectDir,
  "outputs",
  "baseline_metrics_for_workbooks.json",
);

const teal = "#0F766E";
const darkTeal = "#115E59";
const paleTeal = "#E6F4F1";
const paleBlue = "#F2F8FB";
const paleGreen = "#EFFAF3";
const paleYellow = "#FFF7CC";
const paleOrange = "#FFF3E8";
const noteFill = "#F8FAFC";
const grid = "#D5E1E5";
const text = "#1F2937";
const muted = "#475569";

const stageOrder = ["All", "Early", "Moderate", "Severe"];
const regionOrder = [
  "Overall",
  "Supero_Nasal",
  "Supero_Temporal",
  "Macular",
  "Infero_Nasal",
  "Infero_Temporal",
  "Temporal",
];
const regionDisplay = new Map([
  ["Overall", "Overall"],
  ["Supero_Nasal", "Supero-Nasal"],
  ["Supero_Temporal", "Supero-Temporal"],
  ["Macular", "Macular"],
  ["Infero_Nasal", "Infero-Nasal"],
  ["Infero_Temporal", "Infero-Temporal"],
  ["Temporal", "Temporal"],
]);
const displayToRegion = new Map([...regionDisplay.entries()].map(([key, value]) => [value, key]));
const stageFill = new Map([
  ["All", paleBlue],
  ["Early", paleGreen],
  ["Moderate", paleYellow],
  ["Severe", paleOrange],
]);

function colLetter(number) {
  let n = number;
  let result = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    result = String.fromCharCode(65 + rem) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
}

function metricText(row) {
  if (!row || row.sample_mae_db === null || row.sample_sd_db === null) return "N/A";
  return `${Number(row.sample_mae_db).toFixed(2)} ± ${Number(row.sample_sd_db).toFixed(2)}`;
}

function key(dataset, stage, region, baseline) {
  return `${dataset}|${stage}|${region}|${baseline}`;
}

function safeUnmerge(sheet, range) {
  try {
    sheet.unmergeCells(range);
  } catch {
    // The imported workbook may not retain a merge exactly as written; the
    // following full-range clear still removes stale values and formatting.
  }
}

function modelHeaders(dataset, oldHeaderRow) {
  const existing = oldHeaderRow.slice(4).filter((value) => value !== null && value !== undefined && value !== "");
  return existing;
}

function rebuildReadableSummary(workbook, dataset, baselineMetrics) {
  const summary = workbook.worksheets.getItem("Readable Summary");
  const oldLastCol = dataset === "Bern" ? "G" : "H";
  const oldValues = summary.getRange("A1:Z40").values;
  const oldModels = modelHeaders(dataset, oldValues[3]);
  const oldRows = oldValues.slice(4, 32);
  const modelCount = oldModels.length;
  const lastCol = colLetter(4 + 2 + modelCount);

  // Remove the old title/note merges before rebuilding the same first sheet
  // with two additional baseline columns.  Other workbook sheets are not
  // cleared or regenerated.
  for (const row of [1, 2, 34, 35]) safeUnmerge(summary, `A${row}:${oldLastCol}${row}`);
  summary.getRange("A1:Z40").clear({ applyTo: "all" });
  summary.showGridLines = false;
  summary.freezePanes.freezeRows(4);
  summary.freezePanes.freezeColumns(4);

  summary.mergeCells(`A1:${lastCol}1`);
  summary.getRange("A1").values = [[`${dataset} LSTM and Baseline Results by Stage and VF Sector`]];
  summary.getRange(`A1:${lastCol}1`).format = {
    fill: darkTeal,
    font: { name: "Aptos Display", size: 18, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  summary.getRange(`A1:${lastCol}1`).format.rowHeight = 36;

  summary.mergeCells(`A2:${lastCol}2`);
  summary.getRange("A2").values = [[
    "Held-out patient-eye test set · native 59-point TD · sample-weighted MAE ± SD (dB) · baselines and LSTM models shown side by side · lower is better",
  ]];
  summary.getRange(`A2:${lastCol}2`).format = {
    fill: paleTeal,
    font: { name: "Aptos", size: 10, italic: true, color: darkTeal },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  summary.getRange(`A2:${lastCol}2`).format.rowHeight = 30;

  const headers = [
    "Stage",
    "VF region / sector",
    "Patient-eyes (n)",
    "Prediction samples (n)",
    "No-change classifier",
    "PLR (equal-step)",
    ...oldModels,
  ];
  summary.getRange(`A4:${lastCol}4`).values = [headers];
  summary.getRange(`A4:${lastCol}4`).format = {
    fill: teal,
    font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    rowHeight: 56,
    borders: { preset: "all", style: "thin", color: darkTeal },
  };

  const rows = [];
  for (let index = 0; index < oldRows.length; index += 1) {
    const oldRow = oldRows[index];
    const stage = oldRow[0];
    const displayRegion = oldRow[1];
    const region = displayToRegion.get(displayRegion) || displayRegion;
    const baselineRows = [
      baselineMetrics[key(dataset, stage, region, "No-change classifier")],
      baselineMetrics[key(dataset, stage, region, "PLR")],
    ];
    const originalModelCells = oldRow.slice(4, 4 + modelCount).map((value) => (
      value === null || value === undefined || value === "" ? "N/A" : value
    ));
    rows.push([
      oldRow[0],
      oldRow[1],
      oldRow[2],
      oldRow[3],
      metricText(baselineRows[0]),
      metricText(baselineRows[1]),
      ...originalModelCells,
    ]);
  }
  summary.getRange(`A5:${lastCol}${4 + rows.length}`).values = rows;
  summary.getRange(`A5:${lastCol}${4 + rows.length}`).format = {
    font: { name: "Aptos", size: 10, color: text },
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: grid },
    rowHeight: 23,
  };
  for (let index = 0; index < rows.length; index += 1) {
    const rowNumber = 5 + index;
    summary.getRange(`A${rowNumber}:${lastCol}${rowNumber}`).format.fill = stageFill.get(rows[index][0]);
    summary.getRange(`A${rowNumber}:B${rowNumber}`).format.font = { name: "Aptos", size: 10, bold: true, color: text };
  }

  summary.getRange(`C5:D${4 + rows.length}`).format.numberFormat = "#,##0";
  summary.getRange(`A5:B${4 + rows.length}`).format.horizontalAlignment = "left";
  summary.getRange(`C5:${lastCol}${4 + rows.length}`).format.horizontalAlignment = "center";
  const widths = { A: 12, B: 21, C: 16, D: 20, E: 24, F: 20 };
  for (let index = 0; index < modelCount; index += 1) widths[colLetter(7 + index)] = 29;
  for (const [column, width] of Object.entries(widths)) summary.getRange(`${column}:${column}`).format.columnWidth = width;

  summary.mergeCells(`A34:${lastCol}34`);
  summary.getRange("A34").values = [[
    "Stage proxy: baseline global mean TD ≥ −6 dB = Early; −12 dB ≤ TD < −6 dB = Moderate; TD < −12 dB = Severe. It is not a clinician-annotated stage.",
  ]];
  summary.mergeCells(`A35:${lastCol}35`);
  summary.getRange("A35").values = [[
    dataset === "Bern"
      ? "Baselines: No-change classifier copies the last observed VF; PLR extrapolates each point from the last two visits by one equal time step. All cells use the same held-out windows and ± is sample SD. Cluster LSTM + additional data is N/A because Bern has no valid additional modality."
      : "Baselines: No-change classifier copies the last observed VF; PLR extrapolates each point from the last two visits by one equal time step. All cells use the same held-out windows and ± is sample SD. Cluster LSTM + IOP uses visit-aligned longitudinal IOP with the regional LSTM backbone.",
  ]];
  summary.getRange(`A34:${lastCol}35`).format = {
    fill: noteFill,
    font: { name: "Aptos", size: 9, italic: true, color: muted },
    wrapText: true,
    verticalAlignment: "center",
    rowHeight: 36,
  };
  return { lastCol, modelHeaders: oldModels, oldRows };
}

function addBaselineProtocolRows(workbook, dataset) {
  const notes = workbook.worksheets.getItem("Protocol & Availability");
  const oldValues = notes.getRange("A1:F20").values;
  const start = dataset === "Bern" ? 7 : 8;
  notes.getRange(`A${start}:F${start + 1}`).values = [
    ["No-change classifier", "Available", "None", "Copies the last observed VF at every point.", "Post-hoc baseline on the same held-out windows.", "Persistence reference for judging LSTM improvement."],
    ["PLR (equal-step)", "Available", "None", "Pointwise linear extrapolation in dB from the last two visits.", "Post-hoc baseline on the same held-out windows.", "Simple temporal-linear reference; not a learned neural model."],
  ];
  notes.getRange(`A${start}:F${start + 1}`).format = {
    fill: paleBlue,
    font: { name: "Aptos", size: 9, color: text },
    verticalAlignment: "center",
    wrapText: true,
    rowHeight: 52,
    borders: { preset: "all", style: "thin", color: grid },
  };
  notes.getRange(`A${start}:A${start + 1}`).format.font = { name: "Aptos", size: 9, bold: true, color: text };
  notes.getRange(`B${start}:B${start + 1}`).format.horizontalAlignment = "center";
}

async function exportWorkbook(dataset, baselineMetrics) {
  const inputName = dataset === "Bern"
    ? "Bern_LSTM_stage_region_results_updated.xlsx"
    : "GRAPE_LSTM_stage_region_results_updated.xlsx";
  const inputPath = path.join(sourceDir, inputName);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  rebuildReadableSummary(workbook, dataset, baselineMetrics);
  addBaselineProtocolRows(workbook, dataset);

  const safe = dataset.replaceAll(" ", "_");
  const previewSheets = ["Readable Summary", "Robustness Summary", "Detailed Metrics", "Protocol & Availability", "Regional Analysis"];
  for (const sheetName of previewSheets) {
    const rendered = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(path.join(outputDir, `${safe}_${sheetName.replaceAll(" ", "_").replaceAll("&", "and")}.png`), new Uint8Array(await rendered.arrayBuffer()));
  }
  const inspect = await workbook.inspect({
    kind: "table",
    sheetId: "Readable Summary",
    range: "A1:Z40",
    include: "values,formulas",
    tableMaxRows: 40,
    tableMaxCols: 26,
    maxChars: 40000,
  });
  await fs.writeFile(path.join(outputDir, `${safe}_Readable_Summary.inspect.ndjson`), inspect.ndjson ?? String(inspect));
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: `${dataset} baseline workbook error scan`,
  });
  await fs.writeFile(path.join(outputDir, `${safe}_error_scan.ndjson`), errors.ndjson ?? String(errors));
  const outputName = `${dataset}_LSTM_stage_region_results_with_baselines.xlsx`;
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(path.join(outputDir, outputName));
  return path.join(outputDir, outputName);
}

await fs.mkdir(outputDir, { recursive: true });
const baselinePayload = JSON.parse(await fs.readFile(baselinePath, "utf8"));
const baselineMetrics = baselinePayload.metrics;
const bernPath = await exportWorkbook("Bern", baselineMetrics);
const grapePath = await exportWorkbook("GRAPE", baselineMetrics);
console.log(JSON.stringify({ bernPath, grapePath, outputDir }));
