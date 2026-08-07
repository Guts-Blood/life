import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const artifactsDir = path.resolve(scriptDir, "..");
const reportsDir = path.join(artifactsDir, "reports");
const abPath = path.join(reportsDir, "day12-cloud-dev-selection-outcome.json");
const recoveryPath = path.join(reportsDir, "day12-recovery-final-summary.json");
const svgPath = path.join(reportsDir, "day12-checkpoint-score-summary.svg");

const ab = JSON.parse(fs.readFileSync(abPath, "utf8"));
const recovery = JSON.parse(fs.readFileSync(recoveryPath, "utf8"));
const gate = recovery.acceptance_gate;

const normalizeName = (name) => name.replace(/_(percent)$/, "%").replace(/_percent$/, "%");
const scoreRow = (model, run, budget, scores, family) => {
  const general = scores.general;
  const mathScore = scores.math;
  const code = scores.code;
  const finance = scores.finance;
  const total = scores.four_slice_total ?? scores.total;
  const gateFlags = [mathScore >= gate.math_minimum, code >= gate.code_minimum, total >= gate.total_minimum];
  return {
    model,
    run,
    budget,
    family,
    general,
    math: mathScore,
    code,
    finance,
    total,
    deltaBase: total - ab.base.correct.four_slice_total,
    gatesPassed: gateFlags.filter(Boolean).length,
    eligible: gateFlags.every(Boolean),
  };
};

const rows = [
  scoreRow("Base", "Base", "base", ab.base.correct, "base"),
  ...ab.candidates.map((candidate) => {
    const match = candidate.name.match(/^([AB])-(25|60|100)_percent$/);
    return scoreRow(
      normalizeName(candidate.name),
      match?.[1] ?? candidate.name[0],
      match ? `${match[2]}%` : "—",
      candidate.correct,
      "original",
    );
  }),
  ...recovery.rollout_ids.map((id) =>
    scoreRow(`${id}-25%`, id, "25%", recovery.scores[id], "recovery"),
  ),
];

const ranked = [...rows].sort((a, b) =>
  b.total - a.total || b.math - a.math || b.code - a.code || a.model.localeCompare(b.model),
);
let previousTotal = null;
let rank = 0;
ranked.forEach((row, index) => {
  if (row.total !== previousTotal) {
    rank = index + 1;
    previousTotal = row.total;
  }
  row.rank = rank;
});

const W = 2400;
const H = 2000;
const C = {
  bg: "#F5F7FA",
  panel: "#FFFFFF",
  ink: "#17212B",
  muted: "#66717F",
  grid: "#D9DEE7",
  blue: "#2563EB",
  blueDark: "#1E40AF",
  blueLight: "#DBEAFE",
  orange: "#D97706",
  orangeLight: "#FEF3C7",
  neutral: "#334155",
  neutralLight: "#E2E8F0",
  passing: "#EFF6FF",
};

const esc = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");
const f = (value) => Number(value.toFixed(2));
const svg = [];
const add = (line) => svg.push(line);
const text = (x, y, value, size = 28, weight = 400, fill = C.ink, anchor = "start", extra = "") =>
  add(`<text x="${f(x)}" y="${f(y)}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" ${extra}>${esc(value)}</text>`);
const rect = (x, y, width, height, fill, rx = 20, stroke = "none", strokeWidth = 0) =>
  add(`<rect x="${f(x)}" y="${f(y)}" width="${f(width)}" height="${f(height)}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`);
const line = (x1, y1, x2, y2, stroke = C.grid, strokeWidth = 2, dash = "") =>
  add(`<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${stroke}" stroke-width="${strokeWidth}" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`);

add(`<?xml version="1.0" encoding="UTF-8"?>`);
add(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
add(`<title id="title">Day 12 checkpoint 评测总结</title>`);
add(`<desc id="desc">Base、原始 A/B 三阶段 checkpoint 与 recovery C 到 L 的完整分数、总分排名和 math-code 门槛对比。</desc>`);
add(`<style>
  text { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; }
  .mono { font-family: "SFMono-Regular", Menlo, Consolas, monospace; }
</style>`);
rect(0, 0, W, H, C.bg, 0);

// Header and result statement.
text(64, 82, "Day 12 · Checkpoint 评测总结", 48, 760);
text(64, 128, "Base + 原始 A/B 三阶段 + recovery C–L · frozen dev，每个领域 28 题，总分 112", 24, 430, C.muted);
rect(64, 158, 2272, 102, C.blueLight, 18);
text(96, 205, "结论", 23, 760, C.blueDark);
text(188, 205, "E-25% 与 H-25% 并列总分最高：19/112（比 Base +4）", 30, 760, C.ink);
text(188, 238, "但 17 个模型状态中 0 个同时满足 math ≥ 10、code ≥ 5、total ≥ 15；最高总分 ≠ 可验收。", 23, 520, C.ink);

// KPI cards.
const kpis = [
  { x: 64, label: "最高总分", value: "19 / 112", note: "E-25% · H-25%", color: C.blue },
  { x: 636, label: "Base 总分", value: "15 / 112", note: "Math 12 · Code 0", color: C.neutral },
  { x: 1208, label: "通过全部门槛", value: "0 / 17", note: "Frozen test 未启用", color: C.orange },
  { x: 1780, label: "单项峰值", value: "Math 12 · Code 9", note: "Math: Base/G · Code: B25/E", color: C.blueDark },
];
kpis.forEach((kpi) => {
  rect(kpi.x, 286, 556, 142, C.panel, 18, C.grid, 1);
  add(`<rect x="${kpi.x}" y="286" width="8" height="142" rx="4" fill="${kpi.color}"/>`);
  text(kpi.x + 30, 326, kpi.label, 20, 650, C.muted);
  text(kpi.x + 30, 374, kpi.value, 34, 760, C.ink, "start", `class="mono"`);
  text(kpi.x + 30, 408, kpi.note, 19, 480, C.muted);
});

// Panel 1: ranked totals.
const barPanel = { x: 64, y: 460, w: 1070, h: 850 };
rect(barPanel.x, barPanel.y, barPanel.w, barPanel.h, C.panel, 22, C.grid, 1);
text(barPanel.x + 28, barPanel.y + 48, "总分排名", 30, 740);
text(barPanel.x + 28, barPanel.y + 80, "四领域正确数合计；虚线为 Base = 15，横轴缩放至观察区间 0–20", 19, 430, C.muted);

const bx0 = barPanel.x + 215;
const bx1 = barPanel.x + barPanel.w - 60;
const by0 = barPanel.y + 120;
const rowH = 39.2;
const scaleX = (value) => bx0 + (value / 20) * (bx1 - bx0);
[0, 5, 10, 15, 20].forEach((tick) => {
  const x = scaleX(tick);
  line(x, by0 - 12, x, by0 + ranked.length * rowH + 4, tick === 15 ? C.neutral : C.grid, tick === 15 ? 2.5 : 1, tick === 15 ? "8 6" : "");
  text(x, by0 - 20, String(tick), 16, 520, C.muted, "middle", `class="mono"`);
});
ranked.forEach((row, index) => {
  const y = by0 + index * rowH;
  const isLeader = row.total === ranked[0].total;
  const fill = isLeader ? C.blue : row.family === "base" ? C.neutral : row.family === "original" ? C.orangeLight : C.blueLight;
  const stroke = row.family === "original" ? C.orange : isLeader ? C.blueDark : "none";
  text(barPanel.x + 28, y + 25, `${row.rank}. ${row.model}`, 18, isLeader || row.family === "base" ? 720 : 520, C.ink);
  add(`<rect x="${bx0}" y="${f(y + 6)}" width="${f(scaleX(row.total) - bx0)}" height="25" rx="7" fill="${fill}" stroke="${stroke}" stroke-width="${stroke === "none" ? 0 : 1.5}"/>`);
  text(scaleX(row.total) + 12, y + 26, row.total, 18, 720, C.ink, "start", `class="mono"`);
});
text(barPanel.x + 28, barPanel.y + barPanel.h - 26, "色彩：Base 深灰 · 原始 A/B 琥珀 · recovery C–L 蓝 · 并列第一深蓝", 17, 430, C.muted);

// Panel 2: math-code relationship and acceptance gates.
const scatterPanel = { x: 1160, y: 460, w: 1176, h: 850 };
rect(scatterPanel.x, scatterPanel.y, scatterPanel.w, scatterPanel.h, C.panel, 22, C.grid, 1);
text(scatterPanel.x + 28, scatterPanel.y + 48, "Math–Code 权衡", 30, 740);
text(scatterPanel.x + 28, scatterPanel.y + 80, "每个点是一个模型状态；右上角还需同时满足 total ≥ 15 才可验收", 19, 430, C.muted);

const sx0 = scatterPanel.x + 105;
const sx1 = scatterPanel.x + scatterPanel.w - 55;
const sy0 = scatterPanel.y + scatterPanel.h - 135;
const sy1 = scatterPanel.y + 125;
const xMath = (value) => sx0 + (value / 13) * (sx1 - sx0);
const yCode = (value) => sy0 - (value / 10) * (sy0 - sy1);
rect(xMath(gate.math_minimum), sy1, sx1 - xMath(gate.math_minimum), yCode(gate.code_minimum) - sy1, C.passing, 0);
text((xMath(gate.math_minimum) + sx1) / 2, sy1 + 27, "Math/Code 双门槛区", 16, 650, C.blueDark, "middle");
[0, 2, 4, 6, 8, 10].forEach((tick) => {
  const y = yCode(tick);
  line(sx0, y, sx1, y, tick === gate.code_minimum ? C.neutral : C.grid, tick === gate.code_minimum ? 2.5 : 1, tick === gate.code_minimum ? "8 6" : "");
  text(sx0 - 20, y + 6, String(tick), 16, 520, C.muted, "end", `class="mono"`);
});
[0, 2, 4, 6, 8, 10, 12].forEach((tick) => {
  const x = xMath(tick);
  line(x, sy1, x, sy0, tick === gate.math_minimum ? C.neutral : C.grid, tick === gate.math_minimum ? 2.5 : 1, tick === gate.math_minimum ? "8 6" : "");
  text(x, sy0 + 28, String(tick), 16, 520, C.muted, "middle", `class="mono"`);
});
text((sx0 + sx1) / 2, sy0 + 54, "Math 正确数（/28）", 19, 620, C.ink, "middle");
add(`<text x="${scatterPanel.x + 30}" y="${(sy0 + sy1) / 2}" font-size="19" font-weight="620" fill="${C.ink}" text-anchor="middle" transform="rotate(-90 ${scatterPanel.x + 30} ${(sy0 + sy1) / 2})">Code 正确数（/28）</text>`);

const positionGroups = new Map();
rows.forEach((row) => {
  const key = `${row.math},${row.code}`;
  positionGroups.set(key, [...(positionGroups.get(key) ?? []), row]);
});
for (const group of positionGroups.values()) {
  const row = group[0];
  const x = xMath(row.math);
  const y = yCode(row.code);
  const label = group.map((item) => item.model.replace("-25%", "")).join("/");
  const hasBase = group.some((item) => item.family === "base");
  const hasOriginal = group.some((item) => item.family === "original");
  const isLeader = group.some((item) => item.total === 19);
  if (hasBase) {
    add(`<polygon points="${x},${y - 11} ${x + 11},${y} ${x},${y + 11} ${x - 11},${y}" fill="${C.neutral}" stroke="white" stroke-width="2"/>`);
  } else if (hasOriginal) {
    add(`<circle cx="${x}" cy="${y}" r="10" fill="white" stroke="${C.orange}" stroke-width="4"/>`);
  } else {
    add(`<circle cx="${x}" cy="${y}" r="${isLeader ? 12 : 10}" fill="${isLeader ? C.blueDark : C.blue}" stroke="white" stroke-width="2"/>`);
  }
  const anchor = x > sx1 - 130 ? "end" : "start";
  const dx = anchor === "end" ? -14 : 14;
  const dy = y < sy1 + 45 ? 28 : y > sy0 - 35 ? -14 : -12;
  text(x + dx, y + dy, label, 16, isLeader || hasBase ? 720 : 560, C.ink, anchor);
}

// Scatter legend.
const ly = scatterPanel.y + scatterPanel.h - 30;
add(`<polygon points="${scatterPanel.x + 300},${ly - 8} ${scatterPanel.x + 308},${ly} ${scatterPanel.x + 300},${ly + 8} ${scatterPanel.x + 292},${ly}" fill="${C.neutral}"/>`);
text(scatterPanel.x + 316, ly + 6, "Base", 16, 500, C.muted);
add(`<circle cx="${scatterPanel.x + 430}" cy="${ly}" r="8" fill="white" stroke="${C.orange}" stroke-width="3"/>`);
text(scatterPanel.x + 446, ly + 6, "原始 A/B", 16, 500, C.muted);
add(`<circle cx="${scatterPanel.x + 590}" cy="${ly}" r="8" fill="${C.blue}"/>`);
text(scatterPanel.x + 606, ly + 6, "Recovery C–L", 16, 500, C.muted);

// Panel 3: exact score table, split into two reading columns.
const tablePanel = { x: 64, y: 1340, w: 2272, h: 590 };
rect(tablePanel.x, tablePanel.y, tablePanel.w, tablePanel.h, C.panel, 22, C.grid, 1);
text(tablePanel.x + 28, tablePanel.y + 46, "完整精确分数", 30, 740);
text(tablePanel.x + 28, tablePanel.y + 78, "Gate = 已通过门槛数（Math / Code / Total，共 3 项）；所有行最终资格均为 No", 18, 430, C.muted);

const tableRows = [rows.slice(0, 9), rows.slice(9)];
const tableX = [tablePanel.x + 28, tablePanel.x + 1154];
const columns = [
  { key: "model", label: "Checkpoint", width: 200, align: "start" },
  { key: "general", label: "Gen", width: 95, align: "middle" },
  { key: "math", label: "Math", width: 105, align: "middle" },
  { key: "code", label: "Code", width: 105, align: "middle" },
  { key: "finance", label: "Fin", width: 95, align: "middle" },
  { key: "total", label: "Total", width: 115, align: "middle" },
  { key: "gatesPassed", label: "Gate", width: 105, align: "middle" },
  { key: "deltaBase", label: "ΔBase", width: 120, align: "middle" },
];
const headerY = tablePanel.y + 104;
const tableRowH = 45;
tableRows.forEach((group, tableIndex) => {
  let xCursor = tableX[tableIndex];
  rect(xCursor, headerY, 1080, 40, C.neutralLight, 8);
  columns.forEach((column) => {
    const tx = column.align === "start" ? xCursor + 12 : xCursor + column.width / 2;
    text(tx, headerY + 27, column.label, 16, 720, C.ink, column.align);
    xCursor += column.width;
  });
  group.forEach((row, rowIndex) => {
    const y = headerY + 45 + rowIndex * tableRowH;
    const leader = row.total === 19;
    if (leader) rect(tableX[tableIndex], y - 3, 1080, 40, C.blueLight, 7);
    if (rowIndex % 2 === 1 && !leader) rect(tableX[tableIndex], y - 3, 1080, 40, "#F8FAFC", 0);
    let cellX = tableX[tableIndex];
    columns.forEach((column) => {
      let value = row[column.key];
      if (column.key === "gatesPassed") value = `${value}/3`;
      if (column.key === "deltaBase") value = value > 0 ? `+${value}` : String(value);
      const tx = column.align === "start" ? cellX + 12 : cellX + column.width / 2;
      text(tx, y + 25, value, 17, column.key === "total" || column.key === "model" ? 680 : 500, C.ink, column.align, column.key === "model" ? "" : `class="mono"`);
      cellX += column.width;
    });
    line(tableX[tableIndex], y + 37, tableX[tableIndex] + 1080, y + 37, C.grid, 0.8);
  });
});

text(64, 1972, "Source: day12-cloud-dev-selection-outcome.json + day12-recovery-final-summary.json · 2026-08-07", 16, 430, C.muted);
text(2336, 1972, "Frozen test 未消耗", 16, 650, C.muted, "end");
add(`</svg>`);

fs.writeFileSync(svgPath, `${svg.join("\n")}\n`, "utf8");
console.log(svgPath);
