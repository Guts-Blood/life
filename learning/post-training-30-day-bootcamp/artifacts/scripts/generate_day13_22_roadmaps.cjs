#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");

const REPORT_DIR = path.resolve(__dirname, "..", "reports", "day-roadmaps");
const WIDTH = 1800;
const HEIGHT = 2840;

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function textLines(x, y, lines, className = "body", lineHeight = 27, anchor = "start") {
  const content = lines
    .map((line, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${escapeXml(line)}</tspan>`)
    .join("");
  return `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}">${content}</text>`;
}

function pipelineCard(stage, index, x, y, width = 360, height = 210) {
  const tone = stage.tone || ["blue", "cyan", "purple", "green", "amber", "teal", "indigo"][index % 7];
  return [
    `<text x="${x}" y="${y - 14}" class="stage">${String(index + 1).padStart(2, "0")} · ${escapeXml(stage.kicker)}</text>`,
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="16" class="${tone}"/>`,
    textLines(x + 24, y + 40, [stage.title], "label", 28),
    `<line x1="${x + 24}" y1="${y + 58}" x2="${x + width - 24}" y2="${y + 58}" class="divider"/>`,
    textLines(x + 24, y + 92, stage.lines, "small", 28),
  ].join("\n");
}

function knowledgeCard(item, index, x, y, width = 530, height = 205) {
  const tone = item.tone || ["blue-soft", "purple-soft", "green-soft", "amber-soft", "cyan-soft", "slate-soft"][index % 6];
  return [
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="16" class="${tone}"/>`,
    `<circle cx="${x + 36}" cy="${y + 39}" r="18" class="number-dot"/>`,
    `<text x="${x + 36}" y="${y + 45}" class="number" text-anchor="middle">${index + 1}</text>`,
    textLines(x + 66, y + 45, [item.title], "label", 28),
    textLines(x + 26, y + 86, item.lines, "small", 27),
  ].join("\n");
}

function metricCard(metric, x, y, width = 385, height = 140) {
  return [
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="16" class="metric"/>`,
    textLines(x + 24, y + 34, [metric.label], "stage", 24),
    textLines(x + 24, y + 79, [metric.value], "metric-value", 28),
    textLines(x + 24, y + 112, metric.lines, "tiny", 22),
  ].join("\n");
}

function tableSvg(table, x, y, width = 1660) {
  const widths = table.widths;
  const rowHeight = table.rowHeight || 49;
  const headerHeight = 52;
  const height = headerHeight + rowHeight * table.rows.length;
  const result = [
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="12" class="table-bg"/>`,
    `<rect x="${x}" y="${y}" width="${width}" height="${headerHeight}" rx="12" class="table-head"/>`,
  ];
  let cursor = x;
  for (let i = 0; i < table.headers.length; i += 1) {
    result.push(textLines(cursor + 16, y + 33, [table.headers[i]], "stage", 24));
    cursor += widths[i];
    if (i < widths.length - 1) {
      result.push(`<line x1="${cursor}" y1="${y}" x2="${cursor}" y2="${y + height}" class="grid"/>`);
    }
  }
  for (let rowIndex = 0; rowIndex < table.rows.length; rowIndex += 1) {
    const rowY = y + headerHeight + rowIndex * rowHeight;
    if (rowIndex % 2 === 1) {
      result.push(`<rect x="${x}" y="${rowY}" width="${width}" height="${rowHeight}" class="table-alt"/>`);
    }
    result.push(`<line x1="${x}" y1="${rowY}" x2="${x + width}" y2="${rowY}" class="grid"/>`);
    cursor = x;
    for (let columnIndex = 0; columnIndex < table.rows[rowIndex].length; columnIndex += 1) {
      const cell = table.rows[rowIndex][columnIndex];
      result.push(textLines(cursor + 16, rowY + 31, [cell], columnIndex === 0 ? "body-strong" : "body", 24));
      cursor += widths[columnIndex];
    }
  }
  return { svg: result.join("\n"), height };
}

function statusClass(tone) {
  return {
    green: "status-good",
    amber: "status-warn",
    red: "status-stop",
    blue: "status-info",
    slate: "status-slate",
  }[tone] || "status-info";
}

function renderDay(day) {
  const topX = [70, 480, 890, 1300];
  const bottomX = [280, 720, 1160];
  const pipeline = day.pipeline.map((stage, index) => {
    if (index < 4) return pipelineCard(stage, index, topX[index], 405);
    return pipelineCard(stage, index, bottomX[index - 4], 690);
  });

  const arrows = [
    `<path d="M 430 510 L 470 510" class="line"/>`,
    `<path d="M 840 510 L 880 510" class="line"/>`,
    `<path d="M 1250 510 L 1290 510" class="line"/>`,
    `<path d="M 1660 615 C 1660 650 1520 650 1520 680" class="line-dash"/>`,
    `<path d="M 640 795 L 710 795" class="line"/>`,
    `<path d="M 1080 795 L 1150 795" class="line"/>`,
  ];

  const knowledgePositions = [
    [70, 1075], [635, 1075], [1200, 1075],
    [70, 1310], [635, 1310], [1200, 1310],
  ];
  const knowledge = day.knowledge.map((item, index) => knowledgeCard(item, index, ...knowledgePositions[index]));

  const metrics = day.metrics.map((metric, index) => metricCard(metric, 70 + index * 410, 1690));
  const table = tableSvg(day.table, 70, 1860);
  const evidenceNoteY = 1860 + table.height + 36;

  const sources = day.sources.map((line, index) =>
    textLines(70, 2797 + index * 20, [line], "source", 20),
  );

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(day.ariaTitle || day.title)}</title>
  <desc id="desc">${escapeXml(day.description)}</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/>
    </marker>
    <style>
      text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", "PingFang SC", sans-serif; fill: #172033; }
      .title { font-size: 34px; font-weight: 750; }
      .subtitle { font-size: 18px; fill: #526078; }
      .section { font-size: 23px; font-weight: 750; }
      .stage { font-size: 13px; font-weight: 750; letter-spacing: .45px; fill: #475569; }
      .label { font-size: 18px; font-weight: 700; }
      .body { font-size: 15px; }
      .body-strong { font-size: 15px; font-weight: 700; }
      .small { font-size: 14px; fill: #475569; }
      .tiny { font-size: 12px; fill: #526078; }
      .source { font-size: 11px; fill: #64748b; }
      .metric-value { font-family: "SFMono-Regular", Consolas, monospace; font-size: 27px; font-weight: 750; fill: #1e3a8a; }
      .number { font-size: 13px; font-weight: 800; fill: #ffffff; }
      .panel { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.5; }
      .hero { fill: #eef2ff; stroke: #6366f1; stroke-width: 1.8; }
      .blue { fill: #dbeafe; stroke: #2563eb; stroke-width: 1.5; }
      .cyan { fill: #cffafe; stroke: #0891b2; stroke-width: 1.5; }
      .purple { fill: #f3e8ff; stroke: #9333ea; stroke-width: 1.5; }
      .green { fill: #dcfce7; stroke: #16a34a; stroke-width: 1.5; }
      .amber { fill: #fef3c7; stroke: #d97706; stroke-width: 1.5; }
      .teal { fill: #ccfbf1; stroke: #0f766e; stroke-width: 1.5; }
      .indigo { fill: #e0e7ff; stroke: #4f46e5; stroke-width: 1.5; }
      .red { fill: #fee2e2; stroke: #dc2626; stroke-width: 1.5; }
      .blue-soft { fill: #eff6ff; stroke: #93c5fd; stroke-width: 1.3; }
      .purple-soft { fill: #faf5ff; stroke: #d8b4fe; stroke-width: 1.3; }
      .green-soft { fill: #f0fdf4; stroke: #86efac; stroke-width: 1.3; }
      .amber-soft { fill: #fffbeb; stroke: #fcd34d; stroke-width: 1.3; }
      .cyan-soft { fill: #ecfeff; stroke: #67e8f9; stroke-width: 1.3; }
      .slate-soft { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.3; }
      .number-dot { fill: #475569; }
      .metric { fill: #f8fafc; stroke: #94a3b8; stroke-width: 1.4; }
      .table-bg { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.3; }
      .table-head { fill: #e2e8f0; stroke: #94a3b8; stroke-width: 1.2; }
      .table-alt { fill: #f8fafc; }
      .status-good { fill: #dcfce7; stroke: #16a34a; stroke-width: 1.4; }
      .status-warn { fill: #fef3c7; stroke: #d97706; stroke-width: 1.4; }
      .status-stop { fill: #fee2e2; stroke: #dc2626; stroke-width: 1.4; }
      .status-info { fill: #dbeafe; stroke: #2563eb; stroke-width: 1.4; }
      .status-slate { fill: #f1f5f9; stroke: #64748b; stroke-width: 1.4; }
      .decision { fill: #eef2ff; stroke: #6366f1; stroke-width: 1.7; }
      .boundary { fill: #fff7ed; stroke: #ea580c; stroke-width: 1.7; }
      .divider { stroke: #cbd5e1; stroke-width: 1.2; }
      .grid { stroke: #cbd5e1; stroke-width: 1; }
      .line { fill: none; stroke: #475569; stroke-width: 2; marker-end: url(#arrow); }
      .line-dash { fill: none; stroke: #64748b; stroke-width: 1.8; stroke-dasharray: 7 5; marker-end: url(#arrow); }
    </style>
  </defs>

  <rect width="${WIDTH}" height="${HEIGHT}" fill="#f5f7fb"/>
  <text x="50" y="62" class="title">${escapeXml(day.title)}</text>
  <text x="50" y="95" class="subtitle">${escapeXml(day.subtitle)}</text>
  <rect x="1190" y="108" width="570" height="38" rx="19" class="${statusClass(day.statusTone)}"/>
  <text x="1475" y="133" class="stage" text-anchor="middle">${escapeXml(day.status)}</text>

  <rect x="40" y="165" width="1720" height="112" rx="18" class="hero"/>
  <text x="70" y="203" class="section">一句话 Gate</text>
  ${textLines(270, 203, day.gate, "label", 31)}

  <rect x="40" y="310" width="1720" height="650" rx="18" class="panel"/>
  <text x="70" y="352" class="section">1 · Roadmap：从输入边界到可交付结论</text>
  <text x="70" y="382" class="small">每一步都回答“对象是什么、证据在哪里、失败时停在哪一层”；编号代表依赖顺序。</text>
  ${pipeline.join("\n")}
  ${arrows.join("\n")}

  <rect x="40" y="990" width="1720" height="590" rx="18" class="panel"/>
  <text x="70" y="1033" class="section">2 · 真正学到的知识：可迁移的判断框架</text>
  ${knowledge.join("\n")}

  <rect x="40" y="1610" width="1720" height="720" rx="18" class="panel"/>
  <text x="70" y="1653" class="section">3 · Evidence：关键数字、对照与决策依据</text>
  ${metrics.join("\n")}
  ${table.svg}
  ${textLines(70, evidenceNoteY, day.evidenceNote, "small", 26)}

  <rect x="40" y="2360" width="1110" height="405" rx="18" class="decision"/>
  <text x="70" y="2404" class="section">4 · 决策 / Takeaway</text>
  ${textLines(70, 2450, day.decision, "body-strong", 31)}
  <rect x="1180" y="2360" width="580" height="405" rx="18" class="boundary"/>
  <text x="1210" y="2404" class="section">证据边界</text>
  ${textLines(1210, 2450, day.boundaries, "small", 29)}

  ${sources.join("\n")}
</svg>
`;
}

const DAYS = [
  {
    file: "day-13-qwen35-lineage-migration-roadmap.svg",
    title: "Day 13 · Qwen3 v1 → Qwen3.5 v2：Lineage 迁移边界",
    subtitle: "方法可以复用，模型绑定的证据必须重建；迁移不是把旧 runner 里的 model ID 替换掉。",
    status: "ROADMAP RECONSTRUCTED · MIGRATION BOUNDARY VERIFIED",
    statusTone: "blue",
    description: "A Chinese roadmap explaining the immutable Qwen3 v1 history, the new Qwen3.5 v2 lineage, which methods may cross the lineage boundary, which model-specific artifacts must be rebuilt, and why only a promoted S1 may parent DPO or GRPO.",
    gate: [
      "v1 与 v2 没有权重连续性；未经重新 render、tokenize、baseline 与训练验收，v1 artifact 不得跨 lineage 使用。",
      "能带走的是 raw IDs、provenance、审计方法与 scorer 语义；tokens、predictions、checkpoint 与能力结论必须留在原 lineage。",
    ],
    pipeline: [
      { kicker: "FREEZE V1", title: "封存旧证据", lines: ["Day 01–12 作为 immutable history", "Qwen3-0.6B 不再是权重父节点", "保留合法负结果，不回写"] },
      { kicker: "OPEN V2", title: "新建独立 lineage", lines: ["qwen35-4b-day13-plus-v2", "S0 = exact Qwen3.5-4B Base", "run 必须声明 parent 与 lineage_id"] },
      { kicker: "RESOLVE MODEL", title: "重读模型边界", lines: ["ConditionalGeneration + AutoProcessor", "GDN / MTP + vision components", "不是旧 CausalLM 的 drop-in replacement"] },
      { kicker: "REBUILD CONTRACT", title: "重建监督契约", lines: ["template / thinking / special tokens", "assistant spans / causal shift / truncation", "text-only 时仍审计 ViT / aligner"] },
      { kicker: "REBUILD DATA", title: "重新分词与计量", lines: ["raw IDs 可重新审核", "token budget / mixture 全部重算", "旧 rendered text 与 label spans 禁用"] },
      { kicker: "REBUILD EVAL", title: "重新建立比较尺", lines: ["先跑 v2 Base baseline", "冻结 prompt / decoder / scorer identity", "new selection + confirmation contract"] },
      { kicker: "PROMOTE THEN BRANCH", title: "只从 S1 分叉", lines: ["受控 SFT candidate → promotion", "S1 需 immutable downstream identity", "DPO / GRPO 共享正式 parent"] },
    ],
    knowledge: [
      { title: "方法与 artifact 分离", lines: ["清洗、去污染、逐样本审计可以复用；", "model-specific bytes 与 hashes 不能复用。"] },
      { title: "Base 不等于 aligned policy", lines: ["S0 只提供预训练起点；没有通过 promotion", "的 checkpoint 不能冒充 S1。"] },
      { title: "Text-only 仍有多模态边界", lines: ["保留完整 processor；image/video=0；", "用 trainable inventory 证明 ViT/aligner 冻结。"] },
      { title: "负结果也能迁移", lines: ["Day 12 的 no-candidate 教会 fail closed；", "它不证明 4B 也会失败。"] },
      { title: "Eval identity 必须重建", lines: ["scorer 意义可复用；render、prediction、", "comparison key 与 Base score 都绑定模型。"] },
      { title: "Lineage 是因果边界", lines: ["跨模型 aggregate 只能并列描述；", "不能把规模差异误写成训练增益。"] },
    ],
    metrics: [
      { label: "V1 TERMINAL", value: "10 / 10 · 0 accepted", lines: ["Day 12 recovery；frozen test 未消费"] },
      { label: "V2 REPOSITORY", value: "4,659,865,088 params", lines: ["完整 conditional-generation checkpoint"] },
      { label: "NATIVE CONTEXT", value: "262,144 tokens", lines: ["不是首轮训练长度承诺"] },
      { label: "ACTIVE S0 REVISION", value: "1001bb4d…741b", lines: ["独立于 v1 ddc928…e865"] },
    ],
    table: {
      headers: ["对象", "允许跨越 v1→v2", "必须在 v2 重建"],
      widths: [360, 470, 830],
      rows: [
        ["Raw source / provenance", "重新审核后可引用", "新 manifest 记录真实 parent 与 decision ledger"],
        ["清洗 / scorer 方法", "可复用方法与语义", "新 processor、render、comparison identity"],
        ["Rendered / tokens / labels", "不可复用", "全部重新编码、计量、hash 与边界审计"],
        ["Predictions / checkpoint / score", "不可作为新 parent 或增益", "v2 Base、candidate、promotion 与 downstream key"],
      ],
    },
    evidenceNote: [
      "正确迁移顺序：immutable v1 → exact S0 → processor/data/eval acceptance → controlled SFT S1 → DPO 或 GRPO。",
      "后续实验证明这条边界必要；这些后验数字用于完善知识图，不冒充 Day 13 当天的 GPU 结果。",
    ],
    decision: [
      "核心结论：迁移模型时，真正要迁移的是“判断框架”，不是旧模型产生的字节。",
      "S0、S1、S2 是不同角色：Base 不能绕过 SFT promotion 直接充当 DPO/GRPO parent。",
      "任何新 run 先回答五个问题：模型是谁？processor 是谁？监督落在哪些 token？",
      "比较尺是否匹配？checkpoint 是否拥有可验证的 ancestry？",
      "一句话记忆：raw evidence 可以重新审；model-bound evidence 必须重新生成。",
    ],
    boundaries: [
      "• 原 Day 13 README 的阅读 checklist / 三个 takeaway",
      "  没有单独签收记录。",
      "• 本图依据实际迁移计划、机器合同与后来证据",
      "  重建知识，不声称完成了某次未留痕精读。",
      "• Day 12 的 0.6B checkpoint 不可成为 v2 parent。",
      "• native 262K context 不等于训练应直接使用 262K。",
      "• teacher 分支仍为 null / deferred，不能静默选择。",
    ],
    sources: [
      "Sources · QWEN35-4B-MIGRATION-PLAN.md · artifacts/configs/qwen35-4b-migration-contract.json · artifacts/checkpoints/qwen35-4b-base-s0.json",
      "Scope · Day 13 roadmap reconstructed from durable lineage evidence · generated 2026-08-14",
    ],
  },
  {
    file: "day-14-week2-evidence-readiness-roadmap.svg",
    title: "Day 14 · Week 2 Evidence → Qwen3.5 Readiness",
    subtitle: "“合法选择 none”也是实验完成；但 v1 方法闭环不会自动让新模型 ready。",
    status: "ROADMAP RECONSTRUCTED · WEEK 2 EVIDENCE VERIFIED",
    statusTone: "blue",
    description: "A Chinese evidence-review roadmap connecting Day 08 through Day 12 data, evaluation, training and checkpoint-selection lessons to the M0 through M6 readiness ladder required for Qwen3.5.",
    gate: [
      "Claim → Evidence → Alternative explanation → Limit：每个结论都要写出证据对象和不能外推的部分。",
      "Week 2 证明的是 v1 方法链有效；M0–M6 任一缺少 v2 专属证据，都必须阻止正式 SFT 或下游训练。",
    ],
    pipeline: [
      { kicker: "DAY 08", title: "冻结数据契约", lines: ["raw → render → tokens → labels", "attention mask ≠ loss mask", "截断后再次做 encoded gate"] },
      { kicker: "DAY 09", title: "冻结 lineage", lines: ["source / license / revision / hashes", "用 supervised tokens 描述 mixture", "等监督预算 ≠ 等 forward compute"] },
      { kicker: "DAY 10", title: "训练前冻结量尺", lines: ["suite / prompt / decoder / scorer", "Base baseline 早于任何 candidate", "dev 与 sealed confirmation 分开"] },
      { kicker: "DAY 11", title: "证明 pipeline 能学", lines: ["tiny overfit + fresh resume", "按有效 target 聚合 loss", "通过不等于泛化"] },
      { kicker: "DAY 12", title: "接受 no-candidate", lines: ["联合 gate 先于排名", "0 eligible 是合法终点", "不能从失败项中手选最好"] },
      { kicker: "PORT METHODS", title: "只迁移方法证据", lines: ["raw IDs / audit / scorer semantics", "旧 token、score、checkpoint 留在 v1", "明确 alternative explanations"] },
      { kicker: "OPEN M0–M6", title: "建立 v2 readiness", lines: ["lineage → revision → runtime", "processor/data/eval → training/capacity", "candidate set → fail-closed promotion"] },
    ],
    knowledge: [
      { title: "监督语义先于训练", lines: ["先证明哪些 token 产生 CE、由哪个 logit", "预测，再讨论 loss 曲线。"] },
      { title: "Mixture 有多个分母", lines: ["examples、input tokens、supervised tokens", "回答不同问题，不能互换。"] },
      { title: "Baseline 必须早于 candidate", lines: ["训练后再改 prompt、parser 或 scorer，", "就失去纵向比较的因果边界。"] },
      { title: "Tiny overfit 是系统测试", lines: ["它证明 mask、gradient、optimizer、resume；", "不证明真实 recipe 或未见数据能力。"] },
      { title: "None 是一等公民", lines: ["selection policy 必须允许零候选，", "否则实验会被迫产出错误 checkpoint。"] },
      { title: "Held-out 是消耗品", lines: ["只在规则允许的时点查看一次；", "未消费状态本身也是重要证据。"] },
    ],
    metrics: [
      { label: "DAY 08 CONTRACT", value: "20 / 20 golden", lines: ["10 accept · 8 schema · 2 encoded reject"] },
      { label: "DAY 09 CLEAN POOL", value: "7,860 records", lines: ["2,774,246 supervised tokens"] },
      { label: "DAY 10 MEASURE", value: "112 dev · 48 sealed", lines: ["Base 15/112 · Code 0/28"] },
      { label: "DAY 11–12 DECISION", value: "98.89% → 0 eligible", lines: ["pipeline pass；checkpoint promotion none"] },
    ],
    table: {
      headers: ["证据日", "观察对象", "支持的 claim", "不能外推"],
      widths: [230, 420, 470, 540],
      rows: [
        ["Day 08", "token / label / shift audit", "监督位置与拒绝逻辑正确", "真实来源质量或模型能力"],
        ["Day 09", "lineage / mixture / overlap", "数据可重建、预算口径透明", "不存在语义改写污染"],
        ["Day 10", "frozen Base 112-dev", "后续有固定比较锚点", "跨模型、跨 execution 直接相减"],
        ["Day 11", "18-row tiny overfit", "当前 pipeline 能学与恢复", "泛化或上线 recipe"],
        ["Day 12", "17 states / joint gates", "合法结论是 no eligible", "从最高总分反推可晋级"],
      ],
    },
    evidenceNote: [
      "Readiness ladder：M0 lineage · M1 revision/files · M2 runtime/loader · M3 processor/mask · M4 data/eval · M5 train/capacity · M6 promotion。",
      "任何 gate 只证明其本层对象；“后面跑通”不能静默回填前面未生成的 protocol-equivalent artifact。",
    ],
    decision: [
      "Week 2 最强的知识不是某个分数，而是一条 evidence-first 链：",
      "data contract → lineage → frozen baseline → pipeline acceptance → controlled candidates → joint-gate selection。",
      "当迁移到 Qwen3.5 时，保留这条链的结构，同时把所有模型绑定对象重新建立。",
      "复盘的完成标准不是“每项看起来都成功”，而是每项都有 Claim / Evidence / Alternative / Limit。",
      "因此 Day 12 的 no-candidate 与未消费 frozen test 都属于成功的实验治理结果。",
    ],
    boundaries: [
      "• 原计划的 week2-evidence-review.md、",
      "  week2-claim-evidence-table.md 与",
      "  qwen35-v2-readiness-contract.md 未单独生成。",
      "• 本图从 Day 08–12 durable artifacts 与后来",
      "  migration contract 重建审计地图。",
      "• 后来形成的 v2 证据不能改写 v1 历史。",
      "• 旧 dev 多次用于开发后只能保留诊断角色。",
    ],
    sources: [
      "Sources · Day 08–12 reports/SVGs · PROGRESS.md · artifacts/configs/qwen35-4b-migration-contract.json",
      "Scope · Week 2 evidence review roadmap reconstructed 2026-08-14; original named Day 14 review files remain absent",
    ],
  },
  {
    file: "day-15-qwen35-onboarding-close-roadmap.svg",
    title: "Day 15 · Qwen3.5 Onboarding：从 S0 身份到可训练入口",
    subtitle: "不是换 Model ID，而是重建 loader、processor、训练 scope、状态与容量的完整契约。",
    status: "CLOSED BY SUPERSEDING EVIDENCE ≠ ORIGINAL PROTOCOL PASS",
    statusTone: "amber",
    description: "A Chinese roadmap showing the Qwen3.5 onboarding gates and how later Day 18 through Day 20 operational evidence closed the need for a duplicate run without pretending that the original Day 15 protocol artifacts existed.",
    gate: [
      "后续更强实跑证据可以关闭重复 onboarding，但只能按 evidence crosswalk 标注 covered / partial / not executed。",
      "closed_superseded 表示“不再重跑”，不是把原 Day 15 六件未生成产物事后改写成 PASS；onboarding 也不会自动创建 S1。",
    ],
    pipeline: [
      { kicker: "IDENTITY", title: "冻结 S0 文件身份", lines: ["exact revision / license / config", "shards、processor 与 template hashes", "禁止 main/latest 漂移"] },
      { kicker: "LOADER", title: "解析真实模型入口", lines: ["Qwen3_5ForConditionalGeneration", "Qwen3VLProcessor + GDN / MTP", "纯文本也保留完整 checkpoint"] },
      { kicker: "SCOPE", title: "冻结 text-only ownership", lines: ["image/video count = 0", "ViT / aligner 必须 bitwise unchanged", "LoRA targets 先枚举后白名单"] },
      { kicker: "SUPERVISION", title: "验证 processor 与 loss", lines: ["template / thinking / special tokens", "现场编码 supervised tokens", "response boundary 与 truncation audit"] },
      { kicker: "TRAINING", title: "分层证明可训练", lines: ["optimizer update ≠ loader success", "checkpoint continuation + export/reload", "tiny overfit 证明入口可学习"] },
      { kicker: "CAPACITY", title: "在真实 topology 测峰值", lines: ["2×H800 full-state TP/DP", "1×H800 BF16 LoRA @ 2304", "记录 peak / disk / runtime"] },
      { kicker: "CLOSE", title: "保留缺口后关闭", lines: ["逐 Gate 映射 superseding evidence", "不补造原 manifest / split / resume", "Base active；S1 仍需独立 promotion"] },
    ],
    knowledge: [
      { title: "模型身份是文件集合", lines: ["repo ID 不足以复现；revision、config、shards、", "processor、template 和 license 必须共同冻结。"] },
      { title: "能加载不等于能训练", lines: ["forward、真实 update、full-state save/load、", "export/reload 与 learnability 是不同 gate。"] },
      { title: "冻结需要运行时证据", lines: ["配置里写 freeze 不够；必须比较 parameter", "ownership 与训练前后 tensor identity。"] },
      { title: "Supersede 不是回填", lines: ["更长、更真实路径可降低重复运行价值；", "它不能生成从未存在的 protocol artifact。"] },
      { title: "Capacity 与 correctness 正交", lines: ["显存可容纳只解除工程 blocker；", "不会证明数据合同或 checkpoint 可晋级。"] },
      { title: "Onboarding 之后仍是 S0", lines: ["runtime compatible / learnable 只证明入口；", "S1 需要受控 candidate set 与 selection。"] },
    ],
    metrics: [
      { label: "CONVERSION", value: "738 mappings", lines: ["15/15 MTP tensors exact round-trip"] },
      { label: "DISTRIBUTED SMOKE", value: "TP1/DP2 · TP2/DP1", lines: ["两种 topology 各 3 个 optimizer updates"] },
      { label: "LEARNABILITY", value: "150 / 150 updates", lines: ["72/72 teacher-forced targets correct"] },
      { label: "OWNERSHIP", value: "297 unchanged", lines: ["all vision / aligner tensors bitwise stable"] },
    ],
    table: {
      headers: ["原 Day 15 Gate", "后续 evidence", "Close disposition", "仍保留的边界"],
      widths: [300, 430, 390, 540],
      rows: [
        ["Revision / files", "S0 registry + Day 18", "covered", "冻结身份不等于 GPU train"] ,
        ["Runtime / loader", "Day 18 C0–C5", "covered", "只在固定 2×H800 envelope"] ,
        ["Processor / loss", "Day 18 + Day 19/20", "operationally covered", "原 10+10 golden audit 未生成"] ,
        ["7,860 manifest / split", "无 protocol-equivalent artifact", "not executed", "实际六源合同不能改名冒充"] ,
        ["LoRA exact resume", "save + other-stack continuation", "carried / optional", "无 uninterrupted-vs-resumed comparator"] ,
        ["Capacity", "2×H800 full + 1×H800 LoRA", "covered", "packing parity 与 S1 仍独立"] ,
      ],
    },
    evidenceNote: [
      "Day 18 close：C0–C5、10/10 required gates、65 项 hashed evidence、open problems=0；Day 20 另证明单卡 H800 LoRA 可更新与保存。",
      "这些证据足以回答“入口是否可运行”，但不能回答“哪一个 checkpoint 应当晋级”。",
    ],
    decision: [
      "最终处置：Day 15 closed_superseded_by_day18_20，追加 GPU = 0。",
      "允许陈述：exact Qwen3.5 S0、conditional loader、text-only freeze scope、真实 update、",
      "distributed checkpoint/export 与单卡 LoRA capacity 都有运行证据。",
      "禁止陈述：原 Day 15 manifest、独立 split、10+10 audit、LoRA exact resume 全部通过。",
      "最重要的知识：用 evidence level 关闭问题，而不是用一个“done”标签抹平证据差异。",
    ],
    boundaries: [
      "• 原计划 6 个 Day 15 artifact 未生成。",
      "• Day 09 的 7,860 条数据没有 Qwen3.5 v2 manifest。",
      "• 新 selection / confirmation split 未按原协议冻结。",
      "• C4 continuation 没有 uninterrupted comparator。",
      "• packing correctness / throughput parity 未运行。",
      "• Day 15 close 时 Base remains active；无 S1。",
      "• 后续 S1 不能追溯改写本日 closeout。",
    ],
    sources: [
      "Sources · artifacts/reports/day15-close.md · day18-qwen35-megatron-compatibility.md · day-20-qwen35-balanced-lora-sft/README.md",
      "Status · closed_superseded_by_day18_20 · original protocol artifacts intentionally not backfilled",
    ],
  },
  {
    file: "day-16-controlled-lora-no-candidate-roadmap.svg",
    title: "Day 16 · Controlled LoRA：三档 LR Probe，0 个合格 Anchor",
    subtitle: "训练能稳定更新与保存，只是必要条件；fail-closed 比从失败项里挑“最好”更重要。",
    status: "CLOSED NO CANDIDATE · MAIN NOT STARTED · PACKING FALSE",
    statusTone: "red",
    description: "A Chinese roadmap of the controlled Qwen3.5 LoRA learning-rate probes, their shared training identity, the fixed diagnostic guardrails, the no-eligible-candidate decision, and the distinction between safe training evidence and model-quality promotion evidence.",
    gate: [
      "三档 LoRA 都完成 16,000 supervised tokens 与 103 steps，但 0/3 通过与 E2B 无关的必要质量 gate。",
      "因此在 256k main 前合法停止：packing=false、no early/mid/final、no merge、no confirmation、no S1。",
    ],
    pipeline: [
      { kicker: "FREEZE RUN", title: "固定唯一变量", lines: ["同一 exact S0 / data order / seed", "同一 LoRA scope / hardware / eval", "只改变 learning rate"] },
      { kicker: "BUILD DATA", title: "建立等监督预算", lines: ["六源 balanced contract", "821 records · 16,000 targets", "四技能各 4,000 supervised tokens"] },
      { kicker: "AUDIT SCOPE", title: "白名单语言模块", lines: ["248 target modules", "496 trainable tensors", "排除 ViT / aligner / embed / lm_head"] },
      { kicker: "FIX BATCH", title: "冻结训练形状", lines: ["BF16 · max_length 2304", "batch 2 × grad accumulation 4", "packing=false · padding_free=false"] },
      { kicker: "SAFETY", title: "先跑 5-step gate", lines: ["finite loss / grad / update", "warmup 首步 LR=0 单独解释", "三档都通过训练安全检查"] },
      { kicker: "PROBE", title: "各跑 103 steps", lines: ["LR = 1e-5 / 3e-5 / 1e-4", "每档保存 model-only adapter", "同一 32-row diagnostic"] },
      { kicker: "STOP", title: "应用硬 gate", lines: ["retention + Code eligibility", "0/3 eligible，main fail closed", "不制造 selection / anchor / S1"] },
    ],
    knowledge: [
      { title: "稳定训练 ≠ 合格模型", lines: ["finite loss、非零更新和 adapter save", "不能替代能力与 guardrail gate。"] },
      { title: "比较必须只动一个变量", lines: ["Base、数据顺序、预算、LoRA scope、", "hardware 与 eval identity 保持一致。"] },
      { title: "Guardrail 能否决总分", lines: ["Math retention 或 Code execution eligibility", "失败时，即使其它 slice 上升也不能晋级。"] },
      { title: "缺 E2B 也能合法停止", lines: ["当每个候选已触发其它独立硬失败，", "E2B 最好结果也无法恢复完整资格。"] },
      { title: "Packing 默认不批准", lines: ["未验证 boundary / mask / position / isolation，", "正确状态是 false，不是 parity pass。"] },
      { title: "No candidate 是完整结果", lines: ["candidate set 可以为空；不能为了推进日程", "手工挑选“失败中最好”的 LR。"] },
    ],
    metrics: [
      { label: "PROBE DATA", value: "821 records", lines: ["16,000 supervised tokens per LR"] },
      { label: "TRAINING", value: "103 steps × 3", lines: ["three 5-step safety gates passed"] },
      { label: "SCOPE", value: "248 modules", lines: ["496 trainable tensors · language only"] },
      { label: "SELECTION", value: "0 / 3 eligible", lines: ["256k main remained unstarted"] },
    ],
    table: {
      headers: ["Candidate", "General", "Math", "Finance", "Code eligible", "硬失败 / 结论"],
      widths: [280, 170, 170, 190, 230, 620],
      rows: [
        ["Base", "0/8", "7/8", "3/8", "8/8", "diagnostic baseline"] ,
        ["LoRA 1e-5", "0/8", "4/8", "4/8", "8/8", "Math 比 Base −3；ineligible"] ,
        ["LoRA 3e-5", "5/8", "5/8", "4/8", "6/8", "Math −2；Code <7/8；ineligible"] ,
        ["LoRA 1e-4", "6/8", "6/8", "4/8", "0/8", "Code <7/8；ineligible"] ,
      ],
    },
    evidenceNote: [
      "三条 trajectory 都证明 LoRA 在单卡 H800 上可更新、可保存；表中的 Code 是静态/sandbox-execution eligibility，不是 HumanEval pass。",
      "Main 13,197 records / 256,000 tokens 只完成准备，没有 early/mid/final checkpoint；formal PROBE-SELECTION.json 与 E2B 结果均不存在。",
    ],
    decision: [
      "最终决定：closed_no_eligible_candidate_by_day20_evidence。",
      "保留 Base active；不启动 main、不合并 adapter、不消费 confirmation、不登记 provisional anchor。",
      "Packing 继续固定 false，直到新 charter 用同一批输入证明 labels、position IDs、",
      "cross-sample attention isolation、loss 与吞吐都满足预注册 parity。",
      "后续 RSI v0002 属于新的因果分支；它的成功不能回填本日 0/3 历史。",
    ],
    boundaries: [
      "• E2B correctness 未运行；不得声称 HumanEval 分数。",
      "• formal probe selector marker 不存在。",
      "• packed P1 未运行；无吞吐或 semantic parity claim。",
      "• 三个 adapter checkpoint 均 non-resumable。",
      "• main early/mid/final、merge、confirmation 均不存在。",
      "• 结果只说明本冻结 recipe 无 eligible LR。",
      "• 不外推为“Qwen3.5 不适合 LoRA”。",
    ],
    sources: [
      "Sources · artifacts/reports/day16-gap-audit.md · day16-qwen35-sft-trajectory.md · day16-packing-parity.md · probe_metrics.json",
      "Status · closed_no_eligible_candidate · no additional GPU · historical result preserved after later RSI success",
    ],
  },
  {
    file: "day-17-exact-resume-evidence-roadmap.svg",
    title: "Day 17 · Exact Resume：Checkpoint 能加载，不等于轨迹连续",
    subtitle: "完整状态清单、Run A/B comparator 与 first-divergence 证据，才构成训练连续性证明。",
    status: "METHOD CONTRACT COMPLETE · OPTIONAL R NOT RUN · exact_resume=false",
    statusTone: "amber",
    description: "A Chinese roadmap separating checkpoint loadability, generation, fresh-process continuation, uninterrupted-versus-resumed trajectory parity and bitwise exactness, while clearly marking the Qwen3.5 LoRA Run A and Run B experiment as optional and not executed.",
    gate: [
      "selected/resumable S1 已归档，但真正的 exact-resume comparator 仍未运行：Run A 0→40 对 Run B 0→20→fresh process→40。",
      "Day 21 的 checkpoint integrity 与 adapter↔merged parity 不能替代 step 21–40 的数据、优化器、RNG 与 tensor trajectory 对齐。",
    ],
    pipeline: [
      { kicker: "FREEZE CONTRACT", title: "冻结同一训练身份", lines: ["Base / LoRA / data order / seed", "processor / packing / runtime / topology", "Run A 与 Run B 只能差中断"] },
      { kicker: "INVENTORY", title: "审计完整 state", lines: ["adapter + optimizer + scheduler", "Python / NumPy / Torch / CUDA RNG", "sampler cursor / global step / accumulation"] },
      { kicker: "RUN A", title: "连续训练对照", lines: ["同一初始化 step 0 → 40", "每步保存 IDs / loss / LR / grad", "作为 uninterrupted reference"] },
      { kicker: "RUN B1", title: "中断前保存", lines: ["step 0 → 20", "在 accumulation boundary 保存", "退出整个 Python 进程"] },
      { kicker: "RUN B2", title: "新进程恢复", lines: ["验证 state manifest 与 hashes", "从 step 20 继续到 40", "禁止 warm start 冒充 resume"] },
      { kicker: "COMPARE", title: "对齐 step 21–40", lines: ["sample/render/label token identity", "LR / loss / grad / cumulative targets", "adapter / optimizer tensor checksum"] },
      { kicker: "INJECT + CLASSIFY", title: "制造并定位分叉", lines: ["删 scheduler 或 sampler cursor", "跑 3–5 steps 找 first divergence", "报告 tolerance / max diff / exactness"] },
    ],
    knowledge: [
      { title: "Resume 是训练状态问题", lines: ["只有 adapter weights 是 warm start；", "下一次 update 还依赖 optimizer、RNG 与数据位置。"] },
      { title: "证据有五个等级", lines: ["load → generate → continue update → trajectory", "parity → bitwise exact，不能合并成一个 PASS。"] },
      { title: "比较单位是逐 step", lines: ["曲线看起来接近不够；必须核对 sample IDs、", "LR、loss、grad 与 tensor checksums。"] },
      { title: "数据游标属于模型轨迹", lines: ["sampler epoch、batch cursor、accumulation", "boundary 漂移会在首个新 batch 分叉。"] },
      { title: "Export 与 resumable 分开", lines: ["merged HF export 服务推理；完整 checkpoint", "服务继续训练，它们有不同 state inventory。"] },
      { title: "故障注入检验 auditor", lines: ["主动删 scheduler/cursor，确认检测器能指出", "第一个错误 step，而非只报告最终 checksum。"] },
    ],
    metrics: [
      { label: "PLANNED COMPARATOR", value: "0→40 vs 0→20→40", lines: ["comparison window = steps 21–40"] },
      { label: "AVAILABLE WINNER", value: "checkpoint-1904", lines: ["selected · resumable · immutable archive"] },
      { label: "STATE ARCHIVE", value: "11 files · ≈187 MiB", lines: ["optimizer / scheduler / RNG / trainer present"] },
      { label: "RUNTIME RESULT", value: "exact_resume = false", lines: ["Optional R not run; does not block S1"] },
    ],
    table: {
      headers: ["证据等级", "当前 evidence", "能支持的 claim", "仍缺什么"],
      widths: [300, 440, 420, 500],
      rows: [
        ["1 · Checkpoint loads", "Day 21 archive / integrity", "资产完整、可解析", "无训练轨迹比较"] ,
        ["2 · Inference works", "adapter ↔ merged 4/4", "冻结 smoke conversion parity", "不是 optimizer resume"] ,
        ["3 · Fresh continuation", "Day 18 Megatron 3→5", "另一训练栈可继续更新", "没有 target LoRA comparator"] ,
        ["4 · Trajectory parity", "not run", "尚无", "Run A/B step 21–40"] ,
        ["5 · Bitwise exact", "not claimed", "尚无", "kernel determinism + tensor diff"] ,
      ],
    },
    evidenceNote: [
      "Day 11 的 0.6B full-SFT 6-step vs 3+3 exact probe证明过方法；Day 18 证明 fresh continuation；两者都不能跨 lineage / runtime 替代 Qwen3.5 HF LoRA Run A/B。",
      "当前最诚实的结论是：state inventory 与实验设计 ready，runtime comparator 未执行，exactness 等级未授予。",
    ],
    decision: [
      "Day 17 的学习目标已被明确成一份可执行的 continuity contract；真正实验保留为 Optional R。",
      "它不再阻塞 S1，因为 promotion 依赖的是 checkpoint integrity、fixed-suite qualification 与 export parity；",
      "但任何“训练可以无缝恢复”的强 claim 都必须等 Run A/B 证据，而不能引用 load 成功。",
      "若未来出现 loss 跳变、LR 错位或数据重复，才按同一单卡 topology 执行 Optional R。",
      "一句话记忆：resume 的问题不是‘文件能否打开’，而是‘下一次 update 是否还是同一个 update’。",
    ],
    boundaries: [
      "• Run A / Run B 均未执行。",
      "• exact_resume_proven=false；不关闭为 PASS。",
      "• 早期“无 selected candidate”已被后来 S1 取代；",
      "  但新 checkpoint 不会自动产生历史 comparator。",
      "• 4/4 merge parity 只覆盖推理 smoke cohort。",
      "• Day 18 continuation 属于 Megatron full state。",
      "• kernel nondeterminism 若存在，需报告首个分叉",
      "  与 max_abs_diff，不能用“曲线相似”代替。",
    ],
    sources: [
      "Sources · artifacts/reports/day17-gap-audit.md · day17-qwen35-resume-equivalence.md · configs/day17-qwen35-resume-readiness.json",
      "Latest state context · artifacts/reports/day21-qwen35-s1-handoff.md · Optional R not executed",
    ],
  },
  {
    file: "day-18-qwen35-megatron-compatibility-roadmap.svg",
    title: "Day 18 · Qwen3.5 × Megatron：从源码链到 C0–C5 闭环",
    subtitle: "兼容性不是“命令能启动”，而是 conversion、parity、update、checkpoint、export 与 ownership 的证据链。",
    status: "CLOSED PASS · C0–C5 · 10/10 REQUIRED GATES",
    statusTone: "green",
    description: "A Chinese roadmap for the Qwen3.5 Megatron compatibility and learnability run, covering the pinned runtime codepath, HF-to-MCore conversion, parity, DP and TP optimizer smokes, distributed checkpoint continuation, export and 150-step tiny overfit with ownership audit.",
    gate: [
      "10/10 required gates PASS：只在 exact revision、BF16、2×H800、text-only 两行 fixture、冻结 TP/DP/MTP/checkpoint/export envelope 内成立。",
      "它证明兼容与可学习，不证明 vision、长序列、多节点、吞吐 scaling、泛化、exact resume 或可晋级 S1。",
    ],
    pipeline: [
      { kicker: "FREEZE ENVELOPE", title: "冻结运行边界", lines: ["Qwen3.5 revision + ms-swift commit", "2×H800 PCIe · BF16 · same host", "2 rows · 72 supervised targets"] },
      { kicker: "TRACE SOURCE", title: "插桩 8-node codepath", lines: ["loader → processor → conversion", "GDN → loss → optimizer", "distributed save/load → HF export"] },
      { kicker: "C0", title: "HF → MCore conversion", lines: ["核对 parameter names / shapes", "738 tensor mappings accounted", "15 MTP tensors round-trip"] },
      { kicker: "C1", title: "单 rank 数值 parity", lines: ["HF vs MCore logits / loss", "loss-token argmax mismatch = 0", "在预注册 tolerance 内"] },
      { kicker: "C2 / C3", title: "DP 与 TP 真更新", lines: ["TP1/DP2：3 updates", "TP2/DP1：3 updates", "固定 label-token budget 与 ownership"] },
      { kicker: "C4", title: "分布式 checkpoint", lines: ["新进程 iteration 3 → 5", "恢复 model / Adam / RNG", "HF export/reload parity"] },
      { kicker: "C5", title: "同入口 tiny overfit", lines: ["fresh Base → 150 updates", "72/72 teacher-forced targets", "vision / aligner bitwise unchanged"] },
    ],
    knowledge: [
      { title: "Compatibility 是端到端属性", lines: ["loader、conversion、kernel、loss、optimizer、", "checkpoint/export 任一断裂都不能叫兼容。"] },
      { title: "TP/DP 要比较语义", lines: ["不仅看进程存活；还要固定 label-token budget、", "首个 loss、真实 update 与 state ownership。"] },
      { title: "Text-only 仍审计 vision", lines: ["无视觉 token 不是冻结证明；需比较训练前后", "所有 vision / aligner tensors。"] },
      { title: "Parity 要提前写 tolerance", lines: ["先冻结 logits/loss/argmax 标准，", "再运行，避免看结果后放宽阈值。"] },
      { title: "Continuation ≠ exact resume", lines: ["C4 证明新进程能继续；没有 uninterrupted", "5-step comparator 就不能声称 trajectory exact。"] },
      { title: "Tiny overfit 是入口证据", lines: ["它证明相同训练入口能学、保存、导出；", "不证明对未见任务的泛化。"] },
    ],
    metrics: [
      { label: "CONVERSION", value: "738 · 15/15 MTP", lines: ["HF mappings · exact MTP round-trip"] },
      { label: "TOPOLOGY", value: "3 + 3 updates", lines: ["TP1/DP2 and TP2/DP1"] },
      { label: "C5 LEARNING", value: "150/150 · 72/72", lines: ["loss 2.4077 → 3.31e-08"] },
      { label: "CLOSEOUT", value: "65 hashes · 0 open", lines: ["16/16 tests · 17 issue classes resolved/observed"] },
    ],
    table: {
      headers: ["Gate", "关键测试", "运行结果"],
      widths: [170, 650, 840],
      rows: [
        ["C0", "HF→MCore→HF tensor accounting", "738 mappings；15/15 MTP exact round-trip"] ,
        ["C1", "2-row logits / loss parity", "HF 2.407724 vs MCore 2.402899；argmax mismatch 0"] ,
        ["C2", "TP1 / DP2 optimizer + checkpoint", "loss 2.4034→0.2631；peak 52.34 GiB/GPU"] ,
        ["C3", "TP2 / DP1 optimizer + checkpoint", "loss 2.4001→0.2518；peak 38.91 GiB/GPU"] ,
        ["C4", "fresh continuation + export/reload", "iteration 3→5；export argmax mismatch 0"] ,
        ["C5", "150-step overfit + ownership audit", "accuracy 0.541667→1.0；297 visual tensors unchanged"] ,
      ],
    },
    evidenceNote: [
      "C2/C3 first-loss difference = 0.00330782；C4 export max difference = 0.00941098；所有 required gates 在预注册 tolerance 内通过。",
      "Closeout 重新验证 evidence tar、113 members、65 inventory entries、portable SHA sidecar 与 16 tests；追加 GPU = 0。",
    ],
    decision: [
      "结论：Qwen3.5-4B Base 在冻结的 Megatron envelope 内 compatible + learnable。",
      "安全复用方式：把它作为后续 loader/GDN/TP/DP/checkpoint/export 的 runtime evidence，",
      "而不是把 C5 的 two-row overfit checkpoint 当成正式 SFT candidate。",
      "最重要的工程方法：沿真实 source codepath 绑定每个 tensor、rank、state owner 与 artifact，",
      "并让每个失败停在第一个可辨别 gate，而不是用最终 exception 猜根因。",
    ],
    boundaries: [
      "• exact revision + 2×H800 + BF16 + two-row fixture。",
      "• 不覆盖 vision/image/video input path。",
      "• 不覆盖长序列、多节点或其它 TP/PP/CP 组合。",
      "• 不是 throughput benchmark 或 scaling claim。",
      "• C4 无 uninterrupted comparator；exact resume 未证明。",
      "• 不证明泛化，也不产生 S1。",
      "• Git 内 evidence tar 不含大型 checkpoint/export payload。",
    ],
    sources: [
      "Sources · artifacts/reports/day18-qwen35-megatron-compatibility.md · day18-close.md · day18-closeout.json",
      "Run · day18-qwen35-20260808T073811Z · close 2026-08-10 · status closed_pass_c0_c5",
    ],
  },
  {
    file: "day-19-full-sft-regression-diagnostic-roadmap.svg",
    title: "Day 19 · 评测契约错位 × Full-SFT 能力回退",
    subtitle: "代码假零分与真实能力退化可以同时存在；修 parser 不能自动修复被训练损伤的模型。",
    status: "STANDALONE DIAGNOSTIC COMPLETE · SEQUENTIAL TRACK NOT RUN",
    statusTone: "amber",
    description: "A Chinese roadmap for the completed standalone Qwen3.5 Full-SFT A, B and E comparison, showing matched supervised-token budgets, output-contract diagnosis, bounded response and HumanEval adapters, untouched Base control and the conclusion that parser distortion and genuine Full-SFT regression coexisted.",
    gate: [
      "C0 Base 59/112 > Full-SFT B 22 > A 7 > E 5：A/B/E 无一可晋级，且 frozen test 未消费。",
      "旧 Code 0/84 被 thinking wrapper + complete-function/continuation 错位放大；但 Math/Finance 可解析仍全错，说明还有真实训练回退。",
    ],
    pipeline: [
      { kicker: "MATCH INPUTS", title: "固定比较合同", lines: ["同一 Qwen3.5 Base / seed / backend", "A、B、E 各 64,608 targets", "同一 frozen 112-dev protocol"] },
      { kicker: "TRAIN", title: "三条 Full-SFT", lines: ["全语言模型 + MTP 更新", "vision / aligner frozen", "A/B/E 只改变 mixture"] },
      { kicker: "LEGACY EVAL", title: "发现系统性 Code 零分", lines: ["84/84 empty-think wrapper", "84/84 输出完整 def", "旧 scorer 期待函数体 continuation"] },
      { kicker: "SEPARATE ADAPTERS", title: "拆开两层边界", lines: ["response adapter 只去模型族前缀", "HumanEval adapter 判 solution/completion", "绝不改签名或实现"] },
      { kicker: "RESCORE", title: "不可变 sidecars", lines: ["legacy artifacts 保持只读", "eligible code 才进入 fresh E2B", "63/84 syntax salvageable 仅作诊断"] },
      { kicker: "ADD CONTROL", title: "补 untouched C0", lines: ["同一 112-dev / v2 adapter / E2B", "区分 Base 局限与 SFT regression", "不创建新 checkpoint"] },
      { kicker: "DECIDE", title: "拒绝三条 candidate", lines: ["B 虽是 A/B/E 最好，仍远低于 Base", "Best-E 另有 generation ceiling 回退", "不 promote，不归因于单一因素"] },
    ],
    knowledge: [
      { title: "Eval contract 属于系统", lines: ["模型输出边界、task composition 与 sandbox", "都会改变“能否被正确测量”。"] },
      { title: "Adapter 只能做有界转换", lines: ["可去一个已知 wrapper、识别 solution/body；", "不能修代码或重写答案。"] },
      { title: "Syntax salvage ≠ correctness", lines: ["能组成合法 Python 只说明旧零分被放大；", "语义正确必须以隔离 E2B 为准。"] },
      { title: "可解析仍错是真退化信号", lines: ["B 的 Math 28/28、Finance 24/28 可解析，", "却都是 0/28 correct。"] },
      { title: "Untouched Base 是因果控制", lines: ["没有 C0 就无法区分 Base 不适配、", "prompt 问题和 SFT 遗忘。"] },
      { title: "低 train loss 不等于保留能力", lines: ["Full-SFT 能完成 update/checkpoint，仍可能", "被窄数据推向局部输出风格。"] },
    ],
    metrics: [
      { label: "MATCHED BUDGET", value: "64,608 targets × 3", lines: ["A / B / E use the same model start"] },
      { label: "LEGACY CODE SHAPE", value: "84/84 wrapper + def", lines: ["systematic output-contract mismatch"] },
      { label: "BOUNDED DIAGNOSIS", value: "63 / 84 salvageable", lines: ["A15 · B25 · E23; not accuracy"] },
      { label: "PROMOTION", value: "0 eligible S1", lines: ["untouched Base remains the control"] },
    ],
    table: {
      headers: ["Checkpoint", "General", "Math", "Finance", "Code / E2B", "Total", "Decision"],
      widths: [300, 170, 170, 190, 210, 170, 450],
      rows: [
        ["C0 untouched Base", "1/28", "24/28", "15/28", "19/28", "59/112", "causal comparator / fallback"] ,
        ["Full-SFT A", "4/28", "0/28", "0/28", "3/28", "7/112", "reject"] ,
        ["Full-SFT B", "15/28", "0/28", "0/28", "7/28", "22/112", "best A/B/E, still reject"] ,
        ["Full-SFT E", "1/28", "0/28", "1/28", "3/28", "5/112", "reject + response-style regression"] ,
      ],
    },
    evidenceNote: [
      "Code SFT records：A 115 · B 221 · E 142；continuation-only prompt 与 indented-body target 都是 0，模型返回完整 def 与训练分布一致。",
      "因此结论是“测量契约错位 + 真实 Full-SFT 回退”并存，而不是 template 或模型规模的单一解释。",
    ],
    decision: [
      "Day 19 的关键进步是把失败拆成两条证据链：measurement distortion 与 model regression。",
      "先用有边界 adapter 恢复公平测量，再用 untouched Base 控制判断训练是否伤害能力。",
      "结果显示 B 的 Code 从 legacy 假零分恢复一部分，但 Math/Finance 仍归零，且总分远低于 Base。",
      "所以不重训 A/B/E、不降低 gate、不把 parser 修复写成模型能力提升；转向更可回退的 LoRA 与 target contract。",
      "一句话记忆：先修尺子，再判断模型；尺子修好后仍差，才是训练本身要负责。",
    ],
    boundaries: [
      "• 这是 completed standalone Qwen3.5 A/B/E diagnostic。",
      "• 顺序课程 optimizer/LR + failure-injection 未执行。",
      "• 单 seed、每 slice 28 条，只适合作工程 gate。",
      "• frozen test 未消费。",
      "• 63/84 仅为 syntax salvageable，不是 pass@1。",
      "• 0.6B 历史只能作背景，不能直接相减。",
      "• 不支持“LoRA 天生优于 Full-SFT”的跨合同结论。",
    ],
    sources: [
      "Sources · day-19-qwen35-sft-comparison/README.md · diagnostics/DAY19-ROOT-CAUSE-SOURCE-NOTES.md · rsi-control/history/legacy.json",
      "Scope note · sequential day-19-training-diagnostics-failure-injection remains not run",
    ],
  },
  {
    file: "day-20-target-boundary-qualified-lora-roadmap.svg",
    title: "Day 20 · 从 Code Boundary 崩塌到 Qualified LoRA",
    subtitle: "沿 source → target → template/mask → train → verifier 找到 first broken invariant，只改一个因果 lever。",
    status: "RSI V0002 COMPLETE QUALIFIED · MERGE/HANDOFF MOVED TO DAY 21",
    statusTone: "green",
    description: "A Chinese evidence-first SFT roadmap tracing the Qwen3.5 code-indentation failure to target rendering and loss masking, applying a balanced training-only target-boundary repair, rerunning the frozen probe, training early mid and final checkpoints, and qualifying the selected recipe with an independent training seed.",
    gate: [
      "只有 target-boundary 修复后的 Probe、Primary full112 与 independent-training-seed Confirmation 都通过同一冻结门槛，才允许 qualification。",
      "Day 20 停在 fixed-suite qualified checkpoint；winner merge、fresh parity 与 downstream-ready S1 交付属于 Day 21。",
    ],
    pipeline: [
      { kicker: "FREEZE GOAL", title: "冻结目标与 action space", lines: ["Base / six-source data / LoRA recipe", "full112 / scorer / E2B / hard floors", "每版只批准一个 primary lever"] },
      { kicker: "V0001 PROBE", title: "观察 Code 崩塌", lines: ["Non-Code 改善但 Code eligibility 降低", "t12000+ = 0/8 eligible", "继续加 token 已平台化"] },
      { kicker: "ROOT CAUSE", title: "定位 first broken invariant", lines: ["stored target 105/105 以四空格开头", "live nonmasked label 0/105 保留", "故障在 template + loss masking"] },
      { kicker: "MINIMAL FIX", title: "只修训练 target boundary", lines: ["恢复首个四空格监督 token", "交换一个结构性尾 newline mask", "每行 target count 保持不变"] },
      { kicker: "V0002 PROBE", title: "重新预检与选择", lines: ["105/105 Code boundary pass", "四个 checkpoint 全部 8/8 eligible", "23/32 tie → 最早 t12000"] },
      { kicker: "MAIN", title: "Fresh Base 跑 320k", lines: ["1,904 optimizer steps", "early / mid / final 全部冻结", "先过 hard gate，再 deterministic rank"] },
      { kicker: "CONFIRM", title: "独立 seed 同套件复验", lines: ["Primary final 81/112", "Confirmation final 73/112", "complete_qualified；不偷做 merge"] },
    ],
    knowledge: [
      { title: "从症状找第一处不变量破坏", lines: ["最终 Code=0 不等于 LR 太大；逐层排除", "source、E2B、model output 后定位 label construction。"] },
      { title: "最小修复保持因果可解释", lines: ["只改 training target representation + 必要 mask；", "data rows、order、budget、LoRA、eval 全冻结。"] },
      { title: "监督预算可以守恒", lines: ["增加首个 indent token，同时 mask 结构性 newline；", "每行 supervised-token count 完全不变。"] },
      { title: "Probe 用于证伪，不保证成功", lines: ["若四个 checkpoint 仍 eligibility <7/8，", "必须停止，不进入 Main。"] },
      { title: "先 gate 后排名", lines: ["只有全部 hard floors 通过的 checkpoint", "才按 Total→General→Code→earlier tokens 排名。"] },
      { title: "Qualification 与交付分开", lines: ["分数通过、merge parity、downstream key 是", "三个不同状态，必须分别验收。"] },
    ],
    metrics: [
      { label: "ROOT CAUSE", value: "105/105 → 0/105", lines: ["stored four-space prefix → live supervised prefix"] },
      { label: "REPAIRED PROBE", value: "8 / 8 eligible", lines: ["all four checkpoints; winner t12000"] },
      { label: "PRIMARY FINAL", value: "81 / 112", lines: ["G21 · M21 · F19 · Code20"] },
      { label: "CONFIRMATION", value: "73 / 112", lines: ["fresh Base · seed 20260810 · all gates pass"] },
    ],
    table: {
      headers: ["Object", "G", "M", "F", "Code", "Total", "Role / decision"],
      widths: [300, 160, 160, 180, 180, 180, 500],
      rows: [
        ["Base", "1", "24", "16", "18", "59", "comparator / fallback only"] ,
        ["Primary early", "21", "21", "22", "17", "81", "eligible"] ,
        ["Primary mid", "21", "20", "17", "17", "75", "eligible"] ,
        ["Primary final", "21", "21", "19", "20", "81", "winner：Code 20 > early 17"] ,
        ["Confirmation final", "21", "21", "14", "17", "73", "same recipe · independent seed · pass"] ,
      ],
    },
    evidenceNote: [
      "Main hard floors：Total≥65 · G≥5 · M≥17 · F≥10 · Code≥14 · Format≥90 · eligible≥26 · infra=0；early/mid/final 全部通过。",
      "v0002 = 1.8401 GPU-hours；v0001+v0002 = 2.0912 GPU-hours；单张 RTX PRO 6000，训练峰值约 25 GiB。",
    ],
    decision: [
      "RSI v0002 的成功来自一个可证伪的因果修复：恢复 Code continuation 的首个监督缩进。",
      "Primary final 与独立 seed Confirmation 都通过 frozen full112，因此 recipe 获得 same-suite qualification。",
      "这并不抹去 v0001 的失败：旧版本永久记录 target-boundary invariant 被破坏，以及为何继续加 token 无效。",
      "Day 20 到此发布 complete_qualified；不在同一天偷偷 merge，以保持 selection、conversion 与 handoff 的责任边界。",
      "一句话记忆：不要扫参数来掩盖确定性 label bug；修复第一处错误，再重跑同一证伪合同。",
    ],
    boundaries: [
      "• 只证明 frozen full112 qualification。",
      "• Confirmation 复用同一题集，不是独立 held-out。",
      "• 两个 seed 不能估计 seed distribution 或显著性。",
      "• 不证明一般金融 Agent 能力提升。",
      "• Day 20 未做 winner merge / downstream key。",
      "• LoRA exact-resume comparator 未运行。",
      "• 顺序 weekend failure-signature reading 仍未单独执行。",
    ],
    sources: [
      "Sources · rsi-control/versions/rsi-v0001/metrics.json · RSI-V0001-ANALYSIS-AND-V0002-ASSUMPTION.md · rsi-v0002/metrics.json",
      "Result · rsi-control/versions/rsi-v0002/RSI-V0002-RESULT.md · Primary 81/112 · Confirmation 73/112",
    ],
  },
  {
    file: "day-21-s1-selection-handoff-roadmap.svg",
    title: "Day 21 · 从 Qualified Checkpoint 到 Downstream-Ready S1",
    subtitle: "Selection、resumable archive、winner-only merge、fresh parity 与 downstream identity 是五个独立交付 gate。",
    status: "DONE · DOWNSTREAM-READY S1 · 4/4 EXACT PARITY",
    statusTone: "green",
    description: "A Chinese roadmap covering Qwen3.5 SFT candidate roles, frozen hard gates and deterministic ranking, independent-seed same-suite confirmation, immutable resumable checkpoint archival, winner-only merged export, fresh-process token-ID parity and self-hashed downstream identity.",
    gate: [
      "只有同时具备冻结选模证据、完整 resumable archive、winner-only merged export、fresh-process exact token-ID parity 与有效 downstream key，",
      "checkpoint 才是下游可消费的 S1；Base、Probe、裸 adapter URI 或未验证 export 都不能替代。",
    ],
    pipeline: [
      { kicker: "ROLE SET", title: "冻结候选角色", lines: ["Base = comparator / fallback", "Probe = 只授权 LR", "Primary early/mid/final 才参加排名"] },
      { kicker: "BIND MEASURE", title: "绑定同一量尺", lines: ["full112 + sample-order hash", "normalized / E2B comparison keys", "scorer / sandbox / hard floors"] },
      { kicker: "HARD GATE", title: "先判资格", lines: ["Total / slice / format / eligibility", "infrastructure failures = 0", "不合格对象不参加排名"] },
      { kicker: "RANK", title: "确定 operational winner", lines: ["Total↓ → General↓ → Code↓", "checkpoint target tokens↑", "final 以 Code 20 > 17 胜 early"] },
      { kicker: "CONFIRM", title: "独立训练 seed 复验", lines: ["fresh Base + same locked recipe", "Confirmation 73/112 全 gate 通过", "只验 recipe，不参加第二轮排名"] },
      { kicker: "ARCHIVE + MERGE", title: "交付两类资产", lines: ["11-file resumable checkpoint archive", "winner-only merged HF export", "processor / tokenizer assets 完整"] },
      { kicker: "PARITY + KEY", title: "Fresh load 后发布身份", lines: ["Base+adapter vs merged 两个进程", "4 prompts · exact output token IDs", "self-hashed manifest + downstream key"] },
    ],
    knowledge: [
      { title: "候选角色不能混排", lines: ["Probe 只选 LR、Confirmation 只验 recipe；", "把它们加入 Primary 排名会改变选择问题。"] },
      { title: "先 Gate，后 Ranking", lines: ["总分高但 guardrail 不过，不能靠 tie-break", "进入候选集。"] },
      { title: "Operational ≠ Statistical", lines: ["冻结规则可以给唯一执行决策；", "不能凭此制造显著性或唯一最优 claim。"] },
      { title: "Same-suite seed 不是 held-out", lines: ["第二个 seed 显示 recipe 可复现；", "它不提供新题泛化或 seed 分布估计。"] },
      { title: "训练态与推理态分开交付", lines: ["resume 需要 optimizer/scheduler/RNG；", "deployment 需要 merged shards + processor assets。"] },
      { title: "下游消费 identity，不猜路径", lines: ["promotion manifest 与 downstream key 绑定", "parent、hash、template、export 与状态。"] },
    ],
    metrics: [
      { label: "PRIMARY WINNER", value: "81 / 112", lines: ["final · G21 M21 F19 C20"] },
      { label: "CONFIRMATION", value: "73 / 112", lines: ["fresh Base · independent training seed"] },
      { label: "HANDOFF ASSETS", value: "11 + 11 files", lines: ["resumable archive + merged export"] },
      { label: "CONVERSION PARITY", value: "4 / 4 exact", lines: ["two fresh processes · 0 failed"] },
    ],
    table: {
      headers: ["Object", "General", "Math", "Finance", "Code", "Total", "Role / selection"],
      widths: [300, 170, 170, 190, 180, 170, 480],
      rows: [
        ["Base", "1", "24", "16", "18", "59", "fallback only · not ranked"] ,
        ["Primary early", "21", "21", "22", "17", "81", "eligible"] ,
        ["Primary mid", "21", "20", "17", "17", "75", "eligible"] ,
        ["Primary final", "21", "21", "19", "20", "81", "winner by frozen Code tie-break"] ,
        ["Confirmation final", "21", "21", "14", "17", "73", "same-recipe second seed · pass"] ,
      ],
    },
    evidenceNote: [
      "Collector 验证 44 items；resumable archive 含 adapter/optimizer/scheduler/RNG/trainer state；merged export 含 2 safetensors shards 与 processor/tokenizer。",
      "Adapter PID 7411 与 merged PID 7747 使用相同 input IDs、greedy generation：4 passed / 0 failed，没有放宽阈值。",
    ],
    decision: [
      "正式 S1 = main-s20260809-lr1e-4-final；它是 policy-selected operational winner。",
      "downstream key = s1:qwen35-4b:c169e0bb…d8170db0a，状态 active。",
      "Day 22/23/25 只能按该 key 与 promotion manifest 消费 parent，不能从 Base 或路径字符串猜身份。",
      "Primary 81 与 Confirmation 73 提醒我们保留不确定性：两点 spread 不能估计 seed effect，",
      "但足以满足预注册的 one-additional-seed same-suite qualification 与工程 handoff。",
    ],
    boundaries: [
      "• final 是 operational winner，不是 statistically unique best。",
      "• Confirmation 复用 full112，不是新题 held-out。",
      "• 两个 seed 无法估计 seed distribution。",
      "• 4/4 exact 只覆盖冻结 smoke cohort。",
      "• compact JSON 不是 checkpoint / model shard 备份。",
      "• 长期对象存储仍是 durability 工作。",
      "• exact-resume comparator 仍为 Optional R。",
    ],
    sources: [
      "Sources · day-21-weekend-eval-reading/README.md · artifacts/reports/day21-qwen35-s1-handoff.md",
      "Identity · artifacts/checkpoints/day21-qwen35-s1-promotion-manifest.json · day21-qwen35-s1-downstream-key.json",
    ],
  },
  {
    file: "day-22-preference-data-audit-roadmap.svg",
    title: "Day 22 · 从 S1 Rollout 到可审计 Preference Pairs",
    subtitle: "Provenance、双跑 Sandbox、Processor、family split 与盲审缺一不可；pair 数量不是可信度。",
    status: "MACHINE GATES PASS · HUMAN BLIND REVIEW PENDING",
    statusTone: "amber",
    description: "A Chinese roadmap from the downstream-ready Qwen3.5 S1 through on-policy MBPP rollouts, bounded adaptive resampling, duplicate filtering, two-run sandbox classification, preference-pair construction, processor auditing, family-level split isolation and position-swapped human blind review gating.",
    gate: [
      "200 个 promoted-S1 on-policy、non-synthetic pairs 已通过全部 machine gates；formal manifest machine_ready=true。",
      "但 50 对 × primary/swapped 的正式人工盲审尚未完成，formal_dpo_ready=false，Day 23 必须继续 BLOCKED。",
    ],
    pipeline: [
      { kicker: "FREEZE PARENT", title: "绑定正式 S1 与 family", lines: ["只允许 Day 21 downstream key", "351 MBPP main families − 21 probe", "冻结 330 clean train families"] },
      { kicker: "BASE ROUND", title: "同策略 K=6 rollout", lines: ["330 × 6 = 1,980 responses", "temperature 0.8 · top_p 0.95", "family-isolated generation"] },
      { kicker: "FILTER + E2B", title: "只执行独特候选", lines: ["格式过滤 + exact-response dedup", "1,638 unique candidates", "每条两次 fresh、断网 sandbox"] },
      { kicker: "ADAPTIVE R1", title: "只补无 pair 的 family", lines: ["189 families × K=12", "跨轮去重后执行 1,405 new", "同一 S1 / sampler contract"] },
      { kicker: "PAIR", title: "稳定 pass 对 stable wrong", lines: ["chosen = pass,pass", "rejected = wrong_answer,wrong_answer", "runtime/timeout/infra 全 quarantine"] },
      { kicker: "PROCESS + SPLIT", title: "重新编码并隔离 family", lines: ["200/200 processor audits pass", "prompt prefix identical · response-only loss", "train/dev/heldout = 154/17/29"] },
      { kicker: "BLIND REVIEW", title: "Position swap 后 fail closed", lines: ["50 pairs × 2 orderings = 100 rows", "concealed key 与 worksheet 分离", "formal human gate 未完成，不启 DPO"] },
    ],
    knowledge: [
      { title: "Preference 不天然等于正确", lines: ["chosen/rejected 只表达相对偏好；", "必须绑定测试、verifier、环境与 provenance。"] },
      { title: "错误类型不能混用", lines: ["wrong answer、runtime、timeout、infra error", "语义不同；只有稳定 wrong 才能作 rejected。"] },
      { title: "Adaptive 只补 coverage", lines: ["只对首轮没有 pass-vs-wrong 的 family 采样；", "不能换 policy 或改变采样合同。"] },
      { title: "长度是潜在 shortcut", lines: ["pair selection 先最小化相对 token-length delta，", "再考虑失败难度，降低 length bias。"] },
      { title: "DPO processor 必须重审", lines: ["chosen/rejected 共享相同 prompt prefix；", "只对连续 response span 计 log-prob。"] },
      { title: "Split 的单位是 family", lines: ["problem / prompt / test / source family", "不能随机按行切，否则会泄漏同题变体。"] },
    ],
    metrics: [
      { label: "SOURCE FAMILIES", value: "330 clean MBPP", lines: ["promoted S1 only · probe families excluded"] },
      { label: "UNIQUE CANDIDATES", value: "3,043", lines: ["1,230 pass · 1,430 wrong · 383 quarantine"] },
      { label: "FORMAL PAIRS", value: "200 · synthetic 0%", lines: ["processor 200/200 · no truncation"] },
      { label: "HUMAN GATE", value: "0 / 50 complete pairs", lines: ["10 human presentations ≠ 10 completed pairs"] },
    ],
    table: {
      headers: ["Stage", "Raw", "Unique / new", "Stable pass", "Stable wrong", "Pair / quarantine result"],
      widths: [290, 210, 270, 230, 250, 410],
      rows: [
        ["Base round · K6", "1,980", "1,638", "746", "719", "141 base/base pairs"] ,
        ["Adaptive R1 · K12", "2,268", "1,405", "484", "711", "adds cross-round + adaptive pairs"] ,
        ["Combined", "—", "3,043", "1,230", "1,430", "200 families with pass-vs-wrong"] ,
        ["Quarantine", "—", "383", "70 only-pass", "60 only-fail", "runtime/unstable/both-side cases excluded"] ,
      ],
    },
    evidenceNote: [
      "Pair origin：base/base 141 · base/adaptive 11 · adaptive/base 6 · adaptive/adaptive 42；每 family 最多一对。",
      "Split 154/17/29；problem / prompt / test / source overlap = 0/0/0/0；blind packet = 50 unique pairs / 100 position-swapped presentations。",
    ],
    decision: [
      "机器链路已经完成：S1 identity、rollout receipts、dedup、双跑 E2B、pair rubric、processor audit、",
      "family split、hash manifest 与 strict validator 全部可重放；machine_ready=true。",
      "正式 DPO 仍不能启动，因为 human_blind_review_pending 是唯一 blocker。",
      "10 human + 90 AI 的混合审计得到 84/89 directional agreement，只能用于校准；",
      "AI judgment 不能伪装成人工结论。完成 50 对正式盲审并裁决 position disagreement 后，才可发布 ready manifest。",
    ],
    boundaries: [
      "• historical 63-pair smoke 为 synthetic mechanism evidence，",
      "  不进入 formal bundle。",
      "• 双跑 E2B 只证明在现有 tests 下稳定。",
      "• 单一 MBPP source，不能声称 source-held-out。",
      "• heldout 是 preference-stage in-domain，不是 model unseen。",
      "• formal human review 仍为 0/50 complete pairs。",
      "• machine-ready ≠ formal-DPO-ready。",
      "• 任一 pair 被人工排除后仍需重新满足 ≥200 门槛。",
    ],
    sources: [
      "Sources · day-22-preference-data/README.md · artifacts/data/day22-qwen35-formal-s1-manifest.json",
      "Evidence · day22-qwen35-formal-s1-multiround-candidate-e2b-evidence.jsonl · processor-audit.jsonl · assembly-audit.json",
    ],
  },
];

const seenFiles = new Set();

for (const day of DAYS) {
  if (seenFiles.has(day.file)) throw new Error(`${day.file}: duplicate output filename`);
  seenFiles.add(day.file);
  if (day.pipeline.length !== 7) throw new Error(`${day.file}: pipeline must contain 7 stages`);
  if (day.knowledge.length !== 6) throw new Error(`${day.file}: knowledge must contain 6 cards`);
  if (day.metrics.length !== 4) throw new Error(`${day.file}: metrics must contain 4 cards`);
  if (day.table.headers.length !== day.table.widths.length) {
    throw new Error(`${day.file}: table headers and widths must have the same length`);
  }
  if (day.table.widths.reduce((total, width) => total + width, 0) !== 1660) {
    throw new Error(`${day.file}: table widths must add up to 1660`);
  }
  if (day.table.rows.some((row) => row.length !== day.table.headers.length)) {
    throw new Error(`${day.file}: every table row must match the header column count`);
  }
  const target = path.join(REPORT_DIR, day.file);
  fs.writeFileSync(target, renderDay(day), "utf8");
  console.log(`generated ${path.relative(path.resolve(__dirname, "..", ".."), target)}`);
}

console.log(`Generated ${DAYS.length} SVG roadmaps.`);
