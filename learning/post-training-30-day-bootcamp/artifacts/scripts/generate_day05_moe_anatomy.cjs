#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");

const WIDTH = 2400;
const HEIGHT = 4200;
const REPORT_DIR = path.resolve(__dirname, "..", "reports");
const OUTPUT = path.join(REPORT_DIR, "day05-moe-transformer-forward-backward-anatomy.svg");

const cfg = {
  B: 1,
  T: 8192,
  N: 8192,
  D: 2048,
  L: 48,
  V: 151936,
  Hq: 32,
  Hkv: 4,
  headDim: 128,
  E: 128,
  K: 8,
  F: 768,
  bytes: 2,
};

cfg.Q = cfg.Hq * cfg.headDim;
cfg.KV = cfg.Hkv * cfg.headDim;
cfg.A = cfg.N * cfg.K;

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function t(x, y, value, cls = "body", anchor = "start") {
  return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(value)}</text>`;
}

function lines(x, y, values, cls = "small", lineHeight = 25, anchor = "start") {
  const spans = values.map((value, index) => (
    `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${esc(value)}</tspan>`
  )).join("");
  return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${spans}</text>`;
}

function box(x, y, width, height, cls, title, body = [], options = {}) {
  const titleClass = options.titleClass || "label";
  const bodyClass = options.bodyClass || "small";
  const lineHeight = options.lineHeight || 24;
  const titleY = y + (options.titleOffset || 32);
  const bodyY = y + (options.bodyOffset || 62);
  return [
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${options.rx || 13}" class="${cls}"/>`,
    t(x + (options.padX || 20), titleY, title, titleClass),
    body.length ? lines(x + (options.padX || 20), bodyY, body, bodyClass, lineHeight) : "",
  ].join("\n");
}

function arrow(d, cls = "fwd") {
  return `<path d="${d}" class="${cls}"/>`;
}

function table(x, y, width, columns, rows, options = {}) {
  const headerHeight = options.headerHeight || 48;
  const rowHeight = options.rowHeight || 45;
  const height = headerHeight + rowHeight * rows.length;
  const parts = [
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="10" class="table-bg"/>`,
    `<rect x="${x}" y="${y}" width="${width}" height="${headerHeight}" rx="10" class="table-head"/>`,
  ];
  let cursor = x;
  for (let i = 0; i < columns.length; i += 1) {
    parts.push(t(cursor + 14, y + 31, columns[i].label, "table-header"));
    cursor += columns[i].width;
    if (i < columns.length - 1) {
      parts.push(`<line x1="${cursor}" y1="${y}" x2="${cursor}" y2="${y + height}" class="grid"/>`);
    }
  }
  rows.forEach((row, rowIndex) => {
    const rowY = y + headerHeight + rowIndex * rowHeight;
    if (rowIndex % 2 === 1) {
      parts.push(`<rect x="${x}" y="${rowY}" width="${width}" height="${rowHeight}" class="table-alt"/>`);
    }
    parts.push(`<line x1="${x}" y1="${rowY}" x2="${x + width}" y2="${rowY}" class="grid"/>`);
    cursor = x;
    row.forEach((cell, columnIndex) => {
      parts.push(t(cursor + 14, rowY + 29, cell, columnIndex === 0 ? "table-strong" : "table-body"));
      cursor += columns[columnIndex].width;
    });
  });
  return { svg: parts.join("\n"), height };
}

function mb(elements, bytesPerElement = cfg.bytes) {
  return `${(elements * bytesPerElement / 1e6).toFixed(1)} MB`;
}

function gb(elements, bytesPerElement = cfg.bytes) {
  return `${(elements * bytesPerElement / 1e9).toFixed(3)} GB`;
}

const P = {
  wq: cfg.D * cfg.Q,
  wk: cfg.D * cfg.KV,
  wv: cfg.D * cfg.KV,
  wo: cfg.Q * cfg.D,
  router: cfg.D * cfg.E,
  expertMatrix: cfg.D * cfg.F,
};
P.attention = P.wq + P.wk + P.wv + P.wo;
P.oneExpert = 3 * P.expertMatrix;
P.allExperts = cfg.E * P.oneExpert;
P.norms = 2 * cfg.D + 2 * cfg.headDim;
P.layer = P.attention + P.router + P.allExperts + P.norms;
P.activeLayer = P.attention + P.router + cfg.K * P.oneExpert + P.norms;
P.embed = cfg.V * cfg.D;

const A = {
  residual: cfg.N * cfg.D,
  q: cfg.N * cfg.Q,
  kv: cfg.N * cfg.KV,
  score: cfg.B * cfg.Hq * cfg.T * cfg.T,
  routerLogits: cfg.N * cfg.E,
  assignments: cfg.A,
  dispatch: cfg.A * cfg.D,
  expertHidden: cfg.A * cfg.F,
  logits: cfg.N * cfg.V,
};

const F = {
  attnProj: 2 * cfg.N * P.attention,
  qk: 2 * cfg.B * cfg.Hq * cfg.T * cfg.T * cfg.headDim,
  av: 2 * cfg.B * cfg.Hq * cfg.T * cfg.T * cfg.headDim,
  router: 2 * cfg.N * cfg.D * cfg.E,
  experts: 2 * cfg.N * cfg.K * 3 * cfg.D * cfg.F,
  lmHead: 2 * cfg.N * cfg.D * cfg.V,
};
F.layerForward = F.attnProj + F.qk + F.av + F.router + F.experts;

const attentionWeights = table(455, 846, 1440, [
  { label: "Projection", width: 230 },
  { label: "Weight matrix", width: 300 },
  { label: "Parameters", width: 220 },
  { label: "BF16 weight", width: 210 },
  { label: "Forward output", width: 480 },
], [
  ["Q", "[2048, 4096]", "8.389M", "16.8 MB", "[1,32,8192,128] · 67.1 MB"],
  ["K", "[2048, 512]", "1.049M", "2.1 MB", "[1,4,8192,128] · 8.4 MB"],
  ["V", "[2048, 512]", "1.049M", "2.1 MB", "[1,4,8192,128] · 8.4 MB"],
  ["O", "[4096, 2048]", "8.389M", "16.8 MB", "[1,8192,2048] · 33.6 MB"],
], { rowHeight: 43 });

const matrixLedger = table(80, 2415, 1080, [
  { label: "Per decoder layer", width: 250 },
  { label: "Shape / count", width: 320 },
  { label: "Params", width: 210 },
  { label: "BF16", width: 170 },
  { label: "Backward gradient", width: 130 },
], [
  ["RMSNorm ×2 + Q/K norm", "2×[2048] + 2×[128]", "4,352", "8.7 KB", "same shape"],
  ["WQ", "[2048,4096]", "8.389M", "16.8 MB", "dWQ"],
  ["WK / WV", "2 × [2048,512]", "2.097M", "4.2 MB", "dWK / dWV"],
  ["WO", "[4096,2048]", "8.389M", "16.8 MB", "dWO"],
  ["Router", "[2048,128]", "0.262M", "0.52 MB", "dWrouter"],
  ["One expert", "3 matrices · 2048↔768", "4.719M", "9.44 MB", "3 dW"],
  ["All 128 experts", "128 × one expert", "603.980M", "1.208 GB", "all trainable"],
  ["Full layer total", "attention + router + experts", "623.121M", "1.246 GB", "same count"],
  ["Active / token / layer", "attention + router + 8 experts", "56.890M", "113.8 MB", "route-dependent"],
], { rowHeight: 47 });

const activationLedger = table(1220, 2415, 1100, [
  { label: "Logical tensor · B=1,T=8192", width: 330 },
  { label: "Shape", width: 330 },
  { label: "Storage", width: 190 },
  { label: "Lifetime / note", width: 250 },
], [
  ["Residual / layer input", "[8192,2048]", "33.6 MB", "checkpoint candidate"],
  ["Q / K / V", "[8192,4096/512/512]", "67.1+8.4+8.4", "saved or recompute"],
  ["Naive attention score", "[1,32,8192,8192]", "4.295 GB", "FlashAttn avoids full"],
  ["Router logits", "[8192,128]", "2.1 / 4.2 MB", "BF16 / FP32"],
  ["Top-k index + weight", "[8192,8]", "0.52+0.26 MB", "int64 + FP32"],
  ["Dispatched tokens", "[65536,2048]", "268.4 MB", "load-dependent"],
  ["Gate / Up / Hidden", "3 × [65536,768]", "3 × 100.7 MB", "often fused/recomputed"],
  ["Expert output", "[65536,2048]", "268.4 MB", "inverse dispatch"],
  ["LM logits", "[8192,151936]", "2.489 GB", "fused/chunked CE helps"],
], { rowHeight: 47 });

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">Day 05 MoE Transformer anatomy with matrix shapes, temporary tensors, forward and backward propagation</title>
  <desc id="desc">A high-density Chinese poster that expands one Qwen3-30B-A3B decoder layer, labels all major attention, router and expert matrices, quantifies logical temporary tensors for B=1 and T=8192, and traces forward, backward, activation checkpointing and optimizer-state lifecycles.</desc>
  <defs>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#2563eb"/></marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#dc2626"/></marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#16a34a"/></marker>
    <marker id="arrow-amber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#d97706"/></marker>
    <style>
      text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", "PingFang SC", sans-serif; fill:#172033; }
      .title { font-size:38px; font-weight:760; }
      .subtitle { font-size:19px; fill:#526078; }
      .section { font-size:25px; font-weight:760; }
      .subsection { font-size:20px; font-weight:730; }
      .label { font-size:17px; font-weight:700; }
      .body { font-size:15px; }
      .small { font-size:13px; fill:#526078; }
      .tiny { font-size:11px; fill:#64748b; }
      .mono { font-family:"SFMono-Regular",Consolas,monospace; font-size:13px; fill:#273449; }
      .mono-strong { font-family:"SFMono-Regular",Consolas,monospace; font-size:15px; font-weight:750; fill:#1e3a8a; }
      .panel { fill:#ffffff; stroke:#cbd5e1; stroke-width:1.5; }
      .hero { fill:#eef2ff; stroke:#6366f1; stroke-width:1.8; }
      .forward-box { fill:#dbeafe; stroke:#2563eb; stroke-width:1.5; }
      .attention-box { fill:#eff6ff; stroke:#60a5fa; stroke-width:1.5; }
      .moe-box { fill:#fff7ed; stroke:#f59e0b; stroke-width:1.5; }
      .expert-box { fill:#f3e8ff; stroke:#9333ea; stroke-width:1.4; }
      .router-box { fill:#fef3c7; stroke:#d97706; stroke-width:1.5; }
      .backward-box { fill:#fee2e2; stroke:#dc2626; stroke-width:1.5; }
      .saved-box { fill:#dcfce7; stroke:#16a34a; stroke-width:1.5; }
      .state-box { fill:#ede9fe; stroke:#7c3aed; stroke-width:1.5; }
      .neutral-box { fill:#f8fafc; stroke:#94a3b8; stroke-width:1.3; }
      .inactive-box { fill:#f1f5f9; stroke:#cbd5e1; stroke-width:1.1; }
      .warning-box { fill:#fff1f2; stroke:#e11d48; stroke-width:1.5; }
      .fwd { fill:none; stroke:#2563eb; stroke-width:2.2; marker-end:url(#arrow-blue); }
      .bwd { fill:none; stroke:#dc2626; stroke-width:2.2; marker-end:url(#arrow-red); }
      .residual { fill:none; stroke:#16a34a; stroke-width:2; stroke-dasharray:7 5; marker-end:url(#arrow-green); }
      .route { fill:none; stroke:#d97706; stroke-width:2; marker-end:url(#arrow-amber); }
      .thin { fill:none; stroke:#94a3b8; stroke-width:1.4; }
      .divider { stroke:#cbd5e1; stroke-width:1.2; }
      .grid { stroke:#cbd5e1; stroke-width:1; }
      .table-bg { fill:#ffffff; stroke:#cbd5e1; stroke-width:1.2; }
      .table-head { fill:#e2e8f0; stroke:#94a3b8; stroke-width:1.1; }
      .table-alt { fill:#f8fafc; }
      .table-header { font-size:12px; font-weight:750; fill:#475569; letter-spacing:.2px; }
      .table-body { font-size:12px; }
      .table-strong { font-size:12px; font-weight:700; }
      .pill-blue { fill:#dbeafe; stroke:#2563eb; stroke-width:1.2; }
      .pill-red { fill:#fee2e2; stroke:#dc2626; stroke-width:1.2; }
      .pill-green { fill:#dcfce7; stroke:#16a34a; stroke-width:1.2; }
      .pill-purple { fill:#ede9fe; stroke:#7c3aed; stroke-width:1.2; }
      .pill-amber { fill:#fef3c7; stroke:#d97706; stroke-width:1.2; }
    </style>
  </defs>

  <rect width="${WIDTH}" height="${HEIGHT}" fill="#f5f7fb"/>
  ${t(55, 62, "Day 05 · MoE Transformer Anatomy：矩阵、临时张量与 Forward / Backward", "title")}
  ${t(55, 98, "把一整个训练 step 压进一张图：模型结构、单层矩阵账本、logical activation、反向公式与 tensor lifetime。", "subtitle")}

  <rect x="45" y="125" width="2310" height="112" rx="18" class="hero"/>
  ${t(75, 163, "固定 worked example", "subsection")}
  ${t(330, 163, "Qwen3-30B-A3B · decoder-only causal MoE · 每个 layer 都是 dense GQA + sparse MoE", "label")}
  ${t(330, 196, "B=1 · T=N=8,192 · BF16=2 B/elem · L=48 · D=2,048 · Hq/Hkv=32/4 · d=128 · E=128 · top-k=8 · F=768 · V=151,936", "mono")}
  ${t(2215, 163, "30.532B total", "mono-strong", "end")}
  ${t(2215, 196, "3.353B active/token", "mono-strong", "end")}

  <rect x="45" y="257" width="2310" height="55" rx="14" class="panel"/>
  <rect x="75" y="273" width="22" height="22" rx="5" class="forward-box"/>${t(108, 291, "Forward activation", "small")}
  <rect x="315" y="273" width="22" height="22" rx="5" class="backward-box"/>${t(348, 291, "Backward gradient", "small")}
  <rect x="555" y="273" width="22" height="22" rx="5" class="state-box"/>${t(588, 291, "Persistent weights/state", "small")}
  <rect x="835" y="273" width="22" height="22" rx="5" class="router-box"/>${t(868, 291, "Routing / temporary", "small")}
  <rect x="1095" y="273" width="22" height="22" rx="5" class="saved-box"/>${t(1128, 291, "Saved checkpoint", "small")}
  ${t(1510, 291, "蓝线向前 · 红线向后 · 绿虚线为 residual · 橙线为 token routing", "small")}

  <rect x="45" y="332" width="2310" height="1990" rx="20" class="panel"/>
  ${t(75, 376, "1 · 整体模型 + 展开一个 decoder layer：同一 residual stream 上的两次子层更新", "section")}
  ${t(75, 406, "左：48 层全模型 forward　·　中：Layer ℓ 的 attention/MoE 细节　·　右：loss 开始的 reverse-mode backward", "small")}

  <!-- Whole-model forward rail -->
  ${t(82, 455, "MODEL FORWARD", "label")}
  ${box(72, 485, 280, 92, "forward-box", "Token IDs", ["input_ids [1,8192]", "int64 · 65.5 KB"], {bodyClass:"mono", lineHeight:22})}
  ${arrow("M212 577 L212 615", "fwd")}
  ${box(72, 615, 280, 104, "state-box", "Embedding lookup", ["Wemb [151936,2048]", "311.165M · BF16 622.3 MB", "X₀ [8192,2048] · 33.6 MB"], {bodyClass:"mono", lineHeight:21})}
  ${arrow("M212 719 L212 760", "fwd")}
  <rect x="72" y="760" width="280" height="950" rx="18" class="attention-box"/>
  ${t(212, 806, "Decoder blocks × 48", "subsection", "middle")}
  ${t(212, 840, "每层独立 router", "small", "middle")}
  ${t(212, 870, "dense GQA", "label", "middle")}
  ${t(212, 900, "+", "section", "middle")}
  ${t(212, 932, "sparse MoE", "label", "middle")}
  ${t(212, 972, "Xℓ → Xℓ₊₁", "mono-strong", "middle")}
  ${t(212, 1020, "Layer 0", "small", "middle")}
  ${t(212, 1070, "⋮", "section", "middle")}
  ${t(212, 1125, "Layer ℓ · expanded →", "label", "middle")}
  ${arrow("M352 1120 L405 1120", "fwd")}
  ${t(212, 1190, "⋮", "section", "middle")}
  ${t(212, 1245, "Layer 47", "small", "middle")}
  <line x1="102" y1="1310" x2="322" y2="1310" class="divider"/>
  ${lines(98, 1350, ["每个 token 在每一层", "重新 top-8 / 128；", "batch 内通常会触及", "远多于 8 个 experts。"], "small", 25)}
  ${lines(98, 1480, ["MoE 只稀疏 expert compute；", "attention 仍对所有 token dense。", "所有 expert weights 仍是", "persistent trainable state。"], "small", 25)}
  ${arrow("M212 1710 L212 1750", "fwd")}
  ${box(72, 1750, 280, 84, "forward-box", "Final RMSNorm", ["Hfinal [8192,2048] · 33.6 MB"], {bodyClass:"mono"})}
  ${arrow("M212 1834 L212 1870", "fwd")}
  ${box(72, 1870, 280, 108, "state-box", "Untied LM head", ["Wlm [2048,151936]", "311.165M · BF16 622.3 MB", "Z [8192,151936] · 2.489 GB"], {bodyClass:"mono", lineHeight:21})}
  ${arrow("M212 1978 L212 2018", "fwd")}
  ${box(72, 2018, 280, 116, "forward-box", "Shifted masked CE", ["Z[:,:-1,:] ↔ labels[:,1:]", "valid labels only · loss scalar", "fused/chunked CE can avoid full dZ"], {bodyClass:"small", lineHeight:22})}

  <!-- Expanded layer -->
  <rect x="405" y="455" width="1520" height="1790" rx="20" class="neutral-box"/>
  ${t(440, 500, "Layer ℓ · Pre-Norm GQA Attention", "section")}
  ${t(1885, 500, "Xℓ [N,D] · 33.6 MB", "mono-strong", "end")}

  ${box(440, 535, 185, 88, "saved-box", "Xℓ", ["[8192,2048]", "save or recompute"], {bodyClass:"mono", lineHeight:21})}
  ${arrow("M625 579 L660 579", "fwd")}
  ${box(660, 535, 190, 88, "forward-box", "RMSNorm", ["γattn [2048]", "X̂ [N,D]"], {bodyClass:"mono", lineHeight:21})}
  ${arrow("M850 579 L885 579", "fwd")}

  ${box(885, 520, 265, 96, "attention-box", "Q projection", ["WQ [2048,4096]", "Q [N,4096] · 67.1 MB"], {bodyClass:"mono", lineHeight:22})}
  ${box(885, 632, 265, 96, "attention-box", "K projection", ["WK [2048,512]", "K [N,512] · 8.4 MB"], {bodyClass:"mono", lineHeight:22})}
  ${box(885, 744, 265, 96, "attention-box", "V projection", ["WV [2048,512]", "V [N,512] · 8.4 MB"], {bodyClass:"mono", lineHeight:22})}
  ${arrow("M850 579 C865 579 865 568 885 568", "fwd")}
  ${arrow("M850 579 C865 579 865 680 885 680", "fwd")}
  ${arrow("M850 579 C865 579 865 792 885 792", "fwd")}

  ${box(1185, 535, 220, 190, "attention-box", "Q/K Norm + RoPE", ["Q → [1,32,T,128]", "K → [1,4,T,128]", "GQA group = 8 Q / KV", "causal positions + mask"], {bodyClass:"mono", lineHeight:25})}
  ${arrow("M1150 568 L1185 580", "fwd")}
  ${arrow("M1150 680 L1185 650", "fwd")}
  ${arrow("M1405 630 L1440 630", "fwd")}
  ${arrow("M1150 792 C1320 792 1330 710 1440 710", "fwd")}

  ${box(1440, 535, 260, 225, "attention-box", "Scaled causal GQA", ["S = QKᵀ/√128 + mask", "S [1,32,T,T] logical", `naive BF16 = ${gb(A.score)}`, "A = softmax(S)", "C = A·V → [N,4096]", "QKᵀ + AV = 1.100 TFLOPs"], {bodyClass:"mono", lineHeight:27})}
  ${arrow("M1700 648 L1735 648", "fwd")}
  ${box(1735, 590, 155, 116, "attention-box", "O projection", ["WO [4096,2048]", "O [N,D]", "33.6 MB"], {bodyClass:"mono", lineHeight:22, padX:14})}
  ${arrow("M1890 648 L1910 648", "fwd")}
  <circle cx="1910" cy="648" r="16" fill="#dcfce7" stroke="#16a34a" stroke-width="1.7"/>
  ${t(1910, 655, "+", "label", "middle")}
  ${arrow("M532 535 C532 475 1910 475 1910 630", "residual")}

  ${attentionWeights.svg}
  ${t(455, 1090, "Backward inside attention", "label")}
  ${t(675, 1090, "dWO=CᵀdO · dC=dO·WOᵀ → dV,dA → softmax backward → dQ,dK → dWQ/dWK/dWV + dX̂", "mono")}
  ${t(455, 1118, "FlashAttention backward", "label")}
  ${t(675, 1118, "通常分块重算 score/probability；logical [T,T] 存在于数学，不代表完整 tensor 常驻 HBM。", "small")}

  <line x1="430" y1="1145" x2="1900" y2="1145" class="divider"/>
  ${t(440, 1185, "Layer ℓ · Pre-Norm Sparse MoE", "section")}
  ${t(1885, 1185, "Xattn [N,D] · residual after attention", "mono-strong", "end")}

  ${box(440, 1225, 180, 86, "saved-box", "Xattn", ["[8192,2048]", "33.6 MB"], {bodyClass:"mono", lineHeight:21})}
  ${arrow("M620 1268 L655 1268", "fwd")}
  ${box(655, 1225, 185, 86, "forward-box", "RMSNorm", ["γmoe [2048]", "X̃ [N,D]"], {bodyClass:"mono", lineHeight:21})}
  ${arrow("M840 1268 L875 1268", "fwd")}
  ${box(875, 1210, 225, 116, "router-box", "Router", ["Wr [2048,128]", "R [8192,128]", "2.1 MB BF16 / 4.2 FP32"], {bodyClass:"mono", lineHeight:22})}
  ${arrow("M1100 1268 L1130 1268", "route")}
  ${box(1130, 1210, 215, 116, "router-box", "softmax + top-k", ["idx [8192,8] int64", "weight [8192,8] FP32", "65,536 assignments"], {bodyClass:"mono", lineHeight:22})}
  ${arrow("M1345 1268 L1375 1268", "route")}
  ${box(1375, 1210, 225, 116, "router-box", "Dispatch / permute", ["Xd [65536,2048]", "268.4 MB logical", "avg 512 assigns/expert"], {bodyClass:"mono", lineHeight:22})}
  ${arrow("M1600 1268 L1630 1268", "route")}

  <rect x="1630" y="1205" width="255" height="410" rx="16" class="expert-box"/>
  ${t(1757, 1240, "Grouped expert GEMMs", "label", "middle")}
  ${t(1757, 1268, "8 selected / token", "small", "middle")}
  <rect x="1650" y="1290" width="100" height="52" rx="8" class="expert-box"/>${t(1700, 1321, "E₁", "label", "middle")}
  <rect x="1765" y="1290" width="100" height="52" rx="8" class="expert-box"/>${t(1815, 1321, "E₂", "label", "middle")}
  <rect x="1650" y="1355" width="100" height="52" rx="8" class="expert-box"/>${t(1700, 1386, "…", "label", "middle")}
  <rect x="1765" y="1355" width="100" height="52" rx="8" class="expert-box"/>${t(1815, 1386, "E₈", "label", "middle")}
  ${t(1650, 1442, "each expert:", "small")}
  ${t(1650, 1468, "Wgate [2048,768]", "mono")}
  ${t(1650, 1492, "Wup   [2048,768]", "mono")}
  ${t(1650, 1516, "Wdown [768,2048]", "mono")}
  ${t(1650, 1546, "4.719M params · 9.44 MB", "mono")}
  ${t(1650, 1574, "all 128 = 1.208 GB/layer", "mono")}

  ${arrow("M1757 1615 L1757 1650", "route")}
  ${box(1510, 1650, 375, 106, "router-box", "Weighted combine + inverse dispatch", ["Yd [65536,2048] · 268.4 MB", "Y [8192,2048] · 33.6 MB", "top-k weights sum selected expert outputs"], {bodyClass:"mono", lineHeight:22})}
  ${arrow("M1510 1703 L1475 1703", "fwd")}
  <circle cx="1457" cy="1703" r="16" fill="#dcfce7" stroke="#16a34a" stroke-width="1.7"/>
  ${t(1457, 1710, "+", "label", "middle")}
  ${arrow("M440 1268 C410 1268 410 1703 1439 1703", "residual")}
  ${arrow("M1457 1719 L1457 1760", "fwd")}
  ${box(1290, 1760, 335, 82, "saved-box", "Xℓ₊₁", ["[8192,2048] · 33.6 MB", "next layer checkpoint candidate"], {bodyClass:"mono", lineHeight:22})}

  <rect x="440" y="1360" width="1130" height="270" rx="14" class="moe-box"/>
  ${t(465, 1395, "MoE forward temporaries · A=N×top-k=65,536 routed assignments", "label")}
  ${t(465, 1430, "gate = Xd·Wgate", "mono")}${t(820, 1430, "[A,768] · 100.7 MB", "mono")}
  ${t(465, 1462, "up   = Xd·Wup", "mono")}${t(820, 1462, "[A,768] · 100.7 MB", "mono")}
  ${t(465, 1494, "hidden = SiLU(gate) ⊙ up", "mono")}${t(820, 1494, "[A,768] · 100.7 MB", "mono")}
  ${t(465, 1526, "Yd = hidden·Wdown", "mono")}${t(820, 1526, "[A,2048] · 268.4 MB", "mono")}
  ${t(465, 1570, "Fused grouped GEMM / permutation kernels may overwrite or tile these buffers; listed sizes are logical, not additive peak.", "small")}
  ${t(465, 1600, "Expert load is data-dependent: 512 assignments/expert is an average, not a capacity guarantee.", "small")}

  <rect x="440" y="1875" width="1445" height="205" rx="14" class="backward-box"/>
  ${t(465, 1910, "MoE backward · only selected routes carry activation gradient; top-k index is discrete", "label")}
  ${t(465, 1944, "1  combine backward → dYd [A,D]；inverse permutation returns/sums expert-path dX", "mono")}
  ${t(465, 1976, "2  dWdown = hiddenᵀ·dYd；dhidden = dYd·Wdownᵀ", "mono")}
  ${t(465, 2008, "3  dgate = dhidden⊙up⊙SiLU′(gate)；dup = dhidden⊙SiLU(gate)", "mono")}
  ${t(465, 2040, "4  dWgate=Xᵀ·dgate；dWup=Xᵀ·dup；dXd sums gate/up paths", "mono")}
  ${t(1200, 1944, "5  router weights receive gradient through weighted combine", "mono")}
  ${t(1200, 1976, "6  dWr = X̃ᵀ·dR；router dX adds to expert dX", "mono")}
  ${t(1200, 2008, "7  residual gradient bypass + sublayer gradient are added", "mono")}
  ${t(1200, 2040, "8  unselected expert path has no activation gradient for this token", "mono")}

  <rect x="440" y="2105" width="1445" height="105" rx="14" class="state-box"/>
  ${t(465, 2140, "Per-layer accounting", "label")}
  ${t(675, 2140, "623.121M total params · 1.246 GB BF16 weights · attention 18.874M · router 0.262M · experts 603.980M", "mono")}
  ${t(465, 2174, "Per-token active", "label")}
  ${t(675, 2174, "56.890M params/layer = dense attention + router + 8 experts；capacity/state 仍按所有 128 experts。", "mono")}

  <!-- Backward rail -->
  ${t(1980, 455, "REVERSE-MODE BACKWARD", "label")}
  ${box(1975, 2018, 330, 116, "backward-box", "① CE backward", ["dZ = softmax(Z) - onehot(y)", "mask / n_valid scaling", "dZ logical [8192,151936]"], {bodyClass:"mono", lineHeight:22})}
  ${arrow("M2140 2018 L2140 1980", "bwd")}
  ${box(1975, 1860, 330, 118, "backward-box", "② LM head backward", ["dWlm = Hᵀ·dZ", "dH = dZ·Wlmᵀ", "grad Wlm mirrors 622.3 MB"], {bodyClass:"mono", lineHeight:22})}
  ${arrow("M2140 1860 L2140 1818", "bwd")}
  ${box(1975, 1660, 330, 158, "backward-box", "③ Layers 47 → 0", ["MoE backward first", "then attention backward", "residual gradients add", "checkpointed layer: recompute", "internal forward, then free"], {bodyClass:"small", lineHeight:23})}
  ${arrow("M2140 1660 L2140 1410", "bwd")}
  ${box(1975, 1218, 330, 192, "backward-box", "Layer ℓ gradient outputs", ["dWQ,dWK,dWV,dWO", "dWr", "dWgate/up/down for experts", "dγ norms", "dXℓ continues to layer ℓ-1"], {bodyClass:"mono", lineHeight:24})}
  ${arrow("M1975 1314 L1925 1314", "bwd")}
  ${arrow("M2140 1218 L2140 930", "bwd")}
  ${box(1975, 760, 330, 170, "backward-box", "④ Gradient accumulation", ["parameter.grad += microstep", "DP/FSDP/EP sync location", "unscale → clip", "parameters still unchanged"], {bodyClass:"small", lineHeight:24})}
  ${arrow("M2140 760 L2140 720", "bwd")}
  ${box(1975, 565, 330, 155, "state-box", "⑤ optimizer.step()", ["θ ← update(θ,g,m,v)", "Adam m/v persist", "scheduler.step · zero_grad", "only here parameters change"], {bodyClass:"mono", lineHeight:23})}
  ${arrow("M1975 642 C1945 642 1945 2190 1885 2190", "bwd")}

  ${t(75, 2290, "Main-body rule：forward 决定需要哪些 intermediates；backward 决定保存、重算和通信什么；optimizer.step 才改变 persistent parameters。", "label")}

  <rect x="45" y="2345" width="2310" height="675" rx="20" class="panel"/>
  ${t(75, 2390, "2 · 每层矩阵与临时张量总账：参数 shape ≠ activation shape ≠ gradient shape", "section")}
  ${matrixLedger.svg}
  ${activationLedger.svg}
  ${t(80, 2915, "Weight/parameter gradient 的元素数相同，但 dtype、sharding 和是否 materialize 由训练策略决定。", "small")}
  ${t(1220, 2915, "所有大小均为 decimal MB/GB；logical tensors 不能直接相加为 peak memory。", "small")}
  <rect x="80" y="2945" width="2240" height="48" rx="11" class="warning-box"/>
  ${t(100, 2976, "最容易犯的错：把 naive attention score、dispatch、gate/up/hidden、logits 全部相加；真实 peak 取决于 fusion、tiling、recompute、parallel local shape 与 lifetime overlap。", "small")}

  <rect x="45" y="3050" width="2310" height="575" rx="20" class="panel"/>
  ${t(75, 3095, "3 · Tensor lifetime：为什么 activation checkpointing 用计算换显存", "section")}
  ${t(75, 3125, "时间从左到右；绿色长期保存 layer boundary，橙色内部临时只在本层 forward/backward 短暂存在，红色 gradient 在反向逐层产生。", "small")}

  ${t(85, 3190, "FORWARD", "label")}
  ${arrow("M195 3182 L1690 3182", "fwd")}
  ${t(1710, 3190, "LOSS", "label")}
  ${arrow("M1810 3182 L2290 3182", "bwd")}
  ${t(2290, 3190, "BACKWARD", "label", "end")}

  <line x1="195" y1="3245" x2="2290" y2="3245" class="thin"/>
  ${t(85, 3260, "Weights", "label")}
  <rect x="195" y="3225" width="2095" height="40" rx="8" class="state-box"/>
  ${t(1245, 3252, "persistent across forward + backward + optimizer", "small", "middle")}

  ${t(85, 3330, "Boundary", "label")}
  ${[0,1,2,3,4,5].map((i) => {
    const x = 195 + i * 245;
    return `<rect x="${x}" y="3295" width="205" height="46" rx="8" class="saved-box"/>${t(x + 102, 3325, i === 5 ? "X48" : `X${i * 9}`, "mono", "middle")}`;
  }).join("\n")}
  ${t(1700, 3325, "48 × [8192,2048] = 1.611 GB BF16", "mono")}

  ${t(85, 3400, "Internals", "label")}
  <rect x="195" y="3365" width="235" height="46" rx="8" class="router-box"/>
  <rect x="455" y="3365" width="235" height="46" rx="8" class="router-box"/>
  <rect x="715" y="3365" width="235" height="46" rx="8" class="router-box"/>
  ${t(310, 3395, "Q/K/V · score tiles", "small", "middle")}
  ${t(572, 3395, "dispatch · gate/up", "small", "middle")}
  ${t(832, 3395, "logits / CE temp", "small", "middle")}
  ${t(1020, 3395, "freed after forward when checkpointed", "small")}
  <rect x="1800" y="3365" width="235" height="46" rx="8" class="router-box"/>
  <rect x="2055" y="3365" width="235" height="46" rx="8" class="router-box"/>
  ${t(1917, 3395, "recompute layer ℓ", "small", "middle")}
  ${t(2172, 3395, "then free again", "small", "middle")}

  ${t(85, 3470, "Gradients", "label")}
  <rect x="1680" y="3435" width="610" height="46" rx="8" class="backward-box"/>
  ${t(1985, 3465, "dactivation + parameter.grad produced right → left", "small", "middle")}

  <rect x="75" y="3520" width="720" height="75" rx="12" class="saved-box"/>
  ${t(100, 3551, "Checkpointed", "label")}
  ${t(260, 3551, "save Xℓ；backward 重新执行本层 forward", "small")}
  ${t(260, 3578, "降低 saved activations，增加 recompute FLOPs。", "small")}
  <rect x="835" y="3520" width="720" height="75" rx="12" class="router-box"/>
  ${t(860, 3551, "Transient", "label")}
  ${t(990, 3551, "FlashAttention tiles、grouped GEMM workspace、A2A buffers", "small")}
  ${t(990, 3578, "可能主导瞬时 peak，但不该跨 48 层长期保存。", "small")}
  <rect x="1595" y="3520" width="725" height="75" rx="12" class="state-box"/>
  ${t(1620, 3551, "Persistent", "label")}
  ${t(1760, 3551, "weights + grads + optimizer m/v + scheduler/RNG/state", "small")}
  ${t(1760, 3578, "MoE full-SFT 仍按 30.5B total，而非 3.3B active。", "small")}

  <rect x="45" y="3655" width="2310" height="485" rx="20" class="panel"/>
  ${t(75, 3700, "4 · 计算与反向的统一读法：每个矩阵乘法都同时产生 dX 与 dW", "section")}

  <rect x="75" y="3740" width="720" height="190" rx="14" class="attention-box"/>
  ${t(100, 3776, "通用 Linear：Y = XW", "subsection")}
  ${t(100, 3812, "forward     Y = XW        ≈ 2NDF FLOPs", "mono")}
  ${t(100, 3844, "backward dX = dY·Wᵀ      ≈ 2NDF FLOPs", "mono")}
  ${t(100, 3876, "backward dW = Xᵀ·dY      ≈ 2NDF FLOPs", "mono")}
  ${t(100, 3910, "training matmul ≈ 3× forward = 6NDF", "mono-strong")}

  <rect x="835" y="3740" width="720" height="190" rx="14" class="moe-box"/>
  ${t(860, 3776, "本例单层 logical matmul", "subsection")}
  ${t(860, 3812, `attention projections = ${(F.attnProj / 1e12).toFixed(3)} TFLOPs`, "mono")}
  ${t(860, 3844, `QKᵀ + AV (full matrix) = ${((F.qk + F.av) / 1e12).toFixed(3)} TFLOPs`, "mono")}
  ${t(860, 3876, `router + top-8 experts = ${((F.router + F.experts) / 1e12).toFixed(3)} TFLOPs`, "mono")}
  ${t(860, 3910, `layer forward ≈ ${(F.layerForward / 1e12).toFixed(3)}T；train matmul ≈ ${(3 * F.layerForward / 1e12).toFixed(3)}T`, "mono-strong")}

  <rect x="1595" y="3740" width="725" height="190" rx="14" class="backward-box"/>
  ${t(1620, 3776, "Backward 的执行顺序", "subsection")}
  ${t(1620, 3812, "loss → LM head → Layer 47 … Layer 0 → embedding", "mono")}
  ${t(1620, 3844, "每个 residual add：上游 gradient 分流，再在输入处相加", "small")}
  ${t(1620, 3876, "每个 checkpointed layer：先 recompute，再 local backward", "small")}
  ${t(1620, 3910, "grad sync/accumulation 后 optimizer.step 才更新 θ,m,v", "small")}

  <rect x="75" y="3960" width="2245" height="135" rx="14" class="warning-box"/>
  ${t(100, 3996, "证据边界 / 不可直接推断", "subsection")}
  ${t(100, 4030, "• FLOPs 为 logical dense-matmul accounting；causal kernel、fusion、recompute、padding 与 load imbalance 会改变实际工作。", "small")}
  ${t(100, 4058, "• 每卡 local tensor 还受 TP/EP/DP/SP、capacity factor 与 AllToAll layout 影响；此图不是单卡 peak-memory estimator。", "small")}
  ${t(100, 4086, "• 训练时没有 persistent inference KV cache，但 Q/K/V forward activations 仍可能保存或在 backward 重算。", "small")}

  ${t(55, 4170, "Sources · local Day 02 accounting/quiz · training-memory visual guide · Day 05 lifecycle roadmap · Qwen3-30B-A3B config snapshot", "tiny")}
</svg>
`;

const expected = {
  attentionParams: 18874368,
  oneExpertParams: 4718592,
  allExpertParams: 603979776,
  layerParams: 623120640,
  activeLayerParams: 56889600,
  residualElements: 16777216,
  routedAssignments: 65536,
};

const actual = {
  attentionParams: P.attention,
  oneExpertParams: P.oneExpert,
  allExpertParams: P.allExperts,
  layerParams: P.layer,
  activeLayerParams: P.activeLayer,
  residualElements: A.residual,
  routedAssignments: A.assignments,
};

for (const [key, value] of Object.entries(expected)) {
  if (actual[key] !== value) {
    throw new Error(`${key}: expected ${value}, got ${actual[key]}`);
  }
}

if (svg.includes("undefined")) throw new Error("SVG contains undefined");
fs.writeFileSync(OUTPUT, svg, "utf8");
console.log(`generated ${path.relative(path.resolve(__dirname, "..", ".."), OUTPUT)}`);
