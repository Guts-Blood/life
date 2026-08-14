#!/usr/bin/env python3
"""Build a Chinese reviewer UI without changing frozen blind-review cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
BOOTCAMP_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_INPUT = (
    BOOTCAMP_ROOT
    / "artifacts/data/day22-qwen35-formal-s1-blind-review.jsonl"
)
DEFAULT_OUTPUT = (
    BOOTCAMP_ROOT
    / "artifacts/data/day22-qwen35-formal-s1-blind-review-zh.html"
)

REQUIRED_KEYS = {
    "schema_name",
    "schema_version",
    "review_item_id",
    "prompt",
    "response_a",
    "response_b",
    "rubric_version",
    "allowed_verdicts",
    "verdict",
    "confidence",
    "notes",
}
VERDICTS = ["A", "B", "tie", "ambiguous", "reject"]
FROZEN_SOURCE_SHA256 = (
    "cae8b7ae7b52f27f2da9eaf614f536b396d91d04cc18777dd63d4296c3c2f45e"
)


class ChineseReviewBuildError(ValueError):
    """The frozen worksheet or requested output failed validation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ChineseReviewBuildError(f"cannot read worksheet: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ChineseReviewBuildError(f"blank row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ChineseReviewBuildError(
                f"invalid JSON at line {line_number}"
            ) from error
        if not isinstance(row, dict) or set(row) != REQUIRED_KEYS:
            raise ChineseReviewBuildError(f"row {line_number} fields drifted")
        if row.get("schema_name") != "day22.blind_review_item":
            raise ChineseReviewBuildError(f"row {line_number} schema drifted")
        if row.get("allowed_verdicts") != VERDICTS:
            raise ChineseReviewBuildError(f"row {line_number} verdicts drifted")
        if any(row.get(field) != "" for field in ("verdict", "confidence", "notes")):
            raise ChineseReviewBuildError(f"row {line_number} is not blank")
        if not all(
            isinstance(row.get(field), str) and row[field]
            for field in ("review_item_id", "prompt", "response_a", "response_b")
        ):
            raise ChineseReviewBuildError(f"row {line_number} case text is invalid")
        rows.append(row)
    if len(rows) != 100:
        raise ChineseReviewBuildError(f"expected 100 rows, got {len(rows)}")
    identities = [row["review_item_id"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ChineseReviewBuildError("duplicate review_item_id")
    return rows


def render_html(cases: Sequence[Mapping[str, Any]], source_sha256: str) -> str:
    embedded = json.dumps(
        list(cases), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <meta name="day22-source-sha256" content="{source_sha256}">
  <title>Day 22 代码偏好盲审</title>
  <style>
    :root {{ color-scheme: light dark; --accent:#2563eb; --muted:#64748b; --panel:#f8fafc; --border:#cbd5e1; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --accent:#60a5fa; --muted:#94a3b8; --panel:#111827; --border:#334155; }} }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:16px/1.55 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }}
    main {{ max-width:1100px; margin:auto; padding:24px; }}
    h1 {{ margin:0 0 8px; }}
    .notice,.panel {{ border:1px solid var(--border); border-radius:12px; padding:16px; background:var(--panel); }}
    .notice {{ margin:16px 0; }}
    .progress {{ display:flex; justify-content:space-between; gap:12px; margin:18px 0 8px; color:var(--muted); }}
    progress {{ width:100%; height:12px; }}
    h2 {{ font-size:18px; margin:22px 0 8px; }}
    pre {{ margin:0; padding:14px; overflow:auto; white-space:pre-wrap; border:1px solid var(--border); border-radius:9px; background:var(--panel); font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .answers {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    .choices {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; margin:18px 0; }}
    button,label.choice {{ border:1px solid var(--border); border-radius:9px; padding:10px 12px; background:var(--panel); cursor:pointer; text-align:center; }}
    label.choice:has(input:checked) {{ outline:2px solid var(--accent); color:var(--accent); font-weight:650; }}
    input[type=radio] {{ position:absolute; opacity:0; pointer-events:none; }}
    .confidence {{ display:flex; gap:16px; flex-wrap:wrap; margin:12px 0; }}
    textarea {{ width:100%; min-height:82px; padding:10px; border:1px solid var(--border); border-radius:9px; font:inherit; }}
    .nav {{ display:flex; justify-content:space-between; gap:10px; margin-top:18px; flex-wrap:wrap; }}
    button.primary {{ background:var(--accent); color:white; border-color:var(--accent); }}
    button:disabled {{ opacity:.45; cursor:not-allowed; }}
    .error {{ color:#dc2626; min-height:24px; margin-top:8px; }}
    .small {{ color:var(--muted); font-size:13px; }}
    @media (max-width:760px) {{ .answers {{ grid-template-columns:1fr; }} .choices {{ grid-template-columns:1fr 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Day 22 代码偏好盲审</h1>
  <p>页面说明和操作项已中文化；每个 case 的题目、回答 A、回答 B 保持冻结原文。</p>
  <section class="notice">
    <strong>盲审规则</strong>
    <ul>
      <li>共有 50 个独立 pair，每个以 A/B 和 B/A 各展示一次，因此需要完成 100 次判断。</li>
      <li>按页面顺序独立判断，不搜索另一条换序展示，不查看隐藏答案映射，不运行测试。</li>
      <li>优先判断功能正确性，其次判断是否遵守“只返回代码 continuation”等要求；不要因为长度或位置偏好某一边。</li>
      <li>“基本相同 / 无法判断 / 样本有问题”必须填写备注。</li>
    </ul>
  </section>
  <div class="progress"><span id="position"></span><span id="done"></span></div>
  <progress id="bar" max="100" value="0"></progress>
  <section class="panel">
    <h2>题目（原文）</h2><pre id="prompt"></pre>
    <div class="answers">
      <div><h2>回答 A（原文）</h2><pre id="response-a"></pre></div>
      <div><h2>回答 B（原文）</h2><pre id="response-b"></pre></div>
    </div>
    <h2>你的判断</h2>
    <div class="choices" id="verdicts">
      <label class="choice"><input type="radio" name="verdict" value="A">A 更优</label>
      <label class="choice"><input type="radio" name="verdict" value="B">B 更优</label>
      <label class="choice"><input type="radio" name="verdict" value="tie">基本相同</label>
      <label class="choice"><input type="radio" name="verdict" value="ambiguous">无法判断</label>
      <label class="choice"><input type="radio" name="verdict" value="reject">样本有问题</label>
    </div>
    <div class="confidence"><strong>置信度：</strong>
      <label><input type="radio" name="confidence" value="low"> 低</label>
      <label><input type="radio" name="confidence" value="medium"> 中</label>
      <label><input type="radio" name="confidence" value="high"> 高</label>
    </div>
    <label for="notes"><strong>备注</strong>（选择基本相同、无法判断或样本有问题时必填）</label>
    <textarea id="notes" placeholder="可简短说明判断依据"></textarea>
    <div class="error" id="error"></div>
    <div class="nav">
      <button id="previous">上一条</button>
      <button id="next" class="primary">保存并下一条</button>
      <button id="export-draft">导出草稿 JSONL</button>
      <button id="export-completed" class="primary" disabled>导出完成版 JSONL</button>
      <button id="clear">清空本页本地进度</button>
    </div>
    <p class="small" id="storage-note">回答自动保存在当前浏览器的 localStorage。导出不会修改冻结 worksheet；完成版按钮仅在 100 条全部有效填写后启用。</p>
  </section>
</main>
<script id="case-data" type="application/json">{embedded}</script>
<script>
(() => {{
  const cases = JSON.parse(document.getElementById('case-data').textContent);
  const storageKey = 'day22-blind-review-zh:{source_sha256}';
  const $ = id => document.getElementById(id);
  const loadAnswers = () => {{
    try {{ return JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }}
    catch (_) {{ $('storage-note').textContent='当前浏览器未开放 localStorage；请频繁导出草稿备份。'; return {{}}; }}
  }};
  let answers = loadAnswers();
  let index = 0;
  const needsNotes = new Set(['tie','ambiguous','reject']);
  const selected = name => document.querySelector(`input[name="${{name}}"]:checked`)?.value || '';
  const setSelected = (name, value) => document.querySelectorAll(`input[name="${{name}}"]`).forEach(x => x.checked = x.value === value);
  const currentId = () => cases[index].review_item_id;
  const save = validate => {{
    const verdict = selected('verdict');
    const confidence = selected('confidence');
    const notes = $('notes').value;
    if (validate && (!verdict || !confidence)) {{ $('error').textContent='请选择判断和置信度。'; return false; }}
    if (validate && needsNotes.has(verdict) && !notes.trim()) {{ $('error').textContent='该判断必须填写备注。'; return false; }}
    if (verdict || confidence || notes) answers[currentId()] = {{verdict,confidence,notes}};
    else delete answers[currentId()];
    try {{ localStorage.setItem(storageKey, JSON.stringify(answers)); }}
    catch (_) {{ $('storage-note').textContent='当前浏览器无法自动保存；请用“导出草稿 JSONL”备份。'; }}
    $('error').textContent=''; updateProgress(); return true;
  }};
  const completed = answer => answer && answer.verdict && answer.confidence && (!needsNotes.has(answer.verdict) || answer.notes.trim());
  const updateProgress = () => {{
    const count = cases.filter(row => completed(answers[row.review_item_id])).length;
    $('done').textContent = `已完成 ${{count}} / ${{cases.length}}`;
    $('bar').value = count;
    $('export-completed').disabled = count !== cases.length;
  }};
  const render = () => {{
    const row = cases[index], answer = answers[row.review_item_id] || {{}};
    $('position').textContent = `展示 ${{index + 1}} / ${{cases.length}}`;
    $('prompt').textContent = row.prompt;
    $('response-a').textContent = row.response_a;
    $('response-b').textContent = row.response_b;
    setSelected('verdict', answer.verdict || '');
    setSelected('confidence', answer.confidence || '');
    $('notes').value = answer.notes || '';
    $('previous').disabled = index === 0;
    $('next').textContent = index === cases.length - 1 ? '保存本条' : '保存并下一条';
    $('error').textContent=''; updateProgress(); window.scrollTo({{top:0,behavior:'smooth'}});
  }};
  $('previous').onclick = () => {{ save(false); if (index > 0) {{ index--; render(); }} }};
  $('next').onclick = () => {{ if (save(true) && index < cases.length - 1) {{ index++; render(); }} }};
  $('notes').oninput = () => save(false);
  document.querySelectorAll('input').forEach(input => input.onchange = () => save(false));
  const exportRows = mode => {{
    save(false);
    const output = cases.map(row => Object.assign({{}}, row, answers[row.review_item_id] || {{verdict:'',confidence:'',notes:''}}));
    const count = output.filter(completed).length;
    if (mode === 'completed' && count !== cases.length) {{
      $('error').textContent=`还有 ${{cases.length - count}} 条未完成，暂不能导出完成版。`;
      return;
    }}
    const payload = output.map(row => JSON.stringify(row)).join('\\n') + '\\n';
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([payload], {{type:'application/x-ndjson'}}));
    link.download = `day22-qwen35-formal-s1-blind-review-${{mode}}.jsonl`;
    link.click(); URL.revokeObjectURL(link.href);
  }};
  $('export-draft').onclick = () => exportRows('draft');
  $('export-completed').onclick = () => exportRows('completed');
  $('clear').onclick = () => {{
    if (confirm('确定清空这个中文页面保存的全部判断吗？此操作无法撤销。')) {{
      answers={{}};
      try {{ localStorage.removeItem(storageKey); }} catch (_) {{}}
      index=0; render();
    }}
  }};
  render();
}})();
</script>
</body>
</html>
"""


def write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise ChineseReviewBuildError(f"refusing to overwrite: {resolved}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = args.input.expanduser().resolve()
        cases = load_cases(source)
        source_sha = file_sha256(source)
        if source_sha != FROZEN_SOURCE_SHA256:
            raise ChineseReviewBuildError(
                f"frozen worksheet hash drifted: {source_sha}"
            )
        payload = render_html(cases, source_sha).encode("utf-8")
        write_atomic(args.output, payload, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "cases": len(cases),
                    "source_sha256": source_sha,
                    "output": str(args.output.expanduser().resolve()),
                    "output_sha256": hashlib.sha256(payload).hexdigest(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (ChineseReviewBuildError, OSError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
