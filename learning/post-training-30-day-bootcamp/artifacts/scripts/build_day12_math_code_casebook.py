#!/usr/bin/env python3
"""Build a readable Day 12 Math/Code dev-eval casebook from frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARTIFACTS = Path(__file__).resolve().parents[1]
EVAL_DIR = ARTIFACTS / "eval"
REPORT_PATH = ARTIFACTS / "reports" / "day12-math-code-casebook.md"
MODEL_IDS = ["Base", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prediction_path(model: str) -> Path:
    if model == "Base":
        return EVAL_DIR / "day10-qwen3-0.6b-base-predictions.jsonl"
    return EVAL_DIR / f"{model}-25_percent-dev-predictions.jsonl"


def code_result_path(model: str) -> Path:
    if model == "Base":
        return EVAL_DIR / "day10-qwen3-0.6b-base-code-sandbox-e2b.jsonl"
    return EVAL_DIR / f"{model}-25_percent-dev-code-e2b.jsonl"


def math_correct(row: dict[str, Any]) -> bool:
    score = row.get("score")
    if score is None:
        score = row.get("scorer_result", {}).get("score")
    return score == 1 or score == 1.0


def code_correct(row: dict[str, Any]) -> bool:
    return row.get("passed") is True


def short_id(sample_id: str) -> str:
    if "HumanEval/" in sample_id:
        return sample_id.rsplit(":", 1)[-1]
    return sample_id.rsplit(":", 1)[-1]


def fenced(value: Any, language: str = "") -> str:
    text = "" if value is None else str(value).rstrip()
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    fence = "`" * max(4, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def parsed_answer(row: dict[str, Any]) -> Any:
    return row.get("scorer_result", {}).get("parsed_answer")


def score_icon(value: bool) -> str:
    return "✅" if value else "❌"


def main() -> None:
    manifest = json.loads((EVAL_DIR / "day10-frozen-eval-manifest.json").read_text())
    records = {
        row["sample_id"]: row
        for row in manifest["records"]
        if row["evaluation_split"] == "dev" and row["slice"] in {"math", "code"}
    }
    predictions = {
        model: {row["sample_id"]: row for row in read_jsonl(prediction_path(model))}
        for model in MODEL_IDS
    }
    code_results = {
        model: {row["sample_id"]: row for row in read_jsonl(code_result_path(model))}
        for model in MODEL_IDS
    }
    summaries = {
        "Base": json.loads(
            (ARTIFACTS / "reports" / "day10-base-baseline-summary.json").read_text()
        ),
        **{
            model: json.loads(
                (EVAL_DIR / f"{model}-25_percent-dev-summary.json").read_text()
            )
            for model in MODEL_IDS
            if model != "Base"
        },
    }

    math_ids = sorted(sample_id for sample_id, row in records.items() if row["slice"] == "math")
    code_ids = sorted(sample_id for sample_id, row in records.items() if row["slice"] == "code")
    correct: dict[str, dict[str, set[str]]] = {"math": {}, "code": {}}
    for model in MODEL_IDS:
        correct["math"][model] = {
            sample_id for sample_id in math_ids if math_correct(predictions[model][sample_id])
        }
        correct["code"][model] = {
            sample_id for sample_id in code_ids if code_correct(code_results[model][sample_id])
        }

    base_math = correct["math"]["Base"]
    e_math = correct["math"]["E"]
    h_math = correct["math"]["H"]
    e_code = correct["code"]["E"]
    h_code = correct["code"]["H"]

    lines: list[str] = [
        "# Day 12 Math / Code Dev-Eval Casebook",
        "",
        "> 范围：frozen dev 的 28 个 GSM8K Math case + 28 个 HumanEval Code case。",
        "> 对比：Base 与 recovery C–L；逐题展开重点展示 Base、E、H 的原始输出。",
        "> A/B 的逐题输出已不在当前仓库，仅保留 aggregate outcome，因此不伪造其 case-level 对比。",
        "",
        "## 先看结论",
        "",
        f"- Math：Base `{len(base_math)}/28`，E `{len(e_math)}/28`，H `{len(h_math)}/28`。",
        f"- E 只保留 Base 正确集中的 `{len(base_math & e_math)}/12`，同时新增 `{len(e_math - base_math)}` 个 Base 未答对的 case。",
        f"- H 保留 Base 正确集中的 `{len(base_math & h_math)}/12`，同时新增 `{len(h_math - base_math)}` 个 Base 未答对的 case。",
        f"- Code：Base `0/28`，E `{len(e_code)}/28`，H `{len(h_code)}/28`；H 的正确集完全是 E 正确集的子集。",
        "- 终止行为是隐藏的关键变量：Base/H 的 Math 和 Code 都是 `28/28` ceiling hits；E 分别只有 `3/28` 和 `1/28`。",
        "- H 的部分 Code pass 来自 extractor 截取到可执行片段，raw generation 本身仍继续到 512 tokens；复盘时必须同时看 parsed completion 与 raw output。",
        "- 这不是简单的整体能力平移，而是明显的 case redistribution：某些能力被学会，同时另一批原有能力被覆盖。",
        "",
        "### 建议一起复盘的第一批 case",
        "",
        "- Math `0028`：Base/H 对、E 错——典型能力遗忘。",
        "- Math `0053`：Base/E 对、H 错——说明 E 并非所有保留能力都差于 H。",
        "- Math `0327`：只有 E 对——观察 E 新学会了什么模式。",
        "- Code `HumanEval/8`、`/60`、`/122`：E/H 都通过——稳定获得的代码能力。",
        "- Code `HumanEval/48`、`/50`、`/52`、`/97`、`/107`、`/147`：只有 E 通过——E 相对 H 的增量来源。",
        "",
        "## Per-model 正确数",
        "",
        "| Model | Math /28 | Code /28 |",
        "|---|---:|---:|",
    ]
    for model in MODEL_IDS:
        lines.append(
            f"| {model} | {len(correct['math'][model])} | {len(correct['code'][model])} |"
        )

    lines.extend(
        [
            "",
            "## 终止 / ceiling-hit 审计",
            "",
            "| Model | Math ceiling /28 | Code ceiling /28 | Math output tokens | Code output tokens |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in MODEL_IDS:
        math_summary = summaries[model]["per_slice"]["math"]
        code_summary = summaries[model]["per_slice"]["code"]
        lines.append(
            f"| {model} | {math_summary['ceiling_hits']} | {code_summary['ceiling_hits']} | "
            f"{math_summary['output_tokens']} | {code_summary['output_tokens']} |"
        )

    lines.extend(
        [
            "",
            "## Math case matrix",
            "",
            "| Case | " + " | ".join(MODEL_IDS) + " |",
            "|---|" + "---:|" * len(MODEL_IDS),
        ]
    )
    for sample_id in math_ids:
        status = [score_icon(sample_id in correct["math"][model]) for model in MODEL_IDS]
        lines.append(f"| `{short_id(sample_id)}` | " + " | ".join(status) + " |")

    lines.extend(
        [
            "",
            "## Code case matrix",
            "",
            "| Case | " + " | ".join(MODEL_IDS) + " |",
            "|---|" + "---:|" * len(MODEL_IDS),
        ]
    )
    for sample_id in code_ids:
        status = [score_icon(sample_id in correct["code"][model]) for model in MODEL_IDS]
        lines.append(f"| `{short_id(sample_id)}` | " + " | ".join(status) + " |")

    lines.extend(["", "## Math cases：题目、参考答案与 Base/E/H 输出", ""])
    for sample_id in math_ids:
        case = records[sample_id]
        statuses = " · ".join(
            f"{model} {score_icon(sample_id in correct['math'][model])}"
            for model in ("Base", "E", "H")
        )
        lines.extend(
            [
                f"### Math {short_id(sample_id)}",
                "",
                statuses,
                "",
                "**Question**",
                "",
                case["raw_prompt"],
                "",
                "**Reference**",
                "",
                fenced(case["reference"]),
                "",
            ]
        )
        for model in ("Base", "E", "H"):
            row = predictions[model][sample_id]
            lines.extend(
                [
                    f"<details><summary>{model} {score_icon(sample_id in correct['math'][model])} · parsed={parsed_answer(row)!r}</summary>",
                    "",
                    fenced(row.get("raw_output")),
                    "",
                    "</details>",
                    "",
                ]
            )

    lines.extend(["", "## Code cases：函数、参考实现与 Base/E/H 输出", ""])
    for sample_id in code_ids:
        case = records[sample_id]
        statuses = " · ".join(
            f"{model} {score_icon(sample_id in correct['code'][model])}"
            for model in ("Base", "E", "H")
        )
        lines.extend(
            [
                f"### {short_id(sample_id)}",
                "",
                statuses,
                "",
                f"Entry point: `{case['metadata']['entry_point']}` · Test hash: `{case['metadata']['test_sha256']}`",
                "",
                "**Function prompt**",
                "",
                fenced(case["raw_prompt"], "python"),
                "",
                "**Reference completion**",
                "",
                fenced(case["reference"], "python"),
                "",
            ]
        )
        for model in ("Base", "E", "H"):
            pred = predictions[model][sample_id]
            result = code_results[model][sample_id]
            passed = sample_id in correct["code"][model]
            error = result.get("error_type") or result.get("execution_status") or "none"
            parsed = pred.get("scorer_result", {}).get("parsed_answer")
            raw = pred.get("raw_output")
            lines.extend(
                [
                    f"<details><summary>{model} {score_icon(passed)} · E2B={error}</summary>",
                    "",
                    "**Parsed completion executed in E2B**",
                    "",
                    fenced(parsed, "python"),
                    "",
                ]
            )
            if str(raw).strip() != str(parsed).strip():
                lines.extend(
                    [
                        "**Raw model output（extractor 前）**",
                        "",
                        fenced(raw, "python"),
                        "",
                    ]
                )
            lines.extend(["</details>", ""])

    lines.extend(
        [
            "## Evidence boundary",
            "",
            "- 本 casebook 只使用 dev；frozen test 未读取、未生成、未评分。",
            "- Math 是 numeric exact match；Code 是隔离 E2B sandbox 的 pass@1。",
            "- 28 题/切片是受控 pilot，不支持泛化为总体 benchmark 结论。",
            "- HumanEval 隐藏测试正文不在 frozen manifest 中，仅记录 test hash；此处不反推或伪造测试。",
            "",
            "## Source files",
            "",
            "- `artifacts/eval/day10-frozen-eval-manifest.json`",
            "- `artifacts/eval/day10-qwen3-0.6b-base-predictions.jsonl`",
            "- `artifacts/eval/day10-qwen3-0.6b-base-code-sandbox-e2b.jsonl`",
            "- `artifacts/eval/{C-L}-25_percent-dev-predictions.jsonl`",
            "- `artifacts/eval/{C-L}-25_percent-dev-code-e2b.jsonl`",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
