"""
Create a short report for the preliminary FI TODO:

1. SDC effect on reasoning output.
2. SDC effect on execution status.

Input is a run directory produced by eagle3_fault_runner.py.
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def top_examples(records: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    changed = [r for r in records if r.get("output_changed") or r.get("baseline_success_then_fault_error")]
    changed.sort(
        key=lambda r: (
            bool(r.get("baseline_success_then_fault_error")),
            float(r.get("token_edit_distance_norm") or 0.0),
            abs(float(r.get("acceptance_delta") or 0.0)),
        ),
        reverse=True,
    )
    return changed[:limit]


def build_report(run_dir: str) -> str:
    analysis_dir = os.path.join(run_dir, "analysis")
    summary_path = os.path.join(analysis_dir, "analysis_summary.json")
    trials_path = os.path.join(analysis_dir, "analysis_trials.jsonl")
    if not os.path.exists(summary_path) or not os.path.exists(trials_path):
        raise FileNotFoundError(
            "Missing analysis files. Run eagle3_fault_analysis.py first or run "
            "eagle3_fault_runner.py without --no_auto_analyze."
        )

    summary = load_json(summary_path)
    records = load_jsonl(trials_path)
    total = int(summary.get("total_trials") or len(records))

    status_counts = Counter(r.get("fault_execution_status", "success") for r in records)
    error_types = Counter(
        r.get("fault_error_type") or "none"
        for r in records
        if r.get("fault_execution_status", "success") != "success"
    )
    examples = top_examples(records)

    lines = [
        "# Preliminary FI TODO Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Total fault trials: {total}",
        "",
        "## TODO 1: SDC Effect on Reasoning Output",
        "",
        f"- Output changed rate: {pct(float(summary.get('output_changed_rate') or 0.0))}",
        f"- Token changed rate: {pct(float(summary.get('token_changed_rate') or 0.0))}",
        f"- Baseline-correct-then-wrong rate: {pct(float(summary.get('baseline_correct_then_wrong_rate') or 0.0))}",
        f"- Mean token edit distance norm: {float(summary.get('mean_token_edit_distance_norm') or 0.0):.4f}",
        f"- Mean text similarity: {float(summary.get('mean_text_similarity') or 0.0):.4f}",
        f"- Mean acceptance delta: {float(summary.get('mean_acceptance_delta') or 0.0):.4f}",
        f"- Mean verify KL delta: {float(summary.get('mean_verify_kl_delta') or 0.0):.4f}",
        "",
        "Interpretation:",
        "",
        "- `output_changed/token_changed` shows whether FI changes the generated reasoning/output.",
        "- `baseline-correct-then-wrong` is the strongest preliminary SDC signal for task outcome.",
        "- `acceptance_delta` and `verify KL delta` connect the output effect to speculative-decoding traces.",
        "",
        "## TODO 2: SDC Effect on Execution Status",
        "",
        f"- Execution status changed rate: {pct(float(summary.get('execution_status_changed_rate') or 0.0))}",
        f"- Baseline-success-then-fault-error rate: {pct(float(summary.get('baseline_success_then_fault_error_rate') or 0.0))}",
        f"- Fault execution status counts: `{dict(status_counts)}`",
        f"- Fault error type counts: `{dict(error_types)}`",
        "",
        "Interpretation:",
        "",
        "- `baseline-success-then-fault-error` directly addresses whether FI changes execution status.",
        "- Error type counts separate runtime crash, CUDA error, indexing error, timeout-like failures, and clean success.",
        "",
        "## Representative Changed Trials",
        "",
    ]

    if not examples:
        lines.append("- No changed trials found in this preliminary run.")
    else:
        for item in examples:
            lines.extend(
                [
                    (
                        f"- sample={item.get('sample_id')} trial={item.get('trial_idx')} "
                        f"site=`{item.get('fault_site_key')}`"
                    ),
                    (
                        f"  output_changed={item.get('output_changed')} "
                        f"token_changed={item.get('token_changed')} "
                        f"status={item.get('fault_execution_status')} "
                        f"correct_then_wrong={item.get('baseline_correct_then_wrong')}"
                    ),
                    (
                        f"  edit_norm={float(item.get('token_edit_distance_norm') or 0.0):.4f} "
                        f"accept_delta={float(item.get('acceptance_delta') or 0.0):.4f} "
                        f"kl_delta={float(item.get('mean_verify_kl_delta') or 0.0):.4f}"
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Run the same command with more samples/trials after this pilot confirms nonzero output or status effects.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate preliminary FI TODO report.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    report = build_report(args.run_dir)
    output_md = args.output_md or os.path.join(args.run_dir, "analysis", "preliminary_todo_report.md")
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Preliminary TODO report -> {output_md}")


if __name__ == "__main__":
    main()
