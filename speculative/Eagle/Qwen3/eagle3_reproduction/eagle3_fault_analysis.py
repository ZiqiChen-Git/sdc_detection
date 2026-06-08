"""
Analyze EAGLE-3 fault-injection runner outputs.

The runner writes one baseline JSON per sample and one trial JSON per injected
fault. This analysis script compares each fault trial against the matching
baseline and writes:
  * analysis/analysis_summary.json
  * analysis/analysis_trials.jsonl
  * analysis/analysis_by_site.json
  * analysis/analysis_report.md
"""

import argparse
import json
import os
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def divergence_index(a: List[int], b: List[int]) -> Optional[int]:
    n = min(len(a), len(b))
    for idx in range(n):
        if a[idx] != b[idx]:
            return idx
    if len(a) != len(b):
        return n
    return None


def edit_distance(a: List[int], b: List[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, bv in enumerate(b, start=1):
            tmp = dp[j]
            cost = 0 if av == bv else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[-1]


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a or "", b or "").ratio()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def safe_rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def get_acceptance(entry: Dict[str, Any]) -> float:
    return as_float((entry.get("metrics") or {}).get("acceptance_rate"))


def get_trace_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    return entry.get("trace_summary") or {}


def get_execution_status(entry: Dict[str, Any]) -> str:
    return entry.get("execution_status") or "success"


def fault_site_key(fault_log: Dict[str, Any]) -> str:
    parts = [
        fault_log.get("location", "unknown"),
        fault_log.get("fault_type", "unknown"),
        fault_log.get("fault_mode", "unknown"),
    ]
    layer = fault_log.get("layer_idx", fault_log.get("tap_layer_idx"))
    if layer is not None:
        parts.append(f"layer={layer}")
    module = fault_log.get("module_path")
    if module:
        parts.append(f"module={module}")
    phase = fault_log.get("phase_filter")
    if phase:
        parts.append(f"phase={phase}")
    tap = fault_log.get("tap_slot")
    if tap:
        parts.append(f"tap={tap}")
    return "|".join(str(p) for p in parts)


def compare_trial(baseline: Dict[str, Any], trial: Dict[str, Any]) -> Dict[str, Any]:
    base_tokens = baseline.get("tokens") or []
    trial_tokens = trial.get("tokens") or []
    div_idx = divergence_index(base_tokens, trial_tokens)
    edit = edit_distance(base_tokens, trial_tokens)
    max_len = max(1, len(base_tokens), len(trial_tokens))
    base_text = baseline.get("prediction", baseline.get("text", ""))
    trial_text = trial.get("prediction", trial.get("text", ""))

    base_trace = get_trace_summary(baseline)
    trial_trace = get_trace_summary(trial)
    base_accept = get_acceptance(baseline)
    trial_accept = get_acceptance(trial)

    output_changed = trial_text != base_text
    token_changed = base_tokens != trial_tokens
    baseline_correct_then_wrong = bool(baseline.get("is_correct")) and not bool(trial.get("is_correct"))
    fault_log = trial.get("fault_log") or {}
    baseline_status = get_execution_status(baseline)
    fault_status = get_execution_status(trial)

    return {
        "sample_id": trial.get("sample_id"),
        "trial_idx": trial.get("trial_idx"),
        "fault_site_key": fault_site_key(fault_log),
        "fault_log": fault_log,
        "output_changed": output_changed,
        "token_changed": token_changed,
        "baseline_correct_then_wrong": baseline_correct_then_wrong,
        "is_correct": bool(trial.get("is_correct")),
        "baseline_execution_status": baseline_status,
        "fault_execution_status": fault_status,
        "execution_status_changed": baseline_status != fault_status,
        "baseline_success_then_fault_error": baseline_status == "success" and fault_status != "success",
        "fault_error_type": trial.get("error_type"),
        "fault_error_message": trial.get("error_message"),
        "divergence_index": div_idx,
        "token_edit_distance": edit,
        "token_edit_distance_norm": edit / max_len,
        "text_similarity": text_similarity(base_text, trial_text),
        "baseline_acceptance_rate": base_accept,
        "fault_acceptance_rate": trial_accept,
        "acceptance_delta": trial_accept - base_accept,
        "baseline_tokens_emitted": (baseline.get("metrics") or {}).get("tokens_emitted"),
        "fault_tokens_emitted": (trial.get("metrics") or {}).get("tokens_emitted"),
        "baseline_mean_verify_kl": as_float(base_trace.get("mean_verify_kl")),
        "fault_mean_verify_kl": as_float(trial_trace.get("mean_verify_kl")),
        "mean_verify_kl_delta": as_float(trial_trace.get("mean_verify_kl")) - as_float(base_trace.get("mean_verify_kl")),
        "baseline_max_verify_kl": as_float(base_trace.get("max_verify_kl")),
        "fault_max_verify_kl": as_float(trial_trace.get("max_verify_kl")),
        "max_verify_kl_delta": as_float(trial_trace.get("max_verify_kl")) - as_float(base_trace.get("max_verify_kl")),
        "baseline_num_blocks_with_reject": int(base_trace.get("num_blocks_with_reject") or 0),
        "fault_num_blocks_with_reject": int(trial_trace.get("num_blocks_with_reject") or 0),
        "runtime_triggered": (fault_log.get("runtime") or {}).get("triggered"),
    }


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    return {
        "total_trials": total,
        "output_changed": sum(1 for r in records if r["output_changed"]),
        "output_changed_rate": safe_rate(sum(1 for r in records if r["output_changed"]), total),
        "token_changed": sum(1 for r in records if r["token_changed"]),
        "token_changed_rate": safe_rate(sum(1 for r in records if r["token_changed"]), total),
        "baseline_correct_then_wrong": sum(1 for r in records if r["baseline_correct_then_wrong"]),
        "baseline_correct_then_wrong_rate": safe_rate(sum(1 for r in records if r["baseline_correct_then_wrong"]), total),
        "execution_status_changed": sum(1 for r in records if r["execution_status_changed"]),
        "execution_status_changed_rate": safe_rate(sum(1 for r in records if r["execution_status_changed"]), total),
        "baseline_success_then_fault_error": sum(1 for r in records if r["baseline_success_then_fault_error"]),
        "baseline_success_then_fault_error_rate": safe_rate(sum(1 for r in records if r["baseline_success_then_fault_error"]), total),
        "fault_correct": sum(1 for r in records if r["is_correct"]),
        "fault_accuracy": safe_rate(sum(1 for r in records if r["is_correct"]), total),
        "mean_text_similarity": mean(r["text_similarity"] for r in records),
        "mean_token_edit_distance_norm": mean(r["token_edit_distance_norm"] for r in records),
        "mean_acceptance_delta": mean(r["acceptance_delta"] for r in records),
        "mean_fault_acceptance_rate": mean(r["fault_acceptance_rate"] for r in records),
        "mean_verify_kl_delta": mean(r["mean_verify_kl_delta"] for r in records),
        "max_verify_kl_delta": max((r["max_verify_kl_delta"] for r in records), default=0.0),
    }


def group_by_site(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["fault_site_key"]].append(record)
    return {
        key: summarize_records(vals)
        for key, vals in sorted(grouped.items(), key=lambda item: item[0])
    }


def load_run_records(run_dir: str) -> Dict[str, Any]:
    baseline_dir = os.path.join(run_dir, "baselines")
    trial_dir = os.path.join(run_dir, "trials")
    if not os.path.isdir(baseline_dir):
        raise FileNotFoundError(f"Missing baseline directory: {baseline_dir}")
    if not os.path.isdir(trial_dir):
        raise FileNotFoundError(f"Missing trial directory: {trial_dir}")

    baselines: Dict[Any, Dict[str, Any]] = {}
    for filename in sorted(os.listdir(baseline_dir)):
        if not filename.endswith(".json"):
            continue
        entry = load_json(os.path.join(baseline_dir, filename))
        baselines[entry.get("sample_id")] = entry

    trials = []
    for filename in sorted(os.listdir(trial_dir)):
        if not filename.endswith(".json"):
            continue
        trials.append(load_json(os.path.join(trial_dir, filename)))

    return {"baselines": baselines, "trials": trials}


def write_report(path: str, summary: Dict[str, Any], by_site: Dict[str, Any]) -> None:
    lines = [
        "# EAGLE-3 Fault Injection Analysis",
        "",
        "## Overall",
        "",
        f"- Total trials: {summary['total_trials']}",
        f"- Output changed rate: {summary['output_changed_rate']:.4f}",
        f"- Token changed rate: {summary['token_changed_rate']:.4f}",
        f"- Baseline-correct-then-wrong rate: {summary['baseline_correct_then_wrong_rate']:.4f}",
        f"- Execution status changed rate: {summary['execution_status_changed_rate']:.4f}",
        f"- Baseline-success-then-fault-error rate: {summary['baseline_success_then_fault_error_rate']:.4f}",
        f"- Fault accuracy: {summary['fault_accuracy']:.4f}",
        f"- Mean fault acceptance rate: {summary['mean_fault_acceptance_rate']:.4f}",
        f"- Mean acceptance delta: {summary['mean_acceptance_delta']:.4f}",
        f"- Mean verify KL delta: {summary['mean_verify_kl_delta']:.4f}",
        "",
        "## By Fault Site",
        "",
    ]
    for key, vals in by_site.items():
        lines.extend([
            f"### {key}",
            "",
            f"- Trials: {vals['total_trials']}",
            f"- Output changed rate: {vals['output_changed_rate']:.4f}",
            f"- Token changed rate: {vals['token_changed_rate']:.4f}",
            f"- Execution status changed rate: {vals['execution_status_changed_rate']:.4f}",
            f"- Baseline-success-then-fault-error rate: {vals['baseline_success_then_fault_error_rate']:.4f}",
            f"- Fault accuracy: {vals['fault_accuracy']:.4f}",
            f"- Mean acceptance delta: {vals['mean_acceptance_delta']:.4f}",
            f"- Mean verify KL delta: {vals['mean_verify_kl_delta']:.4f}",
            "",
        ])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def analyze_run(run_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    output_dir = output_dir or os.path.join(run_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    loaded = load_run_records(run_dir)
    baselines = loaded["baselines"]
    trial_entries = loaded["trials"]

    records = []
    for trial in trial_entries:
        baseline = baselines.get(trial.get("sample_id"))
        if baseline is None:
            continue
        records.append(compare_trial(baseline, trial))

    summary = summarize_records(records)
    by_site = group_by_site(records)

    summary_path = os.path.join(output_dir, "analysis_summary.json")
    trials_path = os.path.join(output_dir, "analysis_trials.jsonl")
    by_site_path = os.path.join(output_dir, "analysis_by_site.json")
    report_path = os.path.join(output_dir, "analysis_report.md")

    write_json(summary_path, summary)
    write_json(by_site_path, by_site)
    with open(trials_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_report(report_path, summary, by_site)

    return {
        "summary": summary,
        "by_site": by_site,
        "paths": {
            "summary": summary_path,
            "trials": trials_path,
            "by_site": by_site_path,
            "report": report_path,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze EAGLE-3 fault runner outputs.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    result = analyze_run(args.run_dir, args.output_dir)
    print(f"Analysis summary -> {result['paths']['summary']}")
    print(f"Analysis report  -> {result['paths']['report']}")


if __name__ == "__main__":
    main()
