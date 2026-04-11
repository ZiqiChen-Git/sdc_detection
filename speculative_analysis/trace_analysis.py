import argparse
import json
import math
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_distance(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return 1.0 - dot / (norm_a * norm_b)


def l2_distance(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def js_divergence(p: Dict[int, float], q: Dict[int, float]) -> Optional[float]:
    if not p or not q:
        return None
    keys = set(p.keys()) | set(q.keys())
    p_vec = [p.get(k, 0.0) for k in keys]
    q_vec = [q.get(k, 0.0) for k in keys]
    p_sum = sum(p_vec)
    q_sum = sum(q_vec)
    if p_sum == 0.0 or q_sum == 0.0:
        return None
    p_vec = [v / p_sum for v in p_vec]
    q_vec = [v / q_sum for v in q_vec]
    m_vec = [(pv + qv) / 2.0 for pv, qv in zip(p_vec, q_vec)]

    def kl_divergence(a_vec, b_vec) -> float:
        total = 0.0
        for av, bv in zip(a_vec, b_vec):
            if av == 0.0 or bv == 0.0:
                continue
            total += av * math.log2(av / bv)
        return total

    return 0.5 * (kl_divergence(p_vec, m_vec) + kl_divergence(q_vec, m_vec))


def topk_to_map(topk: List[Dict[str, Any]]) -> Dict[int, float]:
    mapping = {}
    for entry in topk:
        token_id = entry.get("token_id")
        prob = entry.get("prob")
        if token_id is None or prob is None:
            continue
        mapping[int(token_id)] = float(prob)
    return mapping


def divergence_index(a: List[int], b: List[int]) -> Optional[int]:
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return min_len
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
            temp = dp[j]
            cost = 0 if av == bv else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[-1]


def sequence_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def extract_trial_info(filename: str) -> Optional[Tuple[int, int]]:
    match = re.match(r"trial(\d+)_sample_(\d+)\.json$", filename)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def extract_baseline_info(filename: str) -> Optional[int]:
    match = re.match(r"sample_(\d+)\.json$", filename)
    if not match:
        return None
    return int(match.group(1))


def analyze_pair(
    baseline: Dict[str, Any],
    trial: Dict[str, Any],
    embedder,
) -> Dict[str, Any]:
    base_tokens = baseline.get("tokens", [])
    trial_tokens = trial.get("tokens", [])
    div_idx = divergence_index(base_tokens, trial_tokens)
    prefix_len = div_idx if div_idx is not None else min(len(base_tokens), len(trial_tokens))
    edit_dist = edit_distance(base_tokens, trial_tokens)
    edit_norm = edit_dist / max(1, max(len(base_tokens), len(trial_tokens)))
    text_sim = sequence_similarity(baseline.get("text", ""), trial.get("text", ""))

    base_trace = baseline.get("trace", [])
    trial_trace = trial.get("trace", [])
    trace_len = min(len(base_trace), len(trial_trace))
    hidden_cos = []
    hidden_l2 = []
    js_scores = []
    base_prob_diffs = []
    hidden_cos_by_step = []
    js_by_step = []
    base_prob_by_step = []
    acceptance_mismatch = 0
    acceptance_total = 0

    for idx in range(trace_len):
        b_step = base_trace[idx]
        t_step = trial_trace[idx]
        b_hidden = b_step.get("hidden_state_slice")
        t_hidden = t_step.get("hidden_state_slice")
        cos = cosine_distance(b_hidden, t_hidden) if b_hidden and t_hidden else None
        if cos is not None:
            hidden_cos.append(cos)
            hidden_cos_by_step.append(cos)
        l2 = l2_distance(b_hidden, t_hidden) if b_hidden and t_hidden else None
        if l2 is not None:
            hidden_l2.append(l2)

        b_topk = topk_to_map(b_step.get("base_topk", []))
        t_topk = topk_to_map(t_step.get("base_topk", []))
        js = js_divergence(b_topk, t_topk) if b_topk and t_topk else None
        if js is not None:
            js_scores.append(js)
            js_by_step.append(js)

        if "base_prob" in b_step and "base_prob" in t_step:
            diff = abs(float(b_step["base_prob"]) - float(t_step["base_prob"]))
            base_prob_diffs.append(diff)
            base_prob_by_step.append(diff)

        if "accepted" in b_step and "accepted" in t_step:
            acceptance_total += 1
            if bool(b_step["accepted"]) != bool(t_step["accepted"]):
                acceptance_mismatch += 1

    def summarize(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "max": 0.0}
        return {"mean": float(sum(values) / len(values)), "max": float(max(values))}

    def before_after(values: List[float], split: Optional[int]) -> Dict[str, float]:
        if split is None or split <= 0:
            return {"before_mean": 0.0, "after_mean": float(sum(values) / len(values)) if values else 0.0}
        before = values[:split]
        after = values[split:]
        return {
            "before_mean": float(sum(before) / len(before)) if before else 0.0,
            "after_mean": float(sum(after) / len(after)) if after else 0.0,
        }

    summary = {
        "divergence_index": div_idx,
        "prefix_match_len": prefix_len,
        "token_edit_distance": edit_dist,
        "token_edit_distance_norm": edit_norm,
        "text_similarity": text_sim,
        "text_embedding_similarity": None,
        "hidden_cosine": summarize(hidden_cos),
        "hidden_l2": summarize(hidden_l2),
        "base_topk_js": summarize(js_scores),
        "base_prob_diff": summarize(base_prob_diffs),
        "acceptance_mismatch_rate": acceptance_mismatch / acceptance_total if acceptance_total else 0.0,
        "hidden_cosine_before_after": before_after(hidden_cos_by_step, div_idx),
        "base_topk_js_before_after": before_after(js_by_step, div_idx),
        "base_prob_diff_before_after": before_after(base_prob_by_step, div_idx),
        "sdc_score": 0.0,
    }
    if embedder is not None:
        try:
            emb_base = embedder.encode(baseline.get("text", ""), normalize_embeddings=True)
            emb_trial = embedder.encode(trial.get("text", ""), normalize_embeddings=True)
            sim = float(sum(a * b for a, b in zip(emb_base, emb_trial)))
            summary["text_embedding_similarity"] = sim
        except Exception:
            summary["text_embedding_similarity"] = None

    divergence_ratio = 1.0 - (prefix_len / max(1, max(len(base_tokens), len(trial_tokens))))
    hidden_l2_mean = summary["hidden_l2"]["mean"]
    emb_sim = summary.get("text_embedding_similarity")
    components = [
        ("divergence_ratio", divergence_ratio, 0.2),
        ("edit_norm", edit_norm, 0.2),
        ("text_sim", 1.0 - text_sim, 0.1),
        ("hidden_cos", summary["hidden_cosine"]["mean"], 0.15),
        ("hidden_l2", hidden_l2_mean / (hidden_l2_mean + 1.0), 0.05),
        ("base_topk_js", summary["base_topk_js"]["mean"], 0.15),
        ("base_prob_diff", summary["base_prob_diff"]["mean"], 0.1),
        ("acceptance_mismatch", summary["acceptance_mismatch_rate"], 0.1),
    ]
    if emb_sim is not None:
        components.append(("embedding_sim", 1.0 - emb_sim, 0.05))
    total_weight = sum(w for _, _, w in components)
    score = sum(value * (weight / total_weight) for _, value, weight in components)
    summary["sdc_score"] = float(score)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze speculative decoding traces.")
    parser.add_argument("--outputs_dir", type=str, default="speculative_outputs")
    parser.add_argument("--baseline_run", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="speculative_analysis_outputs")
    parser.add_argument("--use_embeddings", action="store_true")
    args = parser.parse_args()

    traces_dir = os.path.join(args.outputs_dir, "traces")
    baseline_dir = os.path.join(traces_dir, f"baseline_run_{args.baseline_run}")
    if not os.path.isdir(baseline_dir):
        raise FileNotFoundError(f"Missing baseline trace directory: {baseline_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "trace_analysis_summary.json")
    per_sample_path = os.path.join(args.output_dir, "trace_analysis_samples.jsonl")

    baseline_map: Dict[int, Dict[str, Any]] = {}
    for filename in os.listdir(baseline_dir):
        sample_id = extract_baseline_info(filename)
        if sample_id is None:
            continue
        baseline_map[sample_id] = load_json(os.path.join(baseline_dir, filename))

    embedder = None
    if args.use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            embedder = None

    trial_results: Dict[int, List[Dict[str, Any]]] = {}
    for filename in os.listdir(traces_dir):
        info = extract_trial_info(filename)
        if not info:
            continue
        trial_id, sample_id = info
        baseline = baseline_map.get(sample_id)
        if baseline is None:
            continue
        trial_data = load_json(os.path.join(traces_dir, filename))
        analysis = analyze_pair(baseline, trial_data, embedder)
        record = {
            "trial": trial_id,
            "sample_id": sample_id,
            "analysis": analysis,
        }
        trial_results.setdefault(trial_id, []).append(record)

    summary = {}
    with open(per_sample_path, "w", encoding="utf-8") as fp:
        for trial_id, records in trial_results.items():
            for record in records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics = [r["analysis"] for r in records]
            if metrics:
                summary[trial_id] = {
                    "mean_divergence_index": float(
                        sum(m["divergence_index"] or 0 for m in metrics) / len(metrics)
                    ),
                    "mean_token_edit_distance_norm": float(
                        sum(m["token_edit_distance_norm"] for m in metrics) / len(metrics)
                    ),
                    "mean_text_similarity": float(
                        sum(m["text_similarity"] for m in metrics) / len(metrics)
                    ),
                    "mean_hidden_cosine": float(
                        sum(m["hidden_cosine"]["mean"] for m in metrics) / len(metrics)
                    ),
                    "mean_hidden_l2": float(
                        sum(m["hidden_l2"]["mean"] for m in metrics) / len(metrics)
                    ),
                    "mean_base_topk_js": float(
                        sum(m["base_topk_js"]["mean"] for m in metrics) / len(metrics)
                    ),
                    "mean_base_prob_diff": float(
                        sum(m["base_prob_diff"]["mean"] for m in metrics) / len(metrics)
                    ),
                    "mean_acceptance_mismatch_rate": float(
                        sum(m["acceptance_mismatch_rate"] for m in metrics) / len(metrics)
                    ),
                }

    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
