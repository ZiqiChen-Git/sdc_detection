"""
Lightweight linear-layer profiling for EAGLE-3 Qwen3 reproduction.

Purpose:
  * estimate parameter share of target/draft linear modules;
  * estimate relative forward time of target prefill, target verify, draft_fc,
    and draft_layer linear modules;
  * support FI site selection with a small pre-experiment.

This is not a rigorous performance benchmark. It uses forward hooks and CUDA
synchronization to get a practical module-level signal for experiment planning.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from eagle3_chain_speculative import (
    DEVICE,
    TargetModelWithTaps,
    load_draft_head,
    load_target_model,
    sample_token,
    seed_everything,
)


TARGET_LINEAR_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

DRAFT_LAYER_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


@dataclass
class ModuleStats:
    scope: str
    name: str
    module_type: str
    layer_idx: Optional[int]
    param_count: int
    calls: int = 0
    total_time_s: float = 0.0
    times_s: List[float] = field(default_factory=list)

    def add_time(self, elapsed_s: float) -> None:
        self.calls += 1
        self.total_time_s += elapsed_s
        self.times_s.append(elapsed_s)

    def as_dict(self) -> Dict[str, Any]:
        avg = self.total_time_s / self.calls if self.calls else 0.0
        return {
            "scope": self.scope,
            "name": self.name,
            "module_type": self.module_type,
            "layer_idx": self.layer_idx,
            "param_count": self.param_count,
            "calls": self.calls,
            "total_time_s": self.total_time_s,
            "avg_time_s": avg,
        }


class LinearProfiler:
    def __init__(self):
        self.stats: Dict[str, ModuleStats] = {}
        self.handles: List[Any] = []
        self.enabled = False
        self.phase = "idle"

    def register(
        self,
        module: nn.Module,
        *,
        scope: str,
        name: str,
        module_type: str,
        layer_idx: Optional[int],
    ) -> None:
        key = f"{scope}:{name}"
        params = sum(p.numel() for p in module.parameters(recurse=False))
        self.stats[key] = ModuleStats(
            scope=scope,
            name=name,
            module_type=module_type,
            layer_idx=layer_idx,
            param_count=params,
        )

        def pre_hook(_module, _inputs):
            if not self.enabled:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _module.__profile_start_s = time.perf_counter()

        def post_hook(_module, _inputs, _output):
            if not self.enabled:
                return
            start = getattr(_module, "__profile_start_s", None)
            if start is None:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.stats[key].add_time(time.perf_counter() - start)

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start(self, phase: str) -> None:
        self.phase = phase
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False
        self.phase = "idle"


def get_submodule(root: Any, path: str) -> nn.Module:
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def safe_slug(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "profile"


def make_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = safe_slug(f"{ts}_linear_profile_{args.max_new_tokens}tok_{args.profile_iterations}iter")
    return os.path.join("outputs", "eagle3_reproduction", "profiles", name)


def register_target_linears(profiler: LinearProfiler, target_model: Any, tap_indices: List[int]) -> None:
    tap_set = set(tap_indices)
    for layer_idx, layer in enumerate(target_model.model.layers):
        for module_type in TARGET_LINEAR_MODULES:
            module = get_submodule(layer, module_type)
            tag = "tap" if layer_idx in tap_set else "non_tap"
            profiler.register(
                module,
                scope="target",
                name=f"target.layer{layer_idx}.{module_type}",
                module_type=module_type,
                layer_idx=layer_idx,
            )
            profiler.stats[f"target:target.layer{layer_idx}.{module_type}"].scope = f"target_{tag}"


def register_draft_linears(profiler: LinearProfiler, draft_head: Any) -> None:
    profiler.register(
        draft_head.fc,
        scope="draft",
        name="draft.fc",
        module_type="draft_fc",
        layer_idx=None,
    )
    for module_type in DRAFT_LAYER_MODULES:
        module = get_submodule(draft_head.draft_layer, module_type)
        profiler.register(
            module,
            scope="draft",
            name=f"draft.layer0.{module_type}",
            module_type=module_type,
            layer_idx=0,
        )
    profiler.register(
        draft_head.lm_head,
        scope="draft",
        name="draft.lm_head",
        module_type="draft_lm_head",
        layer_idx=None,
    )


def format_prompt(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.no_grad()
def run_profile_iteration(
    args: argparse.Namespace,
    profiler: LinearProfiler,
    target_wrapped: TargetModelWithTaps,
    draft_head: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
) -> Dict[str, Any]:
    context_ids = input_ids.clone()
    generated = []
    phase_times = defaultdict(float)
    phase_calls = defaultdict(int)
    do_sample = args.temperature > 0

    for _ in range(args.max_new_tokens):
        profiler.start("target_prefill")
        t0 = time.perf_counter()
        prefill_out = target_wrapped.prefill(context_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase_times["target_prefill"] += time.perf_counter() - t0
        phase_calls["target_prefill"] += 1
        profiler.stop()

        anchor_id, _ = sample_token(
            prefill_out["logits"],
            args.temperature,
            args.top_k,
            do_sample,
            args.top_p,
        )
        context_with_anchor = torch.cat(
            [context_ids, torch.tensor([[anchor_id]], device=DEVICE)],
            dim=1,
        )
        generated.append(anchor_id)
        if len(generated) >= args.max_new_tokens:
            context_ids = context_with_anchor
            break

        profiler.start("draft_prefill_context")
        t0 = time.perf_counter()
        draft_hidden, draft_pkv, first_logits, next_position_id = draft_head.prefill_context(
            context_ids,
            anchor_id,
            prefill_out["tapped_full"],
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase_times["draft_prefill_context"] += time.perf_counter() - t0
        phase_calls["draft_prefill_context"] += 1
        profiler.stop()

        proposals = []
        logits = first_logits
        hidden = draft_hidden
        pkv = draft_pkv
        prev_token = None
        remaining = args.max_new_tokens - len(generated)
        steps = min(args.block_size, remaining)

        for step in range(steps):
            if step > 0:
                profiler.start("draft_forward_step")
                t0 = time.perf_counter()
                logits, hidden, pkv = draft_head.forward_step(
                    prev_token_id=prev_token,
                    fused_hidden=hidden,
                    past_key_values=pkv,
                    position_id=next_position_id + step - 1,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                phase_times["draft_forward_step"] += time.perf_counter() - t0
                phase_calls["draft_forward_step"] += 1
                profiler.stop()

            token_id, _ = sample_token(
                logits,
                args.temperature,
                args.top_k,
                do_sample,
                args.top_p,
            )
            proposals.append(token_id)
            prev_token = token_id

        if not proposals:
            context_ids = context_with_anchor
            break

        profiler.start("target_verify")
        t0 = time.perf_counter()
        target_wrapped.verify(context_with_anchor, proposals)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase_times["target_verify"] += time.perf_counter() - t0
        phase_calls["target_verify"] += 1
        profiler.stop()

        # Profiling does not need exact speculative accept/reject. Advance by
        # anchor + proposals to keep sequence length growth realistic enough.
        add_ids = torch.tensor([proposals], device=DEVICE)
        context_ids = torch.cat([context_with_anchor, add_ids], dim=1)
        generated.extend(proposals)
        if len(generated) >= args.max_new_tokens:
            break

    return {
        "phase_times_s": dict(phase_times),
        "phase_calls": dict(phase_calls),
        "tokens_emitted_for_profile": len(generated),
    }


def summarize_modules(stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_module_type: Dict[str, Dict[str, Any]] = {}
    by_scope: Dict[str, Dict[str, Any]] = {}

    def add(group: Dict[str, Dict[str, Any]], key: str, row: Dict[str, Any]) -> None:
        item = group.setdefault(key, {
            "param_count": 0,
            "calls": 0,
            "total_time_s": 0.0,
        })
        item["param_count"] += int(row["param_count"])
        item["calls"] += int(row["calls"])
        item["total_time_s"] += float(row["total_time_s"])

    for row in stats:
        add(by_module_type, row["module_type"], row)
        add(by_scope, row["scope"], row)

    for group in (by_module_type, by_scope):
        for item in group.values():
            item["avg_time_s"] = item["total_time_s"] / item["calls"] if item["calls"] else 0.0

    total_params = sum(row["param_count"] for row in stats)
    total_time = sum(row["total_time_s"] for row in stats)
    for row in stats:
        row["param_share"] = row["param_count"] / total_params if total_params else 0.0
        row["time_share"] = row["total_time_s"] / total_time if total_time else 0.0

    return {
        "total_linear_params_profiled": total_params,
        "total_linear_hook_time_s": total_time,
        "by_module_type": by_module_type,
        "by_scope": by_scope,
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: str, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    module_items = sorted(
        summary["module_summary"]["by_module_type"].items(),
        key=lambda item: item[1]["param_count"],
        reverse=True,
    )
    scope_items = sorted(
        summary["module_summary"]["by_scope"].items(),
        key=lambda item: item[1]["total_time_s"],
        reverse=True,
    )

    lines = [
        "# EAGLE-3 Linear Layer Profile",
        "",
        "## Phase Time",
        "",
    ]
    for phase, elapsed in sorted(summary["phase_times_s"].items(), key=lambda item: item[1], reverse=True):
        calls = summary["phase_calls"].get(phase, 0)
        lines.append(f"- {phase}: {elapsed:.6f}s over {calls} call(s)")

    lines.extend(["", "## Module Type Summary", ""])
    for name, item in module_items:
        lines.append(
            f"- {name}: params={item['param_count']}, calls={item['calls']}, "
            f"time={item['total_time_s']:.6f}s, avg={item['avg_time_s']:.6f}s"
        )

    lines.extend(["", "## Scope Summary", ""])
    for name, item in scope_items:
        lines.append(
            f"- {name}: params={item['param_count']}, calls={item['calls']}, "
            f"time={item['total_time_s']:.6f}s, avg={item['avg_time_s']:.6f}s"
        )

    top_rows = sorted(rows, key=lambda row: row["total_time_s"], reverse=True)[:20]
    lines.extend(["", "## Top Modules By Hook Time", ""])
    for row in top_rows:
        lines.append(
            f"- {row['name']}: params={row['param_count']}, calls={row['calls']}, "
            f"time={row['total_time_s']:.6f}s"
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile EAGLE-3 linear module cost.")
    parser.add_argument("--base_model_id", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft_model_id", default="RedHatAI/Qwen3-8B-Thinking-speculator.eagle3")
    parser.add_argument("--prompt", default="What is 25 * 48?")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--profile_iterations", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = make_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"Loading target : {args.base_model_id}")
    target_model, tokenizer = load_target_model(args.base_model_id, dtype)
    print(f"Loading draft  : {args.draft_model_id}")
    draft_head = load_draft_head(args.draft_model_id, target_model, dtype)
    print(f"Tap layers     : {draft_head.tap_indices}")

    target_wrapped = TargetModelWithTaps(target_model, draft_head.tap_indices)
    profiler = LinearProfiler()
    register_target_linears(profiler, target_model, draft_head.tap_indices)
    register_draft_linears(profiler, draft_head)

    prompt_text = format_prompt(tokenizer, args.prompt, args.enable_thinking)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)

    phase_times = defaultdict(float)
    phase_calls = defaultdict(int)
    iteration_outputs = []
    try:
        for iteration in range(args.profile_iterations):
            seed_everything(args.seed + iteration)
            result = run_profile_iteration(
                args,
                profiler,
                target_wrapped,
                draft_head,
                tokenizer,
                input_ids,
            )
            iteration_outputs.append(result)
            for phase, elapsed in result["phase_times_s"].items():
                phase_times[phase] += elapsed
            for phase, calls in result["phase_calls"].items():
                phase_calls[phase] += calls
    finally:
        profiler.remove()

    rows = [stat.as_dict() for stat in profiler.stats.values()]
    rows.sort(key=lambda row: (row["scope"], row["layer_idx"] if row["layer_idx"] is not None else -1, row["module_type"]))
    module_summary = summarize_modules(rows)

    summary = {
        "profile_args": vars(args),
        "tap_layers": draft_head.tap_indices,
        "phase_times_s": dict(phase_times),
        "phase_calls": dict(phase_calls),
        "module_summary": module_summary,
        "iterations": iteration_outputs,
    }

    summary_path = os.path.join(output_dir, "profile_summary.json")
    csv_path = os.path.join(output_dir, "module_profile.csv")
    report_path = os.path.join(output_dir, "profile_report.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_csv(csv_path, rows)
    write_report(report_path, summary, rows)

    print(f"Profile summary -> {summary_path}")
    print(f"Module CSV      -> {csv_path}")
    print(f"Report          -> {report_path}")


if __name__ == "__main__":
    main()
