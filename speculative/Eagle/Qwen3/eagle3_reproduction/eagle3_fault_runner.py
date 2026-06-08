"""
EAGLE-3 fault-injection runner.

This script keeps the speculative decoding implementation in
eagle3_chain_speculative.py / eagle3_tree_speculative.py unchanged and adds a
small, extensible fault-injection harness around it.

The intended experiment model follows SC2025-style statistical FI:
  * one injected fault per inference trial;
  * weight faults model memory faults;
  * activation faults model computation faults;
  * double-bit flips are the default.

To add a new injection location later, add it to LOCATION_CHOICES and implement
the corresponding branch in install_fault().
"""

import argparse
import json
import os
import random
import time
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from eagle3_chain_speculative import (
    DEVICE,
    Eagle3SpeculativeDecoder,
    TargetModelWithTaps,
    _DATASETS_AVAILABLE,
    is_correct,
    load_benchmark,
    load_draft_head,
    load_target_model,
    seed_everything,
)
from eagle3_tree_speculative import Eagle3TreeSpeculativeDecoder
from eagle3_fault_analysis import analyze_run


TARGET_LINEAR_MODULE_WEIGHTS = {
    "self_attn.v_proj": 1,
    "self_attn.k_proj": 1,
    "self_attn.q_proj": 7,
    "self_attn.o_proj": 7,
    "mlp.up_proj": 37,
    "mlp.gate_proj": 37,
    "mlp.down_proj": 37,
}

DRAFT_LAYER_MODULE_WEIGHTS = {
    "self_attn.v_proj": 1,
    "self_attn.k_proj": 1,
    "self_attn.q_proj": 7,
    "self_attn.o_proj": 7,
    "mlp.up_proj": 37,
    "mlp.gate_proj": 37,
    "mlp.down_proj": 37,
}

LOCATION_CHOICES = [
    "random",
    "target_layer",
    "target_tap",
    "target_embed",
    "target_lm_head",
    "draft_embed",
    "draft_fc",
    "draft_layer",
    "draft_lm_head",
]

WEIGHT_LOCATIONS = {
    "target_layer",
    "target_embed",
    "target_lm_head",
    "draft_embed",
    "draft_fc",
    "draft_layer",
    "draft_lm_head",
}

ACTIVATION_LOCATIONS = {
    "target_layer",
    "target_tap",
    "target_embed",
    "target_lm_head",
    "draft_embed",
    "draft_fc",
    "draft_layer",
    "draft_lm_head",
}


@dataclass
class ActiveFault:
    log: Dict[str, Any]
    cleanup: Callable[[], None]
    live_metadata: Dict[str, Any] = field(default_factory=dict)


def get_submodule(root: Any, path: Optional[str]) -> Any:
    if not path:
        return root
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def weighted_choice(weight_map: Dict[str, int], rng: random.Random) -> str:
    names = list(weight_map.keys())
    weights = list(weight_map.values())
    return rng.choices(names, weights=weights, k=1)[0]


def parse_bit_positions(raw: Optional[str], mode: str, rng: random.Random) -> List[int]:
    if mode == "stuck_at_0":
        return list(range(7, 15))

    expected = 1 if mode == "single_bit" else 2
    if raw:
        bits = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if len(bits) != expected:
            raise ValueError(f"{mode} expects {expected} bit position(s), got {bits}.")
        if any(bit < 0 or bit > 15 for bit in bits):
            raise ValueError(f"Bit positions must be in [0, 15], got {bits}.")
        if len(set(bits)) != len(bits):
            raise ValueError(f"Bit positions must be distinct, got {bits}.")
        return bits

    if mode == "single_bit":
        return [rng.randint(0, 15)]
    if mode == "double_bit":
        return rng.sample(range(16), 2)
    raise ValueError(f"Unsupported fault mode: {mode}")


def flip_scalar(value: torch.Tensor, mode: str, bit_positions: Sequence[int]) -> torch.Tensor:
    """Flip bits of one scalar and return a scalar on the original device."""
    if value.numel() != 1:
        raise ValueError("flip_scalar expects a scalar tensor.")

    fault_dtype = value.dtype if value.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    scalar_cpu = value.detach().to(fault_dtype).cpu().reshape(1)
    bits = int(scalar_cpu.view(torch.int16)[0].item()) & 0xFFFF

    if mode == "single_bit":
        bits ^= 1 << bit_positions[0]
    elif mode == "double_bit":
        bits ^= (1 << bit_positions[0]) | (1 << bit_positions[1])
    elif mode == "stuck_at_0":
        exp_mask = 0x7F80 if fault_dtype == torch.bfloat16 else 0x7C00
        bits &= ~exp_mask
    else:
        raise ValueError(f"Unsupported fault mode: {mode}")

    signed = bits if bits < 0x8000 else bits - 0x10000
    faulted = torch.tensor([signed], dtype=torch.int16).view(fault_dtype)[0]
    return faulted.to(device=value.device, dtype=value.dtype)


def resolve_index(requested: Optional[int], size: int, rng: random.Random, name: str) -> int:
    if size <= 0:
        raise ValueError(f"Cannot choose {name} from empty dimension.")
    if requested is None:
        return rng.randrange(size)
    idx = requested if requested >= 0 else size + requested
    if idx < 0 or idx >= size:
        raise ValueError(f"{name} index {requested} is out of bounds for size {size}.")
    return idx


def resolve_weight_indices(
    weight: torch.Tensor,
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[int, int]:
    if weight.dim() != 2:
        raise ValueError(f"Weight tensor must be 2D, got shape {list(weight.shape)}.")
    row = resolve_index(args.fault_row, weight.shape[0], rng, "fault_row")
    col = resolve_index(args.fault_col, weight.shape[1], rng, "fault_col")
    return row, col


def resolve_activation_indices(
    tensor: torch.Tensor,
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[Optional[int], int]:
    if tensor.dim() >= 3:
        token_idx = resolve_index(args.fault_token_idx, tensor.shape[1], rng, "fault_token_idx")
        hidden_idx = resolve_index(args.fault_hidden_idx, tensor.shape[2], rng, "fault_hidden_idx")
        return token_idx, hidden_idx
    if tensor.dim() == 2:
        if args.fault_token_idx is not None:
            raise ValueError("fault_token_idx was set but activation output has no sequence dimension.")
        hidden_idx = resolve_index(args.fault_hidden_idx, tensor.shape[1], rng, "fault_hidden_idx")
        return None, hidden_idx
    if tensor.dim() == 1:
        if args.fault_token_idx is not None:
            raise ValueError("fault_token_idx was set but activation output has no sequence dimension.")
        hidden_idx = resolve_index(args.fault_hidden_idx, tensor.shape[0], rng, "fault_hidden_idx")
        return None, hidden_idx
    raise ValueError(f"Unsupported activation output shape: {list(tensor.shape)}.")


def read_activation_value(tensor: torch.Tensor, token_idx: Optional[int], hidden_idx: int) -> torch.Tensor:
    if tensor.dim() >= 3:
        return tensor[0, token_idx, hidden_idx]
    if tensor.dim() == 2:
        return tensor[0, hidden_idx]
    return tensor[hidden_idx]


def write_activation_value(
    tensor: torch.Tensor,
    token_idx: Optional[int],
    hidden_idx: int,
    value: torch.Tensor,
) -> torch.Tensor:
    out = tensor.clone()
    if out.dim() >= 3:
        out[0, token_idx, hidden_idx] = value.to(out.dtype)
    elif out.dim() == 2:
        out[0, hidden_idx] = value.to(out.dtype)
    else:
        out[hidden_idx] = value.to(out.dtype)
    return out


def safe_slug(value: Any) -> str:
    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "run"


def make_run_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    dataset_tag = args.dataset or "single"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        timestamp,
        args.decoder,
        dataset_tag,
        args.fault_location,
        args.fault_type,
        args.fault_mode,
    ]
    dirname = safe_slug("_".join(parts))
    return os.path.join("outputs", "eagle3_reproduction", "fault_runs", dirname)


def ensure_run_dirs(run_dir: str) -> Dict[str, str]:
    dirs = {
        "run": run_dir,
        "baselines": os.path.join(run_dir, "baselines"),
        "trials": os.path.join(run_dir, "trials"),
        "analysis": os.path.join(run_dir, "analysis"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_output(output: Any) -> Tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    if isinstance(output, tuple):
        tensor = output[0]

        def rebuild(new_tensor: torch.Tensor) -> Any:
            return (new_tensor,) + output[1:]

        return tensor, rebuild

    def rebuild(new_tensor: torch.Tensor) -> Any:
        return new_tensor

    return output, rebuild


def choose_location(args: argparse.Namespace, rng: random.Random) -> str:
    if args.fault_location != "random":
        return args.fault_location
    pool = WEIGHT_LOCATIONS if args.fault_type == "weight" else ACTIVATION_LOCATIONS
    return rng.choice(sorted(pool))


def choose_tap_layer(
    target_wrapped: TargetModelWithTaps,
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[int, str]:
    tap_indices = list(target_wrapped.tap_indices)
    if args.fault_layer_idx is not None:
        layer_idx = args.fault_layer_idx
        if layer_idx not in tap_indices:
            raise ValueError(
                f"target_tap layer {layer_idx} is not in tap layers {tap_indices}. "
                "Use one of the checkpoint tap layers or omit --fault_layer_idx."
            )
        return layer_idx, f"layer_{layer_idx}"

    slot = args.fault_tap_slot
    if slot == "random":
        pos = rng.randrange(len(tap_indices))
    elif slot == "early":
        pos = 0
    elif slot == "mid":
        pos = len(tap_indices) // 2
    elif slot == "late":
        pos = len(tap_indices) - 1
    else:
        raise ValueError(f"Unsupported tap slot: {slot}")
    return tap_indices[pos], slot


def resolve_module_for_location(
    location: str,
    args: argparse.Namespace,
    target_model: Any,
    draft_head: Any,
    rng: random.Random,
) -> Tuple[Any, Dict[str, Any]]:
    module_path = args.fault_module
    layer_idx = args.fault_layer_idx

    if location == "target_layer":
        if layer_idx is None:
            layer_idx = rng.randrange(len(target_model.model.layers))
        if module_path is None:
            module_path = weighted_choice(TARGET_LINEAR_MODULE_WEIGHTS, rng)
        root = target_model.model.layers[layer_idx]
        return get_submodule(root, module_path), {
            "location": location,
            "layer_idx": layer_idx,
            "module_path": module_path,
        }

    if location == "target_embed":
        return target_model.model.embed_tokens, {
            "location": location,
            "layer_idx": None,
            "module_path": "model.embed_tokens",
        }

    if location == "target_lm_head":
        return target_model.lm_head, {
            "location": location,
            "layer_idx": None,
            "module_path": "lm_head",
        }

    if location == "draft_embed":
        return draft_head.embed_tokens, {
            "location": location,
            "layer_idx": None,
            "module_path": "draft.embed_tokens",
        }

    if location == "draft_fc":
        return draft_head.fc, {
            "location": location,
            "layer_idx": None,
            "module_path": "draft.fc",
        }

    if location == "draft_layer":
        if module_path is None:
            module_path = weighted_choice(DRAFT_LAYER_MODULE_WEIGHTS, rng)
        return get_submodule(draft_head.draft_layer, module_path), {
            "location": location,
            "layer_idx": 0,
            "module_path": f"draft_layer.{module_path}",
        }

    if location == "draft_lm_head":
        return draft_head.lm_head, {
            "location": location,
            "layer_idx": None,
            "module_path": "draft.lm_head",
        }

    raise ValueError(f"Location {location} does not resolve to a normal module.")


def install_weight_fault(
    location: str,
    args: argparse.Namespace,
    target_model: Any,
    draft_head: Any,
    rng: random.Random,
    bit_positions: List[int],
) -> ActiveFault:
    module, base_log = resolve_module_for_location(location, args, target_model, draft_head, rng)
    if not hasattr(module, "weight"):
        raise ValueError(f"Resolved module for {location} has no weight: {module}.")

    weight = module.weight
    row, col = resolve_weight_indices(weight, args, rng)
    original = weight[row, col].detach().clone()
    faulted = flip_scalar(weight[row, col], args.fault_mode, bit_positions)
    with torch.no_grad():
        weight[row, col] = faulted.to(weight.dtype)

    log = {
        **base_log,
        "fault_type": "weight",
        "fault_mode": args.fault_mode,
        "row": row,
        "col": col,
        "bit_positions": bit_positions,
        "weight_shape": list(weight.shape),
        "original_value": float(original.detach().float().cpu().item()),
        "faulted_value": float(faulted.detach().float().cpu().item()),
    }

    def cleanup() -> None:
        with torch.no_grad():
            weight[row, col] = original.to(weight.dtype)

    return ActiveFault(log=log, cleanup=cleanup)


def install_module_activation_fault(
    location: str,
    args: argparse.Namespace,
    target_model: Any,
    target_wrapped: TargetModelWithTaps,
    draft_head: Any,
    rng: random.Random,
    bit_positions: List[int],
) -> ActiveFault:
    module, base_log = resolve_module_for_location(location, args, target_model, draft_head, rng)
    state: Dict[str, Any] = {
        "matched_calls": 0,
        "triggered": False,
        "fault_sites": [],
    }

    phase_filter = args.fault_phase if location.startswith("target_") else "both"

    def phase_matches() -> bool:
        if phase_filter == "both":
            return True
        return getattr(target_wrapped, "current_phase", "idle") == phase_filter

    def hook_fn(_module, _inputs, output):
        if args.fault_trigger_once and state["triggered"]:
            return output
        if not phase_matches():
            return output

        call_idx = state["matched_calls"]
        state["matched_calls"] += 1
        if call_idx != args.fault_call_idx:
            return output

        tensor, rebuild = normalize_output(output)
        token_idx, hidden_idx = resolve_activation_indices(tensor, args, rng)
        original = read_activation_value(tensor, token_idx, hidden_idx).detach().clone()
        faulted = flip_scalar(original, args.fault_mode, bit_positions)
        new_tensor = write_activation_value(tensor, token_idx, hidden_idx, faulted)
        state["triggered"] = True
        state["fault_sites"].append({
            "call_idx": call_idx,
            "phase": getattr(target_wrapped, "current_phase", "idle"),
            "token_idx": token_idx,
            "hidden_idx": hidden_idx,
            "output_shape": list(tensor.shape),
            "bit_positions": bit_positions,
            "original_value": float(original.detach().float().cpu().item()),
            "faulted_value": float(faulted.detach().float().cpu().item()),
        })
        return rebuild(new_tensor)

    handle = module.register_forward_hook(hook_fn)
    log = {
        **base_log,
        "fault_type": "activation",
        "fault_mode": args.fault_mode,
        "phase_filter": phase_filter,
        "call_idx": args.fault_call_idx,
        "bit_positions": bit_positions,
        "trigger_once": args.fault_trigger_once,
    }
    return ActiveFault(log=log, cleanup=handle.remove, live_metadata=state)


def install_target_tap_fault(
    args: argparse.Namespace,
    target_wrapped: TargetModelWithTaps,
    rng: random.Random,
    bit_positions: List[int],
) -> ActiveFault:
    layer_idx, tap_slot = choose_tap_layer(target_wrapped, args, rng)
    layer = target_wrapped.model.model.layers[layer_idx]
    state: Dict[str, Any] = {
        "matched_calls": 0,
        "triggered": False,
        "fault_sites": [],
    }
    phase_filter = args.fault_phase

    def phase_matches() -> bool:
        if phase_filter == "both":
            return True
        return getattr(target_wrapped, "current_phase", "idle") == phase_filter

    def hook_fn(_module, _inputs, output):
        if args.fault_trigger_once and state["triggered"]:
            return output
        if not phase_matches():
            return output

        call_idx = state["matched_calls"]
        state["matched_calls"] += 1
        if call_idx != args.fault_call_idx:
            return output

        tensor, _ = normalize_output(output)
        tapped = tensor.detach().clone()
        token_idx, hidden_idx = resolve_activation_indices(tapped, args, rng)
        original = read_activation_value(tapped, token_idx, hidden_idx).detach().clone()
        faulted = flip_scalar(original, args.fault_mode, bit_positions)
        tapped = write_activation_value(tapped, token_idx, hidden_idx, faulted)

        # This is tap-only: target forward output is unchanged, but the feature
        # saved for EAGLE-3 fusion is overwritten.
        target_wrapped._tapped[layer_idx] = tapped
        state["triggered"] = True
        state["fault_sites"].append({
            "call_idx": call_idx,
            "phase": getattr(target_wrapped, "current_phase", "idle"),
            "tap_slot": tap_slot,
            "tap_layer_idx": layer_idx,
            "token_idx": token_idx,
            "hidden_idx": hidden_idx,
            "output_shape": list(tensor.shape),
            "bit_positions": bit_positions,
            "original_value": float(original.detach().float().cpu().item()),
            "faulted_value": float(faulted.detach().float().cpu().item()),
        })
        return output

    handle = layer.register_forward_hook(hook_fn)
    log = {
        "location": "target_tap",
        "fault_type": "activation",
        "fault_mode": args.fault_mode,
        "tap_slot": tap_slot,
        "tap_layer_idx": layer_idx,
        "phase_filter": phase_filter,
        "call_idx": args.fault_call_idx,
        "bit_positions": bit_positions,
        "trigger_once": args.fault_trigger_once,
        "tap_only": True,
    }
    return ActiveFault(log=log, cleanup=handle.remove, live_metadata=state)


def install_fault(
    args: argparse.Namespace,
    target_model: Any,
    target_wrapped: TargetModelWithTaps,
    draft_head: Any,
    site_seed: int,
) -> ActiveFault:
    rng = random.Random(site_seed)
    location = choose_location(args, rng)

    if args.fault_type == "weight" and location not in WEIGHT_LOCATIONS:
        raise ValueError(f"Location {location} does not support weight faults.")
    if args.fault_type == "activation" and location not in ACTIVATION_LOCATIONS:
        raise ValueError(f"Location {location} does not support activation faults.")
    if location == "target_tap" and args.fault_type != "activation":
        raise ValueError("target_tap is an activation-only, tap-feature fault.")

    bit_positions = parse_bit_positions(args.fault_bit_positions, args.fault_mode, rng)

    if args.fault_type == "weight":
        fault = install_weight_fault(location, args, target_model, draft_head, rng, bit_positions)
    elif location == "target_tap":
        fault = install_target_tap_fault(args, target_wrapped, rng, bit_positions)
    else:
        fault = install_module_activation_fault(
            location, args, target_model, target_wrapped, draft_head, rng, bit_positions
        )

    fault.log["site_seed"] = site_seed
    return fault


def trace_summary(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    verify_events = [e for e in trace if e.get("phase") in {"verify", "verify_tree"}]
    mean_kls = [float(e.get("mean_kl_draft_target", 0.0)) for e in verify_events]
    max_kls = [float(e.get("max_kl_draft_target", 0.0)) for e in verify_events]
    hnorms = [float(e.get("mean_target_hidden_norm", 0.0)) for e in verify_events]
    first_rejects = [e.get("first_reject_pos") for e in verify_events if e.get("first_reject_pos") is not None]
    return {
        "num_events": len(trace),
        "num_verify_events": len(verify_events),
        "mean_verify_kl": sum(mean_kls) / len(mean_kls) if mean_kls else 0.0,
        "max_verify_kl": max(max_kls) if max_kls else 0.0,
        "mean_target_hidden_norm": sum(hnorms) / len(hnorms) if hnorms else 0.0,
        "num_blocks_with_reject": len(first_rejects),
        "first_reject_positions": first_rejects[:20],
    }


def format_prompt(tokenizer: Any, question: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_decode(
    decoder: Any,
    tokenizer: Any,
    sample: Dict[str, Any],
    args: argparse.Namespace,
    run_seed: int,
) -> Dict[str, Any]:
    seed_everything(run_seed)
    started = time.time()
    prompt_text = format_prompt(tokenizer, sample["question"], args.enable_thinking)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    try:
        result = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    except Exception as exc:
        return {
            "sample_id": sample.get("sample_id"),
            "source": sample.get("source"),
            "question": sample["question"],
            "reference": sample.get("answer", ""),
            "prediction": "",
            "is_correct": False,
            "metrics": {},
            "trace_summary": {},
            "run_seed": run_seed,
            "execution_status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_traceback": traceback.format_exc(limit=8),
            "elapsed_s": time.time() - started,
        }
    correct = False
    if args.dataset is not None and sample.get("answer"):
        correct = is_correct(result["text"], sample["answer"], args.dataset)

    entry = {
        "sample_id": sample.get("sample_id"),
        "source": sample.get("source"),
        "question": sample["question"],
        "reference": sample.get("answer", ""),
        "prediction": result["text"],
        "is_correct": correct,
        "metrics": result["metrics"],
        "trace_summary": trace_summary(result["trace"]),
        "run_seed": run_seed,
        "execution_status": "success",
        "error_type": None,
        "error_message": None,
        "elapsed_s": time.time() - started,
    }
    if args.trace_mode == "full":
        entry["trace"] = result["trace"]
    if args.store_tokens:
        entry["tokens"] = result["tokens"]
    return entry


def build_decoder(args: argparse.Namespace) -> Tuple[Any, Any, Any, TargetModelWithTaps, Any]:
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"Loading target : {args.base_model_id}")
    target_model, tokenizer = load_target_model(args.base_model_id, dtype)
    print(f"Loading draft  : {args.draft_model_id}")
    draft_head = load_draft_head(args.draft_model_id, target_model, dtype)

    print("=== Draft head sanity check ===")
    print(f"V_draft: {draft_head.V_draft}, V_target: {draft_head.V_target}")
    print(f"Tap layers: {draft_head.tap_indices}")
    print("================================")

    target_wrapped = TargetModelWithTaps(target_model, draft_head.tap_indices)
    do_sample = args.temperature > 0
    if args.enable_thinking and not do_sample:
        print("[Warning] Qwen3 thinking mode is not usually run greedily; setting temperature=0.6.")
        args.temperature = 0.6
        do_sample = True

    if args.decoder == "chain":
        decoder = Eagle3SpeculativeDecoder(
            target=target_wrapped,
            draft_head=draft_head,
            tokenizer=tokenizer,
            block_size=args.block_size,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=do_sample,
        )
    else:
        decoder = Eagle3TreeSpeculativeDecoder(
            target=target_wrapped,
            draft_head=draft_head,
            tokenizer=tokenizer,
            block_size=args.block_size,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=do_sample,
            tree_depth=args.tree_depth,
            tree_branch_factor=args.tree_branch_factor,
            tree_expand_top_nodes=args.tree_expand_top_nodes,
            tree_verify_nodes=args.tree_verify_nodes,
            verify_backend=args.verify_backend,
            tree_accept_mode=args.tree_accept_mode,
            allow_tree_attention_fallback=not args.no_tree_attention_fallback,
        )
    return decoder, target_model, draft_head, target_wrapped, tokenizer


def load_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset is not None:
        if not _DATASETS_AVAILABLE:
            raise RuntimeError("datasets_loader.py not found.")
        return load_benchmark(args.dataset, num_samples=args.num_samples, seed=args.seed)
    return [{"question": args.prompt, "answer": "", "source": "single", "sample_id": 0}]


def aggregate_results(baselines: List[Dict[str, Any]], trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_by_id = {b["sample_id"]: b for b in baselines}
    changed = 0
    baseline_correct_then_wrong = 0
    correct_count = 0
    accept_rates = []
    for trial in trials:
        base = baseline_by_id.get(trial["sample_id"])
        if base is not None and trial["prediction"] != base["prediction"]:
            changed += 1
        if base is not None and base.get("is_correct") and not trial.get("is_correct"):
            baseline_correct_then_wrong += 1
        if trial.get("is_correct"):
            correct_count += 1
        metrics = trial.get("metrics", {})
        if "acceptance_rate" in metrics:
            accept_rates.append(float(metrics["acceptance_rate"]))

    total = len(trials)
    return {
        "total_trials": total,
        "output_changed": changed,
        "output_changed_rate": changed / total if total else 0.0,
        "baseline_correct_then_wrong": baseline_correct_then_wrong,
        "baseline_correct_then_wrong_rate": baseline_correct_then_wrong / total if total else 0.0,
        "fault_correct_count": correct_count,
        "fault_accuracy": correct_count / total if total else 0.0,
        "avg_fault_acceptance_rate": sum(accept_rates) / len(accept_rates) if accept_rates else 0.0,
    }


def compact_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulky full traces from the run-level index; per-run files keep them."""
    return {
        key: value
        for key, value in entry.items()
        if key not in {"trace", "tokens"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EAGLE-3 statistical fault injection runner")

    parser.add_argument("--decoder", default="chain", choices=["chain", "tree"])
    parser.add_argument("--base_model_id", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft_model_id", default="RedHatAI/Qwen3-8B-Thinking-speculator.eagle3")
    parser.add_argument("--block_size", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--dataset", default=None, choices=[
        "gsm8k", "math500", "aime2024", "aime2025", "gpqa", "livecodebench", "openthoughts"
    ])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--prompt", default="What is 25 * 48?")

    parser.add_argument("--tree_depth", type=int, default=6)
    parser.add_argument("--tree_branch_factor", type=int, default=4)
    parser.add_argument("--tree_expand_top_nodes", type=int, default=2)
    parser.add_argument("--tree_verify_nodes", type=int, default=16)
    parser.add_argument("--verify_backend", default="auto", choices=["auto", "tree_attention", "path"])
    parser.add_argument("--tree_accept_mode", default="auto", choices=["auto", "greedy_tree", "single_path_strict"])
    parser.add_argument("--no_tree_attention_fallback", action="store_true")

    parser.add_argument("--num_fault_trials", type=int, default=1)
    parser.add_argument("--fault_location", default="target_layer", choices=LOCATION_CHOICES)
    parser.add_argument("--fault_type", default="weight", choices=["weight", "activation"])
    parser.add_argument("--fault_mode", default="double_bit", choices=["single_bit", "double_bit", "stuck_at_0"])
    parser.add_argument("--fault_layer_idx", type=int, default=None)
    parser.add_argument("--fault_module", default=None)
    parser.add_argument("--fault_phase", default="both", choices=["prefill", "verify", "both"])
    parser.add_argument("--fault_tap_slot", default="random", choices=["early", "mid", "late", "random"])
    parser.add_argument("--fault_row", type=int, default=None)
    parser.add_argument("--fault_col", type=int, default=None)
    parser.add_argument("--fault_token_idx", type=int, default=None)
    parser.add_argument("--fault_hidden_idx", type=int, default=None)
    parser.add_argument("--fault_bit_positions", default=None, help="Comma-separated bits, e.g. 7,12")
    parser.add_argument("--fault_call_idx", type=int, default=0, help="Which matching forward call to corrupt.")
    parser.add_argument("--fault_seed", type=int, default=None)
    parser.add_argument("--fault_trigger_once", action="store_true", default=True)
    parser.add_argument("--persistent_activation_fault", dest="fault_trigger_once", action="store_false")

    parser.add_argument("--trace_mode", default="full", choices=["summary", "full"])
    parser.add_argument("--store_tokens", action="store_true", default=True)
    parser.add_argument("--no_store_tokens", dest="store_tokens", action="store_false")
    parser.add_argument("--no_auto_analyze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_fault_trials < 0:
        raise ValueError("--num_fault_trials must be non-negative.")
    if args.fault_call_idx < 0:
        raise ValueError("--fault_call_idx must be non-negative.")

    run_dir = make_run_dir(args)
    dirs = ensure_run_dirs(run_dir)
    print(f"Run directory: {run_dir}")

    decoder, target_model, draft_head, target_wrapped, tokenizer = build_decoder(args)
    samples = load_samples(args)

    baselines: List[Dict[str, Any]] = []
    trials: List[Dict[str, Any]] = []

    print(f"Running {len(samples)} baseline sample(s).")
    for sample in samples:
        sample_id = int(sample.get("sample_id", len(baselines)))
        run_seed = args.seed + sample_id
        baseline = run_decode(decoder, tokenizer, sample, args, run_seed=run_seed)
        baselines.append(baseline)
        baseline_path = os.path.join(dirs["baselines"], f"sample_{sample_id}.json")
        write_json(baseline_path, baseline)
        print(
            f"[baseline sample={sample_id}] "
            f"accept={baseline['metrics'].get('acceptance_rate', 0.0):.4f} "
            f"tokens={baseline['metrics'].get('tokens_emitted', 0)}"
        )

    print(f"Running {args.num_fault_trials} fault trial(s) per sample.")
    base_fault_seed = args.fault_seed if args.fault_seed is not None else args.seed + 1_000_003
    for sample in samples:
        sample_id = int(sample.get("sample_id", 0))
        run_seed = args.seed + sample_id
        for trial_idx in range(args.num_fault_trials):
            site_seed = base_fault_seed + sample_id * max(args.num_fault_trials, 1) + trial_idx
            t0 = time.time()
            fault = install_fault(args, target_model, target_wrapped, draft_head, site_seed)
            try:
                trial = run_decode(decoder, tokenizer, sample, args, run_seed=run_seed)
            finally:
                fault.cleanup()

            if fault.live_metadata:
                fault.log["runtime"] = fault.live_metadata
            trial["trial_idx"] = trial_idx
            trial["fault_log"] = fault.log
            trial["elapsed_s"] = time.time() - t0
            trials.append(trial)
            trial_path = os.path.join(dirs["trials"], f"trial_{trial_idx}_sample_{sample_id}.json")
            write_json(trial_path, trial)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                f"[fault sample={sample_id} trial={trial_idx}] "
                f"loc={fault.log.get('location')} type={fault.log.get('fault_type')} "
                f"mode={fault.log.get('fault_mode')} "
                f"accept={trial['metrics'].get('acceptance_rate', 0.0):.4f} "
                f"triggered={fault.log.get('runtime', {}).get('triggered', 'weight')}"
            )

    aggregate = aggregate_results(baselines, trials)
    print("=" * 70)
    print(f"Trials: {aggregate['total_trials']}")
    print(f"Output changed rate: {aggregate['output_changed_rate']:.4f}")
    if args.dataset is not None:
        baseline_correct = sum(1 for b in baselines if b.get("is_correct"))
        print(f"Baseline accuracy: {baseline_correct}/{len(baselines)} = {baseline_correct / len(baselines):.4f}")
        print(f"Fault accuracy: {aggregate['fault_accuracy']:.4f}")
    print(f"Avg fault acceptance rate: {aggregate['avg_fault_acceptance_rate']:.4f}")
    print("=" * 70)

    summary = {
        "implementation": "eagle3_fault_runner",
        "run_dir": run_dir,
        "output_files": {
            "baselines_dir": dirs["baselines"],
            "trials_dir": dirs["trials"],
            "analysis_dir": dirs["analysis"],
            "raw_results": os.path.join(run_dir, "raw_results.json"),
        },
        "decoder": args.decoder,
        "dataset": args.dataset or "single",
        "generation_args": {
            "base_model_id": args.base_model_id,
            "draft_model_id": args.draft_model_id,
            "max_new_tokens": args.max_new_tokens,
            "block_size": args.block_size,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "enable_thinking": args.enable_thinking,
            "seed": args.seed,
            "dtype": args.dtype,
        },
        "tree_args": {
            "tree_depth": args.tree_depth,
            "tree_branch_factor": args.tree_branch_factor,
            "tree_expand_top_nodes": args.tree_expand_top_nodes,
            "tree_verify_nodes": args.tree_verify_nodes,
            "verify_backend": args.verify_backend,
            "tree_accept_mode": args.tree_accept_mode,
        } if args.decoder == "tree" else None,
        "fault_args": {
            "num_fault_trials": args.num_fault_trials,
            "fault_location": args.fault_location,
            "fault_type": args.fault_type,
            "fault_mode": args.fault_mode,
            "fault_layer_idx": args.fault_layer_idx,
            "fault_module": args.fault_module,
            "fault_phase": args.fault_phase,
            "fault_tap_slot": args.fault_tap_slot,
            "fault_row": args.fault_row,
            "fault_col": args.fault_col,
            "fault_token_idx": args.fault_token_idx,
            "fault_hidden_idx": args.fault_hidden_idx,
            "fault_bit_positions": args.fault_bit_positions,
            "fault_call_idx": args.fault_call_idx,
            "fault_seed": args.fault_seed,
            "fault_trigger_once": args.fault_trigger_once,
        },
        "aggregate": aggregate,
        "baselines": [compact_entry(item) for item in baselines],
        "trials": [compact_entry(item) for item in trials],
    }

    raw_summary_path = os.path.join(run_dir, "raw_results.json")
    write_json(raw_summary_path, summary)
    print(f"Raw results -> {raw_summary_path}")

    if args.output_json:
        write_json(args.output_json, summary)
        print(f"Compatibility copy -> {args.output_json}")

    if not args.no_auto_analyze:
        analysis = analyze_run(run_dir, dirs["analysis"])
        print(f"Analysis summary -> {analysis['paths']['summary']}")
        print(f"Analysis report  -> {analysis['paths']['report']}")


if __name__ == "__main__":
    main()
