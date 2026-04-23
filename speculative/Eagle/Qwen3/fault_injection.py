"""
fault_injection.py
==================
SDC fault injection for EAGLE-3 + LLaMA speculative decoding.

设计原则
--------
1. 每层内随机选一个权重元素或激活元素做 bit flip（符合学长要求）。
2. 明确区分注入位置的归属：
     LOCATION_TARGET_ONLY  – 只影响 target model
     LOCATION_DRAFT_ONLY   – 只影响 draft head
     LOCATION_SHARED       – lm_head（target 和 draft 共享）
3. 每种位置类别有对应的预期 SDC 信号描述（见 FAULT_LOCATION_EFFECTS）。

支持两种注入模式
-----------------
  weight fault  : 直接修改 nn.Parameter，注入前保存快照，注入后可还原。
                  每次 forward 都带着错误权重，代表持久性硬件故障。
  activation fault : 注册 forward hook，在激活输出上随机 bit flip。
                    每次 forward 只触发一次（hook 设置 triggered 标志）。
                    代表瞬时 / 单次故障。

用法示例
---------
from fault_injection import FaultInjector, FaultLocation, FaultMode

injector = FaultInjector(target_model, draft_head, tokenizer=tokenizer)

# 在 target 的 layer 16 的 mlp.gate_proj 权重上注入双 bit flip
snapshot = injector.inject_weight_fault(
    location=FaultLocation.TARGET_LAYER,
    layer_idx=16,
    module_path="mlp.gate_proj",
    mode=FaultMode.DOUBLE_BIT,
)

# 跑推理 ...
result = decoder.generate(...)

# 还原权重
injector.restore_weight(snapshot)

# 在 lm_head（共享）注入激活故障
handle = injector.inject_activation_fault(
    location=FaultLocation.SHARED_LM_HEAD,
    mode=FaultMode.SINGLE_BIT,
)
# 跑推理 ...
handle.remove()
"""

import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# Fault location taxonomy
# ============================================================

class FaultLocation(Enum):
    # ── Target-only ──────────────────────────────────────────
    TARGET_LAYER        = "target_layer"         # target.model.layers[i].{attn,mlp}
    TARGET_EMBED        = "target_embed"          # target.model.embed_tokens

    # ── Draft-only ───────────────────────────────────────────
    DRAFT_EMBED         = "draft_embed"           # draft_head.embed_tokens
    DRAFT_FC            = "draft_fc"              # draft_head.fc  (4H→H fusion)
    DRAFT_LAYER         = "draft_layer"           # draft_head.draft_layer

    # ── Shared ───────────────────────────────────────────────
    SHARED_LM_HEAD      = "shared_lm_head"        # target.lm_head == draft_head.lm_head


class FaultMode(Enum):
    SINGLE_BIT  = "single_bit"   # flip 1 bit at a random position
    DOUBLE_BIT  = "double_bit"   # flip 2 bits (higher energy fault)
    STUCK_AT_0  = "stuck_at_0"   # force exponent bits to 0 (NaN / subnormal)


# ============================================================
# Expected SDC effects per location (for documentation & trace)
# ============================================================

FAULT_LOCATION_EFFECTS: Dict[FaultLocation, Dict[str, str]] = {
    FaultLocation.TARGET_LAYER: {
        "affected_steps": "Step-2 (Prefill) AND Step-7 (Verify)",
        "acceptance_rate": "↓ — target probs shift; draft was trained on clean target",
        "kl_draft_target": "↑ — distribution diverges",
        "hidden_norm": "异常 — 该层之后的所有 hidden state 幅值偏移",
        "lm_head_blind_spot": "No — lm_head still clean so output is detectable",
        "detection_difficulty": "Easy",
    },
    FaultLocation.TARGET_EMBED: {
        "affected_steps": "Step-2 AND Step-7 (all tokens re-embedded each pass)",
        "acceptance_rate": "↓ — input representation corrupted from token 0",
        "kl_draft_target": "↑ — large; entire context is wrong",
        "hidden_norm": "f_early_norm 首先异常 (embeds feed layer 0 directly)",
        "lm_head_blind_spot": "No",
        "detection_difficulty": "Easy — f_early_norm spike at step 1",
    },
    FaultLocation.DRAFT_EMBED: {
        "affected_steps": "Step-6 (Draft only)",
        "acceptance_rate": "Varies — draft probs shift, target probs unaffected",
        "kl_draft_target": "↑ — but from draft side only",
        "hidden_norm": "draft_hidden_norm 异常; target hidden 不受影响",
        "lm_head_blind_spot": "No",
        "detection_difficulty": "Medium — acceptance drop visible but target looks clean",
    },
    FaultLocation.DRAFT_FC: {
        "affected_steps": "Step-5 (feature fusion)",
        "acceptance_rate": "↓ — fused context fed to draft head is wrong",
        "kl_draft_target": "↑ — draft distribution diverges",
        "hidden_norm": "fused_feature_norm 异常",
        "lm_head_blind_spot": "No",
        "detection_difficulty": "Medium — visible in fused_feature_norm",
    },
    FaultLocation.DRAFT_LAYER: {
        "affected_steps": "Step-6 (Draft transformer layer)",
        "acceptance_rate": "↓ — draft token predictions wrong",
        "kl_draft_target": "↑ — draft entropy 可能暴涨",
        "hidden_norm": "draft_hidden_norm 异常",
        "lm_head_blind_spot": "No",
        "detection_difficulty": "Medium",
    },
    FaultLocation.SHARED_LM_HEAD: {
        "affected_steps": "Step-6 (Draft token output) AND Step-7 (Target token output)",
        "acceptance_rate": "⚠️ 可能不变 — draft 和 target 同向偏移，ratio 分子分母同时错",
        "kl_draft_target": "⚠️ 可能偏低 — 两个分布同向移动，KL 不增大",
        "hidden_norm": "hidden states 不受影响 (lm_head 只在最后一步)",
        "lm_head_blind_spot": "YES — acceptance_rate 正常但输出 token 全错，是 SDC 盲区",
        "detection_difficulty": "Hard — 需要额外指标（如 output token 与 baseline 比对）",
    },
}


# ============================================================
# Bit manipulation helpers (bfloat16)
# ============================================================

def _flip_single(val: torch.Tensor, bit_pos: int) -> torch.Tensor:
    """Flip one bit of a scalar bfloat16 tensor."""
    with torch.no_grad():
        b = val.to(torch.bfloat16).view(torch.int16)
        b = b ^ (1 << bit_pos)
    return b.view(torch.bfloat16)


def _flip_double(val: torch.Tensor, bit_pos0: int, bit_pos1: int) -> torch.Tensor:
    with torch.no_grad():
        b = val.to(torch.bfloat16).view(torch.int16)
        b = b ^ ((1 << bit_pos0) | (1 << bit_pos1))
    return b.view(torch.bfloat16)


def _stuck_at_0(val: torch.Tensor) -> torch.Tensor:
    """Force exponent bits (bits 7-14 of bfloat16) to 0 → subnormal / zero."""
    with torch.no_grad():
        b = val.to(torch.bfloat16).view(torch.int16)
        # exponent mask for bfloat16: bits 7..14
        exp_mask = torch.tensor(0x7F80, dtype=torch.int16)
        b = b & (~exp_mask)
    return b.view(torch.bfloat16)


def apply_bit_fault(val: torch.Tensor, mode: FaultMode) -> Tuple[torch.Tensor, Dict]:
    """
    Apply a fault to a scalar tensor according to mode.
    Returns (faulted_value, metadata_dict).
    """
    if mode == FaultMode.SINGLE_BIT:
        bit_pos = random.randint(0, 15)
        return _flip_single(val, bit_pos), {"bit_positions": [bit_pos]}
    elif mode == FaultMode.DOUBLE_BIT:
        bits = random.sample(range(16), 2)
        return _flip_double(val, bits[0], bits[1]), {"bit_positions": bits}
    elif mode == FaultMode.STUCK_AT_0:
        return _stuck_at_0(val), {"bit_positions": list(range(7, 15))}
    else:
        raise ValueError(f"Unknown FaultMode: {mode}")


# ============================================================
# Weight fault snapshot
# ============================================================

@dataclass
class WeightFaultSnapshot:
    """Stores enough info to restore a weight fault."""
    location: FaultLocation
    module_path: str          # e.g. "mlp.gate_proj"
    layer_idx: Optional[int]  # None for non-layered modules
    row: int
    col: int
    original_value: torch.Tensor
    fault_metadata: Dict = field(default_factory=dict)

    def as_log(self) -> Dict:
        d = asdict(self)
        d["original_value"] = self.original_value.item()
        d["location"] = self.location.value
        return d


# ============================================================
# Activation fault handle wrapper
# ============================================================

@dataclass
class ActivationFaultHandle:
    location: FaultLocation
    module_path: str
    layer_idx: Optional[int]
    mode: FaultMode
    _hook_handle: Any = field(repr=False)
    metadata: Dict = field(default_factory=dict)

    def remove(self):
        self._hook_handle.remove()

    def as_log(self) -> Dict:
        return {
            "location": self.location.value,
            "module_path": self.module_path,
            "layer_idx": self.layer_idx,
            "mode": self.mode.value,
            "metadata": self.metadata,
        }


# ============================================================
# FaultInjector
# ============================================================

class FaultInjector:
    """
    Central fault injection controller for EAGLE-3 speculative decoding.

    Resolves module references for all FaultLocation categories.
    Distinguishes target-only, draft-only, and shared locations.
    """

    def __init__(self, target_model: nn.Module, draft_head: nn.Module):
        self.target = target_model
        self.draft  = draft_head

    # ------------------------------------------------------------------
    # Module resolution
    # ------------------------------------------------------------------

    def _resolve_module(
        self,
        location: FaultLocation,
        layer_idx: Optional[int],
        module_path: Optional[str],
    ) -> nn.Module:
        """
        Return the nn.Module corresponding to (location, layer_idx, module_path).

        module_path is a dot-separated path relative to the layer root,
        e.g. "self_attn.q_proj" or "mlp.gate_proj".
        For non-layer locations (embed, fc, lm_head) it is ignored.
        """
        if location == FaultLocation.TARGET_LAYER:
            assert layer_idx is not None, "layer_idx required for TARGET_LAYER"
            root = self.target.model.layers[layer_idx]
            return _get_submodule(root, module_path)

        elif location == FaultLocation.TARGET_EMBED:
            return self.target.model.embed_tokens

        elif location == FaultLocation.DRAFT_EMBED:
            return self.draft.embed_tokens

        elif location == FaultLocation.DRAFT_FC:
            return self.draft.fc

        elif location == FaultLocation.DRAFT_LAYER:
            root = self.draft.draft_layer
            if module_path:
                return _get_submodule(root, module_path)
            return root

        elif location == FaultLocation.SHARED_LM_HEAD:
            # target.lm_head and draft_head.lm_head point to same object
            return self.target.lm_head

        else:
            raise ValueError(f"Unknown FaultLocation: {location}")

    # ------------------------------------------------------------------
    # Weight fault (persistent)
    # ------------------------------------------------------------------

    def inject_weight_fault(
        self,
        location: FaultLocation,
        mode: FaultMode = FaultMode.DOUBLE_BIT,
        layer_idx: Optional[int] = None,
        module_path: Optional[str] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> WeightFaultSnapshot:
        """
        Flip bits in a weight tensor at a randomly selected (row, col) position.
        Saves original value and returns a snapshot for later restoration.

        If row/col are None, selects uniformly at random within the weight matrix.
        seed 控制 row/col 和 bit position 的随机选择，传入相同 seed 可复现同一个注入点。
        """
        if seed is not None:
            random.seed(seed)

        module = self._resolve_module(location, layer_idx, module_path)
        weight = module.weight

        if row is None:
            row = random.randint(0, weight.shape[0] - 1)
        if col is None:
            col = random.randint(0, weight.shape[1] - 1)

        original = weight[row, col].clone()
        faulted, meta = apply_bit_fault(weight[row, col], mode)

        with torch.no_grad():
            weight[row, col] = faulted.to(weight.dtype)

        snapshot = WeightFaultSnapshot(
            location=location,
            module_path=module_path or "",
            layer_idx=layer_idx,
            row=row,
            col=col,
            original_value=original,
            fault_metadata={
                "mode": mode.value,
                "seed": seed,
                "weight_shape": list(weight.shape),
                "effects": FAULT_LOCATION_EFFECTS.get(location, {}),
                **meta,
            },
        )
        print(f"[FaultInjector] Weight fault injected: {location.value} "
              f"layer={layer_idx} path={module_path} "
              f"[{row},{col}] mode={mode.value} bits={meta['bit_positions']}")
        return snapshot

    def restore_weight(self, snapshot: WeightFaultSnapshot) -> None:
        """Restore weight to its pre-fault value."""
        module = self._resolve_module(
            snapshot.location, snapshot.layer_idx, snapshot.module_path
        )
        with torch.no_grad():
            module.weight[snapshot.row, snapshot.col] = snapshot.original_value.to(
                module.weight.dtype
            )
        print(f"[FaultInjector] Weight restored: {snapshot.location.value} "
              f"[{snapshot.row},{snapshot.col}]")

    # ------------------------------------------------------------------
    # Activation fault (transient, hook-based)
    # ------------------------------------------------------------------

    def inject_activation_fault(
        self,
        location: FaultLocation,
        mode: FaultMode = FaultMode.SINGLE_BIT,
        layer_idx: Optional[int] = None,
        module_path: Optional[str] = None,
        trigger_once: bool = True,
        seed: Optional[int] = None,
    ) -> ActivationFaultHandle:
        """
        Register a forward hook that flips bits in the activation output.

        trigger_once=True  → fault fires once then becomes a no-op (瞬时故障)
        trigger_once=False → fault fires every forward pass (持续性故障)
        seed 控制激活 hook 内随机选 (seq_idx, feat_idx) 和 bit position 的随机性。
        """
        if seed is not None:
            random.seed(seed)
          To restrict to one phase, set a flag from outside:
              hook_state = {"phase_filter": "prefill_only", "triggered": False}
          and check it inside a custom hook_fn passed via inject_custom_hook().
        """
        module = self._resolve_module(location, layer_idx, module_path)
        hook_meta: Dict = {"triggered_count": 0, "fault_sites": []}

        def hook_fn(mod, inputs, output):
            nonlocal hook_meta
            if trigger_once and hook_meta["triggered_count"] >= 1:
                return output

            tensor = output[0] if isinstance(output, tuple) else output
            seq_dim  = tensor.shape[1]
            feat_dim = tensor.shape[2]
            x = random.randrange(seq_dim)
            y = random.randrange(feat_dim)

            original_val = tensor[0, x, y].clone()
            faulted_val, meta = apply_bit_fault(tensor[0, x, y], mode)

            tensor = tensor.clone()      # avoid in-place on autograd graph
            tensor[0, x, y] = faulted_val.to(tensor.dtype)

            hook_meta["triggered_count"] += 1
            hook_meta["fault_sites"].append({
                "seq_idx": x,
                "feat_idx": y,
                "original": original_val.item(),
                "faulted": faulted_val.item(),
                **meta,
            })

            return (tensor,) + output[1:] if isinstance(output, tuple) else tensor

        handle = module.register_forward_hook(hook_fn)

        afx = ActivationFaultHandle(
            location=location,
            module_path=module_path or "",
            layer_idx=layer_idx,
            mode=mode,
            _hook_handle=handle,
            metadata={
                "trigger_once": trigger_once,
                "hook_meta": hook_meta,
                "effects": FAULT_LOCATION_EFFECTS.get(location, {}),
            },
        )
        print(f"[FaultInjector] Activation fault registered: {location.value} "
              f"layer={layer_idx} path={module_path} mode={mode.value} once={trigger_once}")
        return afx

    def inject_custom_hook(
        self,
        location: FaultLocation,
        hook_fn,
        layer_idx: Optional[int] = None,
        module_path: Optional[str] = None,
    ) -> Any:
        """
        Attach a fully custom forward hook.
        Use this when you need to distinguish Step-2 vs Step-7 via external flags.
        Returns the raw hook handle.
        """
        module = self._resolve_module(location, layer_idx, module_path)
        return module.register_forward_hook(hook_fn)

    # ------------------------------------------------------------------
    # Random layer selection (one random module per layer, weighted)
    # ------------------------------------------------------------------

    def random_target_layer_fault(
        self,
        mode: FaultMode = FaultMode.DOUBLE_BIT,
        fault_type: str = "weight",   # "weight" or "activation"
    ):
        """
        Pick a random layer and a random submodule within it, inject fault.
        Weights are based on parameter count (larger matrices more likely).
        Returns snapshot or handle.
        """
        n_layers = len(self.target.model.layers)
        layer_idx = random.randint(0, n_layers - 1)

        # Weighted by parameter count (mirrors original code's approach)
        candidates = {
            "self_attn.q_proj": 7,
            "self_attn.k_proj": 1,
            "self_attn.v_proj": 1,
            "self_attn.o_proj": 7,
            "mlp.gate_proj":    37,
            "mlp.up_proj":      37,
            "mlp.down_proj":    37,
        }
        names  = list(candidates.keys())
        weights = list(candidates.values())
        path = random.choices(names, weights=weights)[0]

        if fault_type == "weight":
            return self.inject_weight_fault(
                FaultLocation.TARGET_LAYER, mode, layer_idx, path
            )
        else:
            return self.inject_activation_fault(
                FaultLocation.TARGET_LAYER, mode, layer_idx, path
            )

    def random_draft_fault(
        self,
        mode: FaultMode = FaultMode.SINGLE_BIT,
        fault_type: str = "weight",
    ):
        """
        Pick a random draft-only location and inject a fault.
        """
        draft_locations = [
            (FaultLocation.DRAFT_EMBED,  None, None),
            (FaultLocation.DRAFT_FC,     None, None),
            (FaultLocation.DRAFT_LAYER,  None, "self_attn.q_proj"),
            (FaultLocation.DRAFT_LAYER,  None, "mlp.gate_proj"),
        ]
        loc, layer_idx, path = random.choice(draft_locations)

        if fault_type == "weight":
            return self.inject_weight_fault(loc, mode, layer_idx, path)
        else:
            return self.inject_activation_fault(loc, mode, layer_idx, path)

    def inject_shared_lm_head_fault(
        self,
        mode: FaultMode = FaultMode.DOUBLE_BIT,
        fault_type: str = "weight",
    ):
        """
        Inject into the shared lm_head.
        This is the SDC blind-spot: acceptance_rate may look normal.
        """
        if fault_type == "weight":
            return self.inject_weight_fault(FaultLocation.SHARED_LM_HEAD, mode)
        else:
            return self.inject_activation_fault(FaultLocation.SHARED_LM_HEAD, mode)


# ============================================================
# Utility
# ============================================================

def _get_submodule(root: nn.Module, path: Optional[str]) -> nn.Module:
    if not path:
        return root
    m = root
    for attr in path.split("."):
        m = getattr(m, attr)
    return m


def print_fault_taxonomy():
    """Print expected effects for all fault locations. Useful for sanity check."""
    print("\n" + "=" * 70)
    print("EAGLE-3 Fault Location Taxonomy & Expected SDC Effects")
    print("=" * 70)
    for loc, effects in FAULT_LOCATION_EFFECTS.items():
        print(f"\n[{loc.value}]")
        for k, v in effects.items():
            print(f"  {k:25s}: {v}")
    print("=" * 70 + "\n")


# ============================================================
# Example experiment loop (shows how to use with the decoder)
# ============================================================

def run_fault_experiment(
    decoder,            # Eagle3SpeculativeDecoder instance
    input_ids: torch.Tensor,
    target_model: nn.Module,
    draft_head: nn.Module,
    location: FaultLocation,
    mode: FaultMode,
    layer_idx: Optional[int],
    module_path: Optional[str],
    fault_type: str = "weight",
    max_new_tokens: int = 200,
    eos_token_id: Optional[int] = None,
) -> Dict:
    """
    Run one fault injection experiment and return:
      - result from decoder.generate()
      - fault metadata
      - location effects description
    """
    injector = FaultInjector(target_model, draft_head)

    if fault_type == "weight":
        snapshot = injector.inject_weight_fault(location, mode, layer_idx, module_path)
        fault_log = snapshot.as_log()
    else:
        handle = injector.inject_activation_fault(location, mode, layer_idx, module_path)
        fault_log = handle.as_log()

    result = decoder.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )

    if fault_type == "weight":
        injector.restore_weight(snapshot)
    else:
        handle.remove()

    return {
        "fault_log": fault_log,
        "location_effects": FAULT_LOCATION_EFFECTS.get(location, {}),
        "generation_result": result,
    }


if __name__ == "__main__":
    print_fault_taxonomy()
