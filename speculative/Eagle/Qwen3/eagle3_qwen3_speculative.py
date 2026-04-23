"""
EAGLE-3 + Qwen3: Hand-rolled speculative decoding with SDC-oriented trace.
Pure HuggingFace Transformers, no vLLM / SGLang / EAGLE library.

Aligned with the vLLM scheduling diagram (知乎 @Ashan):
  Step 2  – Prefill forward (target model, recorded separately)
  Step 4  – Extract f_l / f_m / f_h from target
  Step 5  – Pass fused features to draft head
  Step 6  – Draft returns [t1..tγ]
  Step 7  – Verify phase: target runs ALL draft tokens in ONE batched forward
             (approximates Tree Attention; hand-rolled = sequential, but the
              trace records the joint verify pass as a single event)
  Step 9  – Per-position true probability distributions recorded
  Step 10 – Rejection sampling; per-position accept/reject recorded
  Step 11 – Effective KV length after accept/reject recorded

SDC-relevant trace fields (see dataclass schemas below):
  PrefillTraceEvent  : phase, input_len, f_l/f_m/f_h norms, top-layer hidden norm
  DraftTraceEvent    : phase, draft_token_id/str, draft_prob, draft_topk
                       draft_entropy, draft_hidden_norm
  VerifyTraceEvent   : phase, all γ positions at once —
                         per_position[i].target_prob
                         per_position[i].draft_prob
                         per_position[i].acceptance_ratio
                         per_position[i].accepted
                         per_position[i].target_topk (top-5)
                         per_position[i].kl_draft_target   ← anomaly signal
                       batch_verify_hidden_norms (one per position)
                       fused_feature_norms (f_l, f_m, f_h per position)
                       num_accepted, num_rejected
                       first_reject_pos  (None if all accepted)
  BridgeTraceEvent   : phase, base_token_id/str, base_prob

Fault injection hooks:
  Attach to target_model.model.layers[N] for Step-2 (Prefill) faults.
  Attach to target_model.model.layers[N] for Step-7 (Verify) faults.
  The code is hook-ready: call decoder.register_fault_hook(layer_idx, hook_fn).

Usage
-----
python eagle3_qwen3_speculative.py \\
    --base_model_id  Qwen/Qwen3-8B \\
    --draft_model_id RedHatAI/Qwen3-8B-Thinking-speculator.eagle3 \\
    --prompt "What is the capital of France?" \\
    --block_size 5 --temperature 0.0 --output_json trace.json
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# fault_injection.py は同じディレクトリに置く
# fault_injection.py 放在同一目录下即可直接 import
try:
    from fault_injection import FaultInjector, FaultLocation, FaultMode
    _FAULT_INJECTION_AVAILABLE = True
except ImportError:
    _FAULT_INJECTION_AVAILABLE = False

try:
    from datasets_loader import load_dataset as load_benchmark, extract_answer, is_correct
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

EPS = 1e-8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_RECORD = 5


# ============================================================
# Trace data schemas  (SDC detection reads these fields)
# ============================================================

@dataclass
class PositionVerifyData:
    """Per-token-position data recorded during the Step-7 verify pass."""
    pos: int
    draft_token_id: int
    draft_token: str
    draft_prob: float
    target_prob: float
    acceptance_ratio: float
    accepted: bool
    target_topk: List[Dict]          # top-5 tokens from target distribution
    kl_draft_target: float           # KL(draft || target) – SDC anomaly signal
    target_hidden_norm: float        # L2 norm of target hidden state here
    fused_feature_norm: float        # filled in after fuse step


@dataclass
class PrefillTraceEvent:
    phase: str = "prefill"
    iteration: int = 0
    input_len: int = 0
    elapsed_s: float = 0.0
    f_early_norm: float = 0.0        # Step 4: norm of early-layer feature
    f_mid_norm: float = 0.0          # Step 4: norm of mid-layer feature
    f_late_norm: float = 0.0         # Step 4: norm of late-layer feature
    top_hidden_norm: float = 0.0
    prefill_topk: List[Dict] = field(default_factory=list)


@dataclass
class DraftTraceEvent:
    phase: str = "draft"
    iteration: int = 0
    draft_step: int = 0
    draft_token_id: int = 0
    draft_token: str = ""
    draft_prob: float = 0.0
    draft_entropy: float = 0.0       # entropy of draft distribution
    draft_topk: List[Dict] = field(default_factory=list)
    draft_hidden_norm: float = 0.0   # norm of draft layer output
    elapsed_s: float = 0.0


@dataclass
class VerifyTraceEvent:
    """Step 7: one event covers the entire draft block verification."""
    phase: str = "verify"
    iteration: int = 0
    block_size_proposed: int = 0
    num_accepted: int = 0
    num_rejected: int = 0
    first_reject_pos: Optional[int] = None
    acceptance_rate_this_block: float = 0.0
    per_position: List[Dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    mean_kl_draft_target: float = 0.0
    max_kl_draft_target: float = 0.0
    mean_target_hidden_norm: float = 0.0
    effective_kv_len_after: int = 0  # Step 11: context length after accept/reject


@dataclass
class BridgeTraceEvent:
    phase: str = "bridge"
    iteration: int = 0
    base_token_id: int = 0
    base_token: str = ""
    base_prob: float = 0.0
    base_topk: List[Dict] = field(default_factory=list)
    elapsed_s: float = 0.0


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(temperature, 1e-6)
    return torch.softmax(logits / t, dim=-1).float()


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    do_sample: bool,
) -> Tuple[int, float]:
    probs = _softmax(logits, temperature)
    if not do_sample or temperature <= 0:
        token_id = logits.argmax(dim=-1).item()
    else:
        topk_p, topk_i = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(topk_p, 1)
        token_id = topk_i.gather(-1, idx).item()
    return token_id, probs[0, token_id].item()


def topk_info(probs: torch.Tensor, tokenizer, k: int = TOP_K_RECORD) -> List[Dict]:
    k = min(k, probs.shape[-1])
    vals, idxs = torch.topk(probs, k, dim=-1)
    return [
        {"token_id": idxs[0, i].item(),
         "token": tokenizer.decode([idxs[0, i].item()]),
         "prob": vals[0, i].item()}
        for i in range(k)
    ]


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q), both (1, vocab)."""
    p = p.float().clamp(min=EPS)
    q = q.float().clamp(min=EPS)
    return (p * (p / q).log()).sum().item()


def entropy(probs: torch.Tensor) -> float:
    p = probs.float().clamp(min=EPS)
    return -(p * p.log()).sum().item()


# ============================================================
# Target model with layer taps (Steps 2, 4, 7)
# ============================================================

class TargetModelWithTaps:
    """
    Wraps LLaMA target model.
    Registers forward hooks on three layers to collect f_l, f_m, f_h (Step 4).
    Exposes prefill() and verify() for clean trace separation.

    Phase tracking
    --------------
    self.current_phase 在每次 prefill() / verify() 调用前后自动切换：
      "prefill"  – 正在执行 Step-2
      "verify"   – 正在执行 Step-7
      "idle"     – 两者之外

    fault hook 可以读这个字段来决定要不要触发，从而实现只注入某一个阶段。
    用 register_phase_aware_fault_hook() 时这一切都自动处理好了。
    """

    def __init__(self, model: nn.Module, tap_indices: List[int]):
        self.model = model
        self.tap_indices = sorted(tap_indices)
        self._tapped: Dict[int, torch.Tensor] = {}
        self._fault_hooks: List[Any] = []
        self._tap_hooks: List[Any] = []
        self.current_phase: str = "idle"   # ← 当前正在跑哪个阶段
        self._register_tap_hooks()

    def _register_tap_hooks(self) -> None:
        layers = self.model.model.layers
        for idx in self.tap_indices:
            def make_hook(i):
                def hook(_, __, output):
                    h = output[0] if isinstance(output, tuple) else output
                    self._tapped[i] = h.detach().clone()
                return hook
            self._tap_hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    def register_phase_aware_fault_hook(
        self,
        layer_idx: int,
        hook_fn,
        phase_filter: str = "both",   # "prefill" | "verify" | "both"
    ) -> Any:
        """
        把 hook_fn 包一层 phase 检查再注册到 target layer `layer_idx`。

        phase_filter:
          "prefill" → hook 只在 Step-2 触发
          "verify"  → hook 只在 Step-7 触发
          "both"    → 两个阶段都触发（和之前的 register_fault_hook 一样）

        用法示例：
            handle = target_wrapped.register_phase_aware_fault_hook(
                layer_idx=16,
                hook_fn=my_bit_flip_fn,
                phase_filter="verify",   # 只在 Step-7 注入
            )
            # 跑推理 ...
            handle.remove()
        """
        target_self = self   # 闭包里要用 self.current_phase

        def wrapped_hook(module, inputs, output):
            if phase_filter == "both" or target_self.current_phase == phase_filter:
                return hook_fn(module, inputs, output)
            return output   # 不是目标阶段，直接透传

        handle = self.model.model.layers[layer_idx].register_forward_hook(wrapped_hook)
        self._fault_hooks.append(handle)
        return handle

    def remove_fault_hooks(self) -> None:
        for h in self._fault_hooks:
            h.remove()
        self._fault_hooks.clear()

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """
        Step 2+4: full forward pass on the current context.
        Records f_l, f_m, f_h at last token position.
        current_phase 在执行期间被设为 "prefill"。
        """
        self._tapped.clear()
        self.current_phase = "prefill"
        t0 = time.time()
        out = self.model(input_ids, use_cache=False, output_hidden_states=False)
        elapsed = time.time() - t0
        self.current_phase = "idle"

        logits = out.logits[:, -1, :]
        tapped = {i: self._tapped[i][:, -1:, :].clone() for i in self.tap_indices}
        return {"logits": logits, "tapped": tapped, "elapsed_s": elapsed, "input_len": input_ids.shape[1]}

    @torch.no_grad()
    def verify(self, context_ids: torch.Tensor, draft_token_ids: List[int]) -> Dict[str, Any]:
        """
        Step 7: ONE batched forward that covers context + all γ draft tokens.
        Per-position logits and hidden states are sliced out at the end.
        current_phase 在执行期间被设为 "verify"。
        """
        γ = len(draft_token_ids)
        draft_t = torch.tensor([draft_token_ids], device=DEVICE)
        full_ids = torch.cat([context_ids, draft_t], dim=1)

        self._tapped.clear()
        self.current_phase = "verify"
        t0 = time.time()
        out = self.model(full_ids, use_cache=False, output_hidden_states=False)
        elapsed = time.time() - t0
        self.current_phase = "idle"

        ctx_len = context_ids.shape[1]
        verify_logits = out.logits[:, ctx_len - 1: ctx_len - 1 + γ, :]   # (1, γ, vocab)

        top_idx = max(self.tap_indices)
        top_hidden = self._tapped.get(top_idx)
        verify_hiddens = None
        if top_hidden is not None:
            verify_hiddens = top_hidden[:, ctx_len - 1: ctx_len - 1 + γ, :]  # (1, γ, H)

        return {"verify_logits": verify_logits, "verify_hiddens": verify_hiddens, "elapsed_s": elapsed}


# ============================================================
# EAGLE-3 Draft Head
# ============================================================

class Eagle3DraftHead(nn.Module):
    """
    Single-layer EAGLE-3 draft head loaded from official checkpoint.
    Input  : token embedding  +  fused (f_l, f_m, f_h) from target
    Output : token logits (LM head shared with target)
    """

    def __init__(self, draft_ckpt_path: str, target_model: nn.Module):
        super().__init__()
        cfg = target_model.config
        H = cfg.hidden_size
        n = len(target_model.model.layers)
        self.tap_indices = [n // 4, n // 2, n - 1]

        self.fc = nn.Linear(4 * H, H, bias=False)

        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        self.draft_layer = Qwen3DecoderLayer(cfg, layer_idx=0)

        self.embed_tokens = nn.Embedding(cfg.vocab_size, H)
        self.lm_head = target_model.lm_head  # shared, no extra params

        self._load_weights(draft_ckpt_path)
        self.to(DEVICE)
        self.eval()

    def _load_weights(self, ckpt_path: str) -> None:
        import glob
        files = glob.glob(os.path.join(ckpt_path, "*.safetensors"))
        if files:
            from safetensors.torch import load_file
            state = {}
            for f in files:
                state.update(load_file(f, device="cpu"))
        else:
            files = glob.glob(os.path.join(ckpt_path, "*.bin"))
            state = {}
            for f in files:
                state.update(torch.load(f, map_location="cpu"))

        def strip(k):
            for p in ("eagle_model.", "model.", ""):
                if k.startswith(p):
                    return k[len(p):]
            return k

        s = {strip(k): v for k, v in state.items()}
        if "fc.weight" in s:
            self.fc.weight = nn.Parameter(s["fc.weight"])
        if "embed_tokens.weight" in s:
            self.embed_tokens.weight = nn.Parameter(s["embed_tokens.weight"])
        layer_state = {k[len("layers.0."):]: v for k, v in s.items() if k.startswith("layers.0.")}
        missing, _ = self.draft_layer.load_state_dict(layer_state, strict=False)
        if missing:
            print(f"[DraftHead] Missing keys (first 5): {missing[:5]}")

    def fuse(self, tapped: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Step 5: project [f_l, f_m, f_h, f_h] → hidden_size.
        Returns (fused (1,1,H), fused_norm).
        """
        idxs = sorted(tapped.keys())
        parts = [tapped[i] for i in idxs]
        while len(parts) < 4:
            parts.append(parts[-1])
        cat = torch.cat(parts[:4], dim=-1)
        fused = self.fc(cat)
        return fused, fused.float().norm().item()

    @torch.no_grad()
    def forward_step(
        self,
        prev_token_id: int,
        fused_hidden: torch.Tensor,
        past_key_values: Optional[Any],
        position_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """One draft autoregressive step. Returns (logits, hidden, new_pkv)."""
        token_t = torch.tensor([[prev_token_id]], device=DEVICE)
        embed = self.embed_tokens(token_t)
        x = embed + fused_hidden
        pos = torch.tensor([[position_id]], device=DEVICE)
        layer_out = self.draft_layer(x, position_ids=pos, past_key_value=past_key_values, use_cache=True)
        hidden = layer_out[0]
        new_pkv = layer_out[1] if len(layer_out) > 1 else None
        logits = self.lm_head(hidden[:, -1, :])
        return logits, hidden[:, 0, :], new_pkv


# ============================================================
# Main Decoder
# ============================================================

class Eagle3SpeculativeDecoder:
    """
    EAGLE-3 speculative decoder with full SDC-oriented trace.

    Per iteration the trace contains (in order):
      PrefillTraceEvent    – Step 2+4  (target features, norms)
      DraftTraceEvent × γ  – Step 5+6  (draft token, entropy, hidden norm)
      VerifyTraceEvent     – Step 7+9+10  (per-position KL, accept/reject, hidden norms)
      BridgeTraceEvent     – Step 12 / bridge token (if all accepted)

    Key SDC signals in VerifyTraceEvent.per_position[i]:
      kl_draft_target       – spikes under fault (distribution shift)
      acceptance_ratio      – drops monotonically under fault
      target_hidden_norm    – deviates under hardware fault
    """

    def __init__(
        self,
        target: TargetModelWithTaps,
        draft_head: Eagle3DraftHead,
        tokenizer: AutoTokenizer,
        block_size: int = 5,
        temperature: float = 0.0,
        top_k: int = 50,
        do_sample: bool = False,
    ):
        self.target = target
        self.draft = draft_head
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.temperature = temperature
        self.top_k = top_k
        self.do_sample = do_sample

    def register_fault_hook(
        self,
        layer_idx: int,
        hook_fn,
        phase_filter: str = "both",   # "prefill" | "verify" | "both"
    ) -> Any:
        """
        把 hook_fn 注册到 target model 的第 layer_idx 层。

        phase_filter 控制在哪个阶段触发：
          "prefill" → 只在 Step-2 (Prefill) 时 hook 生效
          "verify"  → 只在 Step-7 (Verify) 时 hook 生效
          "both"    → 两个阶段都触发

        返回 hook handle，用完后调用 handle.remove() 撤销。
        """
        return self.target.register_phase_aware_fault_hook(layer_idx, hook_fn, phase_filter)

    # ------------------------------------------------------------------
    # Step 5+6: Draft proposal
    # ------------------------------------------------------------------
    def _propose(
        self,
        context_ids: torch.Tensor,
        fused: torch.Tensor,
        fused_norm: float,
        eos_token_id: Optional[int],
        iteration: int,
    ) -> Tuple[List[Dict], List[DraftTraceEvent]]:
        proposals = []
        draft_events = []
        pkv = None
        prev_token = context_ids[0, -1].item()
        pos_offset = context_ids.shape[1] - 1

        for step in range(self.block_size):
            t0 = time.time()
            logits, hidden, pkv = self.draft.forward_step(
                prev_token_id=prev_token,
                fused_hidden=fused,
                past_key_values=pkv,
                position_id=pos_offset + step,
            )
            elapsed = time.time() - t0

            probs = _softmax(logits, self.temperature)
            token_id, token_prob = sample_token(logits, self.temperature, self.top_k, self.do_sample)

            proposals.append({"token_id": token_id, "prob": token_prob, "probs": probs.detach()})
            draft_events.append(DraftTraceEvent(
                iteration=iteration,
                draft_step=step,
                draft_token_id=token_id,
                draft_token=self.tokenizer.decode([token_id]),
                draft_prob=token_prob,
                draft_entropy=entropy(probs),
                draft_topk=topk_info(probs, self.tokenizer),
                draft_hidden_norm=hidden.float().norm().item(),
                elapsed_s=elapsed,
            ))

            prev_token = token_id
            if eos_token_id is not None and token_id == eos_token_id:
                break

        return proposals, draft_events

    # ------------------------------------------------------------------
    # Step 7+9+10: Batched verify + rejection sampling
    # ------------------------------------------------------------------
    def _verify_and_sample(
        self,
        context_ids: torch.Tensor,
        proposals: List[Dict],
        fused_norm: float,
        iteration: int,
    ) -> Tuple[List[int], torch.Tensor, VerifyTraceEvent]:
        draft_token_ids = [p["token_id"] for p in proposals]
        γ = len(proposals)

        # Step 7: ONE batched forward
        verify_out = self.target.verify(context_ids, draft_token_ids)
        verify_logits  = verify_out["verify_logits"]    # (1, γ, vocab)
        verify_hiddens = verify_out["verify_hiddens"]   # (1, γ, H) or None

        per_position: List[PositionVerifyData] = []
        accepted_ids: List[int] = []
        first_reject: Optional[int] = None

        for i, proposal in enumerate(proposals):
            pos_logits    = verify_logits[:, i, :]
            target_probs  = _softmax(pos_logits, self.temperature)
            draft_probs   = proposal["probs"]

            target_prob   = target_probs[0, proposal["token_id"]].item()
            accept_ratio  = min(1.0, (target_prob + EPS) / (proposal["prob"] + EPS))
            accepted      = (not self.do_sample) or (random.random() < accept_ratio)

            kl    = kl_divergence(draft_probs, target_probs)
            h_norm = verify_hiddens[0, i, :].float().norm().item() if verify_hiddens is not None else 0.0

            per_position.append(PositionVerifyData(
                pos=i,
                draft_token_id=proposal["token_id"],
                draft_token=self.tokenizer.decode([proposal["token_id"]]),
                draft_prob=proposal["prob"],
                target_prob=target_prob,
                acceptance_ratio=accept_ratio,
                accepted=accepted,
                target_topk=topk_info(target_probs, self.tokenizer),
                kl_draft_target=kl,
                target_hidden_norm=h_norm,
                fused_feature_norm=fused_norm,
            ))

            token_t = torch.tensor([[proposal["token_id"]]], device=DEVICE)
            if accepted:
                accepted_ids.append(proposal["token_id"])
                context_ids = torch.cat([context_ids, token_t], dim=1)
            else:
                if first_reject is None:
                    first_reject = i
                base_id, _ = sample_token(pos_logits, self.temperature, self.top_k, self.do_sample)
                accepted_ids.append(base_id)
                context_ids = torch.cat([context_ids, torch.tensor([[base_id]], device=DEVICE)], dim=1)
                break

        n_acc  = sum(1 for p in per_position if p.accepted)
        n_rej  = len(per_position) - n_acc
        kls    = [p.kl_draft_target for p in per_position]
        hnorms = [p.target_hidden_norm for p in per_position]

        event = VerifyTraceEvent(
            iteration=iteration,
            block_size_proposed=γ,
            num_accepted=n_acc,
            num_rejected=n_rej,
            first_reject_pos=first_reject,
            acceptance_rate_this_block=n_acc / γ if γ > 0 else 0.0,
            per_position=[asdict(p) for p in per_position],
            elapsed_s=verify_out["elapsed_s"],
            mean_kl_draft_target=sum(kls) / len(kls) if kls else 0.0,
            max_kl_draft_target=max(kls) if kls else 0.0,
            mean_target_hidden_norm=sum(hnorms) / len(hnorms) if hnorms else 0.0,
            effective_kv_len_after=context_ids.shape[1],   # Step 11
        )
        return accepted_ids, context_ids, event

    # ------------------------------------------------------------------
    # Main generate loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, Any]:

        context_ids    = input_ids.clone()
        generated_ids: List[int] = []
        all_events:    List[Dict] = []

        metrics = {
            "proposals_generated":   0,
            "accepted_proposals":    0,
            "rejected_proposals":    0,
            "base_only_tokens":      0,
            "draft_forward_calls":   0,
            "base_forward_calls":    0,
            "draft_time":            0.0,
            "base_time":             0.0,
            "iteration_count":       0,
        }
        start_time = time.time()
        iteration  = 0

        while len(generated_ids) < max_new_tokens:
            iteration += 1
            metrics["iteration_count"] += 1

            # ---- Step 2+4: Prefill ----------------------------------------
            prefill_out = self.target.prefill(context_ids)
            metrics["base_forward_calls"] += 1
            metrics["base_time"]          += prefill_out["elapsed_s"]

            tapped = prefill_out["tapped"]
            idxs   = sorted(tapped.keys())
            f_e = tapped[idxs[0]]
            f_m = tapped[idxs[1]] if len(idxs) > 1 else f_e
            f_l = tapped[idxs[-1]]

            prefill_probs = _softmax(prefill_out["logits"], self.temperature)
            all_events.append(asdict(PrefillTraceEvent(
                iteration=iteration,
                input_len=prefill_out["input_len"],
                elapsed_s=prefill_out["elapsed_s"],
                f_early_norm=f_e.float().norm().item(),
                f_mid_norm=f_m.float().norm().item(),
                f_late_norm=f_l.float().norm().item(),
                top_hidden_norm=f_l.float().norm().item(),
                prefill_topk=topk_info(prefill_probs, self.tokenizer),
            )))

            # ---- Step 5: Fuse features ------------------------------------
            fused, fused_norm = self.draft.fuse(tapped)

            # ---- Step 6: Draft proposals ----------------------------------
            t0 = time.time()
            proposals, draft_events = self._propose(
                context_ids, fused, fused_norm, eos_token_id, iteration
            )
            metrics["draft_time"]         += time.time() - t0
            metrics["draft_forward_calls"] += len(proposals)
            metrics["proposals_generated"] += len(proposals)

            for de in draft_events:
                all_events.append(asdict(de))

            if not proposals:
                break

            # ---- Step 7+9+10: Batched verify + rejection sampling ---------
            accepted_ids, context_ids, verify_event = self._verify_and_sample(
                context_ids, proposals, fused_norm, iteration
            )
            metrics["base_forward_calls"] += 1
            metrics["base_time"]          += verify_event.elapsed_s
            metrics["accepted_proposals"] += verify_event.num_accepted
            metrics["rejected_proposals"] += verify_event.num_rejected
            metrics["base_only_tokens"]   += verify_event.num_rejected

            all_events.append(asdict(verify_event))
            generated_ids.extend(accepted_ids)

            if eos_token_id is not None and generated_ids and generated_ids[-1] == eos_token_id:
                break
            if len(generated_ids) >= max_new_tokens:
                break

            # ---- Step 12 / Bridge: if all γ accepted, get one more token --
            if verify_event.num_rejected == 0:
                t0 = time.time()
                bridge_out = self.target.prefill(context_ids)
                metrics["base_forward_calls"] += 1
                metrics["base_time"]          += time.time() - t0

                bridge_id, bridge_prob = sample_token(
                    bridge_out["logits"], self.temperature, self.top_k, self.do_sample
                )
                bridge_probs = _softmax(bridge_out["logits"], self.temperature)
                all_events.append(asdict(BridgeTraceEvent(
                    iteration=iteration,
                    base_token_id=bridge_id,
                    base_token=self.tokenizer.decode([bridge_id]),
                    base_prob=bridge_prob,
                    base_topk=topk_info(bridge_probs, self.tokenizer),
                    elapsed_s=bridge_out["elapsed_s"],
                )))
                context_ids = torch.cat([context_ids, torch.tensor([[bridge_id]], device=DEVICE)], dim=1)
                generated_ids.append(bridge_id)
                metrics["base_only_tokens"] += 1

                if eos_token_id is not None and bridge_id == eos_token_id:
                    break

        metrics["generation_time"] = time.time() - start_time
        metrics["acceptance_rate"] = (
            metrics["accepted_proposals"] / metrics["proposals_generated"]
            if metrics["proposals_generated"] > 0 else 0.0
        )
        metrics["tokens_emitted"] = len(generated_ids)

        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {"text": decoded, "tokens": generated_ids, "trace": all_events, "metrics": metrics}


# ============================================================
# Load helpers
# ============================================================

def load_target_model(model_id: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
    model.eval()
    return model, tok


def load_draft_head(draft_id: str, target_model: nn.Module, dtype: torch.dtype) -> Eagle3DraftHead:
    from huggingface_hub import snapshot_download
    local = draft_id if os.path.isdir(draft_id) else snapshot_download(draft_id)
    return Eagle3DraftHead(draft_ckpt_path=local, target_model=target_model).to(dtype)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id",  default="Qwen/Qwen3-8B")
    parser.add_argument("--draft_model_id", default="RedHatAI/Qwen3-8B-Thinking-speculator.eagle3")
    parser.add_argument("--block_size",     type=int,   default=5)
    parser.add_argument("--max_new_tokens", type=int,   default=4096)
    parser.add_argument("--temperature",    type=float, default=0.6,
                        help="Qwen3 thinking 모드 권장값: 0.6. 0으로 하면 반복 루프 위험.")
    parser.add_argument("--top_k",          type=int,   default=20,
                        help="Qwen3 thinking 모드 권장값: 20.")
    parser.add_argument("--top_p",          type=float, default=0.95,
                        help="Qwen3 thinking 모드 권장값: 0.95.")
    parser.add_argument("--enable_thinking", action="store_true", default=True,
                        help="Qwen3 thinking 모드 활성화. 기본 True.")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--dtype",          default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--output_json",    default=None)
    # ---- 数据集 / 单题 ----
    parser.add_argument("--dataset",        default=None,
                        choices=["gsm8k", "math500", "aime2024", "aime2025",
                                 "gpqa", "livecodebench", "openthoughts"],
                        help="传入数据集名称则批量跑；不传则用 --prompt 单题模式。")
    parser.add_argument("--num_samples",    type=int, default=None,
                        help="每个数据集最多取多少题。None = 全部。")
    parser.add_argument("--prompt",         default="What is 25 * 48?",
                        help="单题模式下使用的问题（--dataset 不传时生效）。")
    # ---- fault injection ----
    parser.add_argument("--fault_location", default=None,
                        choices=["target_layer", "target_embed",
                                 "draft_embed", "draft_fc", "draft_layer",
                                 "shared_lm_head"],
                        help="注入位置。不填则跑 baseline。")
    parser.add_argument("--fault_mode",     default="double_bit",
                        choices=["single_bit", "double_bit", "stuck_at_0"])
    parser.add_argument("--fault_type",     default="weight",
                        choices=["weight", "activation"],
                        help="weight=持久性权重故障  activation=瞬时激活故障")
    parser.add_argument("--fault_layer_idx", type=int, default=None,
                        help="target_layer / draft_layer 时指定层号。不填则随机选。")
    parser.add_argument("--fault_module",   default=None,
                        help="层内子模块路径，如 mlp.gate_proj。不填则随机选。")
    parser.add_argument("--fault_phase",    default="both",
                        choices=["prefill", "verify", "both"],
                        help="activation fault 专用：限定在哪个阶段触发。")
    parser.add_argument("--fault_seed",     type=int, default=None,
                        help="控制注入点随机性的 seed。不填则由 --seed 统一控制。")
    args = parser.parse_args()

    seed_everything(args.seed)
    # thinking 모드에서는 greedy 금지 — temperature > 0 이면 항상 sampling
    do_sample = args.temperature > 0
    if args.enable_thinking and not do_sample:
        print("[Warning] Qwen3 thinking 모드에서 greedy decoding은 권장하지 않습니다. "
              "temperature=0.6 으로 자동 설정합니다.")
        args.temperature = 0.6
        do_sample = True
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"Loading target : {args.base_model_id}")
    target_model, tokenizer = load_target_model(args.base_model_id, dtype)

    print(f"Loading draft  : {args.draft_model_id}")
    draft_head = load_draft_head(args.draft_model_id, target_model, dtype)

    tap_indices = draft_head.tap_indices
    print(f"Tapping layers : {tap_indices}")

    target_wrapped = TargetModelWithTaps(target_model, tap_indices)
    decoder = Eagle3SpeculativeDecoder(
        target=target_wrapped, draft_head=draft_head, tokenizer=tokenizer,
        block_size=args.block_size, temperature=args.temperature,
        top_k=args.top_k, do_sample=do_sample,
    )

    # ----------------------------------------------------------------
    # Fault injection 准备（注入一次，整个实验期间保持）
    # ----------------------------------------------------------------
    injector = None
    weight_snapshot = None
    activation_handle = None
    fault_log = None

    if args.fault_location is not None:
        if not _FAULT_INJECTION_AVAILABLE:
            raise RuntimeError("fault_injection.py not found.")

        injector = FaultInjector(target_model, draft_head)
        location = FaultLocation(args.fault_location)
        mode     = FaultMode(args.fault_mode)

        if args.fault_type == "weight":
            weight_snapshot = injector.inject_weight_fault(
                location=location, mode=mode,
                layer_idx=args.fault_layer_idx,
                module_path=args.fault_module,
                seed=args.fault_seed,
            )
            fault_log = weight_snapshot.as_log()
            print(f"[Fault] Weight fault injected: {fault_log}")

        else:
            if location == FaultLocation.TARGET_LAYER:
                layer_idx = args.fault_layer_idx
                if layer_idx is None:
                    rng = random.Random(args.fault_seed)
                    layer_idx = rng.randint(0, len(target_model.model.layers) - 1)

                _hook_rng = random.Random(args.fault_seed)

                def _bit_flip_hook(module, inputs, output):
                    tensor = output[0] if isinstance(output, tuple) else output
                    x = _hook_rng.randrange(tensor.shape[1])
                    y = _hook_rng.randrange(tensor.shape[2])
                    val = tensor[0, x, y]
                    if mode == FaultMode.SINGLE_BIT:
                        bit = _hook_rng.randint(0, 15)
                        flipped = val.to(torch.bfloat16).view(torch.int16) ^ (1 << bit)
                        tensor = tensor.clone()
                        tensor[0, x, y] = flipped.view(torch.bfloat16).to(tensor.dtype)
                    elif mode == FaultMode.DOUBLE_BIT:
                        b0, b1 = _hook_rng.sample(range(16), 2)
                        flipped = val.to(torch.bfloat16).view(torch.int16) ^ ((1 << b0) | (1 << b1))
                        tensor = tensor.clone()
                        tensor[0, x, y] = flipped.view(torch.bfloat16).to(tensor.dtype)
                    return (tensor,) + output[1:] if isinstance(output, tuple) else tensor

                activation_handle = decoder.register_fault_hook(
                    layer_idx=layer_idx, hook_fn=_bit_flip_hook,
                    phase_filter=args.fault_phase,
                )
                fault_log = {
                    "location": location.value, "layer_idx": layer_idx,
                    "mode": mode.value, "phase_filter": args.fault_phase,
                    "fault_seed": args.fault_seed,
                }
            else:
                activation_handle = injector.inject_activation_fault(
                    location=location, mode=mode,
                    layer_idx=args.fault_layer_idx,
                    module_path=args.fault_module,
                    seed=args.fault_seed,
                )
                fault_log = activation_handle.as_log()
            print(f"[Fault] registered: {fault_log}")

    # ----------------------------------------------------------------
    # 构造样本列表
    # 传了 --dataset → 批量；否则用 --prompt 单题
    # ----------------------------------------------------------------
    if args.dataset is not None:
        if not _DATASETS_AVAILABLE:
            raise RuntimeError("datasets_loader.py not found.")
        samples = load_benchmark(args.dataset, num_samples=args.num_samples, seed=args.seed)
    else:
        samples = [{"question": args.prompt, "answer": "", "source": "single", "sample_id": 0}]

    # ----------------------------------------------------------------
    # 主循环：对每道题跑 generate
    # ----------------------------------------------------------------
    all_results = []
    n_correct = 0

    for sample in samples:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)

        result = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

        correct = False
        if sample["answer"] and args.dataset is not None:
            correct = is_correct(result["text"], sample["answer"], args.dataset)
            if correct:
                n_correct += 1

        entry = {
            "sample_id":  sample["sample_id"],
            "source":     sample["source"],
            "question":   sample["question"],
            "reference":  sample["answer"],
            "prediction": result["text"],
            "is_correct": correct,
            "metrics":    result["metrics"],
            "trace":      result["trace"],
        }
        if fault_log:
            entry["fault_log"] = fault_log
        all_results.append(entry)

        m = result["metrics"]
        status = "✓" if correct else "✗"
        print(f"[{sample['sample_id']:4d}] {status} "
              f"accept={m['acceptance_rate']:.3f}  "
              f"tokens={m['tokens_emitted']}")

    # ----------------------------------------------------------------
    # 清理 fault
    # ----------------------------------------------------------------
    if weight_snapshot is not None:
        injector.restore_weight(weight_snapshot)
    if activation_handle is not None:
        activation_handle.remove()

    # ----------------------------------------------------------------
    # 汇总输出
    # ----------------------------------------------------------------
    total = len(all_results)
    avg_accept = sum(r["metrics"]["acceptance_rate"] for r in all_results) / total
    print("=" * 60)
    if args.dataset is not None:
        print(f"Dataset  : {args.dataset}  ({total} samples)")
        print(f"Accuracy : {n_correct}/{total} = {n_correct/total:.4f}")
    print(f"Avg acceptance rate : {avg_accept:.4f}")
    print("=" * 60)

    if args.output_json:
        summary = {
            "dataset":             args.dataset or "single",
            "fault_log":           fault_log,
            "total":               total,
            "n_correct":           n_correct,
            "accuracy":            n_correct / total if total > 0 else 0.0,
            "avg_acceptance_rate": avg_accept,
            "results":             all_results,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Results → {args.output_json}")


if __name__ == "__main__":
    main()
