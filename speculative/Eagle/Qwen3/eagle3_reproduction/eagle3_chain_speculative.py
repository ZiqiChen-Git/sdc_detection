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
from types import SimpleNamespace
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
    raw_draft_prob: float            # unfiltered model prob, for diagnosis only
    raw_target_prob: float           # unfiltered model prob, for diagnosis only
    raw_acceptance_ratio: float      # diagnostic ratio; not used for sampling
    raw_target_topk: List[Dict]      # top-5 tokens before top-k/top-p filtering
    raw_kl_draft_target: float       # KL on unfiltered model distributions
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
    raw_prefill_topk: List[Dict] = field(default_factory=list)


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
    raw_draft_prob: float = 0.0
    raw_draft_entropy: float = 0.0
    raw_draft_topk: List[Dict] = field(default_factory=list)
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
    raw_base_prob: float = 0.0
    raw_base_topk: List[Dict] = field(default_factory=list)
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


def sampling_probs(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    do_sample: bool,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    if not do_sample or temperature <= 0:
        return torch.softmax(logits, dim=-1).float()

    scores = logits.float() / max(temperature, 1e-6)

    if top_k is not None and 0 < top_k < scores.shape[-1]:
        kth = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < kth, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = scores.scatter(dim=-1, index=sorted_idx, src=sorted_scores)

    return torch.softmax(scores, dim=-1).float()


def diagnostic_probs(logits: torch.Tensor) -> torch.Tensor:
    """Unfiltered model distribution for trace diagnostics; not used for sampling."""
    return torch.softmax(logits.float(), dim=-1).float()


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    do_sample: bool,
    top_p: Optional[float] = None,
) -> Tuple[int, float]:
    probs = sampling_probs(logits, temperature, top_k, do_sample, top_p)
    if not do_sample or temperature <= 0:
        token_id = logits.argmax(dim=-1).item()
    else:
        token_id = torch.multinomial(probs, 1).item()
    return token_id, probs[0, token_id].item()


def sample_from_probs(probs: torch.Tensor) -> Tuple[int, float]:
    probs = probs.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=EPS)
    token_id = torch.multinomial(probs, 1).item()
    return token_id, probs[0, token_id].item()


def speculative_replacement_probs(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    delta = torch.clamp(target_probs.float() - draft_probs.float(), min=0.0)
    total = delta.sum(dim=-1, keepdim=True)
    if torch.all(total <= EPS):
        return target_probs.float() / target_probs.float().sum(dim=-1, keepdim=True).clamp(min=EPS)
    return delta / total.clamp(min=EPS)


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


class EagleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class EagleMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers.activations import ACT2FN

        bias = getattr(cfg, "mlp_bias", False)
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=bias)
        self.act_fn = ACT2FN[cfg.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class EagleRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=position_ids.device)
        freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


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
        tapped_full = {i: self._tapped[i].clone() for i in self.tap_indices}
        tapped = {i: h[:, -1:, :].clone() for i, h in tapped_full.items()}
        return {
            "logits": logits,
            "tapped": tapped,
            "tapped_full": tapped_full,
            "elapsed_s": elapsed,
            "input_len": input_ids.shape[1],
        }

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

class Eagle3DraftLayer(nn.Module):
    """
    EAGLE-3 custom draft decoder layer.

    RedHat's Qwen3 EAGLE-3 speculator stores a Llama-style draft layer even
    though the target model is Qwen3. Build from the speculator config, not
    from the target config, while keeping the EAGLE 2H attention input.
    """

    def __init__(self, draft_cfg):
        super().__init__()
        H = draft_cfg.hidden_size
        n_heads = draft_cfg.num_attention_heads
        n_kv = getattr(draft_cfg, "num_key_value_heads", n_heads)
        head_dim = getattr(draft_cfg, "head_dim", H // n_heads)
        rms_eps = getattr(draft_cfg, "rms_norm_eps", 1e-6)
        attn_bias = getattr(draft_cfg, "attention_bias", False)
        self.model_type = getattr(draft_cfg, "model_type", "llama")
        self.norm_before_residual = getattr(draft_cfg, "norm_before_residual", False)
        self.H = H
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim

        class _Attn(nn.Module):
            pass
        self.self_attn = _Attn()
        # q: 2H -> n_heads*head_dim
        self.self_attn.q_proj = nn.Linear(2 * H, n_heads * head_dim, bias=attn_bias)
        # k,v: 2H -> n_kv*head_dim
        self.self_attn.k_proj = nn.Linear(2 * H, n_kv * head_dim, bias=attn_bias)
        self.self_attn.v_proj = nn.Linear(2 * H, n_kv * head_dim, bias=attn_bias)
        # o: n_heads*head_dim -> H
        self.self_attn.o_proj = nn.Linear(n_heads * head_dim, H, bias=attn_bias)

        if self.model_type.lower().startswith("qwen3"):
            self.self_attn.q_norm = EagleRMSNorm(head_dim, eps=rms_eps)
            self.self_attn.k_norm = EagleRMSNorm(head_dim, eps=rms_eps)

        self.mlp = EagleMLP(draft_cfg)

        self.input_layernorm = EagleRMSNorm(H, eps=rms_eps)
        self.hidden_norm = EagleRMSNorm(H, eps=rms_eps)
        self.post_attention_layernorm = EagleRMSNorm(H, eps=rms_eps)

        # Rotary embeddings
        self.rotary_emb = EagleRotaryEmbedding(head_dim, base=getattr(draft_cfg, "rope_theta", 10000.0))

    @staticmethod
    def _apply_rope(x, cos, sin):
        # x: (B, n_heads, T, head_dim)  cos/sin: (B, T, head_dim)
        cos = cos.unsqueeze(1)  # (B, 1, T, head_dim)
        sin = sin.unsqueeze(1)
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def _repeat_kv(self, x, n_rep):
        # x: (B, n_kv, T, head_dim) -> (B, n_kv*n_rep, T, head_dim)
        if n_rep == 1:
            return x
        B, n_kv, T, D = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv, n_rep, T, D)
        return x.reshape(B, n_kv * n_rep, T, D)

    def forward(
        self,
        embeds: torch.Tensor,        # (B, T, H)
        fused_hidden: torch.Tensor,  # (B, T, H)
        position_ids: torch.Tensor,
        past_key_value=None,
        use_cache: bool = True,
    ):
        B, T, H = embeds.shape

        # Normalize the two streams, concat to 2H. RedHat/speculators
        # checkpoints use norm_before_residual=True for Qwen3 EAGLE-3.
        emb_n = self.input_layernorm(embeds)
        fh_n = self.hidden_norm(fused_hidden)
        attn_in = torch.cat([emb_n, fh_n], dim=-1)  # (B, T, 2H)

        residual = fh_n if self.norm_before_residual else fused_hidden

        # Q/K/V projections
        q = self.self_attn.q_proj(attn_in)  # (B, T, n_heads*head_dim)
        k = self.self_attn.k_proj(attn_in)  # (B, T, n_kv*head_dim)
        v = self.self_attn.v_proj(attn_in)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, hd)
        k = k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)     # (B, n_kv,   T, hd)
        v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        # Qwen3 draft checkpoints include per-head Q/K RMSNorm; Llama ones do not.
        if hasattr(self.self_attn, "q_norm"):
            q = self.self_attn.q_norm(q)
            k = self.self_attn.k_norm(k)

        # RoPE
        cos, sin = self.rotary_emb(position_ids, q.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # KV cache
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_pkv = (k, v) if use_cache else None

        # Repeat K/V for GQA
        n_rep = self.n_heads // self.n_kv
        k_rep = self._repeat_kv(k, n_rep)
        v_rep = self._repeat_kv(v, n_rep)

        # Scaled dot-product attention (causal)
        # Use F.scaled_dot_product_attention for efficiency & correctness
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k_rep, v_rep,
            is_causal=(k_rep.shape[2] == q.shape[2]),  # only causal during prefill; single-token decode doesn't need mask
        )
        # attn: (B, n_heads, T, head_dim) -> (B, T, n_heads*head_dim)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        attn_out = self.self_attn.o_proj(attn)  # (B, T, H)

        h = residual + attn_out

        # MLP
        h2 = self.post_attention_layernorm(h)
        h2 = self.mlp(h2)
        h = h + h2

        return h, new_pkv


class Eagle3DraftHead(nn.Module):
    """
    EAGLE-3 draft head matching RedHatAI/Qwen3-8B-Thinking-speculator.eagle3.

    Checkpoint layout
    -----------------
      fc:                 [H, 3H]          project concat(f_l, f_m, f_h) → H
      embed_tokens:       [V_target, H]    target vocab (151936)
      layers.0.*:         custom draft layer with 2H attn input
      norm:               [H]              final RMSNorm
      lm_head:            [V_draft, H]     draft vocab (64000, smaller than target)
      d2t:                [V_draft]        target_id = draft_id + d2t[draft_id]
      t2d:                [V_target]       bool mask: which target ids are in draft

    Forward returns target-vocab logits so the existing rejection-sampling
    logic in Eagle3SpeculativeDecoder works unchanged.
    """

    def __init__(self, draft_ckpt_path: str, target_model: nn.Module):
        super().__init__()
        cfg = target_model.config
        draft_cfg = self._load_draft_config(draft_ckpt_path, cfg)
        H = cfg.hidden_size
        V_target = cfg.vocab_size
        n = len(target_model.model.layers)
        self.tap_indices = self._tap_indices_from_config(draft_ckpt_path, n)
        self.H = H
        self.V_target = V_target
        self.V_draft = None
        self.draft_cfg = draft_cfg

        self.fc = nn.Linear(3 * H, H, bias=False)
        self.embed_tokens = nn.Embedding(V_target, H)
        self.draft_layer = Eagle3DraftLayer(draft_cfg)
        self.norm = EagleRMSNorm(H, eps=getattr(draft_cfg, "rms_norm_eps", getattr(cfg, "rms_norm_eps", 1e-6)))

        self.lm_head = None  # built when we see the checkpoint
        self.register_buffer("d2t", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("t2d", torch.zeros(0, dtype=torch.bool), persistent=False)

        self._load_weights(draft_ckpt_path)
        self.to(DEVICE)
        self.eval()

    @staticmethod
    def _load_draft_config(ckpt_path: str, target_cfg) -> SimpleNamespace:
        cfg_path = os.path.join(ckpt_path, "config.json")
        raw: Dict[str, Any] = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            layer_raw = raw.get("transformer_layer_config")
            if isinstance(layer_raw, dict):
                merged = dict(raw)
                merged.update(layer_raw)
                raw = merged
            elif isinstance(raw.get("text_config"), dict):
                merged = dict(raw)
                merged.update(raw["text_config"])
                raw = merged

        def pick(name: str, default: Any) -> Any:
            return raw.get(name, getattr(target_cfg, name, default))

        hidden_act = raw.get(
            "hidden_act",
            raw.get("hidden_activation", getattr(target_cfg, "hidden_act", "silu")),
        )
        cfg = SimpleNamespace(
            model_type=raw.get("model_type", getattr(target_cfg, "model_type", "llama")),
            hidden_size=pick("hidden_size", target_cfg.hidden_size),
            intermediate_size=pick("intermediate_size", getattr(target_cfg, "intermediate_size", 4 * target_cfg.hidden_size)),
            num_attention_heads=pick("num_attention_heads", target_cfg.num_attention_heads),
            num_key_value_heads=pick("num_key_value_heads", getattr(target_cfg, "num_key_value_heads", target_cfg.num_attention_heads)),
            head_dim=pick("head_dim", target_cfg.hidden_size // target_cfg.num_attention_heads),
            rms_norm_eps=pick("rms_norm_eps", getattr(target_cfg, "rms_norm_eps", 1e-6)),
            rope_theta=pick("rope_theta", getattr(target_cfg, "rope_theta", 10000.0)),
            hidden_act=hidden_act,
            attention_bias=pick("attention_bias", False),
            mlp_bias=pick("mlp_bias", False),
            norm_before_residual=pick("norm_before_residual", False),
        )
        if cfg.hidden_size != target_cfg.hidden_size:
            raise ValueError(
                f"Draft hidden_size ({cfg.hidden_size}) must match target hidden_size ({target_cfg.hidden_size})."
            )
        print(
            "[DraftHead] draft layer config: "
            f"model_type={cfg.model_type}, hidden_act={cfg.hidden_act}, "
            f"norm_before_residual={cfg.norm_before_residual}"
        )
        return cfg

    @staticmethod
    def _tap_indices_from_config(ckpt_path: str, num_layers: int) -> List[int]:
        cfg_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for key in (
                "eagle_aux_hidden_state_layer_ids",
                "target_hidden_states_selection",
                "target_hidden_state_layers",
                "target_layer_indices",
            ):
                vals = raw.get(key)
                if isinstance(vals, list) and vals:
                    idxs = [int(v if v >= 0 else num_layers + v) for v in vals[:3]]
                    return sorted(max(0, min(num_layers - 1, i)) for i in idxs)

        return sorted({1, num_layers // 2, max(0, num_layers - 4)})

    def _load_weights(self, ckpt_path: str) -> None:
        import glob
        files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
        if files:
            from safetensors.torch import load_file
            state = {}
            for f in files:
                state.update(load_file(f, device="cpu"))
        else:
            files = sorted(glob.glob(os.path.join(ckpt_path, "*.bin")))
            state = {}
            for f in files:
                state.update(torch.load(f, map_location="cpu"))

        def strip(k):
            for p in ("eagle_model.", "model.", ""):
                if k.startswith(p):
                    return k[len(p):]
            return k

        s = {strip(k): v for k, v in state.items()}

        # ----- fc -----
        if "fc.weight" in s:
            w = s["fc.weight"]
            in_dim = w.shape[1]
            if in_dim != 3 * self.H:
                self.fc = nn.Linear(in_dim, self.H, bias=False)
            self.fc.weight = nn.Parameter(w)
            print(f"[DraftHead] fc: {list(w.shape)}  (in={in_dim // self.H}H)")

        # ----- embed_tokens -----
        if "embed_tokens.weight" in s:
            w = s["embed_tokens.weight"]
            if w.shape != self.embed_tokens.weight.shape:
                self.embed_tokens = nn.Embedding(w.shape[0], w.shape[1])
            self.embed_tokens.weight = nn.Parameter(w)

        # ----- draft layer -----
        layer_state = {k[len("layers.0."):]: v
                       for k, v in s.items() if k.startswith("layers.0.")}
        missing, unexpected = self.draft_layer.load_state_dict(layer_state, strict=False)
        if missing:
            print(f"[DraftHead] draft_layer missing: {missing}")
        if unexpected:
            print(f"[DraftHead] draft_layer unexpected: {unexpected}")

        # ----- final norm -----
        if "norm.weight" in s:
            self.norm.weight = nn.Parameter(s["norm.weight"])

        # ----- lm_head -----
        if "lm_head.weight" in s:
            w = s["lm_head.weight"]
            V_draft, H = w.shape
            self.V_draft = V_draft
            self.lm_head = nn.Linear(H, V_draft, bias=False)
            self.lm_head.weight = nn.Parameter(w)
            print(f"[DraftHead] draft vocab = {V_draft}, target vocab = {self.V_target}")
        else:
            raise RuntimeError("lm_head.weight not found in checkpoint")

        # ----- d2t / t2d -----
        if "d2t" in s:
            self.d2t = s["d2t"].long().to(DEVICE)
        if "t2d" in s:
            self.t2d = s["t2d"].bool().to(DEVICE)

    def fuse(self, tapped: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """Step 5: concat (f_l, f_m, f_h) → 3H and project to H."""
        idxs = sorted(tapped.keys())
        parts = [tapped[i] for i in idxs]
        if len(parts) < 3:
            while len(parts) < 3:
                parts.append(parts[-1])
        cat = torch.cat(parts[:3], dim=-1).to(self.fc.weight.dtype)
        fused = self.fc(cat)
        return fused, fused.float().norm().item()

    @torch.no_grad()
    def prefill_context(
        self,
        context_ids: torch.Tensor,
        next_token_id: int,
        tapped_full: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Any, torch.Tensor, int]:
        """
        Run the EAGLE layer over target hidden states for the existing context.
        EAGLE pairs the previous-position verifier feature with the current
        token embedding. The last context token is replaced by the verifier's
        newly sampled next token, and the last output predicts the first draft
        token after that verifier token.
        """
        fused_full, _ = self.fuse(tapped_full)
        if context_ids.shape[1] > 1:
            input_ids = torch.cat(
                [
                    context_ids[:, 1:],
                    torch.tensor([[next_token_id]], device=DEVICE),
                ],
                dim=1,
            )
        else:
            input_ids = torch.tensor([[next_token_id]], device=DEVICE)

        embeds = self.embed_tokens(input_ids).to(fused_full.dtype)
        prefill_len = input_ids.shape[1]
        pos = torch.arange(prefill_len, device=DEVICE).unsqueeze(0)
        h, pkv = self.draft_layer(
            embeds=embeds,
            fused_hidden=fused_full,
            position_ids=pos,
            past_key_value=None,
            use_cache=True,
        )
        h_for_logits = self.norm(h)
        draft_logits = self.lm_head(h_for_logits[:, -1, :])
        target_logits = self.draft_to_target_logits(draft_logits)
        return h[:, -1:, :], pkv, target_logits, prefill_len

    def draft_to_target_logits(self, draft_logits: torch.Tensor) -> torch.Tensor:
        """
        Map draft-vocab logits (V_draft) -> target-vocab logits (V_target).
        EAGLE-3 convention: target_id = draft_id + d2t[draft_id].
        Positions not covered get -inf.
        """
        B = draft_logits.shape[0]
        device = draft_logits.device
        dtype = draft_logits.dtype
        out = torch.full((B, self.V_target), float("-inf"), device=device, dtype=dtype)

        if self.d2t.numel() == self.V_draft:
            draft_idx = torch.arange(self.V_draft, device=device)
            target_idx = draft_idx + self.d2t
            valid = (target_idx >= 0) & (target_idx < self.V_target)
            target_idx_v = target_idx[valid]
            src = draft_logits[:, valid]
            out[:, target_idx_v] = src
        else:
            # No mapping: assume vocabs align at the low end
            n = min(draft_logits.shape[1], self.V_target)
            out[:, :n] = draft_logits[:, :n]
        return out

    @torch.no_grad()
    def forward_step(
        self,
        prev_token_id: int,
        fused_hidden: torch.Tensor,
        past_key_values: Optional[Any],
        position_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """One draft step. Returns (target_vocab_logits, next_hidden, new_pkv)."""
        token_t = torch.tensor([[prev_token_id]], device=DEVICE)
        embed = self.embed_tokens(token_t).to(fused_hidden.dtype)  # (1,1,H)
        pos = torch.tensor([[position_id]], device=DEVICE)

        h, new_pkv = self.draft_layer(
            embeds=embed,
            fused_hidden=fused_hidden,
            position_ids=pos,
            past_key_value=past_key_values,
            use_cache=True,
        )
        next_hidden = h[:, -1:, :]
        h_for_logits = self.norm(h)
        draft_logits = self.lm_head(h_for_logits[:, -1, :])    # (1, V_draft)
        target_logits = self.draft_to_target_logits(draft_logits)  # (1, V_target)
        return target_logits, next_hidden, new_pkv


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
        top_p: Optional[float] = None,
        do_sample: bool = False,
    ):
        self.target = target
        self.draft = draft_head
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
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
        draft_hidden: torch.Tensor,
        past_key_values: Optional[Any],
        first_logits: torch.Tensor,
        next_position_id: int,
        fused_norm: float,
        eos_token_id: Optional[int],
        iteration: int,
        max_steps: Optional[int] = None,
    ) -> Tuple[List[Dict], List[DraftTraceEvent]]:
        proposals = []
        draft_events = []
        pkv = past_key_values
        prev_token = None
        steps = self.block_size if max_steps is None else min(self.block_size, max_steps)

        for step in range(steps):
            t0 = time.time()
            if step == 0:
                logits = first_logits
                hidden = draft_hidden
            else:
                logits, hidden, pkv = self.draft.forward_step(
                    prev_token_id=prev_token,
                    fused_hidden=draft_hidden,
                    past_key_values=pkv,
                    position_id=next_position_id + step - 1,
                )
            elapsed = time.time() - t0

            probs = sampling_probs(
                logits, self.temperature, self.top_k, self.do_sample, self.top_p
            )
            raw_probs = diagnostic_probs(logits)
            token_id, token_prob = sample_token(
                logits, self.temperature, self.top_k, self.do_sample, self.top_p
            )
            raw_token_prob = raw_probs[0, token_id].item()

            proposals.append({
                "token_id": token_id,
                "prob": token_prob,
                "probs": probs.detach(),
                "raw_prob": raw_token_prob,
                "raw_probs": raw_probs.detach(),
            })
            draft_events.append(DraftTraceEvent(
                iteration=iteration,
                draft_step=step,
                draft_token_id=token_id,
                draft_token=self.tokenizer.decode([token_id]),
                draft_prob=token_prob,
                draft_entropy=entropy(probs),
                draft_topk=topk_info(probs, self.tokenizer),
                raw_draft_prob=raw_token_prob,
                raw_draft_entropy=entropy(raw_probs),
                raw_draft_topk=topk_info(raw_probs, self.tokenizer),
                draft_hidden_norm=hidden.float().norm().item(),
                elapsed_s=elapsed,
            ))

            prev_token = token_id
            draft_hidden = hidden
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
        max_accept_tokens: Optional[int] = None,
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
            if max_accept_tokens is not None and len(accepted_ids) >= max_accept_tokens:
                break

            pos_logits    = verify_logits[:, i, :]
            target_probs  = sampling_probs(
                pos_logits, self.temperature, self.top_k, self.do_sample, self.top_p
            )
            raw_target_probs = diagnostic_probs(pos_logits)
            draft_probs   = proposal["probs"]
            raw_draft_probs = proposal["raw_probs"]

            target_prob   = target_probs[0, proposal["token_id"]].item()
            raw_target_prob = raw_target_probs[0, proposal["token_id"]].item()
            accept_ratio  = min(1.0, (target_prob + EPS) / (proposal["prob"] + EPS))
            raw_accept_ratio = min(1.0, (raw_target_prob + EPS) / (proposal["raw_prob"] + EPS))
            if not self.do_sample or self.temperature <= 0:
                base_id = pos_logits.argmax(dim=-1).item()
                accepted = proposal["token_id"] == base_id
            else:
                base_id = -1
                accepted = random.random() < accept_ratio

            kl    = kl_divergence(draft_probs, target_probs)
            raw_kl = kl_divergence(raw_draft_probs, raw_target_probs)
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
                raw_draft_prob=proposal["raw_prob"],
                raw_target_prob=raw_target_prob,
                raw_acceptance_ratio=raw_accept_ratio,
                raw_target_topk=topk_info(raw_target_probs, self.tokenizer),
                raw_kl_draft_target=raw_kl,
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
                if base_id < 0:
                    replacement_probs = speculative_replacement_probs(target_probs, draft_probs)
                    base_id, _ = sample_from_probs(replacement_probs)
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

            prefill_probs = sampling_probs(
                prefill_out["logits"], self.temperature, self.top_k, self.do_sample, self.top_p
            )
            raw_prefill_probs = diagnostic_probs(prefill_out["logits"])
            all_events.append(asdict(PrefillTraceEvent(
                iteration=iteration,
                input_len=prefill_out["input_len"],
                elapsed_s=prefill_out["elapsed_s"],
                f_early_norm=f_e.float().norm().item(),
                f_mid_norm=f_m.float().norm().item(),
                f_late_norm=f_l.float().norm().item(),
                top_hidden_norm=f_l.float().norm().item(),
                prefill_topk=topk_info(prefill_probs, self.tokenizer),
                raw_prefill_topk=topk_info(raw_prefill_probs, self.tokenizer),
            )))

            # ---- Step 5: sample verifier anchor and prefill EAGLE layer ---
            anchor_id, anchor_prob = sample_token(
                prefill_out["logits"], self.temperature, self.top_k, self.do_sample, self.top_p
            )
            fused, fused_norm = self.draft.fuse(tapped)
            context_with_anchor = torch.cat(
                [context_ids, torch.tensor([[anchor_id]], device=DEVICE)], dim=1
            )
            generated_ids.append(anchor_id)
            all_events.append(asdict(BridgeTraceEvent(
                iteration=iteration,
                base_token_id=anchor_id,
                base_token=self.tokenizer.decode([anchor_id]),
                base_prob=anchor_prob,
                base_topk=topk_info(prefill_probs, self.tokenizer),
                raw_base_prob=raw_prefill_probs[0, anchor_id].item(),
                raw_base_topk=topk_info(raw_prefill_probs, self.tokenizer),
                elapsed_s=prefill_out["elapsed_s"],
            )))

            metrics["base_only_tokens"] += 1
            if eos_token_id is not None and anchor_id == eos_token_id:
                context_ids = context_with_anchor
                break
            if len(generated_ids) >= max_new_tokens:
                context_ids = context_with_anchor
                break

            t0 = time.time()
            draft_hidden, draft_pkv, first_draft_logits, next_position_id = self.draft.prefill_context(
                context_ids, anchor_id, prefill_out["tapped_full"]
            )
            draft_prefill_time = time.time() - t0

            # ---- Step 6: Draft proposals ----------------------------------
            t0 = time.time()
            remaining = max_new_tokens - len(generated_ids)
            proposals, draft_events = self._propose(
                context_with_anchor, draft_hidden, draft_pkv, first_draft_logits,
                next_position_id, fused_norm, eos_token_id, iteration, max_steps=remaining
            )
            metrics["draft_time"]         += draft_prefill_time + time.time() - t0
            metrics["draft_forward_calls"] += len(proposals)
            metrics["proposals_generated"] += len(proposals)

            for de in draft_events:
                all_events.append(asdict(de))

            if not proposals:
                context_ids = context_with_anchor
                break

            # ---- Step 7+9+10: Batched verify + rejection sampling ---------
            accepted_ids, context_ids, verify_event = self._verify_and_sample(
                context_with_anchor, proposals, fused_norm, iteration,
                max_accept_tokens=max_new_tokens - len(generated_ids)
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
     # ===== 诊断 =====
    print(f"=== Draft head sanity check ===")
    print(f"V_draft: {draft_head.V_draft}, V_target: {draft_head.V_target}")
    print(f"d2t shape: {draft_head.d2t.shape}, dtype: {draft_head.d2t.dtype}")
    print(f"d2t[:20]: {draft_head.d2t[:20].tolist()}")
    print(f"d2t min/max: {draft_head.d2t.min().item()} / {draft_head.d2t.max().item()}")
    print(f"t2d shape: {draft_head.t2d.shape}, sum: {draft_head.t2d.sum().item()}")
    print(f"================================")
    # ===============
    tap_indices = draft_head.tap_indices
    print(f"Tapping layers : {tap_indices}")

    target_wrapped = TargetModelWithTaps(target_model, tap_indices)
    decoder = Eagle3SpeculativeDecoder(
        target=target_wrapped, draft_head=draft_head, tokenizer=tokenizer,
        block_size=args.block_size, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, do_sample=do_sample,
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
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        summary = {
            "dataset":             args.dataset or "single",
            "fault_log":           fault_log,
            "generation_args": {
                "base_model_id": args.base_model_id,
                "draft_model_id": args.draft_model_id,
                "max_new_tokens": args.max_new_tokens,
                "block_size": args.block_size,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "do_sample": do_sample,
                "enable_thinking": args.enable_thinking,
                "seed": args.seed,
                "dtype": args.dtype,
            },
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
