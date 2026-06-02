"""
EAGLE-3 + Qwen3 tree-draft reproduction.

This file is intentionally separate from the existing local implementation.
It reuses the same Qwen3 draft head, SDC trace schemas, dataset loader, and
fault injection hooks from eagle3_chain_speculative.py, then changes only the
draft/verify loop to build a dynamic draft tree.

Paper alignment
---------------
EAGLE-3 changes the drafter: it fuses low/mid/high target hidden states and
uses a one-layer decoder to directly predict tokens. The draft-tree policy is
from EAGLE-2: use draft confidence as a proxy for acceptance probability,
expand high-value nodes, rerank all generated nodes, and verify a connected
tree.

Important implementation boundary
---------------------------------
Official EAGLE verifies a flattened tree in one target forward with a custom
tree attention mask. Stock HuggingFace models may reject that 4D mask or ignore
the required tree semantics. This reproduction therefore supports:

  --verify_backend auto
      try one-pass tree attention first, then fall back to per-path verification.
  --verify_backend tree_attention
      require one-pass tree attention.
  --verify_backend path
      verify each selected tree path separately; useful as a correctness/debug
      fallback, not a speed reproduction.

For sampling temperature > 0, the default accept mode is single_path_strict:
it builds the tree, verifies the selected nodes, but applies the strict
Leviathan-style acceptance rule only to the highest-value path. Greedy mode
uses branch alternatives in the tree. This avoids guessing an undocumented
multi-branch stochastic posterior rule.
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from eagle3_chain_speculative import (
    EPS,
    DEVICE,
    BridgeTraceEvent,
    DraftTraceEvent,
    Eagle3DraftHead,
    Eagle3SpeculativeDecoder,
    FaultInjector,
    FaultLocation,
    FaultMode,
    PositionVerifyData,
    PrefillTraceEvent,
    TargetModelWithTaps,
    VerifyTraceEvent,
    _DATASETS_AVAILABLE,
    _FAULT_INJECTION_AVAILABLE,
    diagnostic_probs,
    entropy,
    is_correct,
    kl_divergence,
    load_benchmark,
    load_draft_head,
    load_target_model,
    sample_from_probs,
    sample_token,
    sampling_probs,
    seed_everything,
    speculative_replacement_probs,
    topk_info,
)


@dataclass
class TreeDraftNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    token_id: int
    token_prob: float
    cumulative_prob: float
    draft_entropy: float
    draft_hidden_norm: float
    position_id: int
    # Runtime state used only if this node is expanded.
    parent_hidden: Any = None
    parent_pkv: Any = None
    draft_probs: Any = None
    raw_draft_probs: Any = None


class Eagle3TreeSpeculativeDecoder(Eagle3SpeculativeDecoder):
    def __init__(
        self,
        target: TargetModelWithTaps,
        draft_head: Eagle3DraftHead,
        tokenizer,
        block_size: int = 5,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        tree_depth: int = 6,
        tree_branch_factor: int = 4,
        tree_expand_top_nodes: int = 2,
        tree_verify_nodes: int = 16,
        verify_backend: str = "auto",
        tree_accept_mode: str = "auto",
        allow_tree_attention_fallback: bool = True,
    ):
        super().__init__(
            target=target,
            draft_head=draft_head,
            tokenizer=tokenizer,
            block_size=block_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )
        self.tree_depth = max(1, tree_depth)
        self.tree_branch_factor = max(1, tree_branch_factor)
        self.tree_expand_top_nodes = max(1, tree_expand_top_nodes)
        self.tree_verify_nodes = max(1, tree_verify_nodes)
        self.verify_backend = verify_backend
        self.tree_accept_mode = tree_accept_mode
        self.allow_tree_attention_fallback = allow_tree_attention_fallback

    def _resolved_accept_mode(self) -> str:
        if self.tree_accept_mode != "auto":
            return self.tree_accept_mode
        if not self.do_sample or self.temperature <= 0:
            return "greedy_tree"
        return "single_path_strict"

    def _top_children(
        self,
        parent_id: Optional[int],
        parent_depth: int,
        parent_value: float,
        parent_hidden: torch.Tensor,
        parent_pkv: Any,
        position_id: int,
        logits: torch.Tensor,
        next_node_id: int,
    ) -> Tuple[List[TreeDraftNode], int]:
        probs = sampling_probs(logits, self.temperature, self.top_k, self.do_sample, self.top_p)
        raw_probs = diagnostic_probs(logits)
        k = min(self.tree_branch_factor, probs.shape[-1])
        vals, idxs = torch.topk(probs, k, dim=-1)
        nodes: List[TreeDraftNode] = []
        ent = entropy(probs)
        hidden_norm = parent_hidden.float().norm().item()
        for i in range(k):
            token_id = idxs[0, i].item()
            token_prob = vals[0, i].item()
            if token_prob <= 0.0:
                continue
            nodes.append(
                TreeDraftNode(
                    node_id=next_node_id,
                    parent_id=parent_id,
                    depth=parent_depth + 1,
                    token_id=token_id,
                    token_prob=token_prob,
                    cumulative_prob=parent_value * token_prob,
                    draft_entropy=ent,
                    draft_hidden_norm=hidden_norm,
                    position_id=position_id,
                    parent_hidden=parent_hidden,
                    parent_pkv=parent_pkv,
                    draft_probs=probs.detach(),
                    raw_draft_probs=raw_probs.detach(),
                )
            )
            next_node_id += 1
        return nodes, next_node_id

    def _select_connected_nodes(self, all_nodes: List[TreeDraftNode]) -> List[TreeDraftNode]:
        by_id = {n.node_id: n for n in all_nodes}
        selected: Dict[int, TreeDraftNode] = {}

        def add_with_ancestors(node: TreeDraftNode) -> None:
            chain = []
            cur = node
            while cur is not None:
                chain.append(cur)
                cur = by_id.get(cur.parent_id) if cur.parent_id is not None else None
            for item in reversed(chain):
                if len(selected) >= self.tree_verify_nodes and item.node_id not in selected:
                    return
                selected[item.node_id] = item

        ranked = sorted(all_nodes, key=lambda n: (-n.cumulative_prob, n.depth, n.node_id))
        for node in ranked:
            if len(selected) >= self.tree_verify_nodes:
                break
            add_with_ancestors(node)

        return sorted(selected.values(), key=lambda n: (n.depth, n.node_id))

    def _path_to_root(self, node: TreeDraftNode, by_id: Dict[int, TreeDraftNode]) -> List[TreeDraftNode]:
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = by_id.get(cur.parent_id) if cur.parent_id is not None else None
        return list(reversed(path))

    def _leaf_paths(self, selected_nodes: List[TreeDraftNode]) -> List[List[TreeDraftNode]]:
        by_id = {n.node_id: n for n in selected_nodes}
        parent_ids = {n.parent_id for n in selected_nodes if n.parent_id is not None}
        leaves = [n for n in selected_nodes if n.node_id not in parent_ids]
        paths = [self._path_to_root(n, by_id) for n in leaves]
        return sorted(paths, key=lambda p: (-p[-1].cumulative_prob, len(p), p[-1].node_id))

    def _build_tree(
        self,
        draft_hidden: torch.Tensor,
        draft_pkv: Any,
        first_logits: torch.Tensor,
        next_position_id: int,
        iteration: int,
        max_new_tokens: int,
    ) -> Tuple[List[TreeDraftNode], List[Dict], Dict[str, Any]]:
        all_nodes: List[TreeDraftNode] = []
        draft_events: List[Dict] = []
        next_node_id = 0

        first_nodes, next_node_id = self._top_children(
            parent_id=None,
            parent_depth=0,
            parent_value=1.0,
            parent_hidden=draft_hidden,
            parent_pkv=draft_pkv,
            position_id=next_position_id,
            logits=first_logits,
            next_node_id=next_node_id,
        )
        all_nodes.extend(first_nodes)
        frontier = first_nodes
        draft_forward_calls = 0

        for depth in range(1, min(self.tree_depth, max_new_tokens)):
            expandable = sorted(frontier, key=lambda n: (-n.cumulative_prob, n.node_id))
            expandable = expandable[: self.tree_expand_top_nodes]
            new_frontier: List[TreeDraftNode] = []
            for node in expandable:
                logits, hidden, pkv = self.draft.forward_step(
                    prev_token_id=node.token_id,
                    fused_hidden=node.parent_hidden,
                    past_key_values=node.parent_pkv,
                    position_id=node.position_id,
                )
                draft_forward_calls += 1
                children, next_node_id = self._top_children(
                    parent_id=node.node_id,
                    parent_depth=node.depth,
                    parent_value=node.cumulative_prob,
                    parent_hidden=hidden,
                    parent_pkv=pkv,
                    position_id=node.position_id + 1,
                    logits=logits,
                    next_node_id=next_node_id,
                )
                all_nodes.extend(children)
                new_frontier.extend(children)
            frontier = new_frontier
            if not frontier:
                break

        selected_nodes = self._select_connected_nodes(all_nodes)
        selected_ids = {n.node_id for n in selected_nodes}
        for node in all_nodes:
            event = asdict(
                DraftTraceEvent(
                    iteration=iteration,
                    draft_step=node.depth - 1,
                    draft_token_id=node.token_id,
                    draft_token=self.tokenizer.decode([node.token_id]),
                    draft_prob=node.token_prob,
                    draft_entropy=node.draft_entropy,
                    draft_topk=topk_info(node.draft_probs, self.tokenizer),
                    raw_draft_prob=node.raw_draft_probs[0, node.token_id].item(),
                    raw_draft_entropy=entropy(node.raw_draft_probs),
                    raw_draft_topk=topk_info(node.raw_draft_probs, self.tokenizer),
                    draft_hidden_norm=node.draft_hidden_norm,
                    elapsed_s=0.0,
                )
            )
            event.update(
                {
                    "phase": "draft_tree",
                    "tree_node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "cumulative_draft_prob": node.cumulative_prob,
                    "selected_for_verify": node.node_id in selected_ids,
                }
            )
            draft_events.append(event)

        metadata = {
            "tree_depth": self.tree_depth,
            "tree_branch_factor": self.tree_branch_factor,
            "tree_expand_top_nodes": self.tree_expand_top_nodes,
            "tree_nodes_generated": len(all_nodes),
            "tree_nodes_selected": len(selected_nodes),
            "draft_forward_calls": draft_forward_calls,
            "selection_rule": "EAGLE-2-style confidence product rerank",
        }
        return selected_nodes, draft_events, metadata

    def _build_tree_attention_mask(
        self,
        context_len: int,
        selected_nodes: List[TreeDraftNode],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        total_len = context_len + len(selected_nodes)
        node_offsets = {node.node_id: i for i, node in enumerate(selected_nodes)}
        allowed = torch.zeros((total_len, total_len), device=DEVICE, dtype=torch.bool)

        # Normal causal attention over the verified context.
        for row in range(context_len):
            allowed[row, : row + 1] = True

        by_id = {n.node_id: n for n in selected_nodes}
        for node in selected_nodes:
            row = context_len + node_offsets[node.node_id]
            allowed[row, :context_len] = True
            cur = node
            while cur is not None:
                allowed[row, context_len + node_offsets[cur.node_id]] = True
                cur = by_id.get(cur.parent_id) if cur.parent_id is not None else None

        mask = torch.full((1, 1, total_len, total_len), torch.finfo(dtype).min, device=DEVICE, dtype=dtype)
        mask.masked_fill_(allowed.unsqueeze(0).unsqueeze(0), 0)
        return mask, node_offsets

    @torch.no_grad()
    def _verify_tree_attention(
        self,
        context_ids: torch.Tensor,
        selected_nodes: List[TreeDraftNode],
    ) -> Dict[str, Any]:
        draft_t = torch.tensor([[n.token_id for n in selected_nodes]], device=DEVICE)
        full_ids = torch.cat([context_ids, draft_t], dim=1)
        ctx_len = context_ids.shape[1]
        dtype = next(self.target.model.parameters()).dtype
        mask, node_offsets = self._build_tree_attention_mask(ctx_len, selected_nodes, dtype)

        self.target._tapped.clear()
        self.target.current_phase = "verify"
        t0 = time.time()
        out = self.target.model(full_ids, attention_mask=mask, use_cache=False, output_hidden_states=False)
        elapsed = time.time() - t0
        self.target.current_phase = "idle"

        top_idx = max(self.target.tap_indices)
        top_hidden = self.target._tapped.get(top_idx)
        target_by_node: Dict[int, Dict[str, Any]] = {}
        for node in selected_nodes:
            if node.parent_id is None:
                logit_pos = ctx_len - 1
            else:
                logit_pos = ctx_len + node_offsets[node.parent_id]
            logits = out.logits[:, logit_pos, :]
            h_norm = 0.0
            if top_hidden is not None:
                h_norm = top_hidden[:, logit_pos, :].float().norm().item()
            target_by_node[node.node_id] = {"logits": logits, "hidden_norm": h_norm}

        return {
            "target_by_node": target_by_node,
            "elapsed_s": elapsed,
            "backend": "tree_attention",
            "fallback_reason": None,
        }

    @torch.no_grad()
    def _verify_paths(
        self,
        context_ids: torch.Tensor,
        selected_nodes: List[TreeDraftNode],
    ) -> Dict[str, Any]:
        by_id = {n.node_id: n for n in selected_nodes}
        target_by_node: Dict[int, Dict[str, Any]] = {}
        elapsed = 0.0
        for node in selected_nodes:
            path = self._path_to_root(node, by_id)
            path_tokens = [n.token_id for n in path]
            out = self.target.verify(context_ids, path_tokens)
            elapsed += out["elapsed_s"]
            idx = len(path) - 1
            h_norm = 0.0
            if out["verify_hiddens"] is not None:
                h_norm = out["verify_hiddens"][0, idx, :].float().norm().item()
            target_by_node[node.node_id] = {
                "logits": out["verify_logits"][:, idx, :],
                "hidden_norm": h_norm,
            }
        return {
            "target_by_node": target_by_node,
            "elapsed_s": elapsed,
            "backend": "path",
            "fallback_reason": None,
        }

    def _verify_selected_tree(
        self,
        context_ids: torch.Tensor,
        selected_nodes: List[TreeDraftNode],
    ) -> Dict[str, Any]:
        if self.verify_backend == "path":
            return self._verify_paths(context_ids, selected_nodes)

        try:
            return self._verify_tree_attention(context_ids, selected_nodes)
        except Exception as exc:
            if self.verify_backend == "tree_attention" or not self.allow_tree_attention_fallback:
                raise
            out = self._verify_paths(context_ids, selected_nodes)
            out["backend"] = "path_fallback"
            out["fallback_reason"] = repr(exc)
            return out

    def _accept_single_path_strict(
        self,
        selected_nodes: List[TreeDraftNode],
        target_by_node: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[int], Optional[int], Dict[int, str], int]:
        paths = self._leaf_paths(selected_nodes)
        best_path = paths[0] if paths else []
        accepted_ids: List[int] = []
        first_reject: Optional[int] = None
        status = {n.node_id: "sibling_not_chosen" for n in selected_nodes}

        for i, node in enumerate(best_path):
            pos_logits = target_by_node[node.node_id]["logits"]
            target_probs = sampling_probs(pos_logits, self.temperature, self.top_k, self.do_sample, self.top_p)
            target_prob = target_probs[0, node.token_id].item()
            accept_ratio = min(1.0, (target_prob + EPS) / (node.token_prob + EPS))

            if not self.do_sample or self.temperature <= 0:
                base_id = pos_logits.argmax(dim=-1).item()
                accepted = node.token_id == base_id
            else:
                base_id = -1
                accepted = random.random() < accept_ratio

            if accepted:
                accepted_ids.append(node.token_id)
                status[node.node_id] = "accepted_path"
            else:
                first_reject = i
                if base_id < 0:
                    replacement_probs = speculative_replacement_probs(target_probs, node.draft_probs)
                    base_id, _ = sample_from_probs(replacement_probs)
                accepted_ids.append(base_id)
                status[node.node_id] = "rejected_at_position"
                break

        return accepted_ids, first_reject, status, len(best_path)

    def _accept_greedy_tree(
        self,
        selected_nodes: List[TreeDraftNode],
        target_by_node: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[int], Optional[int], Dict[int, str], int]:
        children: Dict[Optional[int], List[TreeDraftNode]] = {}
        for node in selected_nodes:
            children.setdefault(node.parent_id, []).append(node)
        for vals in children.values():
            vals.sort(key=lambda n: (-n.cumulative_prob, n.node_id))

        accepted_ids: List[int] = []
        first_reject: Optional[int] = None
        status = {n.node_id: "sibling_not_chosen" for n in selected_nodes}
        parent_id: Optional[int] = None
        depth = 0
        chosen_path_len = 0

        while True:
            candidates = children.get(parent_id, [])
            if not candidates:
                break
            chosen_path_len += 1
            probe = candidates[0]
            target_probs = sampling_probs(
                target_by_node[probe.node_id]["logits"],
                self.temperature,
                self.top_k,
                self.do_sample,
                self.top_p,
            )
            target_id = torch.argmax(target_probs, dim=-1).item()
            matched = next((n for n in candidates if n.token_id == target_id), None)
            if matched is None:
                accepted_ids.append(target_id)
                first_reject = depth
                break
            accepted_ids.append(matched.token_id)
            status[matched.node_id] = "accepted_path"
            parent_id = matched.node_id
            depth += 1

        return accepted_ids, first_reject, status, chosen_path_len

    def _verify_accept_tree(
        self,
        context_ids: torch.Tensor,
        selected_nodes: List[TreeDraftNode],
        fused_norm: float,
        iteration: int,
        max_accept_tokens: int,
    ) -> Tuple[List[int], torch.Tensor, VerifyTraceEvent, Dict[str, Any]]:
        verify_out = self._verify_selected_tree(context_ids, selected_nodes)
        target_by_node = verify_out["target_by_node"]
        accept_mode = self._resolved_accept_mode()

        if accept_mode == "greedy_tree":
            accepted_ids, first_reject, status, chosen_path_len = self._accept_greedy_tree(selected_nodes, target_by_node)
        elif accept_mode == "single_path_strict":
            accepted_ids, first_reject, status, chosen_path_len = self._accept_single_path_strict(selected_nodes, target_by_node)
        else:
            raise ValueError(
                "tree_accept_mode must be auto, greedy_tree, or single_path_strict. "
                "Multi-branch stochastic posterior is not implemented here because "
                "the paper does not spell out a standalone HF-compatible rule."
            )

        accepted_ids = accepted_ids[:max_accept_tokens]
        for token_id in accepted_ids:
            context_ids = torch.cat([context_ids, torch.tensor([[token_id]], device=DEVICE)], dim=1)

        per_position = []
        kls = []
        hnorms = []
        for pos, node in enumerate(selected_nodes):
            pos_logits = target_by_node[node.node_id]["logits"]
            target_probs = sampling_probs(pos_logits, self.temperature, self.top_k, self.do_sample, self.top_p)
            raw_target_probs = diagnostic_probs(pos_logits)
            target_prob = target_probs[0, node.token_id].item()
            raw_target_prob = raw_target_probs[0, node.token_id].item()
            accept_ratio = min(1.0, (target_prob + EPS) / (node.token_prob + EPS))
            raw_draft_prob = node.raw_draft_probs[0, node.token_id].item()
            raw_accept_ratio = min(1.0, (raw_target_prob + EPS) / (raw_draft_prob + EPS))
            kl = kl_divergence(node.draft_probs, target_probs)
            raw_kl = kl_divergence(node.raw_draft_probs, raw_target_probs)
            h_norm = target_by_node[node.node_id]["hidden_norm"]
            kls.append(kl)
            hnorms.append(h_norm)

            item = asdict(
                PositionVerifyData(
                    pos=pos,
                    draft_token_id=node.token_id,
                    draft_token=self.tokenizer.decode([node.token_id]),
                    draft_prob=node.token_prob,
                    target_prob=target_prob,
                    acceptance_ratio=accept_ratio,
                    accepted=status.get(node.node_id) == "accepted_path",
                    target_topk=topk_info(target_probs, self.tokenizer),
                    kl_draft_target=kl,
                    raw_draft_prob=raw_draft_prob,
                    raw_target_prob=raw_target_prob,
                    raw_acceptance_ratio=raw_accept_ratio,
                    raw_target_topk=topk_info(raw_target_probs, self.tokenizer),
                    raw_kl_draft_target=raw_kl,
                    target_hidden_norm=h_norm,
                    fused_feature_norm=fused_norm,
                )
            )
            item.update(
                {
                    "tree_node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "cumulative_draft_prob": node.cumulative_prob,
                    "candidate_status": status.get(node.node_id, "sibling_not_chosen"),
                }
            )
            per_position.append(item)

        actual_accepted = sum(1 for v in status.values() if v == "accepted_path")
        actual_rejected = 1 if first_reject is not None else 0
        path_tokens_checked = actual_accepted + actual_rejected
        path_acceptance_rate = actual_accepted / path_tokens_checked if path_tokens_checked else 0.0
        event = VerifyTraceEvent(
            iteration=iteration,
            block_size_proposed=len(selected_nodes),
            num_accepted=actual_accepted,
            num_rejected=actual_rejected,
            first_reject_pos=first_reject,
            acceptance_rate_this_block=actual_accepted / len(selected_nodes) if selected_nodes else 0.0,
            per_position=per_position,
            elapsed_s=verify_out["elapsed_s"],
            mean_kl_draft_target=sum(kls) / len(kls) if kls else 0.0,
            max_kl_draft_target=max(kls) if kls else 0.0,
            mean_target_hidden_norm=sum(hnorms) / len(hnorms) if hnorms else 0.0,
            effective_kv_len_after=context_ids.shape[1],
        )

        metadata = {
            "verify_backend": verify_out["backend"],
            "fallback_reason": verify_out["fallback_reason"],
            "tree_accept_mode": accept_mode,
            "chosen_path_len": chosen_path_len,
            "path_tokens_checked": path_tokens_checked,
            "path_tokens_accepted": actual_accepted,
            "path_acceptance_rate_this_block": path_acceptance_rate,
            "node_acceptance_rate_this_block": actual_accepted / len(selected_nodes) if selected_nodes else 0.0,
            "paper_note": (
                "EAGLE-3 adopts EAGLE-2 dynamic trees. Target verification is "
                "parallel only when verify_backend=tree_attention succeeds."
            ),
        }
        return accepted_ids, context_ids, event, metadata

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        context_ids = input_ids.clone()
        generated_ids: List[int] = []
        all_events: List[Dict] = []

        metrics = {
            "proposals_generated": 0,
            "accepted_proposals": 0,
            "rejected_proposals": 0,
            "base_only_tokens": 0,
            "draft_forward_calls": 0,
            "base_forward_calls": 0,
            "draft_time": 0.0,
            "base_time": 0.0,
            "iteration_count": 0,
            "tree_nodes_generated": 0,
            "tree_nodes_verified": 0,
            "path_tokens_checked": 0,
            "path_tokens_accepted": 0,
        }
        tree_metadata_events: List[Dict[str, Any]] = []
        start_time = time.time()
        iteration = 0

        while len(generated_ids) < max_new_tokens:
            iteration += 1
            metrics["iteration_count"] += 1

            prefill_out = self.target.prefill(context_ids)
            metrics["base_forward_calls"] += 1
            metrics["base_time"] += prefill_out["elapsed_s"]

            tapped = prefill_out["tapped"]
            idxs = sorted(tapped.keys())
            f_e = tapped[idxs[0]]
            f_m = tapped[idxs[1]] if len(idxs) > 1 else f_e
            f_l = tapped[idxs[-1]]

            prefill_probs = sampling_probs(
                prefill_out["logits"], self.temperature, self.top_k, self.do_sample, self.top_p
            )
            raw_prefill_probs = diagnostic_probs(prefill_out["logits"])
            all_events.append(
                asdict(
                    PrefillTraceEvent(
                        iteration=iteration,
                        input_len=prefill_out["input_len"],
                        elapsed_s=prefill_out["elapsed_s"],
                        f_early_norm=f_e.float().norm().item(),
                        f_mid_norm=f_m.float().norm().item(),
                        f_late_norm=f_l.float().norm().item(),
                        top_hidden_norm=f_l.float().norm().item(),
                        prefill_topk=topk_info(prefill_probs, self.tokenizer),
                        raw_prefill_topk=topk_info(raw_prefill_probs, self.tokenizer),
                    )
                )
            )

            anchor_id, anchor_prob = sample_token(
                prefill_out["logits"], self.temperature, self.top_k, self.do_sample, self.top_p
            )
            _, fused_norm = self.draft.fuse(tapped)
            context_with_anchor = torch.cat([context_ids, torch.tensor([[anchor_id]], device=DEVICE)], dim=1)
            generated_ids.append(anchor_id)
            all_events.append(
                asdict(
                    BridgeTraceEvent(
                        iteration=iteration,
                        base_token_id=anchor_id,
                        base_token=self.tokenizer.decode([anchor_id]),
                        base_prob=anchor_prob,
                        base_topk=topk_info(prefill_probs, self.tokenizer),
                        raw_base_prob=raw_prefill_probs[0, anchor_id].item(),
                        raw_base_topk=topk_info(raw_prefill_probs, self.tokenizer),
                        elapsed_s=prefill_out["elapsed_s"],
                    )
                )
            )

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
            remaining = max_new_tokens - len(generated_ids)
            selected_nodes, draft_events, tree_meta = self._build_tree(
                draft_hidden=draft_hidden,
                draft_pkv=draft_pkv,
                first_logits=first_draft_logits,
                next_position_id=next_position_id,
                iteration=iteration,
                max_new_tokens=remaining,
            )
            metrics["draft_time"] += time.time() - t0
            metrics["draft_forward_calls"] += tree_meta["draft_forward_calls"]
            metrics["proposals_generated"] += len(selected_nodes)
            metrics["tree_nodes_generated"] += tree_meta["tree_nodes_generated"]
            metrics["tree_nodes_verified"] += len(selected_nodes)
            all_events.extend(draft_events)

            if not selected_nodes:
                context_ids = context_with_anchor
                break

            accepted_ids, context_ids, verify_event, verify_meta = self._verify_accept_tree(
                context_with_anchor,
                selected_nodes,
                fused_norm,
                iteration,
                max_accept_tokens=remaining,
            )
            metrics["base_forward_calls"] += 1 if verify_meta["verify_backend"] == "tree_attention" else len(selected_nodes)
            metrics["base_time"] += verify_event.elapsed_s
            metrics["accepted_proposals"] += verify_event.num_accepted
            metrics["rejected_proposals"] += verify_event.num_rejected
            metrics["base_only_tokens"] += verify_event.num_rejected
            metrics["path_tokens_checked"] += verify_meta["path_tokens_checked"]
            metrics["path_tokens_accepted"] += verify_meta["path_tokens_accepted"]

            verify_dict = asdict(verify_event)
            verify_dict["phase"] = "verify_tree"
            verify_dict["tree_metadata"] = {**tree_meta, **verify_meta}
            tree_metadata_events.append(verify_dict["tree_metadata"])
            all_events.append(verify_dict)
            generated_ids.extend(accepted_ids)

            if eos_token_id is not None and generated_ids and generated_ids[-1] == eos_token_id:
                break
            if len(generated_ids) >= max_new_tokens:
                break

        metrics["generation_time"] = time.time() - start_time
        metrics["node_acceptance_rate"] = (
            metrics["accepted_proposals"] / metrics["proposals_generated"]
            if metrics["proposals_generated"] > 0
            else 0.0
        )
        metrics["path_acceptance_rate"] = (
            metrics["path_tokens_accepted"] / metrics["path_tokens_checked"]
            if metrics["path_tokens_checked"] > 0
            else 0.0
        )
        metrics["acceptance_rate"] = metrics["path_acceptance_rate"]
        metrics["acceptance_rate_definition"] = (
            "tree path_tokens_accepted/path_tokens_checked; "
            "node_acceptance_rate is accepted_path_nodes/selected_tree_nodes"
        )
        metrics["tokens_emitted"] = len(generated_ids)
        metrics["tree_verify_backends"] = sorted({m["verify_backend"] for m in tree_metadata_events})

        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {"text": decoded, "tokens": generated_ids, "trace": all_events, "metrics": metrics}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft_model_id", default="RedHatAI/Qwen3-8B-Thinking-speculator.eagle3")
    parser.add_argument("--block_size", type=int, default=5, help="Kept for metric comparability; tree_depth controls tree draft length.")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--dataset", default=None, choices=["gsm8k", "math500", "aime2024", "aime2025", "gpqa", "livecodebench", "openthoughts"])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--prompt", default="What is 25 * 48?")

    parser.add_argument("--tree_depth", type=int, default=6)
    parser.add_argument("--tree_branch_factor", type=int, default=4)
    parser.add_argument("--tree_expand_top_nodes", type=int, default=2)
    parser.add_argument("--tree_verify_nodes", type=int, default=16)
    parser.add_argument("--verify_backend", default="auto", choices=["auto", "tree_attention", "path"])
    parser.add_argument("--tree_accept_mode", default="auto", choices=["auto", "greedy_tree", "single_path_strict"])
    parser.add_argument("--no_tree_attention_fallback", action="store_true")

    parser.add_argument("--fault_location", default=None, choices=["target_layer", "target_embed", "draft_embed", "draft_fc", "draft_layer", "shared_lm_head"])
    parser.add_argument("--fault_mode", default="double_bit", choices=["single_bit", "double_bit", "stuck_at_0"])
    parser.add_argument("--fault_type", default="weight", choices=["weight", "activation"])
    parser.add_argument("--fault_layer_idx", type=int, default=None)
    parser.add_argument("--fault_module", default=None)
    parser.add_argument("--fault_phase", default="both", choices=["prefill", "verify", "both"])
    parser.add_argument("--fault_seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    seed_everything(args.seed)

    do_sample = args.temperature > 0
    if args.enable_thinking and not do_sample:
        print("[Warning] Qwen3 thinking mode is not usually run greedily; setting temperature=0.6.")
        args.temperature = 0.6
        do_sample = True
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"Loading target : {args.base_model_id}")
    target_model, tokenizer = load_target_model(args.base_model_id, dtype)
    print(f"Loading draft  : {args.draft_model_id}")
    draft_head = load_draft_head(args.draft_model_id, target_model, dtype)

    print("=== Draft head sanity check ===")
    print(f"V_draft: {draft_head.V_draft}, V_target: {draft_head.V_target}")
    print(f"d2t shape: {draft_head.d2t.shape}, dtype: {draft_head.d2t.dtype}")
    if draft_head.d2t.numel():
        print(f"d2t[:20]: {draft_head.d2t[:20].tolist()}")
        print(f"d2t min/max: {draft_head.d2t.min().item()} / {draft_head.d2t.max().item()}")
    print(f"t2d shape: {draft_head.t2d.shape}, sum: {draft_head.t2d.sum().item() if draft_head.t2d.numel() else 0}")
    print("================================")

    target_wrapped = TargetModelWithTaps(target_model, draft_head.tap_indices)
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

    injector = None
    weight_snapshot = None
    activation_handle = None
    fault_log = None

    if args.fault_location is not None:
        if not _FAULT_INJECTION_AVAILABLE:
            raise RuntimeError("fault_injection.py not found.")
        injector = FaultInjector(target_model, draft_head)
        location = FaultLocation(args.fault_location)
        mode = FaultMode(args.fault_mode)

        if args.fault_type == "weight":
            weight_snapshot = injector.inject_weight_fault(
                location=location,
                mode=mode,
                layer_idx=args.fault_layer_idx,
                module_path=args.fault_module,
                seed=args.fault_seed,
            )
            fault_log = weight_snapshot.as_log()
        else:
            if location == FaultLocation.TARGET_LAYER:
                layer_idx = args.fault_layer_idx
                if layer_idx is None:
                    rng = random.Random(args.fault_seed)
                    layer_idx = rng.randint(0, len(target_model.model.layers) - 1)
                hook_rng = random.Random(args.fault_seed)

                def bit_flip_hook(module, inputs, output):
                    tensor = output[0] if isinstance(output, tuple) else output
                    x = hook_rng.randrange(tensor.shape[1])
                    y = hook_rng.randrange(tensor.shape[2])
                    val = tensor[0, x, y]
                    if mode == FaultMode.SINGLE_BIT:
                        bit = hook_rng.randint(0, 15)
                        flipped = val.to(torch.bfloat16).view(torch.int16) ^ (1 << bit)
                    elif mode == FaultMode.DOUBLE_BIT:
                        b0, b1 = hook_rng.sample(range(16), 2)
                        flipped = val.to(torch.bfloat16).view(torch.int16) ^ ((1 << b0) | (1 << b1))
                    else:
                        flipped = val.to(torch.bfloat16).view(torch.int16) & (~torch.tensor(0x7F80, dtype=torch.int16, device=val.device))
                    tensor = tensor.clone()
                    tensor[0, x, y] = flipped.view(torch.bfloat16).to(tensor.dtype)
                    return (tensor,) + output[1:] if isinstance(output, tuple) else tensor

                activation_handle = decoder.register_fault_hook(layer_idx=layer_idx, hook_fn=bit_flip_hook, phase_filter=args.fault_phase)
                fault_log = {"location": location.value, "layer_idx": layer_idx, "mode": mode.value, "phase_filter": args.fault_phase, "fault_seed": args.fault_seed}
            else:
                activation_handle = injector.inject_activation_fault(
                    location=location,
                    mode=mode,
                    layer_idx=args.fault_layer_idx,
                    module_path=args.fault_module,
                    seed=args.fault_seed,
                )
                fault_log = activation_handle.as_log()
        print(f"[Fault] registered: {fault_log}")

    if args.dataset is not None:
        if not _DATASETS_AVAILABLE:
            raise RuntimeError("datasets_loader.py not found.")
        samples = load_benchmark(args.dataset, num_samples=args.num_samples, seed=args.seed)
    else:
        samples = [{"question": args.prompt, "answer": "", "source": "single", "sample_id": 0}]

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
        result = decoder.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens, eos_token_id=tokenizer.eos_token_id)

        correct = False
        if sample["answer"] and args.dataset is not None:
            correct = is_correct(result["text"], sample["answer"], args.dataset)
            if correct:
                n_correct += 1

        entry = {
            "sample_id": sample["sample_id"],
            "source": sample["source"],
            "question": sample["question"],
            "reference": sample["answer"],
            "prediction": result["text"],
            "is_correct": correct,
            "metrics": result["metrics"],
            "trace": result["trace"],
        }
        if fault_log:
            entry["fault_log"] = fault_log
        all_results.append(entry)
        m = result["metrics"]
        status = "ok" if correct else "na"
        print(
            f"[{sample['sample_id']:4d}] {status} "
            f"path_accept={m['path_acceptance_rate']:.3f} "
            f"node_accept={m['node_acceptance_rate']:.3f} "
            f"tokens={m['tokens_emitted']} backends={m.get('tree_verify_backends')}"
        )

    if weight_snapshot is not None:
        injector.restore_weight(weight_snapshot)
    if activation_handle is not None:
        activation_handle.remove()

    total = len(all_results)
    avg_accept = sum(r["metrics"]["acceptance_rate"] for r in all_results) / total
    print("=" * 60)
    if args.dataset is not None:
        print(f"Dataset  : {args.dataset} ({total} samples)")
        print(f"Accuracy : {n_correct}/{total} = {n_correct / total:.4f}")
    print(f"Avg acceptance rate : {avg_accept:.4f}")
    print("=" * 60)

    if args.output_json:
        summary = {
            "implementation": "eagle3_tree_reproduction",
            "dataset": args.dataset or "single",
            "fault_log": fault_log,
            "tree_args": {
                "tree_depth": args.tree_depth,
                "tree_branch_factor": args.tree_branch_factor,
                "tree_expand_top_nodes": args.tree_expand_top_nodes,
                "tree_verify_nodes": args.tree_verify_nodes,
                "verify_backend": args.verify_backend,
                "tree_accept_mode": args.tree_accept_mode,
            },
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
            "total": total,
            "n_correct": n_correct,
            "accuracy": n_correct / total if total > 0 else 0.0,
            "avg_acceptance_rate": avg_accept,
            "results": all_results,
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Results -> {args.output_json}")


if __name__ == "__main__":
    main()
