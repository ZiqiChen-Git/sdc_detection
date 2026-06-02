# EAGLE-3 Qwen3 Reproduction

This folder is a separate reproduction workspace. It does not modify the older
`../eagle3_qwen3_speculative.py` implementation.

## What the paper says

The source paper used for this reproduction is `arXiv:2503.01840v3`,
"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via
Training-Time Test".

Main points used here:

- EAGLE-3 keeps speculative sampling's two-stage loop: draft first, then target
  verification.
- The draft model is not a standalone small LLM. It reuses target-model hidden
  features from low, middle, and high layers.
- Those three feature streams are concatenated and projected by an FC layer to
  a fused feature `g`.
- The EAGLE-3 draft layer directly predicts token distributions. It removes the
  old EAGLE feature-prediction loss/constraint.
- After the first target-sampled anchor token, later draft steps cannot obtain
  the target fused feature for unverified draft tokens. EAGLE-3 feeds the
  previous draft output vector back as the substitute feature, together with the
  sampled token embedding.
- Target verification is still strict speculative sampling: target scores draft
  tokens in parallel, while accept/reject decisions are applied from left to
  right. If a token is rejected under sampling, the replacement is sampled from
  `norm(max(0, p_target - p_draft))` in the original speculative-sampling rule.
  The local historical implementation uses the same filtered distribution for
  trace/acceptance and samples the target distribution on rejection.
- EAGLE-3 says it is compatible with the EAGLE-2 dynamic draft tree. EAGLE-2's
  policy uses draft confidence as an approximate accept probability, expands
  high-value nodes, reranks all generated nodes, and verifies the selected
  connected tree with tree attention.

## Files

- `eagle3_chain_speculative.py`
  - Chain-style EAGLE-3 speculative decoding.
  - This is copied from the repaired local Qwen3 implementation so existing
    trace fields, dataset flow, and fault injection controls are preserved.

- `eagle3_tree_speculative.py`
  - Tree-style EAGLE-3 draft generation.
  - Reuses the same draft head, target taps, trace schemas, and fault injection.
  - Adds dynamic tree expansion and tree-aware trace fields:
    `tree_node_id`, `parent_id`, `depth`, `cumulative_draft_prob`,
    `selected_for_verify`, `candidate_status`, and `tree_metadata`.

- `fault_injection.py`
  - Copied from the old local version.

- `datasets_loader.py`
  - Copied from the old local version.

## Chain example

```bash
python eagle3_chain_speculative.py \
  --base_model_id /home/czq/models/Qwen3-8B \
  --draft_model_id /home/czq/models/RedHatAI/Qwen3-8B-Thinking-speculator___eagle3 \
  --prompt "What is 25 * 48?" \
  --max_new_tokens 256 \
  --block_size 3 \
  --temperature 0.6 \
  --top_k 20 \
  --top_p 0.95 \
  --output_json outputs/eagle3_reproduction/chain_qwen3_temp06_topk20_topp095.json
```

## Tree example

```bash
python eagle3_tree_speculative.py \
  --base_model_id /home/czq/models/Qwen3-8B \
  --draft_model_id /home/czq/models/RedHatAI/Qwen3-8B-Thinking-speculator___eagle3 \
  --prompt "What is 25 * 48?" \
  --max_new_tokens 256 \
  --temperature 0.6 \
  --top_k 20 \
  --top_p 0.95 \
  --tree_depth 6 \
  --tree_branch_factor 4 \
  --tree_expand_top_nodes 2 \
  --tree_verify_nodes 16 \
  --verify_backend auto \
  --tree_accept_mode auto \
  --output_json outputs/eagle3_reproduction/tree_qwen3_depth6_nodes16_temp06_topk20_topp095.json
```

## Tree verification boundary

Official EAGLE tree verification depends on tree attention. A stock
HuggingFace Qwen3 forward may not accept or correctly apply a custom 4D tree
mask. For this reason, `eagle3_tree_speculative.py` has three verification
backends:

- `--verify_backend tree_attention`
  - Require one target forward with a custom tree mask.
  - This is closest to the paper's intended serving path.

- `--verify_backend path`
  - Verify every selected root-to-node path separately.
  - This is not a speed reproduction. It is a correctness/debug fallback.

- `--verify_backend auto`
  - Try tree attention first, then fall back to path verification.
  - The trace records the backend and fallback reason in
    `VerifyTraceEvent.tree_metadata`.

For non-greedy sampling, the tree script defaults to `single_path_strict`.
It builds the tree and records all selected nodes, but applies strict
Leviathan-style speculative acceptance to the highest-value path only. The
paper does not spell out a standalone HuggingFace-compatible stochastic
multi-branch posterior rule, so this file does not guess one.

## Fault injection

Both scripts keep the previous CLI controls:

```bash
--fault_location target_layer \
--fault_type activation \
--fault_mode single_bit \
--fault_layer_idx 16 \
--fault_module mlp.gate_proj \
--fault_phase verify \
--fault_seed 123
```

The trace still includes the old SDC fields:

- `draft_prob`
- `target_prob`
- `acceptance_ratio`
- `accepted`
- `target_topk`
- `kl_draft_target`
- `target_hidden_norm`
- `fused_feature_norm`

