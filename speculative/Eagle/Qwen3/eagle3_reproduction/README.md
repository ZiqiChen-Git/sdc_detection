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

Both decoder scripts keep the previous single-run CLI controls:

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

For statistical fault injection, use `eagle3_fault_runner.py`. It keeps the
chain/tree decoder logic unchanged and injects one fault per inference trial.
The default fault mode is double-bit, matching the main memory-fault emphasis
in the SC2025 reliability setup.

Each run creates a clear output folder. If `--output_dir` is omitted, the
folder is created under `outputs/eagle3_reproduction/fault_runs/` with a
timestamp and fault-setting slug. The layout is:

```text
<run_dir>/
  raw_results.json
  baselines/sample_<sample_id>.json
  trials/trial_<trial_idx>_sample_<sample_id>.json
  analysis/analysis_summary.json
  analysis/analysis_trials.jsonl
  analysis/analysis_by_site.json
  analysis/analysis_report.md
```

Baseline and trial JSON files include the normal generation metrics, tokens,
and full trace by default. Use `--trace_mode summary` or `--no_store_tokens`
only when the run is too large.

### Preliminary TODO workflow

This workflow directly addresses the two preliminary TODO items:

- SDC effect on reasoning output.
- SDC effect on execution status.

Run a small pilot first:

```bash
cd ~/projects/sdc_detection/speculative/Eagle/Qwen3/eagle3_reproduction

python eagle3_fault_runner.py \
  --base_model_id /home/czq/models/Qwen3-8B \
  --draft_model_id /home/czq/models/RedHatAI/Qwen3-8B-Thinking-speculator___eagle3 \
  --dataset gsm8k \
  --num_samples 5 \
  --num_fault_trials 5 \
  --decoder chain \
  --max_new_tokens 256 \
  --block_size 3 \
  --temperature 0.2 \
  --top_k 10 \
  --top_p 0.9 \
  --fault_location random \
  --fault_type activation \
  --fault_mode double_bit \
  --fault_phase verify \
  --output_dir outputs/eagle3_reproduction/fault_runs/prelim_gsm8k_verify_activation
```

Generate the TODO-specific report:

```bash
python eagle3_prelim_todo_report.py \
  --run_dir outputs/eagle3_reproduction/fault_runs/prelim_gsm8k_verify_activation
```

Read:

```text
outputs/eagle3_reproduction/fault_runs/prelim_gsm8k_verify_activation/analysis/preliminary_todo_report.md
```

For reasoning-output effect, report:

- `output_changed_rate`
- `token_changed_rate`
- `baseline_correct_then_wrong_rate`
- `mean_token_edit_distance_norm`
- `mean_acceptance_delta`
- `mean_verify_kl_delta`

For execution-status effect, report:

- `execution_status_changed_rate`
- `baseline_success_then_fault_error_rate`
- `fault execution status counts`
- `fault error type counts`

Target weight fault with a fully controlled site:

```bash
python eagle3_fault_runner.py \
  --base_model_id /home/czq/models/Qwen3-8B \
  --draft_model_id /home/czq/models/RedHatAI/Qwen3-8B-Thinking-speculator___eagle3 \
  --prompt "What is 25 * 48?" \
  --num_fault_trials 10 \
  --fault_location target_layer \
  --fault_type weight \
  --fault_mode double_bit \
  --fault_layer_idx 16 \
  --fault_module mlp.gate_proj \
  --fault_row 0 \
  --fault_col 0 \
  --fault_bit_positions 7,12 \
  --output_dir outputs/eagle3_reproduction/fault_runs/target_layer_weight_demo
```

Target verify-only activation fault:

```bash
python eagle3_fault_runner.py \
  --base_model_id /home/czq/models/Qwen3-8B \
  --draft_model_id /home/czq/models/RedHatAI/Qwen3-8B-Thinking-speculator___eagle3 \
  --prompt "What is 25 * 48?" \
  --num_fault_trials 10 \
  --fault_location target_layer \
  --fault_type activation \
  --fault_mode double_bit \
  --fault_layer_idx 16 \
  --fault_module mlp.gate_proj \
  --fault_phase verify \
  --fault_token_idx 0 \
  --fault_hidden_idx 128 \
  --output_dir outputs/eagle3_reproduction/fault_runs/target_verify_activation_demo
```

EAGLE-3 tap-only feature fault:

```bash
python eagle3_fault_runner.py \
  --base_model_id /home/czq/models/Qwen3-8B \
  --draft_model_id /home/czq/models/RedHatAI/Qwen3-8B-Thinking-speculator___eagle3 \
  --prompt "What is 25 * 48?" \
  --num_fault_trials 10 \
  --fault_location target_tap \
  --fault_type activation \
  --fault_mode double_bit \
  --fault_tap_slot mid \
  --fault_phase prefill \
  --output_dir outputs/eagle3_reproduction/fault_runs/mid_tap_activation_demo
```

Useful controllable fields:

- `--fault_location`: `target_layer`, `target_tap`, `target_embed`,
  `target_lm_head`, `draft_embed`, `draft_fc`, `draft_layer`, `draft_lm_head`,
  or `random`.
- `--fault_layer_idx` and `--fault_module`: choose transformer layer and module.
  If omitted for `target_layer` / `draft_layer`, the runner samples them.
- `--fault_row`, `--fault_col`: choose exact weight coordinates.
- `--fault_token_idx`, `--fault_hidden_idx`: choose exact activation coordinates.
- `--fault_bit_positions`: choose exact bit positions such as `7,12`.
- `--fault_phase`: for target-side activation faults, choose `prefill`,
  `verify`, or `both`.
- `--decoder tree`: run the same FI harness on the tree reproduction.
- `--no_auto_analyze`: skip automatic analysis if you only want raw traces.

## Linear-layer profile pre-experiment

Use `eagle3_linear_profile.py` before large FI runs if you want a small
evidence table for why FI focuses on high-compute / high-memory linear layers.
It records target linear modules, tap-vs-non-tap target layers, `draft_fc`, and
draft-layer linear modules.

```bash
python eagle3_linear_profile.py \
  --base_model_id /home/czq/models/Qwen3-8B \
  --draft_model_id /home/czq/models/RedHatAI/Qwen3-8B-Thinking-speculator___eagle3 \
  --prompt "What is 25 * 48?" \
  --max_new_tokens 32 \
  --profile_iterations 1 \
  --block_size 3 \
  --temperature 0.2 \
  --top_k 10 \
  --top_p 0.9 \
  --output_dir outputs/eagle3_reproduction/profiles/linear_profile_32tok
```

Outputs:

```text
<output_dir>/
  profile_summary.json
  module_profile.csv
  profile_report.md
```

The report is only for experiment planning, not a rigorous serving benchmark.
Forward hooks add overhead, but the relative module-size and rough module-time
pattern is enough to justify concentrating FI on linear layers.
