import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from speculative_gsm8k_FI import sample_from_logits

torch.manual_seed(0)
# fake logits: vocab=10, batch=1, token 7 has the highest logit
logits = torch.randn(1, 10)
logits[0, 7] = 100.0

# 1. greedy: temperature=0 should always return token 7
results = [sample_from_logits(logits, 0, None, do_sample=False)[0] for _ in range(5)]
assert all(t == 7 for t in results), f"greedy failed: {results}"
print(f"[PASS] temperature=0 greedy: always token 7 -> {results}")

# 2. do_sample=False should also always return token 7
results = [sample_from_logits(logits, 0.7, None, do_sample=False)[0] for _ in range(5)]
assert all(t == 7 for t in results), f"do_sample=False failed: {results}"
print(f"[PASS] do_sample=False greedy: always token 7 -> {results}")

# 3. do_sample=True temperature=0.7 should sometimes differ (stochastic)
results = [sample_from_logits(logits, 0.7, None, do_sample=True)[0] for _ in range(20)]
print(f"[INFO] do_sample=True sampling over 20 draws: {results}")

# 4. top_p: nucleus sampling, should restrict to high-prob tokens
even_logits = torch.zeros(1, 10)  # uniform distribution
even_logits[0, 0] = 3.0  # token 0: ~60%, token 1: ~10%, rest tiny
even_logits[0, 1] = 1.0
results = [sample_from_logits(even_logits, 1.0, None, do_sample=True, top_p=0.65)[0] for _ in range(50)]
assert all(t == 0 for t in results), f"top_p=0.65 should only sample token 0, got: {set(results)}"
print(f"[PASS] top_p=0.65 restricts to token 0: {set(results)}")

# 5. top_k + top_p combined
results = [sample_from_logits(even_logits, 1.0, top_k=3, do_sample=True, top_p=0.65)[0] for _ in range(50)]
assert all(t == 0 for t in results), f"top_k+top_p failed: {set(results)}"
print(f"[PASS] top_k=3 + top_p=0.65: {set(results)}")

print("\nAll tests passed.")
