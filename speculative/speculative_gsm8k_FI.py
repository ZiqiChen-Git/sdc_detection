import argparse
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from modelscope import snapshot_download
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed: int = 196) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6


def safe_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def perform_bit_flip_double(value: torch.Tensor, bit_positions: Tuple[int, int]) -> torch.Tensor:
    with torch.no_grad():
        tensor_bf16 = value.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        mask = (1 << bit_positions[0]) | (1 << bit_positions[1])
        flipped = bits ^ mask
    return flipped.view(torch.bfloat16)


def perform_bit_flip_single(value: torch.Tensor, bit_position: int) -> torch.Tensor:
    with torch.no_grad():
        tensor_bf16 = value.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        mask = 1 << bit_position
        flipped = bits ^ mask
    return flipped.view(torch.bfloat16)


def load_tiny_gsm8k(bundle_root: str, split: str = "test", num_samples: int = 100, seed: int = 42):
    data_dir = os.path.join(bundle_root, "tinyGSM8k", "main")
    split_file = os.path.join(data_dir, f"{split}-00000-of-00001.parquet")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Could not find split file at {split_file}")
    dataset = load_dataset("parquet", data_files={split: split_file})[split]
    if num_samples and num_samples < len(dataset):
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)
    return dataset


def format_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"Question: {question}\nAnswer:"


def build_prompt(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Tuple[str, str]:
    question = example["question"]
    answer = example["answer"]
    prompt = format_prompt(tokenizer, question)
    return prompt, answer


def extract_final_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def extract_last_number(text: str) -> Optional[float]:
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not matches:
        return None
    for raw in reversed(matches):
        candidate = raw[0] if raw[0] else raw[1]
        if not candidate:
            continue
        cleaned = candidate.replace("$", "").replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


def is_answer_correct(prediction: str, reference: str) -> bool:
    pred_number = extract_last_number(prediction)
    try:
        ref_number = float(reference)
    except Exception:
        return False
    if pred_number is None or ref_number is None:
        return False
    return pred_number == ref_number


def download_model_from_modelscope(model_id: str, cache_dir: str, revision: Optional[str] = None) -> str:
    return snapshot_download(model_id, cache_dir=cache_dir, revision=revision)


def load_model_and_tokenizer(model_id: str, cache_dir: str, revision: Optional[str], dtype: torch.dtype) -> Tuple[Any, AutoTokenizer]:
    local_dir = download_model_from_modelscope(model_id, cache_dir, revision)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=DEVICE,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model, tokenizer


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> Tuple[int, float, List[Dict[str, Any]]]:
    if temperature <= 0:
        probs = torch.softmax(logits, dim=-1)
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
    probs = probs.to(torch.float32)
    if top_k is not None and top_k < probs.shape[-1]:
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(topk_probs, num_samples=1)
        token_id = topk_indices.gather(-1, idx).item()
        prob = topk_probs.gather(-1, idx).item()
    else:
        idx = torch.multinomial(probs, num_samples=1)
        token_id = idx.item()
        prob = probs[0, token_id].item()
    view_k = min(20, probs.shape[-1])
    view_probs, view_idx = torch.topk(probs, view_k, dim=-1)
    topk_info = [
        {"token_id": view_idx[0, i].item(), "prob": view_probs[0, i].item()}
        for i in range(view_k)
    ]
    return token_id, prob, topk_info


class SpeculativeDecoder:
    def __init__(
        self,
        base_model,
        base_tokenizer,
        draft_model,
        draft_tokenizer,
        block_size: int = 4,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        hidden_slice: int = 32,
    ):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.block_size = block_size
        self.temperature = temperature
        self.top_k = top_k
        self.hidden_slice = hidden_slice

    @torch.no_grad()
    def _propose_with_draft(
        self,
        context_ids: torch.Tensor,
        eos_token_id: Optional[int],
    ) -> Dict[str, Any]:
        proposals = []
        working_ids = context_ids.clone()
        forward_calls = 0
        draft_time = 0.0
        for _ in range(self.block_size):
            start = time.time()
            outputs = self.draft_model(working_ids, use_cache=False, output_hidden_states=False)
            draft_time += time.time() - start
            forward_calls += 1
            logits = outputs.logits[:, -1, :]
            token_id, token_prob, topk_info = sample_from_logits(logits, self.temperature, self.top_k)
            token_tensor = torch.tensor([[token_id]], device=working_ids.device)
            proposals.append(
                {
                    "token_id": token_id,
                    "token_tensor": token_tensor,
                    "prob": token_prob,
                    "topk": topk_info,
                }
            )
            working_ids = torch.cat([working_ids, token_tensor], dim=1)
            if eos_token_id is not None and token_id == eos_token_id:
                break
        return {
            "proposals": proposals,
            "draft_time": draft_time,
            "forward_calls": forward_calls,
        }

    @torch.no_grad()
    def _forward_teacher(self, context_ids: torch.Tensor) -> Dict[str, Any]:
        start = time.time()
        outputs = self.base_model(
            context_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        elapsed = time.time() - start
        logits = outputs.logits[:, -1, :]
        if self.temperature <= 0:
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
        view_k = min(20, probs.shape[-1])
        topk_probs, topk_idx = torch.topk(probs, view_k, dim=-1)
        topk = []
        for i in range(view_k):
            token_id = topk_idx[0, i].item()
            topk.append(
                {
                    "token_id": token_id,
                    "prob": topk_probs[0, i].item(),
                    "token": self.base_tokenizer.decode([token_id]),
                }
            )
        sample_id, sample_prob, _ = sample_from_logits(logits, self.temperature, self.top_k)
        hidden_state = outputs.hidden_states[-1][0, -1, :].detach().float()
        return {
            "probs": probs,
            "hidden_state": hidden_state,
            "topk": topk,
            "sample_id": sample_id,
            "sample_prob": sample_prob,
            "time": elapsed,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        context_ids = input_ids.clone()
        generated_ids: List[int] = []
        trace: List[Dict[str, Any]] = []
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
        }
        start_time = time.time()
        while len(generated_ids) < max_new_tokens:
            metrics["iteration_count"] += 1
            proposal_pack = self._propose_with_draft(context_ids, eos_token_id)
            proposals = proposal_pack["proposals"]
            metrics["draft_forward_calls"] += proposal_pack["forward_calls"]
            metrics["draft_time"] += proposal_pack["draft_time"]
            if not proposals:
                break
            metrics["proposals_generated"] += len(proposals)
            accept_all = True
            for local_step, proposal in enumerate(proposals):
                if len(generated_ids) >= max_new_tokens:
                    break
                teacher_info = self._forward_teacher(context_ids)
                metrics["base_forward_calls"] += 1
                metrics["base_time"] += teacher_info["time"]
                base_probs = teacher_info["probs"]
                proposed_prob = base_probs[0, proposal["token_id"]].item()
                acceptance_ratio = min(1.0, (proposed_prob + EPS) / (proposal["prob"] + EPS))
                accepted = random.random() < acceptance_ratio
                hidden_slice = teacher_info["hidden_state"][: self.hidden_slice].tolist()
                draft_decoded = self.draft_tokenizer.decode([proposal["token_id"]])
                base_decoded = self.base_tokenizer.decode([proposal["token_id"]])
                draft_topk = [
                    {
                        **entry,
                        "token": self.draft_tokenizer.decode([entry["token_id"]]),
                    }
                    for entry in proposal["topk"]
                ]
                trace.append(
                    {
                        "step": len(trace),
                        "source": "draft",
                        "draft_token_id": proposal["token_id"],
                        "draft_token": draft_decoded,
                        "base_vocab_token": base_decoded,
                        "draft_prob": proposal["prob"],
                        "base_prob": proposed_prob,
                        "acceptance_ratio": acceptance_ratio,
                        "accepted": accepted,
                        "base_topk": teacher_info["topk"],
                        "draft_topk": draft_topk,
                        "hidden_state_slice": hidden_slice,
                        "reason": "draft_token_verification",
                    }
                )
                if accepted:
                    metrics["accepted_proposals"] += 1
                    generated_ids.append(proposal["token_id"])
                    context_ids = torch.cat([context_ids, proposal["token_tensor"]], dim=1)
                else:
                    accept_all = False
                    metrics["rejected_proposals"] += 1
                    metrics["base_only_tokens"] += 1
                    base_token_tensor = torch.tensor([[teacher_info["sample_id"]]], device=context_ids.device)
                    generated_ids.append(teacher_info["sample_id"])
                    context_ids = torch.cat([context_ids, base_token_tensor], dim=1)
                    trace.append(
                        {
                            "step": len(trace),
                            "source": "base",
                            "base_token_id": teacher_info["sample_id"],
                            "base_token": self.base_tokenizer.decode([teacher_info["sample_id"]]),
                            "base_prob": teacher_info["sample_prob"],
                            "hidden_state_slice": hidden_slice,
                            "reason": "base_resample_after_rejection",
                            "base_topk": teacher_info["topk"],
                        }
                    )
                    break
                if eos_token_id is not None and proposal["token_id"] == eos_token_id:
                    accept_all = False
                    break
            if accept_all and len(generated_ids) < max_new_tokens:
                teacher_info = self._forward_teacher(context_ids)
                metrics["base_forward_calls"] += 1
                metrics["base_time"] += teacher_info["time"]
                base_token_id = teacher_info["sample_id"]
                base_token_tensor = torch.tensor([[base_token_id]], device=context_ids.device)
                generated_ids.append(base_token_id)
                context_ids = torch.cat([context_ids, base_token_tensor], dim=1)
                trace.append(
                    {
                        "step": len(trace),
                        "source": "base",
                        "base_token_id": base_token_id,
                        "base_token": self.base_tokenizer.decode([base_token_id]),
                        "base_prob": teacher_info["sample_prob"],
                        "hidden_state_slice": teacher_info["hidden_state"][: self.hidden_slice].tolist(),
                        "reason": "base_bridge_token",
                        "base_topk": teacher_info["topk"],
                    }
                )
            if eos_token_id is not None and generated_ids and generated_ids[-1] == eos_token_id:
                break
        metrics["generation_time"] = time.time() - start_time
        metrics["acceptance_rate"] = (
            metrics["accepted_proposals"] / metrics["proposals_generated"]
            if metrics["proposals_generated"] > 0
            else 0.0
        )
        metrics["base_verification_ratio"] = (
            metrics["base_forward_calls"] / metrics["draft_forward_calls"]
            if metrics["draft_forward_calls"] > 0
            else 0.0
        )
        metrics["tokens_emitted"] = len(generated_ids)
        metrics["sdc_alert"] = metrics["rejected_proposals"] > 0
        if generated_ids:
            generated_tensor = torch.tensor(generated_ids, device=context_ids.device).unsqueeze(0)
            decoded = self.base_tokenizer.decode(generated_tensor[0], skip_special_tokens=True)
        else:
            decoded = ""
        return {
            "text": decoded.strip(),
            "tokens": generated_ids,
            "trace": trace,
            "metrics": metrics,
        }


def select_module(model, model_type: str) -> Tuple[int, str, torch.nn.Module]:
    if model_type == "qwen":
        layer_weights = {
            "self_attn.v_proj": 1,
            "self_attn.k_proj": 1,
            "self_attn.q_proj": 7,
            "self_attn.o_proj": 7,
            "mlp.up_proj": 37,
            "mlp.gate_proj": 37,
            "mlp.down_proj": 37,
        }
        num_layers = len(model.model.layers)
        get_layer = lambda idx: model.model.layers[idx]
    elif model_type == "falcon":
        layer_weights = {
            "self_attn.v_proj": 2,
            "self_attn.k_proj": 2,
            "self_attn.q_proj": 6,
            "self_attn.o_proj": 6,
            "mlp.up_proj": 45,
            "mlp.gate_proj": 45,
            "mlp.down_proj": 45,
        }
        num_layers = len(model.model.layers)
        get_layer = lambda idx: model.model.layers[idx]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    layer_idx = random.randint(0, num_layers - 1)
    layers = list(layer_weights.keys())
    weights = [layer_weights[name] for name in layers]
    module_name = random.choices(layers, weights=weights)[0]
    target = get_layer(layer_idx)
    for attr in module_name.split("."):
        target = getattr(target, attr)
    return layer_idx, module_name, target


def create_activation_hook(bit_positions: Any, fault_mode: str):
    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if not hasattr(hook, "triggered") or not hook.triggered:
            seq_dim = tensor.shape[1]
            feat_dim = tensor.shape[2]
            x = random.randrange(seq_dim)
            y = random.randrange(feat_dim)
            metadata = {
                "seq_index": x,
                "feature_index": y,
                "tensor_shape": list(tensor.shape),
            }
            if fault_mode == "neuron":
                tensor[0, x, y] = perform_bit_flip_double(tensor[0, x, y], bit_positions)
            else:
                tensor[0, x, y] = perform_bit_flip_single(tensor[0, x, y], bit_positions)
            hook.triggered = True
            hook.metadata = metadata
        return output

    hook.triggered = False
    hook.metadata = {}
    return hook


def write_jsonl(fp, payload: Dict[str, Any]) -> None:
    fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    summary_fields = [
        "proposals_generated",
        "accepted_proposals",
        "rejected_proposals",
        "base_only_tokens",
        "draft_forward_calls",
        "base_forward_calls",
        "draft_time",
        "base_time",
        "iteration_count",
        "generation_time",
        "acceptance_rate",
        "base_verification_ratio",
        "tokens_emitted",
    ]
    summary = {}
    for field in summary_fields:
        values = [m.get(field, 0.0) for m in metrics_list]
        summary[field] = float(sum(values) / len(values))
    summary["sdc_alert_rate"] = float(sum(1 for m in metrics_list if m.get("sdc_alert")) / len(metrics_list))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Speculative decoding GSM8K fault injection.")
    parser.add_argument("--bundle_root", type=str, default="data_bundle")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--fault_mode", type=str, default="weight", choices=["weight", "neuron", "single"])
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--hidden_slice", type=int, default=128)
    parser.add_argument("--base_model_id", type=str, default=None)
    parser.add_argument("--draft_model_id", type=str, default=None)
    parser.add_argument("--base_revision", type=str, default=None)
    parser.add_argument("--draft_revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="qwen", choices=["qwen", "falcon"])
    parser.add_argument("--output_dir", type=str, default="speculative_outputs")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--baseline_runs", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    traces_dir = os.path.join(args.output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    if args.cache_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, ".."))
        args.cache_dir = os.path.abspath(os.path.join(repo_root, "..", "modelscope_cache"))
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.base_model_id is None:
        if args.model_type == "qwen":
            args.base_model_id = "qwen/Qwen2.5-7B-Instruct"
        else:
            args.base_model_id = "tiiuae/Falcon3-7B-Instruct"
    if args.draft_model_id is None:
        if args.model_type == "qwen":
            args.draft_model_id = "qwen/Qwen2.5-1.5B-Instruct"
        else:
            args.draft_model_id = "tiiuae/Falcon3-1B"

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    dataset = load_tiny_gsm8k(args.bundle_root, num_samples=args.num_samples)
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model_id, args.cache_dir, args.base_revision, dtype)
    draft_model, draft_tokenizer = load_model_and_tokenizer(args.draft_model_id, args.cache_dir, args.draft_revision, dtype)

    decoder = SpeculativeDecoder(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        block_size=args.block_size,
        temperature=args.temperature,
        top_k=args.top_k,
        hidden_slice=args.hidden_slice,
    )

    all_answers = open(os.path.join(args.output_dir, "all_answers.jsonl"), "w", encoding="utf-8")
    baseline_answers_fp = open(os.path.join(args.output_dir, "baseline_answers.jsonl"), "w", encoding="utf-8")
    diff_answers = open(os.path.join(args.output_dir, "different_answers.jsonl"), "w", encoding="utf-8")

    baseline_predictions = {}
    baseline_metrics_all: List[Dict[str, Any]] = []
    baseline_runs_info: List[Dict[str, Any]] = []
    baseline_correct_first = 0
    print("Running speculative decoding baseline...")
    for run_idx in range(args.baseline_runs):
        print(f"Baseline run {run_idx+1}/{args.baseline_runs}")
        run_trace_dir = os.path.join(traces_dir, f"baseline_run_{run_idx}")
        os.makedirs(run_trace_dir, exist_ok=True)
        run_metrics: List[Dict[str, Any]] = []
        run_correct = 0
        for sample_idx in tqdm(range(len(dataset)), desc=f"Baseline run {run_idx}"):
            prompt, answer = build_prompt(dataset[sample_idx], base_tokenizer)
            reference = extract_final_answer(answer)
            input_ids = base_tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            result = decoder.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=base_tokenizer.eos_token_id,
            )
            is_correct = is_answer_correct(result["text"], reference)
            if run_idx == 0:
                baseline_predictions[sample_idx] = result["text"]
                if is_correct:
                    baseline_correct_first += 1
            if is_correct:
                run_correct += 1
            run_metrics.append(result["metrics"])
            trace_path = os.path.join(run_trace_dir, f"sample_{sample_idx}.json")
            with open(trace_path, "w", encoding="utf-8") as trace_fp:
                json.dump(result, trace_fp, ensure_ascii=False, indent=2)
            baseline_payload = {
                "mode": "baseline",
                "baseline_run": run_idx,
                "sample_id": sample_idx,
                "prompt": prompt,
                "reference": reference,
                "prediction": result["text"],
                "is_correct": is_correct,
                "speculative_metrics": result["metrics"],
                "trace_file": trace_path,
            }
            write_jsonl(baseline_answers_fp, baseline_payload)
            write_jsonl(all_answers, baseline_payload)
        baseline_metrics_all.extend(run_metrics)
        baseline_runs_info.append(
            {
                "run": run_idx,
                "accuracy": run_correct / len(dataset),
                "metric_summary": aggregate_metrics(run_metrics),
            }
        )

    baseline_accuracy = baseline_runs_info[0]["accuracy"]
    baseline_summary = aggregate_metrics(baseline_metrics_all)
    print(f"Baseline speculative accuracy (run 0): {baseline_accuracy:.4f}")

    print("Running fault injection trials...")
    trial_summaries = []
    for trial in range(args.num_trials):
        layer_idx, module_name, target_module = select_module(base_model, args.model_type)
        bit_positions = random.sample(range(16), 2)
        weight_snapshot = None
        hook_handle = None
        hook = None
        if args.fault_mode == "weight":
            weight_tensor = target_module.weight
            x = random.randint(0, weight_tensor.shape[0] - 1)
            y = random.randint(0, weight_tensor.shape[1] - 1)
            weight_snapshot = (x, y, weight_tensor[x, y].clone())
            with torch.no_grad():
                weight_tensor[x, y] = perform_bit_flip_double(weight_tensor[x, y], bit_positions)
        else:
            if args.fault_mode == "single":
                bit_positions = random.randint(0, 15)
            hook = create_activation_hook(bit_positions, args.fault_mode)
            hook_handle = target_module.register_forward_hook(hook)

        trial_desc = {
            "trial": trial,
            "fault_mode": args.fault_mode,
            "layer_idx": layer_idx,
            "module": module_name,
            "bit_positions": bit_positions,
        }

        trial_progress = tqdm(range(len(dataset)), desc=f"Trial {trial}")
        trial_metrics: List[Dict[str, Any]] = []
        trial_correct = 0
        for sample_idx in trial_progress:
            if hook is not None:
                hook.triggered = False
                hook.metadata = {}
            prompt, answer = build_prompt(dataset[sample_idx], base_tokenizer)
            reference = extract_final_answer(answer)
            input_ids = base_tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            result = decoder.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=base_tokenizer.eos_token_id,
            )
            is_correct = is_answer_correct(result["text"], reference)
            trace_path = os.path.join(traces_dir, f"trial{trial}_sample_{sample_idx}.json")
            with open(trace_path, "w", encoding="utf-8") as trace_fp:
                json.dump(result, trace_fp, ensure_ascii=False, indent=2)
            payload = {
                **trial_desc,
                "sample_id": sample_idx,
                "reference": reference,
                "prediction": result["text"],
                "baseline_prediction": baseline_predictions[sample_idx],
                "is_correct": is_correct,
                "speculative_metrics": result["metrics"],
                "trace_file": trace_path,
            }
            if hook is not None and hook.metadata:
                payload["activation_metadata"] = hook.metadata
            write_jsonl(all_answers, payload)
            if result["text"] != baseline_predictions[sample_idx]:
                write_jsonl(diff_answers, payload)
            safe_empty_cache()
            trial_metrics.append(result["metrics"])
            if is_correct:
                trial_correct += 1

        if args.fault_mode == "weight" and weight_snapshot is not None:
            with torch.no_grad():
                x, y, original_value = weight_snapshot
                target_module.weight[x, y] = original_value
        if hook_handle is not None:
            hook_handle.remove()
        trial_summaries.append(
            {
                **trial_desc,
                "accuracy": trial_correct / len(dataset),
                "metric_summary": aggregate_metrics(trial_metrics),
            }
        )

    all_answers.close()
    baseline_answers_fp.close()
    diff_answers.close()
    summary_path = os.path.join(args.output_dir, "speculative_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_fp:
        json.dump(
            {
                "baseline_accuracy": baseline_accuracy,
                "baseline_runs": baseline_runs_info,
                "baseline_metric_summary": baseline_summary,
                "trials": trial_summaries,
            },
            summary_fp,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()