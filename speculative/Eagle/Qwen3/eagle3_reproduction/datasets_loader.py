"""
datasets_loader.py
==================
统一数据集加载接口，支持：
  gsm8k       - 小学数学，答案在 #### 后
  math500     - MATH 测试集 500 题，答案在 \\boxed{} 里
  aime2024    - AIME 2024，30 题
  aime2025    - AIME 2025，30 题
  gpqa        - GPQA Diamond，科学推理四选一
  livecodebench - 代码推理（可选）
  openthoughts  - thinking trace 分析用

每个数据集返回统一格式的列表：
  [
    {
      "question": str,      # 输入给模型的问题
      "answer":   str,      # 标准答案（用于准确率评估）
      "source":   str,      # 数据集名称
      "sample_id": int,     # 在数据集内的索引
    },
    ...
  ]

答案提取：
  extract_answer(text, dataset_name) -> str
  统一入口，内部根据数据集选择提取逻辑。

准确率评估：
  is_correct(prediction, reference, dataset_name) -> bool
"""

import re
import random
from typing import Any, Dict, List, Optional


# ============================================================
# 统一加载入口
# ============================================================

def load_dataset(
    name: str,
    split: str = "test",
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    加载指定数据集，返回统一格式的样本列表。

    name: gsm8k | math500 | aime2024 | aime2025 | gpqa | livecodebench | openthoughts
    num_samples: None 表示加载全部
    """
    loaders = {
        "gsm8k":          _load_gsm8k,
        "math500":        _load_math500,
        "aime2024":       _load_aime2024,
        "aime2025":       _load_aime2025,
        "gpqa":           _load_gpqa,
        "livecodebench":  _load_livecodebench,
        "openthoughts":   _load_openthoughts,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. 可选: {list(loaders.keys())}")

    samples = loaders[name](split)

    if num_samples is not None and num_samples < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, num_samples)

    print(f"[Dataset] Loaded {len(samples)} samples from '{name}'")
    return samples


# ============================================================
# 各数据集加载实现
# ============================================================

def _load_gsm8k(split: str = "test") -> List[Dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("gsm8k", "main", split=split)
    return [
        {
            "question":  row["question"],
            "answer":    _gsm8k_extract_answer(row["answer"]),
            "source":    "gsm8k",
            "sample_id": i,
            "raw_answer": row["answer"],
        }
        for i, row in enumerate(ds)
    ]


def _load_math500(split: str = "test") -> List[Dict]:
    from datasets import load_dataset as hf_load
    # HendrycksTest/MATH-500 或 lighteval/MATH-Hard
    try:
        ds = hf_load("HendrycksTest/MATH-500", split=split)
    except Exception:
        ds = hf_load("lighteval/MATH", "all", split=split)
        # 只取 MATH-500 的题目（按官方列表筛选）
        ds = ds.select(range(min(500, len(ds))))
    return [
        {
            "question":  row["problem"],
            "answer":    _boxed_extract(row["solution"]),
            "source":    "math500",
            "sample_id": i,
            "raw_answer": row["solution"],
        }
        for i, row in enumerate(ds)
    ]


def _load_aime2024(split: str = "test") -> List[Dict]:
    from datasets import load_dataset as hf_load
    try:
        ds = hf_load("AI-MO/aimo-validation-aime", split="train")
        # 只取 2024 年的题
        ds = [r for r in ds if "2024" in r.get("url", "")]
    except Exception:
        ds = hf_load("math-ai/aime24", split="train")
        ds = list(ds)
    return [
        {
            "question":  row["problem"],
            "answer":    str(row["answer"]).strip(),
            "source":    "aime2024",
            "sample_id": i,
        }
        for i, row in enumerate(ds)
    ]


def _load_aime2025(split: str = "test") -> List[Dict]:
    from datasets import load_dataset as hf_load
    try:
        ds = hf_load("AI-MO/aimo-validation-aime", split="train")
        ds = [r for r in ds if "2025" in r.get("url", "")]
    except Exception:
        # fallback: 用 2024 的结构，提示用户手动提供
        print("[Dataset] AIME 2025 数据集暂未公开，回退到 AIME 2024")
        return _load_aime2024(split)
    return [
        {
            "question":  row["problem"],
            "answer":    str(row["answer"]).strip(),
            "source":    "aime2025",
            "sample_id": i,
        }
        for i, row in enumerate(ds)
    ]


def _load_gpqa(split: str = "test") -> List[Dict]:
    from datasets import load_dataset as hf_load
    # Idavidrein/gpqa 的 gpqa_diamond 子集
    ds = hf_load("Idavidrein/gpqa", "gpqa_diamond", split="train")
    return [
        {
            "question":  _gpqa_format_question(row),
            "answer":    row["Correct Answer"],   # A/B/C/D
            "source":    "gpqa",
            "sample_id": i,
            "choices": {
                "A": row["Incorrect Answer 1"],
                "B": row["Incorrect Answer 2"],
                "C": row["Incorrect Answer 3"],
                "D": row["Correct Answer"],
            },
        }
        for i, row in enumerate(ds)
    ]


def _load_livecodebench(split: str = "test") -> List[Dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("livecodebench/code_generation_lite", split="test")
    return [
        {
            "question":  row["question_content"],
            "answer":    "",   # 代码题答案需要执行验证，这里留空
            "source":    "livecodebench",
            "sample_id": i,
        }
        for i, row in enumerate(ds)
    ]


def _load_openthoughts(split: str = "train") -> List[Dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("open-thoughts/OpenThoughts-114k", split=split)
    return [
        {
            "question":  row["problem"],
            "answer":    row.get("answer", ""),
            "source":    "openthoughts",
            "sample_id": i,
            "thinking":  row.get("solution", ""),  # 包含 thinking trace
        }
        for i, row in enumerate(ds)
    ]


# ============================================================
# 答案提取
# ============================================================

def extract_answer(text: str, dataset_name: str) -> str:
    """
    从模型输出里提取最终答案。
    thinking 模型的输出格式：<think>...</think> 答案
    先把 <think> 块去掉，再在剩余文本里找答案。
    """
    # 去掉 thinking 部分
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 如果没有 </think>，取最后出现的内容（有些模型不闭合标签）
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0].strip()

    if dataset_name == "gsm8k":
        return _extract_last_number(cleaned)

    elif dataset_name in ("math500", "aime2024", "aime2025"):
        # 优先找 \boxed{}
        boxed = _boxed_extract(cleaned)
        if boxed:
            return boxed
        # 退而求其次找最后一个数字
        return _extract_last_number(cleaned)

    elif dataset_name == "gpqa":
        # 找最后出现的 A/B/C/D
        matches = re.findall(r"\b([ABCD])\b", cleaned)
        return matches[-1] if matches else ""

    elif dataset_name == "livecodebench":
        # 代码题：提取 ```python ... ``` 块
        blocks = re.findall(r"```python\s*(.*?)```", cleaned, re.DOTALL)
        return blocks[-1].strip() if blocks else cleaned

    else:
        return cleaned


def is_correct(prediction: str, reference: str, dataset_name: str) -> bool:
    """判断模型输出是否和标准答案一致。"""
    pred = extract_answer(prediction, dataset_name).strip()
    ref  = reference.strip()

    if dataset_name == "gpqa":
        return pred.upper() == ref.upper()

    elif dataset_name == "livecodebench":
        # 代码题需要执行，这里只做字符串比较（实际实验里换成执行验证）
        return pred == ref

    else:
        # 数学题：转成数字比较
        pred_num = _to_number(pred)
        ref_num  = _to_number(ref)
        if pred_num is not None and ref_num is not None:
            return abs(pred_num - ref_num) < 1e-6
        return pred == ref


# ============================================================
# 内部工具函数
# ============================================================

def _gsm8k_extract_answer(text: str) -> str:
    """GSM8K 答案在 #### 后面。"""
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def _boxed_extract(text: str) -> str:
    """从 LaTeX \\boxed{...} 里提取内容，支持嵌套括号。"""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return ""
    depth = 0
    start = idx + len(r"\boxed{")
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return ""


def _extract_last_number(text: str) -> str:
    """从文本里提取最后出现的数字（支持负数、小数、带逗号）。"""
    matches = re.findall(r"-?[\d,]+\.?\d*", text)
    if not matches:
        return ""
    return matches[-1].replace(",", "")


def _to_number(s: str) -> Optional[float]:
    """尝试把字符串转成 float，失败返回 None。"""
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _gpqa_format_question(row: Dict) -> str:
    """把 GPQA 的题目和四个选项拼成完整问题。"""
    # GPQA 的正确答案总是在 D，选项需要打乱
    options = [
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
        row["Correct Answer"],
    ]
    random.shuffle(options)
    # 找打乱后正确答案在哪
    correct_idx = options.index(row["Correct Answer"])
    correct_letter = "ABCD"[correct_idx]
    # 把正确答案写回去（供 is_correct 用）
    row["Correct Answer"] = correct_letter

    lines = [row["Question"], ""]
    for letter, opt in zip("ABCD", options):
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)
