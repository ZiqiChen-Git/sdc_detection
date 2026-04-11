---
dataset_info:
  config_name: multiple_choice
  features:
  - name: question
    dtype: string
  - name: mc1_targets
    struct:
    - name: choices
      sequence: string
    - name: labels
      sequence: int32
  - name: mc2_targets
    struct:
    - name: choices
      sequence: string
    - name: labels
      sequence: int32
  - name: input_formatted
    dtype: string
  splits:
  - name: validation
    num_bytes: 136576
    num_examples: 100
  download_size: 50299
  dataset_size: 136576
configs:
- config_name: multiple_choice
  data_files:
  - split: validation
    path: multiple_choice/validation-*
annotations_creators:
  - expert-generated
language_creators:
  - expert-generated
language:
  - en
license:
  - apache-2.0
multilinguality:
  - monolingual
size_categories:
  - n<1K
source_datasets:
  - truthful_qa
task_categories:
  - multiple-choice
  - text-generation
  - question-answering
task_ids:
  - multiple-choice-qa
  - language-modeling
  - open-domain-qa
pretty_name: tinyTruthfulQA
---
# tinyTruthfulQA

Welcome to tinyTruthfulQA! This dataset serves as a concise version of the [truthfulQA](https://huggingface.co/datasets/truthful_qa) dataset, offering a subset of 100 data points selected from the original compilation. 
tinyTruthfulQA is designed to enable users to efficiently estimate the performance of a large language model (LLM) with reduced dataset size, saving computational resources 
while maintaining the essence of the truthfulQA evaluation.

## Features

- **Compact Dataset:** With only 100 data points, tinyTruthfulQA provides a swift and efficient way to evaluate your LLM's performance against a benchmark set, maintaining the essence of the original truthfulQA dataset.
- **Compatibility:** tinyTruthfulQA is compatible with evaluation using the [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/), but can also be integrated into your custom pipeline. See below for more details.

## Model Evaluation
_With lm-eval harness_

Users looking to evaluate a new model with tinyTruthfulQA can use the [lm evaluation harness (v0.4.1 or later)](https://github.com/EleutherAI/lm-evaluation-harness/). 
To do so, you can directly run your evaluation harness with `--tasks=tinyTruthfulQA`:

```shell
lm_eval --model hf --model_args pretrained="<your-model>" --tasks=tinyTruthfulQA --batch_size=1 
```
LM-eval harness will directly output the best accuracy estimator (IRT++), without any additional work required.

_Without lm-eval harness_

Alternatively, the tinyTruthfulQA can be integrated into any other pipeline by downloading the data via

```python
from datasets import load_dataset
tiny_data = load_dataset('tinyBenchmarks/tinyTruthfulQA', 'multiple_choice')['validation']
```

Now, `tiny_data` contains the 100 subsampled data points with the same features as the original dataset, as well as an additional field containing the preformatted data points.
The preformatted data points follow the formatting used in the [open llm leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) including the respective in-context examples.


You can then estimate your LLM's performance using the following code. First, ensure you have the tinyBenchmarks package installed:

```shell
pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
```

Then, use the code snippet below for the evaluation:

```python
import numpy as np
import tinyBenchmarks as tb
### Score vector
y = # your original score vector
### Parameters
benchmark = 'truthfulqa' 
### Evaluation
tb.evaluate(y, benchmark)
```

This process will help you estimate the performance of your LLM against the tinyTruthfulQA dataset, providing a streamlined approach to benchmarking.
Please be aware that evaluating on multiple GPUs can change the order of outputs in the lm evaluation harness. 
Ordering your score vector following the original order in tinyTruthfulQA will be necessary to use the tinyBenchmarks library.

For more detailed instructions on evaluating new models and computing scores, please refer to the comprehensive guides available at [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/) and [tinyBenchmarks GitHub](https://github.com/felipemaiapolo/tinyBenchmarks).

Happy benchmarking!

## More tinyBenchmarks
**Open LLM leaderboard**:
[tiny MMLU](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU),
[tiny Arc-Challenge](https://huggingface.co/datasets/tinyBenchmarks/tinyAI2_arc),
[tiny Winogrande](https://huggingface.co/datasets/tinyBenchmarks/tinyWinogrande),
[tiny TruthfulQA](https://huggingface.co/datasets/tinyBenchmarks/tinyTruthfulQA),
[tiny GSM8k](https://huggingface.co/datasets/tinyBenchmarks/tinyGSM8k)

**AlpacaEval**:
[tiny AlpacaEval](https://huggingface.co/datasets/tinyBenchmarks/tinyAlpacaEval)

**HELM-lite**:
_work-in-progress_

## Citation

    @article{polo2024tinybenchmarks,
      title={tinyBenchmarks: evaluating LLMs with fewer examples}, 
      author={Felipe Maia Polo and Lucas Weber and Leshem Choshen and Yuekai Sun and Gongjun Xu and Mikhail Yurochkin},
      year={2024},
      eprint={2402.14992},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
      }
    @misc{lin2021truthfulqa,
      title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
      author={Stephanie Lin and Jacob Hilton and Owain Evans},
      year={2021},
      eprint={2109.07958},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }