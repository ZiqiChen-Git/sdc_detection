---
dataset_info:
  config_name: main
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: input_formatted
    dtype: string
  splits:
  - name: train
    num_bytes: 27470490
    num_examples: 7473
  - name: test
    num_bytes: 357642
    num_examples: 100
  download_size: 5523427
  dataset_size: 27828132
configs:
- config_name: main
  data_files:
  - split: train
    path: main/train-*
  - split: test
    path: main/test-*
annotations_creators:
- crowdsourced
language_creators:
- crowdsourced
language:
- en
multilinguality:
- monolingual
size_categories:
- n<1K
source_datasets:
- gsm8k
task_categories:
- text2text-generation
task_ids: []
pretty_name: tinyGSM8k
tags:
- math-word-problems
---
# tinyGSM8K

Welcome to tinyGSM8K! This dataset serves as a concise version of the [GSM8K](https://huggingface.co/datasets/gsm8k) dataset, offering a subset of 100 data points selected from the original compilation. 
tinyGSM8K is designed to enable users to efficiently estimate the performance of a large language model (LLM) with reduced dataset size, saving computational resources 
while maintaining the essence of the GSM8K evaluation.

## Features

- **Compact Dataset:** With only 100 data points, tinyGSM8K provides a swift and efficient way to evaluate your LLM's performance against a benchmark set, maintaining the essence of the original GSM8K dataset.
- **Compatibility:** tinyGSM8K is compatible with evaluation using the [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/), but can also be integrated into your custom pipeline. See below for more details.

## Model Evaluation
_With lm-eval harness_
Users looking to evaluate a new model with tinyGSM8k can use the [lm evaluation harness (v0.4.1 or later)](https://github.com/EleutherAI/lm-evaluation-harness/). 
To do so, you can directly run your evaluation harness with `--tasks=tinyGSM8k`:

```shell
lm_eval --model hf --model_args pretrained="<your-model>" --tasks=tinyGSM8k --batch_size=1 
```
LM-eval harness will directly output the best accuracy estimator (IRT++), without any additional steps required.

_Without lm-eval harness_

Alternatively, tinyGSM8k can be integrated into any other pipeline by downloading the data via

```python
from datasets import load_dataset
tiny_data = load_dataset('tinyBenchmarks/tinyGSM8K', 'main')['test']
```

Now, `tiny_data` contains the 100 subsampled data points with the same features as the original dataset, as well as an additional field containing the preformatted data points.
The preformatted data points follow the formatting used in the [open llm leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) including the respective in-context examples.
Ordering your score vector following the original order in tinyGSM8K will be necessary to use the tinyBenchmarks library.

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
benchmark = 'gsm8k' 
### Evaluation
tb.evaluate(y, benchmark)
```

This process will help you estimate the performance of your LLM against the tinyGSM8K dataset, providing a streamlined approach to benchmarking.

For more detailed instructions on evaluating new models and computing scores, please refer to the comprehensive guides available at [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/) and [tinyBenchmarks GitHub](https://github.com/felipemaiapolo/tinyBenchmarks).

Happy benchmarking!

## More tinyBenchmarks
**Open LLM leaderboard**:
[tiny MMLU](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU),
[tiny Arc-Challenge](https://huggingface.co/datasets/tinyBenchmarks/tinyAI2_arc),
[tiny Winogrande](https://huggingface.co/datasets/tinyBenchmarks/tinyWinogrande),
[tiny Hellaswag](https://huggingface.co/datasets/tinyBenchmarks/tinyHellaswag),
[tiny TruthfulQA](https://huggingface.co/datasets/tinyBenchmarks/tinyTruthfulQA),

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
    @article{cobbe2021gsm8k,
      title={Training Verifiers to Solve Math Word Problems},
      author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
      journal={arXiv preprint arXiv:2110.14168},
      year={2021}
    }