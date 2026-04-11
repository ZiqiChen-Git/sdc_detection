---
language:
- en
dataset_info:
  config_name: ARC-Challenge
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: choices
    sequence:
    - name: text
      dtype: string
    - name: label
      dtype: string
  - name: answerKey
    dtype: string
  - name: input_formatted
    dtype: string
  splits:
  - name: train
    num_bytes: 4776965
    num_examples: 1119
  - name: test
    num_bytes: 496912
    num_examples: 100
  - name: validation
    num_bytes: 1281856
    num_examples: 299
  download_size: 1154855
  dataset_size: 6555733
configs:
- config_name: ARC-Challenge
  data_files:
  - split: train
    path: ARC-Challenge/train-*
  - split: test
    path: ARC-Challenge/test-*
  - split: validation
    path: ARC-Challenge/validation-*
task_categories:
- question-answering
pretty_name: tinyArc
size_categories:
- n<1K
multilinguality:
  - monolingual
source_datasets:
  - allenai/ai2_arc
task_ids:
  - open-domain-qa
  - multiple-choice-qa
language_bcp47:
  - en-US
---
# tinyAI2_arc

Welcome to tinyAI2_arc! This dataset serves as a concise version of the [AI2_arc challenge dataset](https://huggingface.co/datasets/allenai/ai2_arc), offering a subset of 100 data points selected from the original compilation. 
tinyAI2_arc is designed to enable users to efficiently estimate the performance of a large language model (LLM) with reduced dataset size, saving computational resources 
while maintaining the essence of the ARC challenge evaluation.

## Features

- **Compact Dataset:** With only 100 data points, tinyAI2_arc provides a swift and efficient way to evaluate your LLM's performance against a benchmark set, maintaining the essence of the original ARC challenge dataset.
- **Compatibility:** tinyAI2_arc is compatible with evaluation using the [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/), but can also be integrated into your custom pipeline. See below for more details.

## Model Evaluation
_With lm-eval harness_
Users looking to evaluate a new model with tinyAI2_arc can use the [lm evaluation harness (v0.4.1 or later)](https://github.com/EleutherAI/lm-evaluation-harness/). 
To do so, you can directly run your evaluation harness with `--tasks=tinyArc`:

```shell
lm_eval --model hf --model_args pretrained="<your-model>" --tasks=tinyArc --batch_size=1 
```
LM-eval harness will directly output the best accuracy estimator (IRT++), without any additional steps required.

_Without lm-eval harness_

Alternatively, tinyAI2_arc can be integrated into any other pipeline by downloading the data via

```python
from datasets import load_dataset
tiny_data = load_dataset('tinyBenchmarks/tinyAI2_arc', 'ARC-Challenge')['test']
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
benchmark = 'arc' 
### Evaluation
tb.evaluate(y, benchmark)
```

This process will help you estimate the performance of your LLM against the tinyAI2_arc dataset, providing a streamlined approach to benchmarking.
Please be aware that evaluating on multiple GPUs can change the order of outputs in the lm evaluation harness. 
Ordering your score vector following the original order in tinyAI2_arc will be necessary to use the tinyBenchmarks library.

For more detailed instructions on evaluating new models and computing scores, please refer to the comprehensive guides available at [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/) and [tinyBenchmarks GitHub](https://github.com/felipemaiapolo/tinyBenchmarks).

Happy benchmarking!

## More tinyBenchmarks
**Open LLM leaderboard**:
[tiny MMLU](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU),
[tiny Winogrande](https://huggingface.co/datasets/tinyBenchmarks/tinyWinogrande),
[tiny Hellaswag](https://huggingface.co/datasets/tinyBenchmarks/tinyHellaswag),
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
    @article{allenai:arc,
      author    = {Peter Clark  and Isaac Cowhey and Oren Etzioni and Tushar Khot and
                    Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
      title     = {Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
      journal   = {arXiv:1803.05457v1},
      year      = {2018},
    }