---
dataset_info:
  features:
  - name: ind
    dtype: int32
  - name: activity_label
    dtype: string
  - name: ctx_a
    dtype: string
  - name: ctx_b
    dtype: string
  - name: ctx
    dtype: string
  - name: endings
    sequence: string
  - name: source_id
    dtype: string
  - name: split
    dtype: string
  - name: split_type
    dtype: string
  - name: label
    dtype: string
  - name: input_formatted
    dtype: string
  splits:
  - name: train
    num_bytes: 160899446
    num_examples: 39905
  - name: test
    num_bytes: 40288101
    num_examples: 10003
  - name: validation
    num_bytes: 473652
    num_examples: 100
  download_size: 50109798
  dataset_size: 201661199
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
language:
- en
pretty_name: tinyHellaswag
size_categories:
- n<1K
multilinguality:
  - monolingual
source_datasets:
  - Rowan/hellaswag
language_bcp47:
  - en-US
---
# tinyHellaswag

Welcome to tinyHellaswag! This dataset serves as a concise version of the [hellaswag](https://huggingface.co/datasets/hellaswag) dataset, offering a subset of 100 data points selected from the original compilation. 
tinyHellaswag is designed to enable users to efficiently estimate the performance of a large language model (LLM) with reduced dataset size, saving computational resources 
while maintaining the essence of the hellaswag evaluation.

## Features

- **Compact Dataset:** With only 100 data points, tinyHellaswag provides a swift and efficient way to evaluate your LLM's performance against a benchmark set, maintaining the essence of the original hellaswag dataset.
- **Compatibility:** tinyHellaswag is compatible with evaluation using the [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/), but can also be integrated into your custom pipeline. See below for more details.

## Model Evaluation

_With lm-eval harness_

Users looking to evaluate a new model with tinyHellaswag can use the [lm evaluation harness (v0.4.1 or later)](https://github.com/EleutherAI/lm-evaluation-harness/). 
To do so, you can directly run your evaluation harness with `--tasks=tinyHellaswag`:

```shell
lm_eval --model hf --model_args pretrained="<your-model>" --tasks=tinyHellaswag --batch_size=1 
```
LM-eval harness will directly output the best accuracy estimator (IRT++), without any additional steps required.

_Without lm-eval harness_

Alternatively, tinyHellaswag can be integrated into any other pipeline by downloading the data via

```python
from datasets import load_dataset
tiny_data = load_dataset('tinyBenchmarks/tinyHellaswag')['validation']
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
benchmark = 'hellaswag' 
### Evaluation
tb.evaluate(y, benchmark)
```

This process will help you estimate the performance of your LLM against the tinyHellaswag dataset, providing a streamlined approach to benchmarking.
Please be aware that evaluating on multiple GPUs can change the order of outputs in the lm evaluation harness. 
Ordering your score vector following the original order in tinyHellaswag will be necessary to use the tinyBenchmarks library.

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
    @inproceedings{zellers2019hellaswag,
      title={HellaSwag: Can a Machine Really Finish Your Sentence?},
      author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
      booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
      year={2019}
    }