---
dataset_info:
  config_name: winogrande_xl
  features:
  - name: sentence
    dtype: string
  - name: option1
    dtype: string
  - name: option2
    dtype: string
  - name: answer
    dtype: string
  - name: input_formatted
    dtype: string
  splits:
  - name: train
    num_bytes: 29034018
    num_examples: 40398
  - name: test
    num_bytes: 1273510
    num_examples: 1767
  - name: validation
    num_bytes: 74654
    num_examples: 100
  download_size: 5558675
  dataset_size: 30382182
configs:
- config_name: winogrande_xl
  data_files:
  - split: train
    path: winogrande_xl/train-*
  - split: test
    path: winogrande_xl/test-*
  - split: validation
    path: winogrande_xl/validation-*
language:
- en
pretty_name: tinyWinogrande
multilinguality:
- monolingual
source_datasets:
- winogrande
language_bcp47:
- en-US
size_categories:
- n<1K
---
# tinyWinogrande

Welcome to tinyWinogrande! This dataset serves as a concise version of the [Winogrande](https://huggingface.co/datasets/winogrande) dataset, offering a subset of 100 data points selected from the original compilation. 
tinyWinogrande is designed to enable users to efficiently estimate the performance of a large language model (LLM) with reduced dataset size, saving computational resources 
while maintaining the essence of the Winogrande evaluation.

## Features

- **Compact Dataset:** With only 100 data points, tinyWinogrande provides a swift and efficient way to evaluate your LLM's performance against a benchmark set, maintaining the essence of the original Winogrande dataset.
- **Compatibility:** tinyWinogrande is compatible with evaluation using the [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/), but can also be integrated into your custom pipeline. See below for more details.

## Model Evaluation

_With lm-eval harness_

Users looking to evaluate a new model with tinyWinogrande can use the [lm evaluation harness (v0.4.1 or later)](https://github.com/EleutherAI/lm-evaluation-harness/). 
To do so, you can directly run your evaluation harness with `--tasks=tinyWinogrande`:

```shell
lm_eval --model hf --model_args pretrained="<your-model>" --tasks=tinyWinogrande --batch_size=1 
```
LM-eval harness will directly output the best accuracy estimator (IRT++), without any additional work required.

_Without lm-eval harness_

Alternatively, tinyWinogrande can be integrated into any other pipeline by downloading the data via

```python
from datasets import load_dataset
tiny_data = load_dataset('tinyBenchmarks/tinyWinogrande', 'winogrande_xl')['validation']
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
benchmark = 'winogrande' 
### Evaluation
tb.evaluate(y, benchmark)
```

This process will help you estimate the performance of your LLM against the tinyWinogrande dataset, providing a streamlined approach to benchmarking.
Please be aware that evaluating on multiple GPUs can change the order of outputs in the lm evaluation harness. 
Ordering your score vector following the original order in tinyWinogrande will be necessary to use the tinyBenchmarks library.

For more detailed instructions on evaluating new models and computing scores, please refer to the comprehensive guides available at [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/) and [tinyBenchmarks GitHub](https://github.com/felipemaiapolo/tinyBenchmarks).

Happy benchmarking!

## More tinyBenchmarks
**Open LLM leaderboard**:
[tiny MMLU](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU),
[tiny Arc-Challenge](https://huggingface.co/datasets/tinyBenchmarks/tinyAI2_arc),
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
    @InProceedings{ai2:winogrande,
      title = {WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
      authors={Keisuke, Sakaguchi and Ronan, Le Bras and Chandra, Bhagavatula and Yejin, Choi},
      year={2019}
      }