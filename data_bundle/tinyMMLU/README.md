---
annotations_creators:
- no-annotation
language_creators:
- expert-generated
language:
- en
multilinguality:
- monolingual
source_datasets:
- cais/mmlu
task_categories:
- question-answering
task_ids:
- multiple-choice-qa
pretty_name: tinyMMLU
dataset_info:
  config_name: all
  features:
  - name: question
    dtype: string
  - name: subject
    dtype: string
  - name: choices
    sequence: string
  - name: answer
    dtype:
      class_label:
        names:
          '0': A
          '1': B
          '2': C
          '3': D
  - name: input_formatted
    dtype: string
  splits:
  - name: test
    num_bytes: 337628
    num_examples: 100
  - name: dev
    num_bytes: 858526
    num_examples: 285
  download_size: 1671192
  dataset_size: 6621454
configs:
- config_name: all
  data_files:
  - split: test
    path: all/test-*
  - split: dev
    path: all/dev-*
language_bcp47:
- en-US
---
# tinyMMLU

Welcome to tinyMMLU! This dataset serves as a concise version of the [MMLU](https://huggingface.co/datasets/cais/mmlu) dataset, offering a subset of 100 data points selected from the original compilation. 
tinyMMLU is designed to enable users to efficiently estimate the performance of a large language model (LLM) with reduced dataset size, saving computational resources 
while maintaining the essence of the MMLU evaluation.

## Features

- **Compact Dataset:** With only 100 data points, tinyMMLU provides a swift and efficient way to evaluate your LLM's performance against a benchmark set, maintaining the essence of the original MMLU dataset.
- **Compatibility:** tinyMMLU is compatible with evaluation using the [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/), but can also be integrated into your custom pipeline. See below for more details.

## Model Evaluation

_With lm-eval harness_

Users looking to evaluate a new model with tinyMMLU can use the [lm evaluation harness (v0.4.1 or later)](https://github.com/EleutherAI/lm-evaluation-harness/). 
To do so, you can directly run your evaluation harness with `--tasks=tinyMMLU` but without the `--num_fewshot` argument:
<!--To do so, download this [task config folder](https://drive.google.com/uc?export=download&id=1IMeCFfcWyYVEzJ2hoMZn0cPftWcxYd82), and add the uncompressed folder
to your version of the evaluation harness at `lm-evaluation-harness/lm_eval/tasks/`. Afterwards, run your evaluation harness as usual with `--tasks=tinyMMLU` and `--log_samples` but without the `--num_fewshot` argument: -->
```shell
lm_eval --model hf --model_args pretrained="<your-model>" --tasks=tinyMMLU --batch_size=1 
```
LM-eval harness will directly output the best accuracy estimator (IRT++), without any additional work required.

_Without lm-eval harness_

tinyMMLU can be integrated into any other pipeline by downloading the data via

```python
from datasets import load_dataset
tiny_data = load_dataset('tinyBenchmarks/tinyMMLU')['test']
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
benchmark = 'mmlu' 
### Evaluation
tb.evaluate(y, benchmark)
```

This process will help you estimate the performance of your LLM against the tinyMMLU dataset, providing a streamlined approach to benchmarking. 
Please be aware that evaluating on multiple GPUs can change the order of outputs in the lm evaluation harness. 
Ordering your score vector following the original order in tinyMMLU will be necessary to use the tinyBenchmarks library.

For more detailed instructions on evaluating new models and computing scores, please refer to the comprehensive guides available at [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/) and [tinyBenchmarks GitHub](https://github.com/felipemaiapolo/tinyBenchmarks).

Happy benchmarking!

## More tinyBenchmarks
**Open LLM leaderboard**: 
[tinyArc-Challenge](https://huggingface.co/datasets/tinyBenchmarks/tinyAI2_arc),
[tinyWinogrande](https://huggingface.co/datasets/tinyBenchmarks/tinyWinogrande),
[tinyHellaswag](https://huggingface.co/datasets/tinyBenchmarks/tinyHellaswag),
[tinyTruthfulQA](https://huggingface.co/datasets/tinyBenchmarks/tinyTruthfulQA),
[tinyGSM8k](https://huggingface.co/datasets/tinyBenchmarks/tinyGSM8k)

**AlpacaEval**:
[tinyAlpacaEval](https://huggingface.co/datasets/tinyBenchmarks/tinyAlpacaEval)

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
    @article{hendryckstest2021,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }