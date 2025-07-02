# EncodEval: Evaluating Pretrained Encoders Across NLP Tasks with Confidence-Aware Rankings

## Overview

**EncodEval** is a lightweight evaluation framework designed to benchmark general-purpose pre-trained encoders on a diverse set of downstream NLP tasks:

- Sequence Classification (SC)  
- Token Classification (TC)
- Question Answering (QA)  
- Information Retrieval (IR)

This repository was used for evaluation in the paper  
[Should We Still Pretrain Encoders with Masked Language Modeling?](https://arxiv.org/abs/2507.00994). 


## Installation

To install EncodEval directly via pip:

```bash
pip install git+https://github.com/hgissbkh/EncodEval.git@MLM_vs_CLM
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/hgissbkh/EncodEval.git
cd EncodEval
git checkout MLM_vs_CLM
pip install -e .
```


## Running Evaluations

To run a task evaluation from the command line:

```bash
python main.py \ 
    --config_file <config_file_path> \ 
    --model_path <model_path>
```

This will generate a `results.json` file with instance-level scores.


## Task Evaluation Modules

Task-specific evaluation logic is implemented in [encodeval/eval_tasks/](encodeval/eval_tasks/). These modules handle both fine-tuning and evaluation.

Example usage in Python:
```python
from encodeval.eval_tasks import EvalConfig, SequenceClassificationEval

config_file = "./configs/SC/sst2_lr1e-04_sd0.yaml"
eval_config: EvalConfig = configue.load(
    config_file,  
    sub_path="eval_config",
)
evaluator = SequenceClassificationEval(eval_config)
evaluator.train() # Fine-tune on the target task
evaluator.validate() # Evaluate on the validation set
evaluator.test() # Evaluate on the test set
```


## Datasets

Dataset loading and preprocessing are managed in [encodeval/datasets.py](encodeval/datasets.py).

Example (loading the sst2 dataset):

```python
from encodeval.datasets import sst2
dataset = sst2()
```


## Configuration Files

Configuration files are available in the [configs/](configs/) folder.


## Results

All evaluation results are available in the [results/](results/) directory as JSON files.


## Citation

If you use this framework in your research, please consider citing:

```bibtex
@misc{gisserotboukhlef2025pretrainencodersmaskedlanguage,
      title={Should We Still Pretrain Encoders with Masked Language Modeling?}, 
      author={Hippolyte Gisserot-Boukhlef and Nicolas Boizard and Manuel Faysse and Duarte M. Alves and Emmanuel Malherbe and André F. T. Martins and Céline Hudelot and Pierre Colombo},
      year={2025},
      eprint={2507.00994},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.00994}, 
}
```

```bibtex
@misc{boizard2025eurobertscalingmultilingualencoders,
  title={EuroBERT: Scaling Multilingual Encoders for European Languages}, 
  author={Nicolas Boizard and Hippolyte Gisserot-Boukhlef and Duarte M. Alves and André Martins and Ayoub Hammal and Caio Corro and Céline Hudelot and Emmanuel Malherbe and Etienne Malaboeuf and Fanny Jourdan and Gabriel Hautreux and João Alves and Kevin El-Haddad and Manuel Faysse and Maxime Peyrard and Nuno M. Guerreiro and Patrick Fernandes and Ricardo Rei and Pierre Colombo},
  year={2025},
  eprint={2503.05500},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2503.05500}
}
```
