# LoRA-CLM
## Overview

## Installation
To install the necessary software, follow the following command:
```bash
pip install -r requirements.txt
```

## Usage
Before training, you need to specify some configuration in `config.yaml`, it will be loaded internally when executing `finetune.py` or `pl_finetune.py`. 

Remark: In case if you want to push the fine-tuned model to HuggingFace hub, please visit your HF account and copy the Access Tokens (as WRITE mode) then paste it after executing:
```bash
huggingface-cli login
```
The HF repo should be specified with `huggingface_hub: "your/huggingface/repo/name"` and `push_to_hub: True` in the `config.yaml`.

### With Accelerate framework
To fine-tune with `accelerate` framework, follow the steps:

1. Generate config and follow the instruction (to specify number gpus, machines, precision, etc).
```bash
accelerate config
```

2. Perform fine-tuning.
```bash
accelerate launch finetune.py --config "config.yaml"
```

### With PyTorch-Lightning framework
To fine-tune with `pytorch-lightning` framework, run the following command:
```bash
python pl_finetune.py --config "config.yaml"
```