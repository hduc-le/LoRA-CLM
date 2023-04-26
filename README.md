# LoRA-CLM
## Overview

## Installation
To install the necessary software, follow the following command:
```
pip install -r requirement.txt
```
## Usage
Before training, you need to specify some configuration in `config.yaml`, it will be loaded internally when executing `finetune.py` or `pl_finetune.py`

### With Accelerate framework
To fine-tune with `accelerate` framework, follow the steps:

1. Generate config and follow the instruction (to specify number gpus, machines, precision, etc).
```bash
accelerate config
```

2. Perform fine-tuning.
```bash
accelerate launch finetune.py
```

### With PyTorch-Lightning framework
To fine-tune with `pytorch-lightning` framework, run the following command:
```bash
python pl_finetune.py
```