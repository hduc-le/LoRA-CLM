# LoRA-CLM
## Overview
A simple and friendly code for fine-tuning CausalLM with `LoRA` (Low-Rank Adaptation) method such as Dolly 2.0 that can be applied to the custom datasets and provide distributed training mode on multi-GPUs using `Accelerate`.

## Installation
To install the necessary software, follow the following command:
```bash
pip install -r requirements.txt
```

## Usage
Before training, you need to specify some configuration in `configs/finetune.yaml`, it will be loaded internally when executing `finetune.py`. 

### Fine tuning with PyTorch + Accelerate framework
To fine-tune with `Accelerate` framework, follow the steps:

1. Generate config and follow the instruction (to specify number gpus, machines, mixed-precision, etc).
```bash
accelerate config
```

2. Perform fine-tuning.
```bash
accelerate launch finetune.py --config configs/finetune.yaml
```

### Text Generation
Configure `configs/generate.yaml` then execute the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --config configs/generate.yaml\
                                          --prompt "enter your instruction here!!!"
```

## Web Application
Step 1. Download SocketXP for publishing local-host
```bash
curl -O https://portal.socketxp.com/download/linux/socketxp && chmod +wx socketxp && sudo mv socketxp /usr/local/bin
```
Step 2. Authenticate
```
socketxp login $ACCESS_TOKENS
```
Step 3. Run flask app
```python
python app/app.py
```
Step 4. Create secure tunnels
```bash
socketxp connect $local_host_ip
```