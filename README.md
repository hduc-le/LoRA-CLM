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

Remark: In case if you want to push the fine-tuned model to HuggingFace hub, please visit your HuggingFace account's settings and copy the Access Tokens (as WRITE mode) then paste it after executing:
```bash
huggingface-cli login
```
The HuggingFace repository where you want to push the model should be specified with `huggingface_hub: "your/huggingface/repo/name"` and `push_to_hub: true` in the `configs/finetune.yaml`.

### Fine tuning with PyTorch + Accelerate framework
To fine-tune with `accelerate` framework, follow the steps:

1. Generate config and follow the instruction (to specify number gpus, machines, precision, etc).
```bash
accelerate config
```

Example: Training with 2-GPUs
```
In which compute environment are you running?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  This machine
    AWS (Amazon SageMaker)
--------------------------------------------------------------------------------
Which type of machine are you using?           
Please select a choice using the arrow or number keys, and selecting with enter
    No distributed training
    multi-CPU                                                           
 ➔  multi-GPU
    TPU
--------------------------------------------------------------------------------
How many different machines will you use (use more than 1 for multi-node training)? [1]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]:                                                           
Do you want to use FullyShardedDataParallel? [yes/NO]:
Do you want to use Megatron-LM ? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]: 2
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
--------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
no
```
2. Perform fine-tuning.
```bash
accelerate launch finetune.py --config configs/finetune.yaml
```

### Text Generation
Configure `configs/generate.yaml` then execute the following command:
```bash
python generate.py --config configs/generate.yaml\
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