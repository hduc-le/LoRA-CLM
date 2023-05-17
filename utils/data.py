import os
import json
import pickle
import torch

def save_pkl(save_object, save_file):
    with open(save_file, 'wb') as f:
        pickle.dump(save_object, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(load_file):
    with open(load_file, 'rb') as f:
        output = pickle.load(f)
    return output

def load_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    json_file.close()
    return data

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

def write_to_json(output_path, docs):
    with open(output_path, 'w', encoding="utf-8") as fw:
        json.dump(docs, fw, ensure_ascii=False, indent=4)
    fw.close()

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def batch_to_device(batch, except_keys=['tgt_mask', 'tgt_nsp_mask'], device=torch.device('cuda')):
    for key in batch:
        if key not in except_keys:
            batch[key] = batch[key].to(device)
    return batch

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def generate_prompt(data_point):
    # taken from https://github.com/tloen/alpaca-lora
    if "#### Question conclude:" in data_point["instruction"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Let's think it step by step:
{data_point["input"]}

### Response:
{data_point["response"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["input"]}

### Response:
{data_point["response"]}"""

import numpy as np
from .consts import RESPONSE_KEY_NL
from transformers import DataCollatorForLanguageModeling
from typing import List, Union, Any, Dict

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch