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

from .consts import (
    CONCLUDING_QUESTION_PREFIX, 
    PROMPT_WITH_INPUT_FORMAT, 
    PROMPT_NO_INPUT_FORMAT
)

def generate_prompt(data_point):
    if CONCLUDING_QUESTION_PREFIX in data_point["input"]:
        return PROMPT_WITH_INPUT_FORMAT.format(
            instruction=data_point["instruction"],
            input=data_point["input"],
            response=data_point["output"],
        )
    else:
        return PROMPT_NO_INPUT_FORMAT.format(
            instruction=data_point["input"],
            response=data_point["output"]
        )