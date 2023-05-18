import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from peft import PeftConfig, PeftModel, PeftModelForCausalLM
from typing import Tuple
from utils.consts import (
    DEFAULT_INPUT_MODEL,
    END_KEY,    
    INSTRUCTION_KEY,
    RESPONSE_KEY,
    RESPONSE_KEY_NL
)

def read_config(path):
    # read yaml and return contents 
    with open(path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left")
    return tokenizer

def load_model(
        pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, 
        *,
        load_in_8bit: bool = False,
        gradient_checkpointing: bool = False
    ) -> AutoModelForCausalLM:

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        device_map="auto" if load_in_8bit else None
    )
    return model

def load_peft_model(pretrained_model_name_or_path: str, **kwargs) -> PeftModel:
    peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
    base_model = load_model(peft_config.base_model_name_or_path, **kwargs)
    peft_model = PeftModelForCausalLM.from_pretrained(
        base_model, pretrained_model_name_or_path, torch_dtype=torch.bfloat16
    )
    peft_model.to(dtype=torch.bfloat16)
    return peft_model

def get_model_tokenizer(
        pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, 
        *,
        load_in_8bit: bool = False, 
        gradient_checkpointing: bool = False
    ) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, 
        load_in_8bit=load_in_8bit, 
        gradient_checkpointing=gradient_checkpointing
    )
    
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY, RESPONSE_KEY_NL]}
    )

    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer