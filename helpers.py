import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from datasets import load_dataset
from peft import (
    PeftConfig,
    PeftModel,
)
from typing import Tuple, Union, Dict
from consts import (
    LOG,
    DEFAULT_INPUT_MODEL,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY,
    RESPONSE_KEY_NL,
)
# %%
def load_tokenizer(
        pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    ) -> PreTrainedTokenizer:
    
    LOG.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, padding_side="left"
    )
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY, RESPONSE_KEY_NL]}
    )
    return tokenizer


def load_model(
        pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
        *,
        load_in_8bit: bool = False,
        gradient_checkpointing: bool = False,
    ) -> AutoModelForCausalLM:
    
    LOG.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True,
        torch_dtype=torch.bfloat16,
        # load_in_8bit=load_in_8bit,
        # device_map="auto",
    )
    return model


def load_peft_model(
        pretrained_model_name_or_path: str, 
        peft_config: PeftConfig
    ) -> PeftModel:

    LOG.info(f"Loading peft model for {pretrained_model_name_or_path}")
    peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
    peft_model = load_model(peft_config.base_model_name_or_path)
    peft_model = PeftModel.from_pretrained(peft_model, pretrained_model_name_or_path)
    return peft_model


def get_model_tokenizer(
        pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
        *,
        load_in_8bit: bool = False,
        gradient_checkpointing: bool = False,
    ) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:

    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, load_in_8bit=load_in_8bit, gradient_checkpointing=gradient_checkpointing
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer