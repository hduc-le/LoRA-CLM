import argparse
import yaml
import os 
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from datasets import load_dataset, DatasetDict
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
)
from consts import DEFAULT_SEED
from train_utils import TrainingArguments, get_model_tokenizer, train, test
from data_utils import generate_prompt, print_trainable_parameters

# %% 
def parse_args():
    parser = argparse.ArgumentParser(description='Config Arguments')
    parser.add_argument('--config', type=str, default="config.yaml")
    return parser.parse_args()

def read_config(path: str):
    """read yaml file"""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data
# %%
def main():
    args = parse_args()
    # Load the YAML file
    yaml_data = read_config(args.config)
    args = TrainingArguments(**yaml_data)
    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(args.model_name_or_path, load_in_8bit=args.load_8bit)
    
    # LoRA fine tune
    if args.lora_finetune:
        # Peft model
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.config.use_cache = False
        print_trainable_parameters(model)

    # Load dataset
    data = load_dataset("json", data_files=args.data_path, split="train")
    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=args.cutoff_len,
            padding="max_length",
        ),
        remove_columns=["instruction", "input", "response"]
    )
    # 90% train, 10% test + validation
    train_testvalid = data.train_test_split(test_size=args.test_ratio, seed=DEFAULT_SEED)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict({
                                    'train': train_testvalid['train'],
                                    'test': test_valid['test'],
                                    'valid': test_valid['train']
                                })

    train_sampler = RandomSampler(train_test_valid_dataset["train"])
    eval_sampler = SequentialSampler(train_test_valid_dataset["valid"])
    test_sampler = SequentialSampler(train_test_valid_dataset["test"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataloader = DataLoader(
        train_test_valid_dataset["train"],
        batch_size=args.train_bsz,
        collate_fn=collator,
        sampler=train_sampler,
    )

    eval_dataloader = DataLoader(
        train_test_valid_dataset["valid"],
        batch_size=args.test_bsz,
        collate_fn=collator,
        sampler=eval_sampler,
    )

    test_dataloader = DataLoader(
        train_test_valid_dataset["test"],
        batch_size=args.test_bsz,
        collate_fn=collator,
        sampler=test_sampler,
    )
    
    train(args, model, tokenizer, train_dataloader, eval_dataloader)
    test(args, model, test_dataloader)

if __name__ == "__main__":
    main()
