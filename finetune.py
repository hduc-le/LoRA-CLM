import argparse
import yaml
import torch
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from functional.train import TrainingArguments, get_model_tokenizer, train
from utils.consts import LOG, DEFAULT_SEED
from utils.data import generate_prompt, print_trainable_parameters

def parse_args():
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--config', type=str, default="configs/config.yaml")
    return parser.parse_args()

def read_config(path):
    # read yaml and return contents 
    with open(path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

# %% 
def main():
    p_args = parse_args()
    # Load the training config file
    yaml_data = read_config(p_args.config)
    args = TrainingArguments(**yaml_data)

    # Accelerator setup
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.print(f"Using {accelerator.num_processes} GPUs")
    set_seed(DEFAULT_SEED)

    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(args.model_name_or_path, 
                                           load_in_8bit=args.load_8bit, 
                                           gradient_checkpointing=args.gradient_checkpointing)
    
    # LoRA fine tune
    if args.lora_finetune:
        # Peft model
        peft_config = LoraConfig(r=args.lora_r,
                                 inference_mode=False,
                                 lora_alpha=args.lora_alpha,
                                 lora_dropout=args.lora_dropout,
                                 bias="none",
                                 task_type="CAUSAL_LM")
        
        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)

    LOG.info(f"Mem needed: {model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB")
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
    
    train_sampler = RandomSampler(train_testvalid["train"])
    eval_sampler = SequentialSampler(train_testvalid["test"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataloader = DataLoader(train_testvalid["train"],
                                  batch_size=args.train_bsz,
                                  collate_fn=collator,
                                  sampler=train_sampler)

    eval_dataloader = DataLoader(train_testvalid["test"],
                                 batch_size=args.test_bsz,
                                 collate_fn=collator,
                                 sampler=eval_sampler)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.learning_rate * accelerator.num_processes, 
                                  weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.ExponentialLR_gamma)
    
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )
    train(args, model, optimizer, scheduler, tokenizer, train_dataloader, eval_dataloader, accelerator)

if __name__ == "__main__":
    main()
