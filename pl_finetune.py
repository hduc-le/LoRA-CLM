import os
import yaml
import argparse
import torch
import pytorch_lightning as pl
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from utils.data import print_trainable_parameters
from functional.training import get_model_tokenizer, TrainingArguments
from pl_module import LitModel, LegalDataModule
from utils.read import read_config

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

def main():
    p_args = parse_args()
    yaml_data = read_config(p_args.config)
    args = TrainingArguments(**yaml_data)

    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(args.model_name_or_path, 
                                           load_in_8bit=args.load_8bit, 
                                           gradient_checkpointing=args.gradient_checkpointing)
    
    peft_config = None
    if args.lora:
        # Peft model
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)
        peft_config = LoraConfig(r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 lora_dropout=args.lora_dropout,
                                 bias="none",
                                 task_type="CAUSAL_LM")
        
        model = get_peft_model(model, peft_config)
        model.config.use_cache = False
        print_trainable_parameters(model)

    dm = LegalDataModule(tokenizer=tokenizer, args=args)
    model = LitModel(model=model, args=args)
    trainer = pl.Trainer(max_epochs=args.num_epochs, accelerator="auto", devices=torch.cuda.device_count())
    trainer.fit(model, datamodule=dm)
    
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "saved-runs"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if peft_config:
        save_model_id = os.path.join(out_dir, f"best_{args.save_name}_{peft_config.peft_type}_{peft_config.task_type}")
    else:
        save_model_id = os.path.join(out_dir, f"best_{args.save_name}")

    model.save_pretrained(save_model_id)
    tokenizer.save_pretrained(save_model_id)

    if args.push_to_hub:
        model.push_to_hub(args.huggingface_hub)
        tokenizer.push_to_hub(args.huggingface_hub)

if __name__=="__main__":
    main()