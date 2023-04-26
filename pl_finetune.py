import os
import argparse
import pytorch_lightning as pl
import yaml
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
)
from utils import print_trainable_parameters
from helpers import *
from pl_module import LitModel, LegalDataModule


# %% 
def parse_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    # model config
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--model_name_or_path', type=str, default="databricks/dolly-v2-3b")

    # optimizer config
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=[None, "ExponentialLR"])
    parser.add_argument('--ExponentialLR_gamma', type=float, default=0.96)

    # training config
    parser.add_argument('--lora_finetune', type=bool, default=True, help="set to False for normally fine-tuning.")
    parser.add_argument('--load_8bit', type=bool, default=False, help="load model in 8bit.")
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--save_model', type=int, default=1, help="save model everywhen val loss is improved.")
    parser.add_argument('--save_best', type=bool, default=True, help="restore and save the best model at the end of training.")
    parser.add_argument('--patience', type=int, default=5, help="stop training after `patience` epoch if the model will not have improvement on val loss.")
    parser.add_argument('--show_progress_bar', type=bool, default=True)
    parser.add_argument('--push_to_hub', type=bool, default=False)
    parser.add_argument('--huggingface_hub', type=str)

    # data config
    parser.add_argument('--data_path', type=str, help="path to json data")
    parser.add_argument('--cutoff_len', type=int, default=512)
    parser.add_argument('--train_bsz', type=int, default=4)
    parser.add_argument('--test_bsz', type=int, default=4)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # LoRA config
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    if os.path.exists("config.yaml"):
        # Load the configuration file
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        # Set default values for the parser arguments based on the configuration file
        parser.set_defaults(**config)
        
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(args.model_name_or_path, load_in_8bit=args.load_8bit)
    
    peft_config = None
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

    dm = LegalDataModule(tokenizer=tokenizer, args=args)
    model = LitModel(model=model, args=args)
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=torch.cuda.device_count(),
    )
    trainer.fit(model, datamodule=dm)
    
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "saved-runs"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if peft_config is not None:
        save_model_id = os.path.join(
                            out_dir, 
                            f"best_{args.save_name}_{peft_config.peft_type}_{peft_config.task_type}"
                        )
    else:
        save_model_id = os.path.join(out_dir, f"best_{args.save_name}")

    trainer.model.save_pretrained(save_model_id)
    tokenizer.save_pretrained(save_model_id)

    if args.push_to_hub:
        trainer.model.push_to_hub(args.huggingface_hub)
        tokenizer.push_to_hub(args.huggingface_hub)

if __name__=="__main__":
    main()