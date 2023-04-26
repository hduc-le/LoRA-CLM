import os
import numpy as np
import time
import torch
import torch.nn as nn
import bitsandbytes as bnb
import argparse
import yaml
from tqdm import tqdm
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from datasets import load_dataset, DatasetDict
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    PeftConfig,
    PeftModel,
)
from typing import Tuple, Union, Dict
from consts import (
    LOG,
    DEFAULT_SEED
)
from termcolor import colored
from helpers import load_model, load_peft_model, load_tokenizer, get_model_tokenizer
from utils import generate_prompt, print_trainable_parameters

# %% 
def parse_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    # model config
    parser.add_argument('--save_name', type=str, default="LoRA-CLM-fine-tune")
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

    # Load the configuration file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set default values for the parser arguments based on the configuration file
    parser.set_defaults(**config)

    return parser.parse_args()

# %% 
def train_one(model, optimizer, train_dataloader, accelerator):
    # Set to training mode
    model.train()
    # Iterate over batches
    loss_sum = 0
    for batch_idx, batch in enumerate(train_dataloader):
        # Zero out gradients for optimizer
        optimizer.zero_grad()
        # Run model
        output = model(**batch)
        # Save loss
        loss = output.loss
        # Backprop
        accelerator.backward(loss)
        # Update params
        optimizer.step()
        
        # Gather losses
        loss_sum += loss.item()
        train_dataloader.set_postfix(loss=loss.item())

    overall_loss = loss_sum / len(train_dataloader)
    return overall_loss


def test_one(model, test_loader):
    # Set to eval mode
    model.eval()
    # Iterate over batches
    all_loss = []
    for batch_idx, batch in enumerate(test_loader):
        with torch.no_grad():
            # Run model
            output = model(**batch)
            loss = output.loss

        all_loss.append(loss.item())
        test_loader.set_postfix(loss=loss.item())

    # Report overall test performance
    avg_loss = np.mean(all_loss)
    return avg_loss


def train(
        args: argparse.ArgumentParser,
        model: Union[PeftModel, AutoModelForCausalLM],
        tokenizer: PreTrainedTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: str = "cpu",
    ) -> None:
    # creating a tmp directory to save the models at each epoch
    out_dir = os.path.abspath(
        os.path.join(os.path.curdir, "tmp-runs", str(int(time.time() * 1e7)))
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_path = None
    is_peft_model = True if isinstance(model, PeftModel) else False

    LOG.info("Setting up optimizer...")
    if args.load_8bit:  # ERROR: Fix later
        optimizer = bnb.optim.Adam8bit(
            model.parameters(), 
            lr=args.learning_rate, 
            betas=(0.9, 0.995)
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    if args.lr_scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.ExponentialLR_gamma
        )

    LOG.info("Training begins...")

    if is_peft_model and hasattr(model, "peft_config"):
        peft_config = model.peft_config["default"]

    accelerator = Accelerator()
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    min_val_loss = np.inf
    sub_cycle = 0

    epoch_loss = {"train": [], "val": []}
    for epoch in range(1, args.num_epochs + 1):
        # Progress bar
        train_dataloader = tqdm(
            train_dataloader,
            leave=True,
            disable=not args.show_progress_bar,
            desc=colored(f"Training on train - Epoch {epoch}", "blue"),
        )
        # Training
        overall_train_loss = train_one(model, optimizer, train_dataloader, accelerator)
        epoch_loss["train"].append(overall_train_loss)
        LOG.info("[Epoch: " + str(epoch) + "] " + "[Loss = " + "{:.4f}".format(overall_train_loss) + "] ")
        
        # Evaluation
        eval_dataloader = tqdm(
            eval_dataloader,
            leave=True,
            disable=not args.show_progress_bar,
            desc=colored(f"Testing on evaluation - Epoch {epoch}", "yellow"),
        )
        overall_val_loss = test_one(model, eval_dataloader, accelerator)
        epoch_loss["val"].append(overall_val_loss)
        LOG.info("[Test Summary] " + "[Loss = " + "{:.4f}".format(overall_val_loss) + "] ")

        # Update the current best model if val loss is better
        if args.save_model == 1 and overall_val_loss < min_val_loss:
            LOG.info(f"Val loss improves from {min_val_loss} to {overall_val_loss}.")

            # save current model
            best_path = os.path.join(out_dir, args.save_name + "_epoch_" + str(epoch))
            LOG.info("Save cur best model to {}".format(best_path))

            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

            min_val_loss = overall_val_loss
            sub_cycle = 0
        else:
            LOG.info(f"Val loss does NOT improve from previous.")
            sub_cycle += 1

        # Break if the val loss hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

        if args.lr_scheduler == "ExponentialLR":
            scheduler.step()

    LOG.info("End of training. Restore the best weights")

    # restore the best saved model
    if is_peft_model:
        model = load_peft_model(best_path)
        tokenizer = load_tokenizer(best_path)
    else:
        model, tokenizer = get_model_tokenizer(best_path)

    if args.save_best:
        # save the current model
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, "saved-runs", str(int(time.time() * 1e7)))
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # save the current model
        if is_peft_model:
            save_model_id = (
                f"best_{args.save_name}_{peft_config.peft_type}_{peft_config.task_type}"
            )
        else:
            save_model_id = f"best_{args.save_name}"

        best_path = os.path.join(out_dir, save_model_id)
        LOG.info("Save final best model to {}".format(best_path))

        # save current model to disk
        model.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)

        # push model to HF hub 
        if args.push_to_hub:
            model.push_to_hub(args.huggingface_hub)
            tokenizer.push_to_hub(args.huggingface_hub)

        # save model config to file
        with open(best_path + "_args.txt", "w") as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

        # save loss report to file
        with open(best_path + "_summary.txt", "w") as f:
            f.write(
                "{} = {}\n".format(
                    "Avg. Train loss",
                    sum(epoch_loss["train"]) / len(epoch_loss["train"]),
                )
            )
            f.write(
                "{} = {}\n".format(
                    "Avg. Val loss", sum(epoch_loss["val"]) / len(epoch_loss["val"])
                )
            )


def test(args, model, test_loader, accelerator) -> None:
    LOG.info("Testing begins...")
    test_loader = tqdm(
            test_loader,
            leave=True,
            disable=not args.show_progress_bar,
            desc=colored(f"Testing on test", "yellow"),
        )
    # Test model
    test_one(model, test_loader, accelerator)

# %%
def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(args.model_name_or_path, load_in_8bit=args.load_8bit)
    
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

    test_loader = DataLoader(
        train_test_valid_dataset["valid"],
        batch_size=args.test_bsz,
        collate_fn=collator,
        sampler=test_sampler,
    )
    
    train(args, model, tokenizer, train_dataloader, eval_dataloader)
    test(args, model, test_loader)

if __name__ == "__main__":
    main()
