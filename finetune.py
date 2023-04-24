import numpy as np
import time
import torch
import torch.nn as nn
import bitsandbytes as bnb
import argparse
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from datasets import load_dataset
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
    DEFAULT_SEED,
    DEFAULT_INPUT_MODEL,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY,
    RESPONSE_KEY_NL,
)
from termcolor import colored
from utils import generate_prompt, print_trainable_parameters, batch_to_device

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

    return parser.parse_args()

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
        load_in_8bit=load_in_8bit,
        device_map="auto",
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
        gradient_checkpointing: bool = False,
    ) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:

    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def train_one(model, optimizer, train_loader, device):
    # Set to training mode
    model.train()
    # Iterate over batches
    loss_sum = 0
    for batch_idx, batch in enumerate(train_loader):
        # Load data to device
        assert isinstance(batch, dict), f"The input batch is required as a dictionary, but was received as `{type(batch)}"
        batch = batch_to_device(batch, device=device)
        # Zero out gradients for optimizer
        optimizer.zero_grad()
        # Run model
        output = model(**batch)
        # Save loss
        loss = output.loss
        # Backprop
        loss.backward()
        # Update params
        optimizer.step()
        
        # Gather losses
        loss_sum += loss.item()
        train_loader.set_postfix(loss=loss.item())

    overall_loss = loss_sum / len(train_loader)
    return overall_loss


def test_one(model, test_loader, device):
    # Set to eval mode
    model.eval()
    # Iterate over batches
    all_loss = []
    for batch_idx, batch in enumerate(test_loader):
        # Load data to device
        assert isinstance(batch, dict), f"The input batch is required as a dictionary, but was received as `{type(batch)}"
        batch = batch_to_device(batch, device=device)

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
    train_loader: DataLoader,
    val_loader: DataLoader,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    min_val_loss = np.inf
    sub_cycle = 0

    epoch_loss = {"train": [], "val": []}
    for epoch in range(1, args.num_epochs + 1):
        # Progress bar
        train_loader = tqdm(
            train_loader,
            leave=True,
            disable=not args.show_progress_bar,
            desc=colored(f"Training on train - Epoch {epoch}", "blue"),
        )
        # Training
        overall_train_loss = train_one(model, optimizer, train_loader, device)
        epoch_loss["train"].append(overall_train_loss)
        LOG.info("[Epoch: " + str(epoch) + "] " + "[Loss = " + "{:.4f}".format(overall_train_loss) + "] ")
        
        # Evaluation
        val_loader = tqdm(
            val_loader,
            leave=True,
            disable=not args.show_progress_bar,
            desc=colored(f"Testing on evaluation - Epoch {epoch}", "yellow"),
        )
        overall_val_loss = test_one(model, val_loader, device)
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


def test(args, model, test_loader) -> None:
    LOG.info("Testing begins...")
    test_loader = tqdm(
            test_loader,
            leave=True,
            disable=not args.show_progress_bar,
            desc=colored(f"Testing on test", "yellow"),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transfer model to cuda if not in cuda mode
    if not next(model.parameters()).is_cuda:
        model.to(device)
    # Test model
    test_one(model, test_loader, device)

# %%
def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(args.model_name_or_path)

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

    # Training on multi-gpus
    model = nn.DataParallel(model)

    # Load dataset
    data = load_dataset("json", data_files=args.data_path, split="train")
    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=args.cutoff_len,
            padding="max_length",
        ),
        remove_columns=["instruction", "input", "response"],
    )
    data = data.train_test_split(test_size=args.test_ratio, seed=DEFAULT_SEED)

    train_sampler = RandomSampler(data["train"])
    val_sampler = SequentialSampler(data["test"])

    train_loader = DataLoader(
        data["train"],
        batch_size=args.train_bsz,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        data["test"],
        batch_size=args.test_bsz,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        sampler=val_sampler,
    )

    train(args, model, tokenizer, train_loader, val_loader)

if __name__ == "__main__":
    main()
