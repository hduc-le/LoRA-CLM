import os
import numpy as np
import time
import torch
import bitsandbytes as bnb
from tqdm import tqdm
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from torch.utils.data import DataLoader
from termcolor import colored
from peft import PeftConfig, PeftModel
from typing import Tuple, Union
from consts import (
    LOG,
    DEFAULT_INPUT_MODEL,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY,
    RESPONSE_KEY_NL,
)

class TrainingArguments:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        
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
        # load_in_8bit=load_in_8bit,    # WARNING: uncomment this line may cause some unexpected errors during training, fix it later
        # device_map="auto",            # WARNING: uncomment this line may cause some unexpected errors during training, fix it later
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

# %% Torch scripts
def _train(model, optimizer, train_dataloader, accelerator):
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

def _test(model, test_loader):
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
        args: TrainingArguments,
        model: Union[PeftModel, AutoModelForCausalLM],
        tokenizer: PreTrainedTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
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
    if args.load_8bit:                                              # WARNING: Fix later
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
        overall_train_loss = _train(model, optimizer, train_dataloader, accelerator)
        epoch_loss["train"].append(overall_train_loss)
        LOG.info("[Epoch: " + str(epoch) + "] " + "[Loss = " + "{:.4f}".format(overall_train_loss) + "] ")
        
        # Evaluation
        eval_dataloader = tqdm(
            eval_dataloader,
            leave=True,
            disable=not args.show_progress_bar,
            desc=colored(f"Testing on evaluation - Epoch {epoch}", "yellow"),
        )
        overall_val_loss = _test(model, eval_dataloader)
        epoch_loss["val"].append(overall_val_loss)
        LOG.info("[Test Summary] " + "[Loss = " + "{:.4f}".format(overall_val_loss) + "] ")

        # Update the current best model if val loss is better
        if args.save_model == 1 and overall_val_loss < min_val_loss:
            LOG.info(f"Val loss improves from {min_val_loss} to {overall_val_loss}.")
            # save the current model
            if is_peft_model:
                save_model_id = (
                    f"{args.save_name}_{peft_config.peft_type}_{peft_config.task_type}_epoch_{epoch}"
                )
            else:
                save_model_id = f"{args.save_name}_epoch_{epoch}"
            best_path = os.path.join(out_dir, save_model_id)

            LOG.info("Save cur best model to {}".format(best_path))
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            if args.push_to_hub:
                accelerator.print(f"Epoch {epoch} finished")
                accelerator.print(f"Pushing to HF hub")
                try:
                    if accelerator.is_main_process:
                        unwrapped_model.push_to_hub(args.huggingface_hub)
                        tokenizer.push_to_hub(args.huggingface_hub)
                except Exception as e:
                    accelerator.print(e)
                    accelerator.print(f"Failed to push to hub")

            unwrapped_model.save_pretrained(
                best_path,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
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
        model = load_peft_model(best_path, peft_config)
        tokenizer = load_tokenizer(best_path)
        save_model_id = (
                f"best_{args.save_name}_{peft_config.peft_type}_{peft_config.task_type}"
            )
    else:
        model, tokenizer = get_model_tokenizer(best_path)
        save_model_id = f"best_{args.save_name}"

    if args.save_best:
        # save the current model
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, "saved-runs", str(int(time.time() * 1e7)))
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, save_model_id)

        LOG.info("Save final best model to {}".format(best_path))

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            best_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(best_path)

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

        accelerator.end_training()

def test(args, model, test_dataloader) -> None:
    LOG.info("Testing begins...")
    test_dataloader = tqdm(
            test_dataloader,
            leave=True,
            disable=not args.show_progress_bar,
            desc=colored(f"Testing on test", "yellow"),
        )
    # Test model
    _test(model, test_dataloader)