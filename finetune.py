import os
import torch
import argparse
import numpy as np
import logging
import datasets
import transformers
from torchmetrics import MeanMetric
from tqdm import tqdm
from torch.optim import AdamW
from datasets import load_dataset
from accelerate import Accelerator
from colorlog import ColoredFormatter
from utils.load import read_config, get_model_tokenizer
from utils.data import generate_prompt, print_trainable_parameters
from transformers import DataCollatorForLanguageModeling, set_seed, get_scheduler
from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from peft.utils.other import fsdp_auto_wrap_policy
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green,bold",
        "INFOV": "cyan,bold",
        "WARNING": "yellow",
        "ERROR": "red,bold",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)
ch.setFormatter(formatter)

logger = logging.getLogger("rn")
logger.setLevel(logging.DEBUG)
logger.handlers = []  # No duplicated handlers
logger.propagate = False  # workaround for duplicated logs in ipython
logger.addHandler(ch)

torch.backends.cuda.matmul.allow_tf32 = True


def train(config):
    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(config["train"]["seed"])

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"]
    )
    logger.info(f"Using {accelerator.num_processes} GPUs")

    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(
        config["model"]["name"],
        load_in_8bit=config["model"]["load_in_8bit"],
        gradient_checkpointing=config["train"]["gradient_checkpointing"],
    )
    # LoRA fine tune
    if config["lora"]["active"]:
        if config["lora"]["ckpt_path"]:
            model = PeftModel.from_pretrained(
                model, config["lora"]["ckpt_path"], is_trainable=True
            )
        else:
            peft_config = LoraConfig(
                r=config["lora"]["r"],
                inference_mode=False,
                lora_alpha=config["lora"]["alpha"],
                lora_dropout=config["lora"]["dropout"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)

    logger.info(
        f"Mem needed: {model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB"
    )

    # FSDP plugin with accelerate
    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    # When using FSDP, it is efficient and recommended to call prepare for the model before creating the optimizer
    model = accelerator.prepare(model)

    # Load dataset
    data = load_dataset("json", data_files=config["data"]["path"], split="train")
    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=config["data"]["max_length"],
            padding="max_length",
        ),
        num_proc=os.cpu_count(),
        remove_columns=data.column_names,
    )
    # split data
    split_dataset = data.train_test_split(
        test_size=config["data"]["test_size"], seed=config["train"]["seed"]
    )

    # Data Loaders
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        split_dataset["train"],
        batch_size=config["data"]["train_batch_size"],
        collate_fn=data_collator,
        shuffle=True,
    )

    eval_dataloader = DataLoader(
        split_dataset["test"],
        batch_size=config["data"]["eval_batch_size"],
        collate_fn=data_collator,
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config["optim"]["lr"],
        weight_decay=config["optim"]["weight_decay"],
    )

    gradient_accumulation_steps = config["train"]["gradient_accumulation_steps"]

    # decay to min_lr instead of 0
    lr_ratio = config["optim"]["min_lr"] / config["optim"]["lr"]
    accelerator.print(f"Len of train_dataloader: {len(train_dataloader)}")
    total_num_steps = (len(train_dataloader) / gradient_accumulation_steps) * config["train"]["num_epochs"]
    # instead of decaying to zero, decay to ratio of min_lr / lr
    total_num_steps += int(total_num_steps * lr_ratio) + config["optim"]["warmup_steps"]
    accelerator.print(f"Total training steps: {total_num_steps}")

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config["optim"]["warmup_steps"] * accelerator.num_processes,
        num_training_steps=total_num_steps,
    )

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        optimizer, scheduler, train_dataloader, eval_dataloader
    )

    # Now we train the model
    epochs_no_improve = 0
    min_val_loss = np.inf
    for epoch in range(config["train"]["num_epochs"]):
        train_loss = MeanMetric(nan_strategy="error").to(model.device)
        for step, batch in enumerate(
            pbar := tqdm(
                train_dataloader,
                desc=f"Epoch {epoch} - Training",
                disable=not accelerator.is_main_process,
            )
        ):
            model.train()
            outputs = model(**batch)
            loss = outputs.loss

            # progress bar
            pbar.set_postfix({"loss": loss.item()})
            
            # gather loss before backprop in case of gradient accumulation
            loss_values = accelerator.gather_for_metrics(
                {"loss": loss.detach().float()}
            )
            train_loss.update(loss_values["loss"])

            # Gradient accumulation and backprop
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Evaluate at the end of the epoch (distributed evaluation as we have all GPU cores)
        model.eval()
        val_loss = MeanMetric(nan_strategy="error").to(model.device)
        with torch.no_grad():
            for batch in (
                pbar := tqdm(
                    eval_dataloader,
                    desc=f"Epoch {epoch} - Validation",
                    disable=not accelerator.is_main_process,
                )
            ):
                loss = model(**batch).loss

                pbar.set_postfix({"loss": loss.item()})
                
                loss_values = accelerator.gather_for_metrics({"loss": loss.detach()})

                val_loss.update(loss_values["loss"])

        # Compute average train and validation loss
        log_items = {"train_loss": train_loss.compute(), "val_loss": val_loss.compute()}

        # Use accelerator.print to print only on the main process.
        accelerator.print(
            f"Summary epoch {epoch}: train loss: {log_items['train_loss'].item()} || validation loss: {log_items['val_loss'].item()}"
        )

        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss

            accelerator.print(f"Epoch {epoch} finished")
            accelerator.print(f"Pushing to HF hub")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            try:
                if accelerator.is_main_process:
                    unwrapped_model.push_to_hub(
                        config["model"]["save_name"] + f"-epoch-{epoch}", private=True
                    )
                    tokenizer.push_to_hub(
                        config["model"]["save_name"] + f"-epoch-{epoch}", private=True
                    )

            except Exception as e:
                accelerator.print(e)
                accelerator.print(f"Failed to push to hub")

            unwrapped_model.save_pretrained(
                f"{config['model']['output_dir']}/epoch-{epoch}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == config["train"]["patience"]:
                accelerator.print("Early stopping!")
                break

        train_loss.reset()

    save_dir = f"{config['model']['output_dir']}/final"
    logger.info("Training finished.")
    logger.info(f"Unpacking and saving the final checkpoint at {save_dir}")

    # save trained model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # Use accelerator.save to save
    unwrapped_model.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    accelerator.print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    args = parser.parse_args()

    # Load the training config file
    config = read_config(args.config)
    train(config)