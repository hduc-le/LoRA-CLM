import torch
import argparse
import numpy as np
import logging
from tqdm import tqdm
from datasets import load_dataset
from accelerate import Accelerator
from colorlog import ColoredFormatter
from utils.read import read_config, get_model_tokenizer
from utils.data import generate_prompt, print_trainable_parameters
from transformers import DataCollatorForLanguageModeling, set_seed
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

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

def main():
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--config', type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    # Load the training config file
    config = read_config(args.config)

    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(config["model_name_or_path"], 
                                           load_in_8bit=config["load_8bit"], 
                                           gradient_checkpointing=config["gradient_checkpointing"])
    # LoRA fine tune
    if config["lora"]:
        # Peft model
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)
        peft_config = LoraConfig(r=config["lora_r"],
                                 inference_mode=False,
                                 lora_alpha=config["lora_alpha"],
                                 lora_dropout=config["lora_dropout"],
                                 bias="none",
                                 task_type="CAUSAL_LM")
        
        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)

    logger.info(f"Mem needed: {model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB")

    # Load datasetimport numpy as np
    data = load_dataset("json", data_files=config["data_path"], split="train")
    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=config["max_length"],
            padding="max_length",
        ),
        remove_columns=["instruction", "input", "response"]
    )
    # 90% train, 10% test + validation
    split_dataset = data.train_test_split(test_size=config["test_size"], seed=config["seed"])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(split_dataset["train"], 
                                  batch_size=config["train_batch_size"], 
                                  collate_fn=data_collator, 
                                  sampler=RandomSampler(split_dataset["train"]))
    
    eval_dataloader = DataLoader(split_dataset["test"], 
                                 batch_size=config["eval_batch_size"], 
                                 collate_fn=data_collator, 
                                 sampler=SequentialSampler(split_dataset["test"]))

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config["lr"], 
                                  weight_decay=config["weight_decay"])

    # Initialize accelerator
    accelerator = Accelerator()
    accelerator.print(f"Using {accelerator.num_processes} GPUs")

    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(config["seed"])

    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    # Now we train the model
    epochs_no_improve = 0
    min_val_loss = np.inf
    for epoch in range(config["num_epochs"]):
        model.train()
        for batch in (pbar:=tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({'loss': loss.item()})

        # Evaluate at the end of the epoch (distributed evaluation as we have 8 TPU cores)
        model.eval()
        validation_losses = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss

            # We gather the loss from the 8 TPU cores to have them all.
            validation_losses.append(accelerator.gather(loss[None]))

        # Compute average validation loss
        val_loss = torch.stack(validation_losses).sum().item() / len(validation_losses)
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: validation loss:", val_loss)
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            continue
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == config["patience"]:
                accelerator.print("Early stopping!")
                break

    # save trained model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # Use accelerator.save to save
    unwrapped_model.save_pretrained(config["output_dir"], save_function=accelerator.save)


if __name__ == "__main__":
    main()
