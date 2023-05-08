import os
import numpy as np
import time
import torch
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import DataLoader
from termcolor import colored
from peft import PeftConfig, PeftModel, PeftModelForCausalLM
from typing import Tuple, Union
from utils.consts import (LOG,
                          DEFAULT_INPUT_MODEL,
                          END_KEY,
                          INSTRUCTION_KEY,
                          RESPONSE_KEY,
                          RESPONSE_KEY_NL)

class TrainingArguments:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        
def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    LOG.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left")
    return tokenizer

def load_model(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, * ,
               load_in_8bit: bool = False,
               gradient_checkpointing: bool = False) -> AutoModelForCausalLM:
    
    LOG.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path,
                                                 trust_remote_code=True,
                                                 use_cache=False if gradient_checkpointing else True,
                                                 torch_dtype=torch.bfloat16,
                                                 load_in_8bit=load_in_8bit,                   # WARNING: Only support Single-GPU training, not support DDP
                                                 device_map="auto" if load_in_8bit else None)
    return model

def load_peft_model(pretrained_model_name_or_path: str, **kwargs) -> PeftModel:
    LOG.info(f"Loading peft model for {pretrained_model_name_or_path}")
    peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
    base_model = load_model(peft_config.base_model_name_or_path, **kwargs)
    peft_model = PeftModelForCausalLM.from_pretrained(base_model, pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    peft_model.to(dtype=torch.bfloat16)
    return peft_model

def get_model_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, * ,
                        load_in_8bit: bool = False, 
                        gradient_checkpointing: bool = False) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)

    model = load_model(pretrained_model_name_or_path, 
                       load_in_8bit=load_in_8bit, 
                       gradient_checkpointing=gradient_checkpointing)
    
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY, RESPONSE_KEY_NL]})

    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def train(args: TrainingArguments = None,
          model: Union[PeftModel, AutoModelForCausalLM] = None,
          optimizer: torch.optim.Optimizer = None,
          scheduler: torch.optim.Optimizer = None,
          tokenizer: PreTrainedTokenizer = None,
          train_dataloader: DataLoader = None,
          eval_dataloader: DataLoader = None,
          accelerator: Accelerator = None,
          peft_config: PeftConfig = None) -> None:
    
    # creating a tmp directory to save the models at each epoch
    out_dir = os.path.abspath(
        os.path.join(os.path.curdir, "tmp-runs", str(int(time.time() * 1e7)))
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_path = None

    LOG.info("Training begins...")

    min_val_loss = np.inf
    sub_cycle = 0
    
    for epoch in range(1, args.num_epochs+1):
        # Progress bar
        train_dataloader = tqdm(train_dataloader,
                                leave=True,
                                disable=not args.show_progress_bar,
                                desc=colored(f"Training on train - Epoch {epoch}", "blue"))
        # Training
        model.train()
        overall_train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

                train_dataloader.set_postfix(loss=loss.item())
                overall_train_loss += loss.item()

        LOG.info("[Epoch: " + str(epoch) + "] " + "[Loss = " + "{:.4f}".format(overall_train_loss/len(train_dataloader)) + "] ")
        
        # Evaluation
        eval_dataloader = tqdm(eval_dataloader,
                               leave=True,
                               disable=not args.show_progress_bar,
                               desc=colored(f"Testing on evaluation - Epoch {epoch}", "yellow"))

        model.eval()
        overall_val_loss = 0.0
        for eval_step, eval_batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**eval_batch)
                loss = outputs.loss

            overall_val_loss += loss.item()
            eval_dataloader.set_postfix(loss=loss.item())

        overall_val_loss /= len(eval_dataloader)

        LOG.info("[Test Summary] " + "[Loss = " + "{:.4f}".format(overall_val_loss) + "] ")

        # Update the current best model if val loss is better
        if args.save_model == 1 and overall_val_loss < min_val_loss:
            LOG.info(f"Val loss improves from {min_val_loss} to {overall_val_loss}.")
            # save the current model
            if args.lora:
                save_model_id = f"{args.save_name}_{peft_config.peft_type}_{peft_config.task_type}_epoch_{epoch}"
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

            unwrapped_model.save_pretrained(best_path,
                                            is_main_process=accelerator.is_main_process,
                                            save_function=accelerator.save,
                                            state_dict=accelerator.get_state_dict(model))
            
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
    if args.lora:
        model = load_peft_model(best_path)
        tokenizer = load_tokenizer(best_path)
        save_model_id = f"best_{args.save_name}_{peft_config.peft_type}_{peft_config.task_type}"
    else:
        model, tokenizer = get_model_tokenizer(best_path, load_in_8bit=args.load_8bit, gradient_checkpointing=False)
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
        unwrapped_model.save_pretrained(best_path,
                                        is_main_process=accelerator.is_main_process,
                                        save_function=accelerator.save,
                                        state_dict=accelerator.get_state_dict(model))
        
        tokenizer.save_pretrained(best_path)