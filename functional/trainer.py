import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import Optional, Union, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from accelerate import Accelerator
from peft import PeftModel
from tqdm import tqdm
from termcolor import colored
from utils.consts import LOG
from train import TrainingArguments, load_peft_model, load_tokenizer, get_model_tokenizer

class DataCollator: pass

class Trainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None,),
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.args = args
        self.model = model
        self.optimizer, self.lr_scheduler = optimizers
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

        self._setup()

        self._is_peft_model = True if isinstance(self.model, PeftModel) else False
        if self._is_peft_model and hasattr(self.model, "peft_config"):
            self._peft_config = model.peft_config["default"]

    def _setup(self):
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()

        self._accelerator = Accelerator()
        (
            self.train_dataloader,
            self.eval_dataloader,
            self.model,
            self.optimizer,
        ) = self._accelerator.prepare(
            train_dataloader, eval_dataloader, self.model, self.optimizer
        )

    def _get_train_sampler(self):
        return RandomSampler(self.train_dataset)

    def _get_eval_sampler(self):
        return SequentialSampler(self.eval_dataset)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
        )

    def get_eval_dataloader(self):
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a eval_dataset.")

        eval_dataset = self.eval_dataset
        data_collator = self.data_collator

        eval_sampler = self._get_eval_sampler()

        return DataLoader(
            eval_dataset,
            batch_size=self._test_batch_size,
            sampler=eval_sampler,
            collate_fn=data_collator,
        )

    def _train(self, dataloader):
        self.model.train()
        loss_sum = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # zero out gradient
            self.optimizer.zero_grad()
            # forward
            output = self.model(**batch)
            loss = output.loss
            # update model
            self._accelerator.backward(loss)
            self.optimizer.step()

            loss_sum += loss.item()
            dataloader.set_postfix(loss=loss.item())

        overall_loss = loss_sum / len(dataloader)
        return overall_loss

    def _eval(self, dataloader):
        # Set to eval mode
        self.model.eval()
        # Iterate over batches
        all_loss = []
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                # Run model
                output = self.model(**batch)
                loss = output.loss

            all_loss.append(loss.item())
            dataloader.set_postfix(loss=loss.item())

        # Report overall test performance
        avg_loss = np.mean(all_loss)
        return avg_loss

    def train(self, save_dir: str = "saved-runs"):
        # creating a tmp directory to save the models at each epoch
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "tmp-runs"))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        LOG.info("Training begins...")

        min_eval_loss = np.inf
        sub_cycle = 0

        for epoch in range(1, self.args.num_epochs + 1):
            # Progress bar
            self.train_dataloader = tqdm(
                self.train_dataloader,
                leave=True,
                disable=not self.args.show_progress_bar,
                desc=colored(f"Training on train - Epoch {epoch}", "blue"),
            )
            # Training
            overall_train_loss = self._train(self.train_dataloader)
            LOG.info(
                "[Epoch: "
                + str(epoch)
                + "] "
                + "[Loss = "
                + "{:.4f}".format(overall_train_loss)
                + "] "
            )

            # Evaluation
            self.eval_dataloader = tqdm(
                self.eval_dataloader,
                leave=True,
                disable=not self.args.show_progress_bar,
                desc=colored(f"Testing on evaluation - Epoch {epoch}", "yellow"),
            )
            overall_eval_loss = self._eval(self.eval_dataloader)
            LOG.info(
                "[Test Summary] "
                + "[Loss = "
                + "{:.4f}".format(overall_eval_loss)
                + "] "
            )

            if self.args.save_model == 1 and overall_eval_loss < min_eval_loss:
                min_eval_loss = overall_eval_loss
                sub_cycle = 0

                # named the current model
                if self._is_peft_model:
                    save_model_id = f"{self.args.save_name}_{self._peft_config.peft_type}_{self.peft_config.task_type}_epoch_{epoch}"
                else:
                    save_model_id = f"{self.args.save_name}_epoch_{epoch}"
                best_path = os.path.join(out_dir, save_model_id)
                self._save_checkpoint(
                    self.model,
                    self.tokenizer,
                    out_dir,
                    save_model_id,
                    self.args.push_to_hub,
                )
            else:
                LOG.info(f"Eval loss does NOT improve from previous.")
                sub_cycle += 1

            # Break if the val loss hasn't improved in the past patience epochs
            if sub_cycle == self.args.patience:
                break

            if self.args.lr_scheduler == "ExponentialLR":
                self.scheduler.step()

        LOG.info("End of training. Restore the best weights")
        # restore the best saved model
        if self._is_peft_model:
            best_model = load_peft_model(best_path, self._peft_config)
            best_tokenizer = load_tokenizer(best_path)
            save_model_id = f"best_{self.args.save_name}_{self._peft_config.peft_type}_{self._peft_config.task_type}"
        else:
            best_model, best_tokenizer = get_model_tokenizer(best_path)
            save_model_id = f"best_{self.args.save_name}"

        if self.args.save_best:
            # save the current model
            out_dir = os.path.abspath(os.path.join(os.path.curdir, save_dir))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            best_path = os.path.join(out_dir, save_model_id)
            LOG.info("Save final best model to {}".format(best_path))
            self._save_checkpoint(best_model, best_tokenizer, best_path)
            self._accelerator.end_training()

    def _save_checkpoint(
        self, model, tokenizer, save_dir: str, push_to_hub: bool = False
    ):
        LOG.info("Save cur best model to {}".format(save_dir))

        self._accelerator.wait_for_everyone()
        unwrapped_model = self._accelerator.unwrap_model(model)

        if push_to_hub:
            self._accelerator.print(f"Pushing to HF hub")
            try:
                if self._accelerator.is_main_process:
                    unwrapped_model.push_to_hub(self.args.huggingface_hub)
                    self.tokenizer.push_to_hub(self.args.huggingface_hub)
            except Exception as e:
                self._accelerator.print(e)
                self._accelerator.print(f"Failed to push to hub")

        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=self._accelerator.is_main_process,
            save_function=self._accelerator.save,
            state_dict=self._accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(save_dir)
