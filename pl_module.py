import pytorch_lightning as pl
import torch
import bitsandbytes as bnb
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)
from peft import PeftModel
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from datasets import load_dataset, DatasetDict
from typing import Tuple, Union, Dict
from utils.consts import DEFAULT_SEED
from utils.data import generate_prompt

class LegalDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, args) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def prepare_data(self) -> None:
        load_dataset("json", data_files=self.args.data_path, split="train")
        
    def setup(self, stage: str) -> None:
        data = load_dataset("json", data_files=self.args.data_path, split="train")
        data = data.shuffle().map(
            lambda data_point: self.tokenizer(
                generate_prompt(data_point),
                truncation=True,
                max_length=self.args.cutoff_len,
                padding="max_length",
            ),
            remove_columns=["instruction", "input", "response"]
        )
        # 90% train, 10% test + validation
        train_testvalid = data.train_test_split(test_size=self.args.test_ratio, seed=DEFAULT_SEED)
        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        train_test_valid_dataset = DatasetDict({
                                        'train': train_testvalid['train'],
                                        'test': test_valid['test'],
                                        'valid': test_valid['train']
                                    })
        
        if stage in (None, "fit"):
            self.train, self.valid = train_test_valid_dataset['train'], train_test_valid_dataset['valid']
            self.train_sampler = RandomSampler(self.train)
            self.val_sampler = SequentialSampler(self.valid)

        if stage in (None, "test"):
            self.test = train_test_valid_dataset['test']
            self.test_sampler = SequentialSampler(self.test)

    def train_dataloader(self):
        return DataLoader(self.train, 
                          batch_size=self.args.train_bsz, 
                          collate_fn=self.collator, 
                          sampler=self.train_sampler)
    
    def val_dataloader(self):
        return DataLoader(self.valid, 
                          batch_size=self.args.test_bsz, 
                          collate_fn=self.collator, 
                          sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.test, 
                          batch_size=self.args.test_bsz, 
                          collate_fn=self.collator, 
                          sampler=self.test_sampler)
    
class LitModel(pl.LightningModule):
    def __init__(self, model: Union[PeftModel, AutoModelForCausalLM], args: argparse.ArgumentParser):
        super().__init__()
        self.model = model
        self.args = args

    def forward(self, batch):
        return self.model(**batch)
    
    def configure_optimizers(self):
        if self.args.load_8bit:
            optimizer = bnb.optim.Adam8bit(
                self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.995)
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
            )

        if self.args.lr_scheduler == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.args.ExponentialLR_gamma)
            return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def get_backbone(self):
        return self.model