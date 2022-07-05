# Standard
import os
import sys
from datetime import date
import re

# PIP
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Custom
from config import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES
)

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils import load_json, CustomTokenizer

class CustomDataset(Dataset):
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg

        self.config = CONFIG_CLASSES[self.cfg.model_type].from_pretrained(
                self.cfg.model_name,
                max_seq_length = 512
        )

        self.tokenizer = TOKENIZER_CLASSES[self.cfg.model_type].from_pretrained(
                self.cfg.model_name,
                config = self.config
        )

        special_tokens_dict = {
                'additional_special_tokens': self.cfg.special_tokens
        }
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


        xs, ys = self.gather_data()

        print(self.tokenizer.decode(xs[0]))
        print(self.tokenizer.decode(xs[2]))
        print(self.tokenizer.decode(xs[1]))

        self.X = torch.tensor(xs, dtype=torch.long)
        self.y = torch.tensor(ys, dtype=torch.long)

    def gather_data(self):
        xs = []
        ys = []

        c_tokenizer = CustomTokenizer('../names.txt')

        file_list = sorted([f for f in os.listdir(self.cfg.DATASET_DIR) if f[0].isdigit()])
        for file_name in file_list:
            data = self.load_json(os.path.join(self.cfg.DATASET_DIR,file_name))
            for question in data:
                conv, data_dict = c_tokenizer.tokenize(question['Question'])
                conv = f'[cls{file_name[0]}] '+conv
                print(conv)
                tok = self.tokenizer.encode(
                        conv,
                        padding='max_length',
                        truncation=True
                )
                xs.append(tok)
            ys.extend([int(file_name[0])]*len(data))
        return xs, ys

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CustomDataModule():
    def __init__(
        self,
        cfg,
        batch_size=1,
        num_workers=0,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.set_datasets()

    def set_datasets(self):
        self.dataset = CustomDataset(
            cfg = self.cfg
        )

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.seed(self.cfg.SEED)
        np.random.shuffle(indices)

        train_size = int(dataset_size * 0.8)
        val_size = int(dataset_size * 0.1)

        train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            sampler = self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            sampler = self.val_sampler
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            sampler = self.test_sampler
        )

    
