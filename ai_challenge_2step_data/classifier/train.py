import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import os
import csv
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    ElectraConfig,
    ElectraForSequenceClassification,
    ElectraTokenizer
)

from config import Config

import sys
from os import path
print(path.dirname( path.dirname( path.abspath(__file__) ) ))
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
from utils import load_json, CustomTokenizer
from model import CustomModule



class CustomDataset(Dataset):
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg

        self.config = ElectraConfig.from_pretrained(
                self.cfg.model_name,
        )

        self.tokenizer = ElectraTokenizer.from_pretrained(
                self.cfg.model_name,
                config = self.config
        )

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

        print(file_list)
        for file_name in file_list:
            # print(os.path.join(self.cfg.DATASET_DIR,file_name))
            data = load_json(os.path.join(self.cfg.DATASET_DIR,file_name))
            for question in data:
                conv, data_dict = c_tokenizer.tokenize(question['Question'])
                # print(conv)
                tok = self.tokenizer.encode(
                        conv,
                        max_length=512,
                        padding='max_length',
                        truncation=True
                )
                xs.append(tok)
            ys.extend([int(file_name[0])-1]*len(data))
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
        self.num_workers = num_workers
        self.set_datasets()

    def set_datasets(self):
        self.dataset = CustomDataset(
            cfg = self.cfg
        )

        dataset_size = len(self.dataset)
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
            num_workers=self.num_workers,
            sampler = self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler = self.val_sampler
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler = self.test_sampler
        )

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def train(model,criterion,optimizer,train_loader, val_loader, cfg):
    data_module = CustomDataModule(
        cfg=cfg,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_stats = {
            'train':[],
            'val':[]
        }
    accuracy_stats = {
            'train':[],
            'val':[]
        }

    early_stopping = EarlyStopping(patience=20,verbose=True)

    for e in tqdm(range(1, cfg.max_epochs+1)):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        
        model.train()
        
        for X_train_batch, y_train_batch in tqdm(train_loader):
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            x_ = (X_train_batch.cpu().numpy())
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred[0], y_train_batch)
            train_acc = multi_acc(y_train_pred[0], y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()


        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred[0], y_val_batch)
                val_acc = multi_acc(y_val_pred[0], y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))


        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

        early_stopping(loss_stats['val'][-1],model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, loss_stats, accuracy_stats

def test(model, test_loader, criterion, cfg):
    test_loss = 0.

    class_correct = [0] *  8
    class_total = [0] * 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    for data, target in test_loader:
        
        data, target = data.to(device), target.to(device)

        if len(target.data) != cfg.batch_size:
            break

        print(data)

        output = model(data)

        loss = criterion(output[0],target)
        test_loss += loss.item()*data.size(0)

        _, pred = torch.max(output[0],1)

        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        for i in range(cfg.batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.4f}\n'.format(test_loss))

    for i in range(8):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i+1), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

if __name__ == "__main__":
    cfg = Config()
    data_module = CustomDataModule(
        cfg=cfg,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    
    """
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    """
    test_loader = data_module.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",num_labels=8)

    model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",config=config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=cfg.learning_rate)
    """
    model, loss, acc = train(model,criterion,optimizer,train_loader, val_loader, cfg)
    
    with open('loss.csv','w') as f:
        writer = csv.writer(f)
        for k,v in loss.items():
            writer.writerow([k,v])

    with open('acc.csv','w') as f:
        writer = csv.writer(f)
        for k,v in acc.items():
            writer.writerow([k,v])
    """
    model.load_state_dict(torch.load('checkpoint.pt'))
    test(model,test_loader,criterion,cfg)
