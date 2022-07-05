# Standard

# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

# Custom
from config import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION,
)

class CustomModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-5,
        criterion_name='RMSE',
        optimizer_name='Adam',
        momentum=0.9,
        model_type='koelectra-base-v3',
        **kwargs
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model_type = model_type

        self.model_name = "monologg/koelectra-base-v3-discriminator"
        self.config = CONFIG_CLASSES[self.model_type].from_pretrained(
                self.model_name,
                num_labels = 8,
        )

        self.model = MODEL_FOR_SEQUENCE_CLASSIFICATION[self.model_type].from_pretrained(
                self.model_name,
                config = self.config
        )
        
        
        self.tokenizer = TOKENIZER_CLASSES[self.model_type].from_pretrained(
                self.model_name,
                config = self.config
        )
        
        special_tokens_dict = {
                'additional_special_tokens': self.cfg.special_tokens
         }
     
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.optimizer = self.get_optimizer(optimizer_name)

    def get_optimizer(self, optimizer_name):
        name = optimizer_name.lower()

        if name == 'SGD'.lower():
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        if name == 'Adam'.lower():
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if name == 'AdamW'.lower():
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        raise ValueError(f'{optimizer_name} is not on the custom optimizer list!')

    def forward(self, **kwargs):

        out = self.model(**kwargs)

        return out

    def step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        self.log('loss', float(loss))
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        loss_arr = []

        for i in outputs:
            loss_cpu = i['loss'].cpu().detach()
            loss += loss_cpu
            loss_arr.append(float(loss_cpu))
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true.extend(i['y_true'])
            y_pred.extend(i['y_pred'])

        y_true = [int(y) for y in y_true]
        y_pred = [int(y) for y in y_pred]

        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', accuracy_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_precision', precision_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_recall', recall_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_f1', f1_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_mcc', matthews_corrcoef(y_true, y_pred), on_epoch=True, prog_bar=True)

        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, state='test')
