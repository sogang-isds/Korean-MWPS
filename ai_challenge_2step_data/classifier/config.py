# Standard
from os.path import dirname, abspath, join
import random

# PIP
import numpy as np
import torch
import torch.nn as nn

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    XLMRobertaConfig,
    ElectraTokenizer,
    XLMRobertaTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    ElectraForSequenceClassification,
    XLMRobertaForSequenceClassification,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification,
    XLMRobertaForTokenClassification,
    BertForQuestionAnswering,
    DistilBertForQuestionAnswering,
    ElectraForQuestionAnswering,
    XLMRobertaForQuestionAnswering,
)

CONFIG_CLASSES = {
    "kobert": BertConfig,
    "distilkobert": DistilBertConfig,
    "hanbert": BertConfig,
    "koelectra-base": ElectraConfig,
    "koelectra-small": ElectraConfig,
    "koelectra-base-v2": ElectraConfig,
    "koelectra-base-v3": ElectraConfig,
    "koelectra-small-v2": ElectraConfig,
    "koelectra-small-v3": ElectraConfig,
    "xlm-roberta": XLMRobertaConfig,
}

TOKENIZER_CLASSES = {
    "koelectra-base": ElectraTokenizer,
    "koelectra-small": ElectraTokenizer,
    "koelectra-base-v2": ElectraTokenizer,
    "koelectra-base-v3": ElectraTokenizer,
    "koelectra-small-v2": ElectraTokenizer,
    "koelectra-small-v3": ElectraTokenizer,
    "xlm-roberta": XLMRobertaTokenizer,
}

MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    "kobert": BertForSequenceClassification,
    "distilkobert": DistilBertForSequenceClassification,
    "hanbert": BertForSequenceClassification,
    "koelectra-base": ElectraForSequenceClassification,
    "koelectra-small": ElectraForSequenceClassification,
    "koelectra-base-v2": ElectraForSequenceClassification,
    "koelectra-base-v3": ElectraForSequenceClassification,
    "koelectra-small-v2": ElectraForSequenceClassification,
    "koelectra-small-v3": ElectraForSequenceClassification,
    "xlm-roberta": XLMRobertaForSequenceClassification,
}

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "kobert": BertForTokenClassification,
    "distilkobert": DistilBertForTokenClassification,
    "hanbert": BertForTokenClassification,
    "koelectra-base": ElectraForTokenClassification,
    "koelectra-small": ElectraForTokenClassification,
    "koelectra-base-v2": ElectraForTokenClassification,
    "koelectra-base-v3": ElectraForTokenClassification,
    "koelectra-small-v2": ElectraForTokenClassification,
    "koelectra-small-v3": ElectraForTokenClassification,
    "koelectra-small-v3-51000": ElectraForTokenClassification,
    "xlm-roberta": XLMRobertaForTokenClassification,
}

MODEL_FOR_QUESTION_ANSWERING = {
    "kobert": BertForQuestionAnswering,
    "distilkobert": DistilBertForQuestionAnswering,
    "hanbert": BertForQuestionAnswering,
    "koelectra-base": ElectraForQuestionAnswering,
    "koelectra-small": ElectraForQuestionAnswering,
    "koelectra-base-v2": ElectraForQuestionAnswering,
    "koelectra-base-v3": ElectraForQuestionAnswering,
    "koelectra-small-v2": ElectraForQuestionAnswering,
    "koelectra-small-v3": ElectraForQuestionAnswering,
    "xlm-roberta": XLMRobertaForQuestionAnswering,
}

class Config:
    # User Setting
    SEED = 94
    DATASET_DIR = ('../data')  

    model_type = 'koelectra-base-v3'
    model_name = 'monologg/koelectra-base-v3-discriminator'
    
    num_gpus = 1
    max_epochs = 60
    batch_size = 4 
     
    criterion = 'RMSE'
    optimizer = 'AdamW'
    learning_rate = 1.6585510780186816e-06
    num_workers = 4
    verbose = 0  # 0: quiet, 1: with log

    special_tokens = [f'[cls{i+1}]' for i in range(8)]

    def __init__(self, SEED=None):
        if SEED:
            self.SEED = SEED
        self.set_random_seed()

    def set_random_seed(self):
        print(f'=> SEED : {self.SEED}')

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(self.SEED)  # if use multi-GPU

