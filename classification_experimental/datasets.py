from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import csv
import xml.etree.ElementTree as ET
from hackashop_datasets.panbot import load_panbot
from hackashop_datasets.ynacc import load_ynacc_data


# def ynacc_loader(args):
#     '''
#     Read single annotated file as pandas dataframe.
#     '''
#     data_dir = Path(args.dataset_path)
#     data_file = data_dir / 'ydata-ynacc-v1_0_expert_annotations.tsv'
#     data = pd.read_csv(data_file, sep="\t", engine='python', quoting=csv.QUOTE_NONE)
#     return data
#
#
# class YnaccDataset(torch.utils.data.Dataset):
#     LABEL_MAPS = {
#         'constructiveclass': {'Not constructive': 0, 'Constructive': 1}
#     }
#     ID2LABEL = {
#         'constructiveclass': {0: 'Not constructive', 1: 'Constructive'}
#     }
#
#     def __init__(self, data, max_len, label, tokenizer):
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
#         self.label = label
#         data.dropna(subset=[label], inplace=True)
#         self.label_map = YnaccDataset.LABEL_MAPS[self.label]
#         self.encodings = self.tokenizer(data['text'].tolist(), truncation=True, padding=True, max_length=max_len)
#         self.labels = [self.label_map[label] for label in data[self.label]]
#
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
#
#     def __len__(self):
#         return len(self.labels)


class TaskDataset(torch.utils.data.Dataset):
    '''
    This is dataset wrapper for conntecting hackathlon datasets
    '''

    def __init__(self, texts, labels, max_len, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.labels = labels
        self.texts = self.tokenizer(texts, truncation=True, padding=True, max_length=max_len)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


DATA_LOADERS = {
    'ynacc': load_ynacc_data,
    'pan_bot': load_panbot
}
