from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import csv


def load_en_cro_tweet_data():
    dataset = load_dataset(
        'csv',
        data_files={
            'train': 'data/twitter_sentiment/English_Twitter_sentiment.csv',
            'test': 'data/twitter_sentiment/Croatian_Twitter_sentiment.csv'
        },
        column_names=['sentence', 'label', 'annotator_id']
    )
    for item in dataset: print(item)
    return dataset


def ynacc_loader(args):
    '''
    Read single annotated file as pandas dataframe.
    '''
    data_dir = Path(args.dataset_path)
    data_file = data_dir / 'ydata-ynacc-v1_0_expert_annotations.tsv'
    data = pd.read_csv(data_file, sep="\t", engine='python', quoting=csv.QUOTE_NONE)
    return data


constructiveLabels = {'Not constructive': 0, 'Constructive': 1}


def load_ynacc_data(file="ydata-ynacc-v1_0_expert_annotations.tsv",
                    label='constructiveclass',
                    label_map=constructiveLabels,
                    trunc_max_chars=1000):
    '''
    Load and prepare training data.
    returns: texts, labels
    '''
    data = ynacc_loader('/data/resources/hackashop/ydata-ynacc-v1_0/', file)
    text_column = data['text']
    label_column = data[label]
    # print all labels
    # print(set([label_column[i] for i in data.index]))
    # label map, and filter -> label not recognized, example discarded
    texts, labels = [], []
    for i in data.index:
        label = label_column[i]
        if label in label_map:  # ignore labels such as NaN
            text = text_column[i]
            # truncate text because of the bug in BERT preproc (probably tokenizer)
            #   it crashes for some long texts
            if trunc_max_chars: text = text[:trunc_max_chars]
            texts.append(text)
            labels.append(label_map[label])
            # print(label, text)
    # print(len(texts))
    return texts, labels


class YnaccDataset(torch.utils.data.Dataset):
    LABEL_MAPS = {
        'constructiveclass': {'Not constructive': 0, 'Constructive': 1}
    }
    ID2LABEL = {
        'constructiveclass': {0: 'Not constructive', 1: 'Constructive'}
    }

    def __init__(self, data, max_len, label, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.label = label
        data.dropna(subset=[label], inplace=True)
        self.label_map = YnaccDataset.LABEL_MAPS[self.label]
        self.encodings = self.tokenizer(data['text'].tolist(), truncation=True, padding=True, max_length=max_len)
        self.labels = [self.label_map[label] for label in data[self.label]]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


DATA_LOADERS = {
    'ynacc': ynacc_loader
}

DATASETS = {
    'ynacc': YnaccDataset
}
