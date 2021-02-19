from transformers import AutoTokenizer
import torch
from hackashop_datasets.panbot import load_panbot
from hackashop_datasets.ynacc import load_ynacc_data
from hackashop_datasets.load_data import load_toxic_en_data
from hackashop_datasets.cro_24sata import load_cro_train, load_cro_train2, load_cro_dev, load_cro_dev2
from hackashop_datasets.est_express import load_est_train, load_est_train2, load_est_dev, load_est_dev2

class TaskDataset(torch.utils.data.Dataset):
    '''
    This is dataset wrapper for conntecting hackathlon datasets
    '''

    def __init__(self, texts, labels, max_len, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.labels = labels
        self.len = len(texts)
        self.texts = self.tokenizer(texts, truncation=True, padding=True, max_length=max_len)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.len


DATA_LOADERS = {
    'ynacc': load_ynacc_data,
    'en_toxic': load_toxic_en_data,
    'pan_bot': load_panbot,
    'cro_train': load_cro_train, # 40.000 examples
    'cro_train_large': load_cro_train2, # 80.000 examples
    'cro_dev': load_cro_dev,  # 10.000 examples
    'cro_dev_large': load_cro_dev2,  # 15.000 examples
    'est_train': load_est_train, # 40.000 examples
    'est_train_large': load_est_train2, # 80.000 examples
    'est_dev': load_est_dev,  # 10.000 examples
    'est_dev_large': load_est_dev2  # 15.000 examples

}

def croest_loaders_test():
    for l in DATA_LOADERS.keys():
        if l.startswith('cro') or l.startswith('est'):
            loader = DATA_LOADERS[l]
            texts, labels = loader()
            print(l)
            print(labels[:3]); print(texts[:3])
            print()

if __name__ == '__main__':
    croest_loaders_test()