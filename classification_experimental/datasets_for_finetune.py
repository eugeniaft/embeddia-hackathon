from transformers import AutoTokenizer
import torch
from hackashop_datasets.panbot import load_panbot
from hackashop_datasets.ynacc import load_ynacc_data


class TaskDataset(torch.utils.data.Dataset):
    '''
    This is dataset wrapper for conntecting hackathlon datasets
    '''

    def __init__(self, texts, labels, max_len, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.labels = labels
        self.texts = self.tokenizer(texts, truncation=True, padding=True, max_length=max_len)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


DATA_LOADERS = {
    'ynacc': load_ynacc_data,
    'pan_bot': load_panbot
}
