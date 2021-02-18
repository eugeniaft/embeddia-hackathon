'''
Modified feature extraction and prediction code from 'generic_transformer_classifier.py'
Put in a separate file to avoid potential code clashes.
'''

import torch
from classification_experimental.datasets_for_finetune import DATA_LOADERS, TaskDataset
from transformers import AutoModelForSequenceClassification, AutoModel
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F


def predict_fn(fine_tuned_model, max_len, texts,
               tokenizer='EMBEDDIA/crosloengual-bert', torch_device='cpu'):
    '''
    device: torch.device('cuda') or torch.device('cpu')
    fine_tuned_model: directory of fine tuned model, inside BERT_FOLDER
    max_len: use the length that had been used for fine-tuning, generally 128 since the texts are short
    texts: array of strings from data loaders
    tokenizer: pretrained model name for the tokenizer

    the function returns array of dictionary containing labels (0 or 1) and probabilities of assigned label
    '''

    device = torch.device(torch_device)
    model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model)
    dataset = TaskDataset(texts=texts, labels=None, max_len=max_len,
                          tokenizer=tokenizer)
    data_loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=1)
    model.eval()
    model.to(device)

    results = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits
            # probs = torch.max(F.softmax(logits, dim=1)).cpu().detach().numpy().item()
            probs = F.softmax(logits, dim=1).cpu().detach().numpy()[0]
            label = torch.argmax(logits).cpu().detach().numpy().item()
            results.append({'probs': probs,
                            'label': label
                            })
    return results


def features_finetuned_model(text, labels, fine_tuned_model, max_len, torch_device):
    '''
    Get embedding layer from fine tuned model to use for classification task
    device: torch.device('cuda') or torch.device('cpu')
    fine_tuned_model: directory of fine tuned model
    max_len: use the length that had been used for fine-tuning
    torch_device: 'cpu' or 'cuda'
    texts: array of strings from data loaders
    '''
    device = torch.device(torch_device)
    dataset = TaskDataset(texts=text, labels=labels, max_len=max_len,
                          tokenizer='EMBEDDIA/crosloengual-bert')
    data_loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=1)
    finetuned_model = AutoModel.from_pretrained(fine_tuned_model)
    finetuned_model.to(device)
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = finetuned_model(input_ids, token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)
            pooled_output = torch.mean(outputs[0], 1)
            embeddings = pooled_output.cpu().detach().numpy()
            yield embeddings


if __name__ == '__main__':
    pass
