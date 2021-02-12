# pip install transformers datasets sklearn numpy torch torchvision
# code adapted from:
# https://colab.research.google.com/drive/1ayU3ERpzeJ8fHFJoEBCVCklxvvgjEz_P?usp=sharing#scrollTo=xxcHlNP21An8
import torch
from classification_experimental.datasets_for_finetune import DATA_LOADERS, TaskDataset
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, \
    TrainingArguments, Trainer
from argparse import ArgumentParser
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

metric = load_metric('glue', 'sst2')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def trainer(args):
    random_seed = args.random_seed
    task_name = args.task_name
    data, labels = DATA_LOADERS[args.dataset]('train')
    pretrained_model = args.pretrained_model

    # TODO add cross validation
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.8, random_state=random_seed)

    dataset_clf = TaskDataset
    train_dataset = dataset_clf(texts=train, labels=train_labels, max_len=args.max_len,
                                tokenizer=pretrained_model)
    val_dataset = dataset_clf(texts=test, labels=test_labels, max_len=args.max_len,
                              tokenizer=pretrained_model)

    # fine-tune/train BERT model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, num_labels=args.num_label)

    model_name = f"{pretrained_model}_{random_seed}_{task_name}"

    training_args = TrainingArguments(
        model_name,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    if args.fine_tune:
        print(f'Finetuning the model {model_name}')
        trainer.train()

    print(f'Evaluating the model {model_name}')
    eval_results = trainer.evaluate()
    print(f'Eval results \n{eval_results}')

    trainer.save_model(output_dir=model_name)
    print(f'Saved the model {model_name}')


def predict(test_data, classifier, test_batch):
    # TODO Support for gpu
    model = AutoModelForSequenceClassification.from_pretrained(classifier)
    model.eval()
    data_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=test_batch)

    _probs = []
    _labels = []
    for batch in tqdm(data_loader):
        labels = batch['labels']
        _labels.append(labels.detach().numpy())
        outputs = model(**batch, return_dict=True)
        probs = outputs['logits']
        probs = np.argmax(probs.detach().numpy())
        _probs.append(probs)

    probs = np.asarray(_probs).flatten()
    return probs, _labels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--pretrained_model', choices=['EMBEDDIA/crosloengual-bert'])
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--num_label', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--label_name', type=str)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--per_device_train_batch_size', type=int)
    parser.add_argument('--per_device_eval_batch_size', type=int)
    parser.add_argument('--weight_decay', type=float)
    args = parser.parse_args()
    trainer(args)
