# pip install transformers datasets sklearn numpy torch torchvision
# code adapted from:
# https://colab.research.google.com/drive/1ayU3ERpzeJ8fHFJoEBCVCklxvvgjEz_P?usp=sharing#scrollTo=xxcHlNP21An8

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer
import numpy as np

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

def build_twitter_sentiment_classifier():
    # load data
    dataset = load_en_cro_tweet_data()
    metric = load_metric('glue', 'sst2')
    tokenizer = AutoTokenizer.from_pretrained(
        'EMBEDDIA/crosloengual-bert',
        use_fast=True
    )
    label2id = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
    id2label = ['Negative', 'Neutral', 'Positive']
    def preprocess_function(examples):
        print(examples)
        result = tokenizer(examples['sentence'], truncation=True, max_length=512)
        result['label'] = [label2id[l] for l in examples['label']]
        return result
    encoded_dataset = dataset.map(preprocess_function,
                                  batched=True)  # add load_from_cache_file=False to reload from scratch
    # fine-tune/train BERT model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        'EMBEDDIA/crosloengual-bert', num_labels=3)
    args = TrainingArguments(
        "tweet-sentiment",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=None,
    )
    eval_results = trainer.evaluate()
    print(eval_results)
    trainer.save_model(output_dir='tweet-sentiment-model')
    # use model for sequence prediction
    model = AutoModelForSequenceClassification.from_pretrained('tweet-sentiment-model')
    # predict sentiment
    examples = ['Today, the president met with the CEO of the fastest growing company in the last year.',
                'Thank you so much for this wonderful present, I really love it!',
                'stupid politicians screwed it all up again',
                'Don\'t touch me or I will kick you',
                'can\'t wait to finally see the eclipse!',
                'Ne bih volio letjeti u 737 Max.']
    inputs = tokenizer(examples, padding='longest', return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].detach().numpy()
    for i in range(len(examples)):
        print(examples[i], '\t', id2label[np.argmax(probs[i])])

if __name__ == '__main__':
    build_twitter_sentiment_classifier()