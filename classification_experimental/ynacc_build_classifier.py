from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import random

from classification_experimental.BertFeatureExtractor import BertFeatureExtractor

#import warnings
#from sklearn.exceptions import *
#warnings.filterwarnings(action='ignore', category=Warning)
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")

def ynacc_load(path, file="ydata-ynacc-v1_0_expert_annotations.tsv"):
    '''
    Read single annotated file as pandas dataframe.
    '''
    import pandas as pd
    from pathlib import Path
    import csv
    data_dir = Path(path)
    data_file = data_dir / file
    data = pd.read_csv(data_file, sep="\t", engine='python', quoting=csv.QUOTE_NONE)
    return data

constructiveLabels = {'Not constructive':0, 'Constructive':1}

def load_ynacc_data(file="ydata-ynacc-v1_0_expert_annotations.tsv",
                    label='constructiveclass',
                    label_map=constructiveLabels,
                    trunc_max_chars=1000):
    '''
    Load and prepare training data.
    returns: texts, labels
    '''
    data = ynacc_load('/data/resources/hackashop/ydata-ynacc-v1_0/', file)
    text_column = data['text']
    label_column = data[label]
    # print all labels
    # print(set([label_column[i] for i in data.index]))
    # label map, and filter -> label not recognized, example discarded
    texts, labels = [], []
    for i in data.index:
        label = label_column[i]
        if label in label_map: # ignore labels such as NaN
            text = text_column[i]
            # truncate text because of the bug in BERT preproc (probably tokenizer)
            #   it crashes for some long texts
            if trunc_max_chars: text = text[:trunc_max_chars]
            texts.append(text)
            labels.append(label_map[label])
            #print(label, text)
    #print(len(texts))
    return texts, labels

def build_and_test_classifier(features="bert", subsample=1000, rseed=572):
    # setup pipeline with feat.extract and classif. model
    if features == "bert": fextr = BertFeatureExtractor();
    elif features == "tfidf": fextr = TfidfVectorizer()
    pipe = [("fextr", fextr),
            ("logreg", LogisticRegression(solver='liblinear', max_iter=1000, penalty='l1'))]
    pipe = Pipeline(pipe)
    # prepare data
    texts, labels = load_ynacc_data()
    N = len(texts)
    if subsample:
        random.seed(rseed)
        ids = random.sample(range(N), subsample)
        texts, labels = [texts[i] for i in ids], [labels[i] for i in ids]
    texts_train, texts_test, labels_train, labels_test = \
        train_test_split(texts, labels, test_size=0.33, random_state=rseed)
    # train
    pipe.fit(texts_train, labels_train)
    labels_predict = pipe.predict(texts_test)
    # calculate
    f1 = f1_score(labels_test, labels_predict)
    precision = precision_score(labels_test, labels_predict)
    recall = recall_score(labels_test, labels_predict)
    print(f"f1: {f1:1.3f}, precision: {precision:1.3f}, recall: {recall:1.3f}")

if __name__ == '__main__':
    #load_ynacc_data()
    #build_and_test_classifier("tfidf", subsample=10000)
    build_and_test_classifier("bert", subsample=5000)