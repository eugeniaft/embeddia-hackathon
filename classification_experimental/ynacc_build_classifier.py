import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from classification_experiments.BertFeatureExtractor import BertFeatureExtractor
# import warnings
# from sklearn.exceptions import *
# warnings.filterwarnings(action='ignore', category=Warning)
# with warnings.catch_warnings():
#    warnings.simplefilter("ignore")

from hackashop_datasets.ynacc import load_ynacc_data, ynacc_constructive_labels

def build_and_test_classifier(features="bert", subsample=1000, rseed=572):
    # setup pipeline with feat.extract and classif. model
    if features == "bert": fextr = BertFeatureExtractor();
    elif features == "tfidf": fextr = TfidfVectorizer()
    pipe = [("fextr", fextr),
            ("logreg", LogisticRegression(solver='liblinear', max_iter=1000, penalty='l1'))]
    pipe = Pipeline(pipe)
    # prepare data
    texts, labels = load_ynacc_data(ynacc_constructive_labels)
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
    build_and_test_classifier("tfidf", subsample=10000)
    #build_and_test_classifier("bert", subsample=10000)