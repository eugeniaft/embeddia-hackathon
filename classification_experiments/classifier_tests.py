import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from classification_experiments.BertFeatureExtractor import BertFeatureExtractor


def get_classifier(c):
    '''
    Factory method building scikit-learn classifiers.
    :param c: classifier label
    '''
    if c == 'logreg':
        return LogisticRegression(solver='liblinear', max_iter=1000, penalty='l1')
    elif c == 'svm':
        return SVC()

def build_and_test_classifier(data, features="bert", classifier="logreg",
                              subsample=None, rseed=572):
    '''
    Build and test binary classifier.
    :param data: (texts, labels) pair - list of texts, list of binary labels
    :param features: 'bert' of 'tfidf'
    :param subsample: if not False, first subsample data to given size
    :return:
    '''
    # setup pipeline with feat.extract and classif. model
    if features == "bert": fextr = BertFeatureExtractor();
    elif features == "tfidf": fextr = TfidfVectorizer()
    classifier = get_classifier(classifier)
    pipe = [("fextr", fextr),
            ("classifier", classifier)]
    pipe = Pipeline(pipe)
    # prepare data
    texts, labels = data
    N = len(texts)
    print(f'Dataset size: {N}')
    if subsample and subsample < N:
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

def hasoc_classifer_test():
    from hackashop_datasets.hasoc2019 import load_hasoc_data, hasoc_labels_hateoffensive
    data = load_hasoc_data(hasoc_labels_hateoffensive) # task is hate+offensive vs rest
    #build_and_test_classifier(data, "bert", subsample=20000)
    build_and_test_classifier(data, "tfidf", classifier='logreg', subsample=None)

def ynacc_classifier_test():
    from hackashop_datasets.ynacc import load_ynacc_data, ynacc_constructive_labels
    data = load_ynacc_data(ynacc_constructive_labels) # task is constructive vs rest
    build_and_test_classifier(data, "tfidf", classifier='logreg')

if __name__ == '__main__':
    #hasoc_classifer_test()
    ynacc_classifier_test()