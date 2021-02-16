import random, numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from classification_experiments.BertFeatureExtractor import BertFeatureExtractor
from classification_experiments.classification_models import build_classifier
from hackashop_datasets.cro_24sata import cro24_load_tfidf
from hackashop_datasets.est_express import est_load_tfidf


def get_classifier(c):
    '''
    Factory method building scikit-learn classifiers.
    :param c: classifier label
    '''
    if c == 'logreg':
        return LogisticRegression(solver='liblinear', max_iter=1000, penalty='l1', C=1.0)
    elif c == 'svm':
        return SVC()

def build_and_test_classifier(data, features="bert", classifier="logreg",
                              subsample=None, rseed=572):
    '''
    Build and test binary classifier -
     subsample data if specified, create train/test split, train and test.
    :param data: (texts, labels) pair - list of texts, list of binary labels
    :param features: 'bert' of 'tfidf'
    :param subsample: if not False, first subsample data to given size
    :return:
    '''
    classif = create_classifier(features, classifier)
    # prepare data
    texts, labels = subsample_data(data, subsample, rseed)
    N = len(texts); print(f'Dataset size: {N}')
    texts_train, texts_test, labels_train, labels_test = \
        train_test_split(texts, labels, test_size=0.33, random_state=rseed)
    # train
    classif.fit(texts_train, labels_train)
    labels_predict = classif.predict(texts_test)
    # calculate
    test_classifier(classif, (texts_test, labels_test), subsample=False)

def build_and_test_classifier_split(train, test,
                                    features='bert', classifier='logreg', rseed=572):
    '''
    Build and test binary classifier on a train/test split data.
    :param train: (texts, labels) pair - list of texts, list of binary labels
    :param test: (texts, labels) pair - list of texts, list of binary labels
    :param features: 'bert' of 'tfidf'
    :return:
    '''
    np.random.seed(rseed)
    # prepare data
    texts_train, labels_train  = train
    texts_test, labels_test = test
    N = len(texts_train); print(f'train size: {N}')
    # feature extraction
    if features == 'bert': fextr = BertFeatureExtractor();
    elif features == 'tfidf': # fit tfidf on train+test texts
        fextr = TfidfVectorizer(sublinear_tf=True)
        all_texts = []; all_texts.extend(texts_train); all_texts.extend(texts_test)
        fextr.fit(all_texts)
    elif features == 'tfidf-cro': fextr = cro24_load_tfidf()
    elif features == 'tfidf-est': fextr = est_load_tfidf()
    feats_train = fextr.transform(texts_train)
    feats_test = fextr.transform(texts_test)
    # train model
    classif = create_classifier_grid(classifier)
    classif.fit(feats_train, labels_train)
    if ('grid' in classifier):
        print(classif.best_estimator_)
        print(classif.best_params_)
        print(classif.best_score_)
    # calculate
    test_classifier(classif, (feats_test, labels_test), subsample=False)

def create_classifier(features='bert', classifier='logreg'):
    '''
    Compose a scikit-learn classifier ready for training.
    :param features: 'bert' of 'tfidf'
    :parameter classifier: 'logreg' or 'svm'
    :return:
    '''
    if features == 'bert': fextr = BertFeatureExtractor();
    elif features == 'tfidf': fextr = TfidfVectorizer()
    classifier = get_classifier(classifier)
    pipe = [('fextr', fextr),
            ('classifier', classifier)]
    pipe = Pipeline(pipe)
    return pipe

def create_classifier_grid(classifier='logreg'):
    '''
    Factory method building scikit-learn classifiers, possibly
     wrapped in a crossvalidation fitter using grid search.
    :param c: classifier label
    '''
    #if features == 'bert': fextr = BertFeatureExtractor();
    #elif features == 'tfidf': fextr = TfidfVectorizer()
    if 'grid' in classifier:
        model, paramgrid = build_classifier(classifier)
        # model_label = 'classifier'
        # pipe = [('feature_extr', fextr),
        #         (model_label, model)]
        # pipe = Pipeline(pipe)
        # grid = {'%s__%s' % (model_label, k): v for k, v in paramgrid.items()}
        # cvFitter = GridSearchCV(estimator=pipe, param_grid=grid, cv=5,
        #                         scoring='f1', verbose=True, n_jobs=3)
        cvFitter = GridSearchCV(estimator=model, param_grid=paramgrid, cv=5,
                                scoring='f1', verbose=True, n_jobs=3)
        return cvFitter

def create_train_classifier(train, features='bert', classifier='logreg', subsample=False, rseed=883):
    '''
    Create classifier, fit to data, possibly with subsampling
    :param train: texts, labels
    '''
    classif = create_classifier(features, classifier)
    train = subsample_data(train, subsample=subsample, rseed=rseed)
    texts, data = train
    classif.fit(texts, data)
    return classif


def subsample_data(data, subsample, rseed=572):
    '''
    :param data: texts, labels
    :param subsample: subsample size
    :return: subsampled texts, labels
    '''
    if not subsample: return data
    texts, labels = data; N = len(texts)
    if subsample >= N or subsample < 0: return data
    random.seed(rseed)
    ids = random.sample(range(N), subsample)
    texts, labels = [texts[i] for i in ids], [labels[i] for i in ids]
    return texts, labels


def test_classifier(c, data, subsample=False, rseed=883):
    '''
    Compare predictions of classifier with correct labels
    :param c: trained sklearn classifier
    :param data: text, labels - reference (correct) labeling
    :param subsample: if true, test on a subsample of data
    :return:
    '''
    if subsample: texts, labels = subsample_data(data, subsample, rseed)
    else: texts, labels = data
    labels_predict = c.predict(texts)
    # calculate
    f1 = f1_score(labels, labels_predict)
    precision = precision_score(labels, labels_predict)
    recall = recall_score(labels, labels_predict)
    print(f"f1: {f1:1.3f}, precision: {precision:1.3f}, recall: {recall:1.3f}")