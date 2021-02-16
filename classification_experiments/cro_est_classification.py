'''
Classification experiments with detection of blocked comments
 on the Croatian and Estonian datasets.
'''

from classification_experiments.classification_helpers import \
    build_and_test_classifier_split

from hackashop_datasets.cro_24sata import cro24_load_forclassif
from hackashop_datasets.est_express import est_load_forclassif

def cro_classifier_v0():
    # todo: build/test simple
    # todo: ? pre-built tfidf
    # todo: Xvalid param opt
    train = cro24_load_forclassif('train')
    dev = cro24_load_forclassif('dev')
    build_and_test_classifier_split(train, dev, features='tfidf')

def est_classifier_v0():
    train = est_load_forclassif('train')
    dev = est_load_forclassif('dev')
    build_and_test_classifier_split(train, dev, features='tfidf')

if __name__ == '__main__':
    #cro_classifier_v0()
    est_classifier_v0()
