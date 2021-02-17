'''
Classification experiments with detection of blocked comments
 on the Croatian and Estonian datasets.
'''

from classification_experiments.classification_helpers import \
    build_and_test_classifier_split

from hackashop_datasets.cro_24sata import cro24_load_forclassif
from hackashop_datasets.est_express import est_load_forclassif

def cro_classifier_v0():
    train = cro24_load_forclassif('train')
    dev = cro24_load_forclassif('dev')
    build_and_test_classifier_split(train, dev,
                                    features='tfidf-cro', classifier='svc-grid')

def cro_classifier_grid(label='CRO GRID'):
    train = cro24_load_forclassif('train')
    dev = cro24_load_forclassif('dev')
    for classif in ['logreg-grid', 'svc-grid']:
        for bal in [False, True]:
            for feats in ['tfidf', 'wcount']:
                for bigrams in [True, False]:
                    build_and_test_classifier_split(train, dev,
                                    classifier=classif, balanced=bal,
                                    features=feats, bigrams=bigrams, label=label)
                    print()

def est_classifier_v0():
    train = est_load_forclassif('train')
    dev = est_load_forclassif('dev')
    build_and_test_classifier_split(train, dev,
                                    features='tfidf-est', classifier='svc-grid')

def est_classifier_grid(label='EST GRID'):
    train = est_load_forclassif('train')
    dev = est_load_forclassif('dev')
    for classif in ['logreg-grid', 'svc-grid']:
        for bal in [False, True]:
            for feats in ['tfidf', 'wcount']:
                for bigrams in [True, False]:
                    build_and_test_classifier_split(train, dev,
                                    classifier=classif, balanced=bal,
                                    features=feats, bigrams=bigrams, label=label)
                    print()

if __name__ == '__main__':
    #cro_classifier_v0()
    #cro_classifier_grid()
    #est_classifier_v0()
    est_classifier_grid()
