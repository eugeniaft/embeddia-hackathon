'''
Classification experiments with detection of blocked comments
 on the Croatian and Estonian datasets.
'''

from classification_experiments.classification_helpers import \
    build_and_test_classifier_split, evaluate_predictions

from hackashop_datasets.cro_24sata import cro24_load_forclassif
from hackashop_datasets.est_express import est_load_forclassif
from classification_experiments.feature_extraction import \
    bert_feature_loader, BERT_CRO_V0, BERT_CRO_V1, BERT_EST_V1, BERT_CRO_FINETUNE

def cro_classifier_v0():
    train = cro24_load_forclassif('train')
    dev = cro24_load_forclassif('dev')
    build_and_test_classifier_split(train, dev,
                                    features='tfidf-cro', classifier='svc-grid')

def classifier_grid(lang='cro', label='CRO GRID', opt_metrics='f1'):
    if lang == 'cro':
        train = cro24_load_forclassif('train')
        dev = cro24_load_forclassif('dev')
    elif lang == 'est':
        train = est_load_forclassif('train')
        dev = est_load_forclassif('dev')
    for classif in ['logreg-grid', 'svc-grid']:
        for bal in [False, True]:
            for feats in ['tfidf', 'wcount']:
                for bigrams in [True, False]:
                    build_and_test_classifier_split(train, dev,
                                    classifier=classif, balanced=bal,
                                    features=feats, bigrams=bigrams, label=label,
                                                    opt_metrics=opt_metrics)
                    print()

def classifier_grid_bert(lang='cro', label='CRO GRID BERT'):
    if lang == 'cro':
        train = cro24_load_forclassif('train')
        dev = cro24_load_forclassif('dev')
    elif lang == 'est':
        train = est_load_forclassif('train')
        dev = est_load_forclassif('dev')
    for classif in ['logreg-grid', 'svc-grid']:
        for bal in [False, True]:
            for feats in ['tfidf+bert', 'wcount+bert']:
                for bigrams in [True, False]:
                    build_and_test_classifier_split(train, dev,
                                    classifier=classif, balanced=bal,
                                    features=feats, bigrams=bigrams, label=label,
                                    bert_loader=
                                    {
                                        'dset': lang, 'train_label': 'train', 'test_label': 'dev',
                                        'features': 'predict',
                                        'bert': BERT_CRO_V1}
                                    )
                    print()

def est_classifier_v0():
    train = est_load_forclassif('train')
    dev = est_load_forclassif('dev')
    build_and_test_classifier_split(train, dev,
                                    features='tfidf-est', classifier='svc-grid')


def evaluate_bert_labels(bert, dset='cro', split='dev'):
    '''
    Evaluate labels of a pre-trained BERT classifier.
    '''
    import numpy as np
    if dset == 'cro': _, labels = cro24_load_forclassif(split)
    elif dset == 'est': _, labels = est_load_forclassif(split)
    _, labels_bert = bert_feature_loader(dset, split, bert=bert, features='predict')
    #ones = np.ones(shape=labels_bert.shape)
    #labels_bert = ones - labels_bert
    evaluate_predictions(labels_bert, labels)

def test_combined_features():
    train = cro24_load_forclassif('train')
    dev = cro24_load_forclassif('dev')
    build_and_test_classifier_split(train, dev,
                                    classifier='logreg-grid', balanced=False,
                                    features='tfidf+bert', bigrams=False, label='COMB_TEST',
                                    bert_loader=
                                    {
                                     'dset': 'cro', 'train_label': 'train', 'test_label': 'dev',
                                     'features': 'predict',
                                     'bert': BERT_CRO_V1}
                                    )

def cro_classifier_best(label='CRO BEST', classifier='logreg-cro', balanced=False):
    train = cro24_load_forclassif('train2')
    dev = cro24_load_forclassif('test2')
    build_and_test_classifier_split(train, dev,
                                    classifier=classifier, balanced=balanced,
                                    features='wcount', bigrams=True, label=label,
                                    bert_loader=
                                    {
                                        'dset': 'cro', 'train_label': 'train', 'test_label': 'dev',
                                        'features': 'predict',
                                        'bert': BERT_CRO_V1}
                                    )

def est_classifier_best(label='EST BEST', classifier='logreg-est', balanced=False):
    train = est_load_forclassif('train2')
    dev = est_load_forclassif('test2')
    build_and_test_classifier_split(train, dev,
                                    classifier=classifier, balanced=balanced,
                                    features='wcount', bigrams=True, label=label)

def f1_baselines():
    from classification_experiments.classification_helpers import calculate_baseline_f1
    # minority class proportions
    cro_min, est_min = 0.0777, 0.0899
    print('CRO'); calculate_baseline_f1(cro_min)
    print('EST'); calculate_baseline_f1(est_min)

def cro_subcategories_recall():
    # labels: 1 (Disallowed content), 3 (Hate Speech), 5 (Deception& trolling), 6 (Vulgarity), 8 (Abuse)
    for label in [1.0, 3.0, 5.0, 6.0, 8.0]:
        texts, labels = cro24_load_forclassif('test2', label)
        print(f'LABEL: {label}')
        # only trained on english
        #_, labels_bert = bert_feature_loader('cro', 'test2', bert=BERT_CRO_V1, features='predict')
        #print('BERT-EN')
        #evaluate_predictions(labels_bert, labels)
        #fine tuned on cro
        _, labels_bert = bert_feature_loader('cro', 'test2', bert=BERT_CRO_FINETUNE, features='predict')
        print('BERT-CRO')
        evaluate_predictions(labels_bert, labels)
        # print('NATIVE-RECALL')
        # train = cro24_load_forclassif('train2')
        # dev = texts, labels
        # build_and_test_classifier_split(train, dev,
        #                             classifier='logreg-cro-recall', balanced=True,
        #                             features='wcount', bigrams=True, label='')

if __name__ == '__main__':
    #cro_classifier_v0()
    #cro_classifier_grid(opt_metrics='precision')
    #est_classifier_v0()
    #classifier_grid(lang='est', label='EST GRID', opt_metrics='precision')
    #evaluate_bert_labels(bert=BERT_CRO_FINETUNE, dset='cro', split='test2')
    #test_combined_features()
    #cro_classifier_best(classifier='logreg-cro', balanced=False)
    #cro_classifier_best(classifier='logreg-cro-recall', balanced=True)
    #est_classifier_best(classifier='logreg-est', balanced=False)
    #est_classifier_best(classifier='logreg-est-recall', balanced=True)
    #cro_classifier_grid_bert()
    #evaluate_bert_labels(bert=BERT_CRO_V1, dset='cro', split='test2')
    #f1_baselines()
    cro_subcategories_recall()
