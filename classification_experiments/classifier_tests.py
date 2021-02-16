from classification_experiments.classification_helpers import \
    build_and_test_classifier, create_train_classifier, test_classifier

def hasoc_classifer_test():
    from hackashop_datasets.hasoc2019 import load_hasoc_data, hasoc_labels_hateoffensive
    data = load_hasoc_data(hasoc_labels_hateoffensive) # task is hate+offensive vs rest
    build_and_test_classifier(data, "bert", subsample=100)
    #build_and_test_classifier(data, "tfidf", classifier='logreg', subsample=None)

def ynacc_classifier_test():
    from hackashop_datasets.ynacc import load_ynacc_data, ynacc_constructive_labels
    data = load_ynacc_data(ynacc_constructive_labels) # task is constructive vs rest
    build_and_test_classifier(data, "tfidf", classifier='logreg', subsample=20000)

def cro24sata_offensive_classifier_test():
    from hackashop_datasets.cro_24sata import cro24sata_unbalanced_offensive
    data = cro24sata_unbalanced_offensive() # task is constructive vs rest
    build_and_test_classifier(data, "tfidf", classifier='logreg', subsample=30000)

def test_offensive_transfer_hasoc24sata(subsample=10000, rseed=1312):
    '''
    Test transfer of offensive labes from hasoc to 24sata,
    using bert features and a separate classifier.
    '''
    from hackashop_datasets.hasoc2019 import load_hasoc_data, hasoc_labels_hateoffensive
    from hackashop_datasets.cro_24sata import cro24sata_unbalanced_offensive
    train = load_hasoc_data(hasoc_labels_hateoffensive)  # task is hate+offensive vs rest
    classif_hasoc = create_train_classifier(train, 'bert', 'logreg', subsample, rseed)
    test = cro24sata_unbalanced_offensive()
    test_classifier(classif_hasoc, test, subsample, rseed)

if __name__ == '__main__':
    hasoc_classifer_test()
    #ynacc_classifier_test()
    #test_offensive_transfer_hasoc24sata()
    #cro24sata_offensive_classifier_test()