import numpy as np

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib as mpl
import matplotlib.pyplot as plt

from modAL.models import ActiveLearner
from modAL.density import information_density


def modAL_example(base_learner = KNeighborsClassifier(n_neighbors=3), rseed=7712):
    '''
    Adapted from:
    https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html
    '''
    np.random.seed(rseed)
    # load data
    iris = load_iris()
    X_raw = iris['data']
    y_raw = iris['target']
    # prepare data for AL
    n_labeled_examples = X_raw.shape[0]
    ## separate inital learning data
    training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=3)
    X_train = X_raw[training_indices]
    y_train = y_raw[training_indices]
    ## delete inital data to create pool for querying
    X_pool = np.delete(X_raw, training_indices, axis=0)
    y_pool = np.delete(y_raw, training_indices, axis=0)
    # create learner
    # cosine_density = information_density(X, 'cosine')
    learner = ActiveLearner(estimator=base_learner, X_training=X_train, y_training=y_train)
    ## initial score on the raw data
    unqueried_score = learner.score(X_raw, y_raw)
    # RUN ACTIVE LEARNING LOOP
    N_QUERIES = 20
    performance_history = [unqueried_score]
    # Allow our model to query our unlabeled dataset for the most
    # informative points according to our query strategy (uncertainty sampling).
    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(X_pool)
        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)
        # Remove the queried instance from the unlabeled pool.
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
        # Calculate and report our model's accuracy.
        model_accuracy = learner.score(X_raw, y_raw)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        # Save our model's performance for plotting.
        performance_history.append(model_accuracy)
    # Plot our performance over time.
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    ax.plot(performance_history)
    ax.scatter(range(len(performance_history)), performance_history, s=13)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)
    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')
    plt.show()

def al_baseline(data, rseed=7712):
    '''
    Baseline method for binary text classification.
    '''
    from classification_experiments.classification_helpers import subsample_data
    from sklearn.feature_extraction.text import TfidfVectorizer
    np.random.seed(rseed)
    text, labels = data
    tfidf = TfidfVectorizer()
    X_raw = tfidf.fit_transform(text)
    y_raw = np.array(labels)
    print(X_raw.shape); print(y_raw.shape)
    base_learner = LogisticRegression()
    CHUNK = 1000
    # prepare data for AL
    n_labeled_examples = X_raw.shape[0]
    ## separate inital learning data
    training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=CHUNK)
    X_train = X_raw[training_indices]
    y_train = y_raw[training_indices]
    print(training_indices)
    ## delete inital data to create pool for querying
    X_pool = np.delete(X_raw, training_indices, axis=0)
    y_pool = np.delete(y_raw, training_indices, axis=0)
    # create learner
    # cosine_density = information_density(X, 'cosine')
    learner = ActiveLearner(estimator=base_learner, X_training=X_train, y_training=y_train)
    ## initial score on the raw data
    unqueried_score = learner.score(X_raw, y_raw)
    # RUN ACTIVE LEARNING LOOP
    N_QUERIES = 20
    performance_history = [unqueried_score]
    # Allow our model to query our unlabeled dataset for the most
    # informative points according to our query strategy (uncertainty sampling).
    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(X_pool)
        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)
        # Remove the queried instance from the unlabeled pool.
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
        # Calculate and report our model's accuracy.
        model_accuracy = learner.score(X_raw, y_raw)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        # Save our model's performance for plotting.
        performance_history.append(model_accuracy)
    # Plot our performance over time.
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    ax.plot(performance_history)
    ax.scatter(range(len(performance_history)), performance_history, s=13)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)
    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')
    plt.show()

def hasoc_alearn():
    from hackashop_datasets.hasoc2019 import load_hasoc_data, hasoc_labels_hateoffensive
    data = load_hasoc_data(hasoc_labels_hateoffensive) # task is hate+offensive vs rest
    #build_and_test_classifier(data, "bert", subsample=20000)
    al_baseline(data, 2134)

def ynacc_alearn():
    from hackashop_datasets.ynacc import load_ynacc_data, ynacc_constructive_labels
    data = load_ynacc_data(ynacc_constructive_labels) # task is constructive vs rest
    al_baseline(data, 5445)

if __name__ == '__main__':
    #modAL_example()
    #modAL_example(LogisticRegression())
    hasoc_alearn()