
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from classification_experiments.BertFeatureExtractor import BertFeatureExtractor

from hackashop_datasets import load_data

def build_and_test_classifier(features="bert",
                              fine_tuned_model=None,
                              clf=SGDClassifier(alpha = 0.0001, 
                                                penalty='l2', 
                                                class_weight='balanced'),
                              max_length=128,
                              rseed=42):

    # setup pipeline with feat.extract and clf
    
    if features == "bert": 
        fextr = BertFeatureExtractor()
    elif features == "tfidf": 
        fextr = TfidfVectorizer()
    
    pipe = [("fextr", fextr), ("clf", clf)]
    pipe = Pipeline(pipe)
    
    # prepare data
    texts, labels = load_data.load_toxic_en_data()
    data_split = train_dev_test(texts, labels, rseed)
    
    # train
    pipe.fit(data_split['train'][0], data_split['train'][1])
    labels_predict = pipe.predict(data_split['test'][0])

    # calculate metrics
    f1 = f1_score(data_split['test'][1], labels_predict)
    precision = precision_score(data_split['test'][1], labels_predict)
    recall = recall_score(data_split['test'][1], labels_predict)
    print(f"f1: {f1:1.3f}, precision: {precision:1.3f}, recall: {recall:1.3f}")

if __name__ == '__main__':
    #load_ynacc_data()
    build_and_test_classifier("tfidf")
