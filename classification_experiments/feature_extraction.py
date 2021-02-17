from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion

from pathlib import Path
from project_settings import BERT_FOLDER
from classification_experiments.bert_features_predictions import predict_fn, features_finetuned_model

def tfidf_features(max_feats=None, bigrams=False):
    fextr = TfidfVectorizer(max_features=max_feats, sublinear_tf=True)
    if not bigrams: return fextr
    fextr_2g = TfidfVectorizer(max_features=25000, sublinear_tf=True, ngram_range=(2, 2))
    union = FeatureUnion([("words", fextr),
                          ("bigrams", fextr_2g)])
    return union

def wcount_features(max_feats=None, binary=True, bigrams=False):
    fextr = CountVectorizer(max_features=max_feats, binary=binary)
    if not bigrams: return fextr
    fextr_2g = CountVectorizer(max_features=max_feats, binary=binary, ngram_range=(2, 2))
    union = FeatureUnion([("words", fextr),
                          ("bigrams", fextr_2g)])
    return union

def bert_features(bert_folder, texts, features, max_len=256):
    '''
    :param bert_folder: folder within BERT_FOLDER (from project_settings.py)
    :param texts: list of texts
    :param features: 'predict' (class probabilities and labels), 'transformer' (transformer states)
    :return:
    '''
    data_dir = Path(BERT_FOLDER)
    model_folder = data_dir / bert_folder
    texts = texts[:5]
    if features == 'transformer':
        feats = features_finetuned_model(texts, labels=None, fine_tuned_model=model_folder,
                                 max_len=max_len)
        for f in feats:
            f = f[0]
            print(f.shape, f.dtype)
            print(f)
    if features == 'predict':
        results = predict_fn(model_folder, texts=texts, max_len=max_len)
        probs = [r['probs'] for r in results]
        labels = [r['label'] for r in results]
        print(probs)
        print(labels)

def bert_feature_test():
    from hackashop_datasets.cro_24sata import cro24_load_forclassif
    bert_folder = 'crosloengual-bert-42-toxicity-2e-5-256'
    texts, labels = cro24_load_forclassif('train')
    #bert_features(bert_folder, texts, features='transformer')
    bert_features(bert_folder, texts, features='predict')

if __name__ == '__main__':
    bert_feature_test()