import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForPreTraining, \
    FeatureExtractionPipeline

class BertFeatureExtractor():
    '''
    Converts a (string) text to a numpy feature vector using FeatureExtractionPipeline.
    '''

    def __init__(self, bert=None, strategy="avg"):
        '''
        :param strategy: avg, first-layer, last-layer
        '''
        if bert is None: self._bert = load_feature_extractor()
        else: self._bert = bert
        self._strategy = strategy

    def __call__(self, txt):
        #print(txt)
        features = self._bert(txt)[0]
        if self._strategy == "avg": features = np.average(features, axis=0)
        elif self._strategy == "first-layer": features = features[0]
        elif self._strategy == "last-layer": features = features[-1]
        return features

    ######### scikit-learn interface

    def fit(self, data, y=None): return self

    def transform(self, data):
        td = [self(txt) for txt in data]
        return np.array(td)

    ######### scikit-learn interface


def load_feature_extractor():
    '''
    Load a pretrained BERT wrapped as a tranformers.FeatureExtractionPipeline.
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        'EMBEDDIA/crosloengual-bert',
        use_fast=True
    )
    def tokenizer_wrapper(*args, **kwargs):
        kwargs['truncation'] = True; kwargs['max_length'] = 512
        return tokenizer(*args, **kwargs)
    #print(tokenizer(txt))
    #model = AutoModelForSequenceClassification.from_pretrained('EMBEDDIA/crosloengual-bert', num_labels=3)
    #model = AutoModelForPreTraining.from_pretrained('EMBEDDIA/crosloengual-bert')
    model = AutoModel.from_pretrained('EMBEDDIA/crosloengual-bert')
    fextr = FeatureExtractionPipeline(model, tokenizer_wrapper)
    return fextr