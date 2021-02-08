from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForPreTraining, \
    AutoModelForSequenceClassification, FeatureExtractionPipeline

import numpy as np
from scipy.spatial.distance import cosine

def load_feature_extractor():
    '''
    Load a pretrained BERT wrapped as a tranformers.FeatureExtractionPipeline.
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        'EMBEDDIA/crosloengual-bert',
        use_fast=True
    )
    #print(tokenizer(txt))
    #model = AutoModelForSequenceClassification.from_pretrained('EMBEDDIA/crosloengual-bert', num_labels=3)
    model = AutoModelForPreTraining.from_pretrained('EMBEDDIA/crosloengual-bert')
    fextr = FeatureExtractionPipeline(model, tokenizer)
    return fextr

class BertFeatureExtractor():
    '''
    Converts a (string) text to a numpy feature vector using FeatureExtractionPipeline.
    '''

    def __init__(self, bert):
        self._bert = bert

    def __call__(self, txt):
        features = self._bert(txt)[0]
        features = np.average(features, axis=0)
        return features

def extract_features_try0(txt="ovo je testni tekst"):
    fextr = load_feature_extractor()
    features = np.array(fextr(txt))[0]
    features = np.average(features, axis=0)
    features = features.flatten()
    print(features.__class__)
    print(features.shape)
    print(features[0], features[1])
    return features

def compare_texts():
    txt1 = "two fast cars raced down the road. drivers were good. tires screeched."
    txt2 = "road or highway determines the maximum speed for the vehicle."
    txt3 = "on a meadow, flowers and trees grow and bees buzz."
    txt4 = "in a forrest by a creek, grass grows and birds and insects fly."
    fextr = BertFeatureExtractor(load_feature_extractor())
    f1 = fextr(txt1); f2 = fextr(txt2)
    f3 = fextr(txt3); f4 = fextr(txt4)
    print(cosine(f1, f2))
    print(cosine(f3, f4))
    print(cosine(f1, f3))
    print(cosine(f2, f4))

if __name__ == '__main__':
    #extract_features_try0()
    compare_texts()