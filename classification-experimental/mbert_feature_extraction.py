from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForPreTraining, \
    AutoModelForSequenceClassification, FeatureExtractionPipeline

import numpy as np
from scipy.spatial.distance import cosine

def load_feature_extractor():
    tokenizer = AutoTokenizer.from_pretrained(
        'EMBEDDIA/crosloengual-bert',
        use_fast=True
    )
    #print(tokenizer(txt))
    #model = AutoModelForSequenceClassification.from_pretrained('EMBEDDIA/crosloengual-bert', num_labels=3)
    model = AutoModelForPreTraining.from_pretrained('EMBEDDIA/crosloengual-bert')
    fextr = FeatureExtractionPipeline(model, tokenizer)
    return fextr


def extract_features_try0(txt="ovo je testni tekst"):
    fextr = load_feature_extractor()
    features = np.array(fextr(txt)).flatten()
    print(features)

def compare_texts():
    txt1 = "auto cesta semafor"
    txt2 = "motor kotač brzina"
    txt3 = "cvijet drvo trava"
    fextr = load_feature_extractor()
    f1 = np.array(fextr(txt1)).flatten()[:200000]
    f2 = np.array(fextr(txt3)).flatten()[:200000]
    print(cosine(f1, f2))

if __name__ == '__main__':
    #extract_features()
    compare_texts()