import numpy as np
from scipy.spatial.distance import cosine

from classification_experimental.BertFeatureExtractor import BertFeatureExtractor, load_feature_extractor

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
    fextr = BertFeatureExtractor(strategy="first-layer")
    f1 = fextr(txt1); f2 = fextr(txt2)
    f3 = fextr(txt3); f4 = fextr(txt4)
    print(cosine(f1, f2))
    print(cosine(f3, f4))
    print(cosine(f1, f3))
    print(cosine(f2, f4))

def batch_process_test():
    txt1 = "two fast cars raced down the road. drivers were good. tires screeched."
    txt2 = "road or highway determines the maximum speed for the vehicle."
    fextr = BertFeatureExtractor(load_feature_extractor())
    res = fextr([txt1, txt2])

if __name__ == '__main__':
    #extract_features_try0()
    compare_texts()
    #batch_process_test()