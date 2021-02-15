from functools import reduce
import pandas as pd
import glob
from pathlib import Path
import csv

from project_settings import WIKI_DATASET

def wiki_load(path, 
              labels = ('aggression', 'attack', 'toxicity'), 
              thresh=0.5):
    '''
    Reads all relevant wikipedia comment tsv files and
    returns 1 merged dataframe
    '''
    data_dir = Path(path)
    comments = sorted([c for c in data_dir.glob('*_comments.tsv')])
    annots = sorted([a for a in data_dir.glob('*_annotations.tsv')])
    data_sets = []
    for l, c, a in zip(labels, comments, annots):
        df_c = pd.read_csv(c, sep="\t", engine='python', quoting=csv.QUOTE_NONE)
        df_a = pd.read_csv(a, sep="\t", engine='python', quoting=csv.QUOTE_NONE)
        df = df_c.merge((df_a.groupby('rev_id')[l].mean() > thresh).reset_index())
        data_sets.append(df)
    data = reduce(lambda  left,right: pd.merge(left,right), data_sets)
    return data
    
wiki_toxicity_labels = {True:1, False:0}
wiki_aggression_labels = {True:1, False:0}
wiki_attack_labels = {True:1, False:0}
wiki_tox_agg_att_labels = {True:1, False:0}

def load_wiki_data(label_map=wiki_tox_agg_att_labels, label='all'):
    '''
    Load and prepare training data.
    :param label_map: defines the classfication problem: relevant string labels -> integer labels;
    returns: texts, labels
    '''
    data = wiki_load(WIKI_DATASET)
    text_column = data['comment']
    
    # create custom toxic/aggressive/attack label
    if label == 'all':
        label_column = data.toxicity | data.aggression | data.attack
    else: 
        label_column = data[label]
    texts, labels = [], []
    for i in data.index:
        label = label_column[i]
        if label in label_map: # ignore labels such as NaN
            text = text_column[i]
            texts.append(text)
            labels.append(label_map[label])
    return texts, labels

if __name__ == '__main__':
    texts, labels = load_wiki_data(wiki_toxicity_labels)