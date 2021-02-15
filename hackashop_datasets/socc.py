import pandas as pd
from pathlib import Path

from project_settings_template import SOCC_DATASET

def socc_load(path, file='SFU_constructiveness_toxicity_corpus.csv'):
    '''
    Read single annotated file as pandas dataframe.
    '''
    data_dir = Path(path)
    data_file = data_dir / file
    data = pd.read_csv(data_file)
    return data

socc_toxic_labels = {True:1, False:0}

def load_socc_data(label_map=socc_toxic_labels, 
                   file='SFU_constructiveness_toxicity_corpus.csv',
                   label='toxic'):
    '''
    Load and prepare training data.
    :param label_map: defines the classfication problem: relevant string labels -> integer labels;
    returns: texts, labels
    '''
    data = socc_load(SOCC_DATASET, file)
    text_column = data['comment_text']
    
    # create custom toxic label
    # Very toxic (4) and Toxic (3)
    # Each comment was annotated by at least three annotators 
    # and the most popular answer is the first one
    if label == 'toxic':
        toxic_level = data.toxicity_level.apply(lambda x: x.startswith('3') or x.startswith('4'))
        not_constructive = data.is_constructive != 'yes'
        label_column = toxic_level & not_constructive

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