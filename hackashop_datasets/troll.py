import pandas as pd
from pathlib import Path

from project_settings_template import TROLL_DATASET


def troll_load(path, file):
    '''
    Reads single annotated excel file and returns dataframe
    '''
    data_dir = Path(path)
    data_file = data_dir / file
    data = pd.read_excel(data_file)
    return data
    
troll_toxicity_labels = {True:1, False:0}

def load_troll_data(label_map=troll_toxicity_labels, 
                    file='Dataset_to_upload.xlsx'):
    '''
    Load and prepare training data.
    returns: texts, labels
    '''
    data = troll_load(TROLL_DATASET, file)
    # remove comments with: [removed] and 0
    data = data.loc[~(data['Sample'].isin(['[removed]', 0])), :]
    # create toxic label
    data['toxic'] = data.Majority_label.apply(lambda x: x != 'Normal')
    # drop duplicated comments that might have different labels
    data = data.drop_duplicates(subset=['Sample']).reset_index(drop=True)
    label_column = data['toxic']
    text_column = data['Sample']

    texts, labels = [], []
    for i in data.index:
        label = label_column[i]
        if label in label_map: # ignore labels such as NaN
            text = text_column[i]
            texts.append(text)
            labels.append(label_map[label])
    return texts, labels