import pandas as pd
from pathlib import Path
import csv

from project_settings import YNACC_DATASET


def ynacc_load(path, file="ydata-ynacc-v1_0_expert_annotations.tsv"):
    '''
    Read single annotated file as pandas dataframe.
    '''
    data_dir = Path(path)
    data_file = data_dir / file
    data = pd.read_csv(data_file, sep="\t", engine='python', quoting=csv.QUOTE_NONE)
    return data


ynacc_constructive_labels = {'Not constructive': 0, 'Constructive': 1}


# ynacc_toxic_labels = {True:1, False:0}
def load_ynacc_data(label_map=ynacc_constructive_labels, file="ydata-ynacc-v1_0_expert_annotations.tsv",
                    label='constructiveclass'):
    '''
    Load and prepare training data.
    :param label_map: defines the classfication problem: relevant string labels -> integer labels;
    returns: texts, labels
    '''
    data = ynacc_load(YNACC_DATASET, file)
    text_column = data['text']

    # create custom toxic label
    if label == 'toxic':

        insulting = data.sd_type.fillna('').apply(lambda x: 'insulting' in x)
        mean = data.tone.fillna('').apply(lambda x: 'mean' in x.lower())
        constructive = data.constructiveclass == 'Constructive'
        label_column = (insulting | mean) & (~constructive)

    else:
        label_column = data[label]
    # print all labels
    # print(set([label_column[i] for i in data.index]))
    # label map, and filter -> label not recognized, example discarded
    texts, labels = [], []
    for i in data.index:
        label = label_column[i]
        if label in label_map:  # ignore labels such as NaN
            text = text_column[i]
            texts.append(text)
            labels.append(label_map[label])
            # print(label, text)
    # print(len(texts))
    return texts, labels
