import pandas as pd
from pathlib import Path
import csv

from project_settings import HASOC2019_DATASET

def hasoc_load_raw(path=HASOC2019_DATASET, part='all'):
    '''
    Read and return data as pandas dataframe.
    :param path: folder of the dataset
    :param part: 'train', 'test', or 'all'
    :return:
    '''
    data_dir = Path(path)
    if part in ['train', 'test']:
        if part == 'train': file = 'english_dataset.tsv'
        elif part == 'test': file = 'hasoc2019_en_test-2919.tsv'
        data_file = data_dir / file
        data = pd.read_csv(data_file, sep="\t", engine='python', quoting=csv.QUOTE_NONE)
        return data
    else:
        train = hasoc_load_raw(path, "train")
        test = hasoc_load_raw(path, "test")
        return pd.concat([train, test], ignore_index=True)

hasoc_labels_hateoffensive = {'NOT':0, 'HOF':1}

def load_hasoc_data(label_map, text_column="text", label_column="task_1"):
    '''
    Load hasoc dataset ready for binary classification: hate/offensive or NO
    returns: texts, labels [0, 1]
    '''
    data = hasoc_load_raw(part="all")
    text_column = data[text_column]
    label_column = data[label_column]
    # print all labels
    #print(set([label_column[i] for i in data.index]))
    # label map, and filter -> label not recognized, example discarded
    texts, labels = [], []
    for i in data.index:
        label = label_column[i]
        if label in label_map: # ignore labels such as NaN
            text = text_column[i]
            texts.append(text)
            labels.append(label_map[label])
            #print(label, text)
    #print(len(texts))
    return texts, labels


def hasoc_explore_dataset():
    dset = hasoc_load_raw(part="all")
    print(dset[:5])
    print("All labels for each of the three tasks")
    print(set(dset["task_1"]))
    print(set(dset["task_2"]))
    print(set(dset["task_3"]))

if __name__ == '__main__':
    hasoc_explore_dataset()