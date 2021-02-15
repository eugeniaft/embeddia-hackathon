import pandas as pd
from pathlib import Path
from math import isnan
from random import shuffle
import csv

from project_settings import EST_EXPRESS_DATASET


def estexpress_load_raw(path=EST_EXPRESS_DATASET):
    '''
    Read and return data as pandas dataframe.
    :param path: folder of the dataset
    '''
    data_dir = Path(path)
    file = 'comments_2019.csv'
    data_file = data_dir / file
    data = pd.read_csv(data_file, sep='\t', lineterminator='\n', escapechar='\\', #quoting=csv.QUOTE_NONE,
                       parse_dates=['created_time'])#, nrows=2000000)
    return data

def print_dataset(dset):
    print('shape: ', dset.shape)
    print(set(r for r in dset['is_enabled']))
    print(set(r for r in dset['channel_language']))
    dates = [d for d in dset['created_time']]; shuffle(dates)
    print(dates[:1000])
    texts = [t for t in dset['content']]
    shuffle(texts)
    for t in texts[:1000]: print('['+str(t)+']')

if __name__ == '__main__':
    #estexpress_load_raw()
    print_dataset(estexpress_load_raw())
