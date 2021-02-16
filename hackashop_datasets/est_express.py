import pandas as pd
from pathlib import Path
from math import isnan
from random import shuffle
import csv

from project_settings import EST_EXPRESS_DATASET


def estexpress_load_raw(path=EST_EXPRESS_DATASET, fname='comments_2019.csv'):
    '''
    Read and return data as pandas dataframe.
    :param path: folder of the dataset
    '''
    data_dir = Path(path)
    file = fname
    data_file = data_dir / file
    #data = pd.read_csv(data_file, sep=';')
    #data = pd.read_csv(data_file, sep='\t', quoting=csv.QUOTE_NONE) #, nrows=1100000)
    data = pd.read_csv(data_file, sep='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                        doublequote=False, parse_dates=['created_time'], nrows=1100000)
    #, lineterminator='\n') #, escapechar='\\', #quoting=csv.QUOTE_NONE,
    #parse_dates=['created_time'])#, nrows=2000000)
    return data

def print_dataset(dset):
    print('shape: ', dset.shape)
    print(set(r for r in dset['is_enabled']))
    print(sorted(set(r for r in dset['create_user_id'])))
    print(set(r for r in dset['channel_language']))
    dates = [d for d in dset['created_time']]; shuffle(dates)
    print(dates[:1000])
    texts = [t for t in dset['content']]
    shuffle(texts)
    for t in texts[:1000]: print('['+str(t)+']')


def label_distribution(dset):
    enabled = dset['is_enabled']
    userid = dset['create_user_id']
    N = float(len(dset)); nonblck = 0.0
    for i in dset.index:
        e, u = int(enabled[i]), userid[i]
        if e == 1: nonblck += 1 # not blocked
        #if u != '0': nonblck += 1
    blck = N - nonblck
    print(f"all: {int(N)}")
    print(f"nonblck: {nonblck/N*100:2.2f}%, {int(nonblck)}, blocked: {blck/N*100:2.2f}%, {blck}")

def clean_dataset(fname='comments_2019.csv'):
    ''' Filter out posts in russian, leaving only estonian. '''
    dset = estexpress_load_raw(fname=fname)
    print(dset.shape)
    indices = []; lang_labels = dset['channel_language']
    for i in dset.index:
        lang = lang_labels[i]
        if lang == 'nat': indices.append(i)
    dset = dset.iloc[indices]
    print_dataset(dset)
    fname = fname + f"_estonly.pickle"
    save_dir = Path(EST_EXPRESS_DATASET); save_file = save_dir / fname
    dset.to_pickle(save_file, compression=None)

def est_express_load(fname='comments_2019.csv', label='_estonly'):
    fname = fname + f"{label}.pickle"
    save_dir = Path(EST_EXPRESS_DATASET); save_file = save_dir / fname
    dset = pd.read_pickle(save_file, compression=None)
    return dset

def est_balance_dataset(train_sz, dev_sz, test_sz, fname='comments_2019.csv', label='_estonly'):
    from sklearn.model_selection import train_test_split
    dset = est_express_load()
    label_column = dset['is_enabled']
    indices = []; labels = []
    for i in dset.index:
        indices.append(i)
        l = label_column[i]
        if l == 1: labels.append(0);
        else: labels.append(1)
    train_ix, td_ix, train_lab, td_lab = train_test_split(indices, labels, train_size=train_sz,
                                           test_size=dev_sz+test_sz, stratify=labels)
    test_ix, dev_ix, test_lab, dev_lab = train_test_split(td_ix, td_lab, train_size=test_sz,
                                                              test_size=dev_sz, stratify=td_lab)
    parts = {
        'train' : dset.loc[train_ix],
        'test' : dset.loc[test_ix],
        'dev' : dset.loc[dev_ix]
    }
    for part in ['train', 'test', 'dev']:
        file_name = fname+f"{label}_{part}.pickle"
        save_dir = Path(EST_EXPRESS_DATASET); save_file = save_dir / file_name
        parts[part].to_pickle(save_file, compression=None)

def est_validate_split():
    train = est_express_load(label='_estonly_train')
    dev = est_express_load(label='_estonly_dev')
    test = est_express_load(label='_estonly_test')
    print('--------- train'); label_distribution(train)
    print('--------- dev'); label_distribution(dev)
    print('--------- test'); label_distribution(test)
    ixtr = set(train.index)
    ixts = set(test.index)
    ixdv = set(dev.index)
    assert(len(ixtr.intersection(ixts))==0)
    assert(len(ixtr.intersection(ixdv)) == 0)
    assert(len(ixts.intersection(ixdv)) == 0)

if __name__ == '__main__':
    #estexpress_load_raw()
    #print_dataset(estexpress_load_raw())
    #label_distribution(estexpress_load_raw(), 'is_enabled')
    #clean_dataset()
    #print_dataset(est_express_load())
    #label_distribution(estexpress_load_raw())
    label_distribution(est_express_load())
    #est_balance_dataset(40000, 10000, 10000)
    #est_validate_split()
