import pandas as pd
from pathlib import Path
from math import isnan
from random import shuffle

from project_settings import CRO_24SATA_DATASET


def cro24sata_load_raw(path=CRO_24SATA_DATASET):
    '''
    Read and return data as pandas dataframe.
    :param path: folder of the dataset
    '''
    data_dir = Path(path)
    file = 'STY_24sata_comments_hr_001.csv'
    data_file = data_dir / file
    data = pd.read_csv(data_file, lineterminator='\n', parse_dates=['created_date', 'last_change'])
    #data = pd.read_csv(data_file, nrows=1000000, lineterminator='\n', parse_dates=['created_date', 'last_change'])
    return data

def print_dataset(dset):
    print('shape: ', dset.shape)
    print(set(r for r in dset['infringed_on_rule'] if not isnan(r)))
    print(set(d.year for d in dset['created_date']))
    print(set(s for s in dset['site']))
    texts = [t for t in dset['content']]
    shuffle(texts)
    for t in texts[:1000]: print('['+t+']')

def label_distribution(dset):
    N = float(len(dset))
    lmap = {}; nonblck = 0.0
    for label in dset['infringed_on_rule']:
        if isnan(label): nonblck += 1 # not blocked
        else:
            if not label in lmap: lmap[label] = 0.0
            else: lmap[label] += 1
    blck = N - nonblck
    print(f"all: {int(N)}")
    print(f"nonblck: {nonblck/N*100:2.2f}%, {int(nonblck)}, blocked: {blck/N*100:2.2f}%, {blck}")
    print("label dist: " + "; ".join(f"{l}: {cnt/N*100:2.2f}%, {int(cnt)}" for l, cnt in sorted(lmap.items())))

def cro24sata_explore():
    dset = cro24sata_load_raw()
    print_dataset(dset)

def cro24sata_filterbyyear(year, fname='STY_24sata_comments_hr_001'):
    dset = cro24sata_load_raw()
    print(dset.shape)
    indices = []; dates = dset['created_date']
    for i in dset.index:
        date = dates[i]
        if date.year == year: indices.append(i)
    dset = dset.iloc[indices]
    print_dataset(dset)
    fname = fname + f"_year{year}.pickle"
    save_dir = Path(CRO_24SATA_DATASET); save_file = save_dir / fname
    dset.to_pickle(save_file, compression=None)

def cro24sata_load_byyear(year, fname='STY_24sata_comments_hr_001'):
    fname = fname + f"_year{year}.pickle"
    save_dir = Path(CRO_24SATA_DATASET); save_file = save_dir / fname
    dset = pd.read_pickle(save_file, compression=None)
    return dset

cro24_labelset_offensive = ([2.0, 3.0, 8.0], [1.0])

def cro24sata_unbalanced_offensive():
    dset = cro24sata_load_byyear(2019)
    label_column = dset['infringed_on_rule']
    text_column = dset['content']
    filter, remove = cro24_labelset_offensive
    texts, labels = [], []
    for i in dset.index:
        label = label_column[i]
        text = text_column[i]
        if isnan(label): addlabel = 0; # not blocked
        else:
            if label in remove: continue # remove (ex. introduces noise)
            if label in filter: addlabel = 1; # target labels
            else: addlabel = 0 # other blocked labels
        texts.append(text); labels.append(addlabel)
    return texts, labels

if __name__ == '__main__':
    #cro24sata_explore()
    #cro24sata_filterbyyear(2019)
    #print_dataset(cro24sata_load_byyear(2019))
    #label_distribution(cro24sata_load_byyear(2019))
    cro24sata_unbalanced_offensive()
