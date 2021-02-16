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
            if not label in lmap: lmap[label] = 1.0
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

def cro24sata_load_byyear(year, fname='STY_24sata_comments_hr_001', label=''):
    fname = fname + f"_year{year}{label}.pickle"
    save_dir = Path(CRO_24SATA_DATASET); save_file = save_dir / fname
    dset = pd.read_pickle(save_file, compression=None)
    return dset

cro24_labelset_offensive = ([2.0, 3.0, 8.0], [1.0])

def cro24data_label_sample(label, sample_size=200):
    dset = cro24sata_load_byyear(2019)
    label_column = dset['infringed_on_rule']
    text_column = dset['content']
    texts = [text_column[i] for i in dset.index if label_column[i] == label]
    shuffle(texts)
    for t in texts[:sample_size]:
        print(t.replace('\n', ' '))

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

def cro24sata_filter_categories(year, categories, label):
    dset = cro24sata_load_byyear(2019)
    print(dset.shape)
    indices = []; labels = dset['infringed_on_rule']
    for i in dset.index:
        l = labels[i]
        if isnan(l) or l not in categories: indices.append(i)
    #print(indices)
    dset = dset.loc[indices]
    print_dataset(dset)
    fname = f"STY_24sata_comments_hr_001_year{year}{label}.pickle"
    save_dir = Path(CRO_24SATA_DATASET); save_file = save_dir / fname
    dset.to_pickle(save_file, compression=None)

def cro24sata_balance_dataset(year, label, train_sz, dev_sz, test_sz):
    from sklearn.model_selection import train_test_split
    dset = cro24sata_load_byyear(year, label=label)
    label_column = dset['infringed_on_rule']
    indices = []; labels = []
    for i in dset.index:
        indices.append(i)
        l = label_column[i]
        if isnan(l): labels.append(0);
        else: labels.append(int(l))
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
        fname = f"STY_24sata_comments_hr_001_year{year}{label}_{part}.pickle"
        save_dir = Path(CRO_24SATA_DATASET); save_file = save_dir / fname
        parts[part].to_pickle(save_file, compression=None)

def cro24_validate_split():
    train = cro24sata_load_byyear(2019, label='_nosmallcat_train')
    dev = cro24sata_load_byyear(2019, label='_nosmallcat_dev')
    test = cro24sata_load_byyear(2019, label='_nosmallcat_test')
    print('--------- train'); label_distribution(train)
    print('--------- dev'); label_distribution(dev)
    print('--------- test'); label_distribution(test)
    ixtr = set(train.index)
    ixts = set(test.index)
    ixdv = set(dev.index)
    assert(len(ixtr.intersection(ixts))==0)
    assert(len(ixtr.intersection(ixdv)) == 0)
    assert(len(ixts.intersection(ixdv)) == 0)

def cro24_texts_labels_from_dframe(dset):
    label_column = dset['infringed_on_rule']
    text_column = dset['content']
    texts, labels = [], []
    for i in dset.index:
        l = label_column[i]
        if isnan(l): label = 0
        else: label = 1;
        labels.append(label)
        texts.append(text_column[i])
    return texts, labels

def cro24_load_forclassif(part):
    '''
    Loading of final datasets used for classification experiments
    :param part - 'train', 'dev', or 'test'
    :returns texts, labels
    '''
    dset = cro24sata_load_byyear(2019, label=f'_nosmallcat_{part}')
    return cro24_texts_labels_from_dframe(dset)

def cro24_build_tfidf():
    from pickle import dump
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    dset = cro24sata_load_byyear(2019, label=f'_nosmallcat')
    texts, _ = cro24_texts_labels_from_dframe(dset)
    fextr = TfidfVectorizer(max_features=25000, sublinear_tf=True)
    #fextr.fit(texts)
    fextr_2g = TfidfVectorizer(max_features=25000, sublinear_tf=True, ngram_range=(2, 2))
    #fextr_2g.fit(texts)
    union = FeatureUnion([("words", fextr),
                          ("bigrams", fextr_2g)])
    union.fit(texts)
    save_dir = Path(CRO_24SATA_DATASET); savefile = save_dir / 'cro24_tfidf_2g.pickle'
    dump(union, open(savefile, 'wb'))

def cro24_load_tfidf():
    from pickle import load
    save_dir = Path(CRO_24SATA_DATASET); savefile = save_dir / 'cro24_tfidf_2g.pickle'
    return load(open(savefile, 'rb'))

if __name__ == '__main__':
    #cro24sata_explore()
    #cro24sata_filterbyyear(2019)
    #print_dataset(cro24sata_load_byyear(2019, label='_nosmallcat'))
    #label_distribution(cro24sata_load_byyear(2019))
    #cro24sata_unbalanced_offensive()
    #cro24data_label_sample(4.0)
    #cro24sata_filter_categories(2019, [2.0, 4.0, 7.0], '_nosmallcat')
    #cro24sata_balance_dataset(2019, '_nosmallcat', 40000, 10000, 10000)
    #cro24_validate_split()
    cro24_build_tfidf()
    #cro24_load_tfidf()

