import pandas as pd
import glob
from pathlib import Path


def nyt_load(path):
    '''
    From a path, reads all csv files 
    with comments as pandas dataframe.
    '''
    data_dir = Path(path)
    appended_ds = []
    for f in data_dir.glob('Comments*.csv'):
        d = pd.read_csv(f)
        appended_ds.append(d)
    data = pd.concat(appended_ds)
    return data