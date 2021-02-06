import pandas as pd
from pathlib import Path
import csv


def ynacc(path, mode="train"):
    data_dir = Path(path)
    ids_file = data_dir / f"ydata-ynacc-v1_0_{mode}-ids.txt"
    data_file = data_dir / "ydata-ynacc-v1_0_unlabeled_conversations.tsv"

    ids = []
    f = open(ids_file, "r")
    _ids = f.readlines()
    for _id in _ids:
        ids.append(_id)

    data = pd.read_csv(data_file, sep="\t", engine='python', quoting=csv.QUOTE_NONE)
    data = data[data.sdid.isin(ids)]
    return data


if __name__ == '__main__':
    ynacc("data/ydata-ynacc-v1_0", mode="train")
