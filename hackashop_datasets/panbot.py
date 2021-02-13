import pandas as pd
from pathlib import Path
import csv
import xml.etree.ElementTree as ET
from project_settings import PANBOT_DATASET
from bs4 import BeautifulSoup
from utils import clean_helper

'''
The base code is from https://github.com/EMBEDDIA/PAN2019/blob/master/gender/src/parse_data.py
'''

LABEL_MAPPING = {'bot': 0, 'human': 1}


def load_panbot(part='all'):
    '''
    Read single annotated file as pandas dataframe.
    '''

    if 'train' == part:
        data_dir = Path(PANBOT_DATASET) / 'pan19-author-profiling-training-2019-02-18' / 'en'
        ground_truth = data_dir / 'truth.txt'
        texts, labels = process_xml_files(data_dir, ground_truth)
    elif 'test' == part:
        data_dir = Path(PANBOT_DATASET) / 'pan19-author-profiling-test-2019-04-29' / 'en'
        ground_truth = data_dir / 'en.txt'
        texts, labels = process_xml_files(data_dir, ground_truth)

    elif 'all' == part:
        train_data_dir = Path(PANBOT_DATASET) / 'pan19-author-profiling-training-2019-02-18' / 'en'
        train_ground_truth = train_data_dir / 'truth.txt'
        texts, labels = process_xml_files(train_data_dir, train_ground_truth)

        test_data_dir = Path(PANBOT_DATASET) / 'pan19-author-profiling-test-2019-04-29' / 'en'
        test_ground_truth = test_data_dir / 'en.txt'
        test_labels, test_texts = process_xml_files(test_data_dir, test_ground_truth)

        texts.extend(test_texts)
        labels.extend(test_labels)

    return texts, labels


def process_xml_files(data_dir, ground_truth):
    ground_truth_dict = {}
    for line in open(ground_truth):
        l = line.split(':::')
        ground_truth_dict[l[0]] = (l[1].strip(), l[2].strip())
    texts = []
    labels = []

    for file in data_dir.glob('*.xml'):
        name = file.stem
        try:
            tree = ET.parse(file)
        except:
            continue
        root = tree.getroot()

        if name in ground_truth_dict:
            type, _ = ground_truth_dict[name]

        concatenated_text = ""
        for document in root.iter('document'):
            if document.text:
                txt = beautify(document.text)
                tweet = txt.replace("\n", " ").replace("\t", " ")
                concatenated_text += tweet + "\n"

        # remove empty strings
        if concatenated_text:
            texts.append(clean_helper(concatenated_text.strip()))
            labels.append(LABEL_MAPPING[type])

    assert len(labels) == len(texts)
    return texts, labels


# remove html tags, used in PAN corpora
def beautify(text):
    return BeautifulSoup(text, 'html.parser').get_text()


if __name__ == '__main__':
    texts, labels = load_panbot(part='train')
