from hackashop_datasets import socc, ynacc, wiki, hasoc2019, troll
from sklearn.model_selection import train_test_split


def load_toxic_en_data():
	'''
	Loads all english data sets with 
	defined toxic labels for the socc
	ynacc dataset and defined toxic
	aggresive and attack label for wiki dataset 
	hasoc hate/ofensive label
	troll derogatory, hate speech, profanity

	'''
	
	d1 = wiki.load_wiki_data()
	d2 = socc.load_socc_data()
	d3 = ynacc.load_ynacc_data()
	d4 = hasoc2019.load_hasoc_data()
	d5 = troll.load_troll_data()

    data = tuple(x + y + z + g + f 
    	for x,y,z,g,f in zip(d1, d2, d3, d4, d5))

	return data


def train_dev_test(data, labels, random_seed):

	# split data into train and test
    train, test, train_labels, test_labels = train_test_split(data, 
    	labels, test_size=0.1, stratify=labels, random_state=random_seed)

    # split train into train and dev
    train, dev, train_labels, dev_labels = train_test_split(train, 
    	train_labels, test_size=0.1/0.9, stratify=train_labels, 
    	random_state=random_seed)

    data_split = {'train': (train, train_labels),
                  'dev': (dev, dev_labels),
                  'test': (test, test_labels)
                  }

    return data_split





