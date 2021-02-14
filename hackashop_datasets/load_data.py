from hackashop_datasets import socc, ynacc, wiki, 
from hackashop_datasets import socc_toxic_labels, ynacc_toxic_labels, wiki_toxicity_labels


def load_toxic_en_data():
	'''
	Loads all english data sets with toxic labels
	'''
	d1 = load_wiki_data(wiki_toxicity_labels)
	d2 = load_socc_data(socc_toxic_labels)
	d3 = load_ynacc_data(ynacc_toxic_labels, label='toxic')

	return tuple(x + y + z for x, y, z in zip(d1, d2, d3))



