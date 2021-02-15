from hackashop_datasets import socc, ynacc, wiki, 


def load_toxic_en_data():
	'''
	Loads all english data sets with 
	defined toxic labels for the socc
	and ynacc dataset and defined toxic
	aggresive and attack label for wiki dataset 
	'''
	d1 = wiki.load_wiki_data()
	d2 = socc.load_socc_data()
	d3 = ynacc.load_ynacc_data()

	return tuple(x + y + z for x, y, z in zip(d1, d2, d3))