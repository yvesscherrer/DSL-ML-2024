#! /usr/bin/env python3

import pandas as pd
import sys
import gzip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


def get_data(task, mode, split):
	if task == "FR":
		with open(f"../{task}/{split}.labels") as labelfile:
			labels = [line.rstrip() for line in labelfile]
		if split == "train":
			with gzip.open(f"../{task}/{split}.txt.gz", mode='rt') as textfile:
				texts = [line.rstrip() for line in textfile]
		else:
			with open(f"../{task}/{split}.txt") as textfile:
				texts = [line.rstrip() for line in textfile]
		df = pd.DataFrame({'Label': labels, 'Text': texts})
	else:
		df = pd.read_csv(f"../{task}/{task}_{split}.tsv", sep="\t", header=None, names=["Label", "Text"])
	print(f"{df.shape[0]} instances loaded from {task}/{split}")
	# print(task, mode, split)
	# print(df.isna().any())
	
	if mode == "expand" and split == "train":
		df['ExpandLabel'] = df['Label'].str.split(',')
		df = df.explode('ExpandLabel')
		df = df.drop('Label', axis=1)
		df = df.rename({'ExpandLabel': 'Label'}, axis=1)
		print(f"{df.shape[0]} instances after expansion")
	return df


def train(train_data, word_only=False):
	print("Start training")
	if word_only:
		vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=10)#, decode_error='replace', encoding='utf-8')
	else:
		char_vec = TfidfVectorizer(analyzer='char', ngram_range=(1,4), min_df=10)#, decode_error='replace', encoding='utf-8')
		word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=10)#, decode_error='replace', encoding='utf-8')
		vectorizer = FeatureUnion([("char", char_vec), ("word", word_vec)])
	cl = LinearSVC(max_iter=100, verbose=True, dual='auto')

	train_ngrams = vectorizer.fit_transform(train_data.Text)
	cl.fit(train_ngrams, train_data.Label)
	print()
	print("Training finished")
	return vectorizer, cl


def predict(predict_data, vectorizer, cl):
	print("Start predicting")
	dev_x_grams = vectorizer.transform(predict_data.Text)
	dev_pred = cl.predict(dev_x_grams)
	print("Predicting finished")
	return dev_pred

def save_results(prediction, filename):
	f = open(filename, 'w')
	for p in prediction:
		f.write(p + "\n")
	f.close()

if __name__ == "__main__":
	task = sys.argv[1]
	assert(task in ["BCMS", "EN", "ES", "FR", "PT"])
	mode = sys.argv[2]
	assert(mode in ["atomic", "expand"])
	print(task, mode)
	
	training_data = get_data(task, mode, 'train')
	dev_data = get_data(task, mode, 'dev')
	vec, svm = train(training_data)
	dev_out = predict(dev_data, vec, svm)
	save_results(dev_out, f"{task}.{mode}.dev.out")
