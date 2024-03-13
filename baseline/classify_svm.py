#! /usr/bin/env python3

import pandas as pd
import csv
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


def get_data(task, mode, split):
	if split == "test":
		suffix = "txt" if task == "FR" else "tsv"
		df = pd.read_csv(f"../{task}/{task}_{split}.{suffix}", sep="\t", header=None, quoting=csv.QUOTE_NONE, names=["Text"])
	else:
		if task == "FR":
			labels_df = pd.read_csv(f"../{task}/{split}.labels", sep="\t", header=None, quoting=csv.QUOTE_NONE, names=["Label"])
			suffix = ".gz" if split == "train" else ""
			text_df = pd.read_csv(f"../{task}/{split}.txt{suffix}", sep="\t", header=None, quoting=csv.QUOTE_NONE, names=["Text"])
			df = pd.concat([labels_df, text_df], axis=1)
		else:
			df = pd.read_csv(f"../{task}/{task}_{split}.tsv", sep="\t", header=None, names=["Label", "Text"])
	print(f"{df.shape[0]} instances loaded from {task}/{split}")

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
	if len(sys.argv) > 3:
		split = sys.argv[3]
	else:
		split = "dev"
	assert(split in ["dev", "test"])
	print(task, mode, split)

	training_data = get_data(task, mode, 'train')
	test_data = get_data(task, mode, split)
	vec, svm = train(training_data)
	test_out = predict(test_data, vec, svm)
	save_results(test_out, f"{task}.{mode}.{split}.out")
