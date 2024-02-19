#! /usr/bin/env python3

import sys
import pandas as pd
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import *
import matplotlib.pyplot as plt


def load_answers(goldfile, systemfile):
	with open(goldfile) as g:
		gold_df = pd.read_csv(goldfile, sep="\t", header=None)
		if gold_df.shape[1] == 2:
			gold_df.columns = ["Label", "Text"]
		else:
			gold_df.columns = ["Label"]
	with open(systemfile) as s:
		sys_list = [line.rstrip() for line in s]
		sys_df = pd.DataFrame({"Label": sys_list})
	
	if gold_df.shape[0] != sys_df.shape[0]:
		print("Number of lines does not match!")
		print("Gold length:  ", gold_df.shape[0])
		print("System length:", sys_df.shape[0])
		return [], [], []

	cv = CountVectorizer(binary=True, lowercase=False, tokenizer=lambda x: x.split(","), token_pattern=None)
	gold_array = cv.fit_transform(gold_df.Label).toarray()
	label_list = cv.get_feature_names_out()
	sys_array = cv.transform(sys_df.Label).toarray()
	return label_list, gold_array, sys_array


def only_ambiguous_answers(gold_array, sys_array):
	amb_lines = numpy.where(gold_array.sum(axis=1) > 1)
	amb_gold_array = gold_array[amb_lines[0],:]
	amb_sys_array = sys_array[amb_lines[0],:]
	return amb_gold_array, amb_sys_array


if __name__ == "__main__":
	systemfile = sys.argv[1]
	goldfile = sys.argv[2]
	labels, gold_answers, system_answers = load_answers(goldfile, systemfile)
	print("Evaluated file:", systemfile)
	print()
	
	# mcm = multilabel_confusion_matrix(gold_answers, system_answers)
	# print(mcm)

	print("Scores for entire dataset ({} instances):".format(gold_answers.shape[0]))
	class_f1 = f1_score(gold_answers, system_answers, average=None, zero_division="warn")
	macro_f1 = f1_score(gold_answers, system_answers, average='macro', zero_division="warn")
	weighted_f1 = f1_score(gold_answers, system_answers, average='weighted', zero_division="warn")
	for label, score in zip(labels, class_f1):
		print("F1-score for class {}: {:.2f}%".format(label, 100*score))
	print("Macro-avg F1 score:    {:.2f}%".format(100*macro_f1))
	print("Weighted-avg F1 score: {:.2f}%".format(100*weighted_f1))
	print()

	amb_gold_answers, amb_system_answers = only_ambiguous_answers(gold_answers, system_answers)
	print("Scores for ambiguous subset ({} instances):".format(amb_gold_answers.shape[0]))
	class_f1 = f1_score(amb_gold_answers, amb_system_answers, average=None, zero_division="warn")
	macro_f1 = f1_score(amb_gold_answers, amb_system_answers, average='macro', zero_division="warn")
	weighted_f1 = f1_score(amb_gold_answers, amb_system_answers, average='weighted', zero_division="warn")
	for label, score in zip(labels, class_f1):
		print("F1-score for class {}: {:.2f}%".format(label, 100*score))
	print("Macro-avg F1 score:    {:.2f}%".format(100*macro_f1))
	print("Weighted-avg F1 score: {:.2f}%".format(100*weighted_f1))
	print()