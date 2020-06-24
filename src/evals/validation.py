#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import statistics
import tensorflow as tf
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from utils.data_helper import load_test

flags = tf.app.flags
FLAGS = flags.FLAGS

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def get_normalized_inner_product_score(vector1, vector2):
	"""
	adapted from https://github.com/THUDM/GATNE
	Calculate normalized inner product for vector1 and vector2.
	"""
	return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def get_sigmoid_score(vector1, vector2):
	"""
	Calculate sigmoid score of the dot product for vector1 and vector2.
	"""
	return sigmoid(np.dot(vector1, vector2))

def get_average_score(vector1, vector2):
	"""
	Calculate average (element-wise average) for vector1 and vector2.
	Note the result is not a scalar, it has dimension same as vector1 or vector2.
	"""
	return (vector1 + vector2)/2

def get_hadamard_score(vector1, vector2):
	"""
	Calculate Hadamard product score (element-wise multiplication) for vector1 and vector2.
	Note the result is not a scalar, it has dimension same as vector1 or vector2.
	"""
	return np.multiply(vector1, vector2)

def get_l1_score(vector1, vector2):
	"""
	Calculate Weighted-L1 for vector1 and vector2.
	Note the result is not a scalar, it has dimension same as vector1 or vector2.
	"""
	return np.abs(vector1 - vector2)

def get_l2_score(vector1, vector2):
	"""
	Calculate Weighted-L2 for vector1 and vector2.
	Note the result is not a scalar, it has dimension same as vector1 or vector2.
	"""
	return np.square(vector1 - vector2)

def get_link_score(embeds, node1, node2, score_type):
	"""
	Calculate link_score for node1 and node2 according to score_type.
	"""
	if score_type not in ["cosine", "sigmoid", "hadamard", "average", "l1", "l2"]:
		raise NotImplementedError

	vector_dimension = len(embeds[random.choice(list(embeds.keys()))])
	try:
		vector1 = embeds[node1]
		vector2 = embeds[node2]
	except Exception as e:
		# print(e)
		if score_type in ["cosine", "sigmoid"]:
			return 0
		elif score_type in ["hadamard", "average", "l1", "l2"]:
			return np.zeros(vector_dimension)

	if score_type == "cosine":
		score = get_normalized_inner_product_score(vector1, vector2)
	elif score_type == "sigmoid":
		score = get_sigmoid_score(vector1, vector2)
	elif score_type == "hadamard":
		score = get_hadamard_score(vector1, vector2)
	elif score_type == "average":
		score = get_average_score(vector1, vector2)
	elif score_type == "l1":
		score = get_l1_score(vector1, vector2)
	elif score_type == "l2":
		score = get_l2_score(vector1, vector2)

	return score

def get_links_scores(embeds, links, score_type):
	"""
	Calculate link score for a list of links (node pairs).
	"""
	features = []
	num_links = 0
	for l in links:
		num_links = num_links + 1
		# if num_links % 1000 == 0:
		#	 print("get_links_score, num of edges processed: {}".format(num_links))
		node1, node2 = l[0], l[1]
		f = get_link_score(embeds, node1, node2, score_type)
		features.append(f)
	return features


def evaluate_classifier(embeds, train_pos_edges, train_neg_edges, test_pos_edges, test_neg_edges, score_type):
	"""
	Use Logistic Regression to do link prediction.
	"""
	train_pos_feats = np.array(get_links_scores(embeds, train_pos_edges, score_type))
	train_neg_feats = np.array(get_links_scores(embeds, train_neg_edges, score_type))

	train_pos_labels = np.ones(train_pos_feats.shape[0])
	train_neg_labels = np.zeros(train_neg_feats.shape[0])

	# train_data = np.vstack((train_pos_feats, train_neg_feats))
	train_data = np.concatenate((train_pos_feats, train_neg_feats), axis=0)
	train_labels = np.append(train_pos_labels, train_neg_labels)

	test_pos_feats = np.array(get_links_scores(embeds, test_pos_edges, score_type))
	test_neg_feats = np.array(get_links_scores(embeds, test_neg_edges, score_type))

	test_pos_labels = np.ones(test_pos_feats.shape[0])
	test_neg_labels = np.zeros(test_neg_feats.shape[0])

	# test_data = np.vstack((test_pos_feats, test_neg_feats))
	test_data = np.concatenate((test_pos_feats, test_neg_feats), axis=0)
	test_labels = np.append(test_pos_labels, test_neg_labels)

	train_data_indices_not_zero = train_data != 0
	# train_data_indices_not_zero = np.prod(train_data, axis=1) == 0
	train_data = train_data[train_data_indices_not_zero]
	train_labels = train_labels[train_data_indices_not_zero]

	test_data_indices_not_zero = test_data != 0
	# test_data_indices_not_zero = np.prod(test_data, axis=1) == 0
	test_data = test_data[test_data_indices_not_zero]
	test_labels = test_labels[test_data_indices_not_zero]

	# # training: eliminate the edges formed by new nodes. Their edge feats are 0s.
	# train_data_filtered = train_data[(np.product(train_data, axis=1) + np.sum(train_data, axis=1)) != 0, :]
	# train_labels_filtered = train_labels[(np.product(train_data, axis=1) + np.sum(train_data, axis=1)) != 0]
	train_data_filtered = train_data
	train_labels_filtered = train_labels

	if len(train_data_filtered.shape) == 1:
		train_data_filtered = np.expand_dims(train_data_filtered, axis=1)

	if len(test_data.shape) == 1:
		test_data = np.expand_dims(test_data, axis=1)

	logistic_regression = linear_model.LogisticRegression()
	logistic_regression.fit(train_data_filtered, train_labels_filtered)

	# test: here don't eliminate the edges formed by new nodes.
	test_predict_prob = logistic_regression.predict_proba(test_data)
	test_predict = logistic_regression.predict(test_data)

	# calculate roc auc score
	auroc = roc_auc_score(test_labels, test_predict_prob[:, 1])

	# calculate precision_recall auc score
	precisions, recalls, _ = precision_recall_curve(test_labels, test_predict_prob[:, 1])
	auprc = auc(recalls, precisions)

	return auroc, auprc


def cross_validation(embed, score_type, n_trials=5):
	edges = load_test(FLAGS.train_prefix) #(6170,4)
	# print(len(train_test_edges))
	neg_edges = edges[edges[:,2] == 0,:]
	pos_edges = edges[edges[:,2] == 1,:]
	# print(len(neg_edges),len(pos_edges)) #(3085,4)

	# shuffle and split training and test sets
	trials = ShuffleSplit(n_splits=n_trials, test_size=0.2, random_state=None)
	ss = trials.split(pos_edges)
	trial_splits = []
	for train_idx, test_idx in ss:
		trial_splits.append((train_idx, test_idx))
	# print(len(trial_splits),len(trial_splits[0]))

	list_auroc = []
	list_auprc = []
	for idx in range(n_trials):
		test_idx,train_idx = trial_splits[idx]
		# print(len(test_idx),len(train_idx))
		train_neg = neg_edges[train_idx,:]
		train_pos = pos_edges[train_idx,:]
		test_neg = neg_edges[test_idx,:]
		test_pos = pos_edges[test_idx,:]

		auroc, auprc = evaluate_classifier(embed,train_pos,train_neg,test_pos,test_neg,score_type)
		list_auroc.append(auroc)
		list_auprc.append(auprc)
	# print(list_auroc,list_auprc)
	avg_auroc = statistics.mean(list_auroc)
	std_auroc = statistics.stdev(list_auroc)
	avg_auprc = statistics.mean(list_auprc)
	std_auprc = statistics.stdev(list_auprc)

	return avg_auroc,std_auroc,avg_auprc,std_auprc

