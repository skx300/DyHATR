#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import collections
import numpy as np
import networkx as nx
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def run_random_walks(G, nodes, N_WALKS=50, WALK_LEN=5):
	pairs = []
	for count, node in enumerate(nodes):
		if G.degree(node) == 0:
			continue
		for i in range(N_WALKS):
			curr_node = node
			for j in range(WALK_LEN):
				next_node = random.choice(G.neighbors(curr_node))
				# self co-occurrences are useless
				if curr_node != node:
					pairs.append((node,curr_node))
				curr_node = next_node
		if count % 1000 == 0:
			print("Done walks for", count, "nodes")
	return pairs


def load_graph(fileppath):

	data = np.genfromtxt(fileppath, dtype=int, delimiter='\t', encoding=None)
	time_stamp = [datetime.strptime(str(ele), '%Y%m%d') for ele in data[:,-1]]
	START_DATE = min(time_stamp)
	DAY_DURATION = 1

	graphs = dict()
	slice_id = 0
	for i in range(len(data)):
		edge = data[i, :]
		current_date = datetime.strptime(str(edge[-1]), '%Y%m%d')
		slice_id = (current_date - START_DATE).days // DAY_DURATION

		start_node,end_node,edge_type = edge[0],edge[1],edge[2]

		if not graphs.get(slice_id):
			graphs[slice_id] = dict()

		if not graphs.get(slice_id).get(edge_type):
			graphs[slice_id][edge_type] = nx.Graph()
		# undirected graph
		if graphs[slice_id][edge_type].has_edge(start_node, end_node):
			graphs[slice_id][edge_type][start_node][end_node]['weight'] += 1
		else:
			graphs[slice_id][edge_type].add_edge(start_node, end_node, weight=1, type=edge_type, date=current_date)

	graphs = dict(sorted(graphs.items()))
	list_graphs = []
	for k, graph_views in graphs.items():
		list_graphs.append(list(collections.OrderedDict(sorted(graph_views.items())).values()))

	return list_graphs

def load_feat(fileppath):
	feats = {}


	return feats

def read_dynamic_graph(graph_snapshots_file, val_edge_list_file, feat_file, walk_pairs_file, normalize=True, load_walks=False):

	# graphs_with_views = np.load(graph_snapshots_file, allow_pickle=True, encoding = 'latin1')['graph']
	graphs_with_views = load_graph(graph_snapshots_file)
	print("### Loaded {} graphs...".format(len(graphs_with_views)))
	print("### Each graph has {} views...".format(len(graphs_with_views[0])))
	# print(len(graphs_with_views[0][0]))

	# create the final graph snapshot for random walk sampling.
	final_graph_snapshot = nx.Graph()
	for graph_view in graphs_with_views[-1]:
		final_graph_snapshot.add_edges_from(graph_view.edges())

	for graph_views in graphs_with_views:
		for graph_view in graph_views:
			nx.set_node_attributes(graph_view, 'val', False)
			nx.set_node_attributes(graph_view, 'test', False)

			for edge in graph_view.edges():
				if (graph_view.node[edge[0]]['val'] or graph_view.node[edge[1]]['val'] or
						graph_view.node[edge[0]]['test'] or graph_view.node[edge[1]]['test']):
					graph_view[edge[0]][edge[1]]['train_removed'] = True
				else:
					graph_view[edge[0]][edge[1]]['train_removed'] = False

	if isinstance(next(iter(graphs_with_views[0])).nodes()[0], int) or \
			np.issubdtype(next(iter(graphs_with_views[0])).nodes()[0], np.dtype(int).type):
		conversion = lambda n: int(n)
	else:
		conversion = lambda n: n

	# load feats
	if os.path.exists(feat_file):
		feats = np.load(feat_file, allow_pickle=True)
	else:
		print("### No features present... (Only identity features will be used)")
		feats = None

	# Create id_map
	nodes_list = []
	for graph_views in graphs_with_views:
		for graph_view in graph_views:
			for node in graph_view.nodes():
				nodes_list.append(node)
	nodes_list = np.sort(list(set(nodes_list)))
	id_map = {conversion(str(nodes_list[i])): int(i) for i in range(len(nodes_list))} #35981
	
	if normalize and not feats is None:
		# train_ids = np.array([id_map[n] for n in g.nodes() if not g.node[n]['val'] and not g.node[n]['test'] for g in graphs_with_views])
		temp_ids = []
		for graph_views in graphs_with_views:
			for graph_view in graph_views:
				for n in graph_view.nodes():
					if not graph_view.node[n]['val'] and not graph_view.node[n]['test']:
						temp_ids.append(id_map[n])
		train_ids = temp_ids
		# print(train_ids.shape)
		# train_feats = feats[train_ids]
		train_feats = []
		nodes = feats["node"]
		for ids in train_ids:
			idx = nodes.index(ids)
			train_feats.append(feats["faets"][idx])
		# train_feats = [feats[i] for i in train_ids]
		scaler = StandardScaler()
		scaler.fit(train_feats)
		feats = scaler.transform(feats)

	# load random walk node pairs.
	walk_pairs = []
	if os.path.isfile(walk_pairs_file):
		print("### Find walk_pair file, read walk_pair from file...")
		with open(walk_pairs_file, 'r') as f:
			for line in f:
				# walk_pairs.append(map(conversion, line.split()))
				walk_pairs.append([conversion(ele) for ele in line.split()])
		print("### Done read walk_pair from file.")
	else:
		print("### Can't find walk_pair file, run run_random_walks...")
		# only do random walk for the last (latest in time) snapshot graph.
		walks_pairs = run_random_walks(final_graph_snapshot, final_graph_snapshot.nodes())
		with open(walk_pairs_file, 'w') as f:
			f.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in walks_pairs]))
		print("### Done run_random_Walks and saved them into file...")

	# load validation edges
	val_edges_data = np.genfromtxt(val_edge_list_file, dtype=int, delimiter='\t', encoding=None)
	val_edges = []
	for edge in val_edges_data:
		# we only need positive edges of val set
		if edge[2] == 1 and edge[3] == 0:
			if edge[0] not in id_map or edge[1] not in id_map:
				continue
			val_edges.append((edge[0], edge[1]))
	# print("### num of val edges: {}".format(len(val_edges)))

	return graphs_with_views, feats, id_map, walk_pairs, val_edges


def load_data(train_prefix):
	if train_prefix == "EComm":
		edge_train_file = "../dataset/ecomm/ecomm_edge_train.txt"
		edges_val_lr_train_test_file = "../dataset/ecomm/ecomm_edge_val_lr_train_test.txt"
		walk_pairs_file = "../dataset/ecomm/ecomm_graphsage_walk_pairs_dynamic.txt"
		# feat_file = "./feats.npz"
		# feat_file = ""

	train_data = read_dynamic_graph(graph_snapshots_file=edge_train_file,
		val_edge_list_file=edges_val_lr_train_test_file, feat_file="", 
		walk_pairs_file=walk_pairs_file, load_walks=True)

	return train_data


def load_train_test(train_prefix):
	if train_prefix == "EComm":
		edges_val_lr_train_test_file = "../dataset/ecomm/ecomm_edge_val_lr_train_test.txt"

	edges = np.genfromtxt(edges_val_lr_train_test_file, dtype=int)

	train_edges = edges[edges[:,3] == 1,:]
	test_edges = edges[edges[:,3] == 2,:]

	train_pos_edges = train_edges[train_edges[:, 2] == 1, :]
	train_neg_edges = train_edges[train_edges[:, 2] == 0, :]
	test_pos_edges = test_edges[test_edges[:, 2] == 1, :]
	test_neg_edges = test_edges[test_edges[:, 2] == 0, :]

	return train_pos_edges, train_neg_edges, test_pos_edges, test_neg_edges


def load_test(train_prefix):
	if train_prefix == "EComm":
		edges_val_lr_train_test_file = "../dataset/ecomm/ecomm_edge_val_lr_train_test.txt"

	edges = np.genfromtxt(edges_val_lr_train_test_file, dtype=int)

	train_test_edges = np.concatenate((edges[edges[:,3] == 1,:],edges[edges[:,3] == 2,:]), axis=0)
	# print(train_test_edges.shape) #(6170,4)
	return train_test_edges


if __name__ == '__main__':
	# graphs = load_graph("../processed/ecomm/ecomm_edge_train.txt")
	# print(len(graphs))
	train_data = load_train("EComm")
	print("### Done loading training data...")