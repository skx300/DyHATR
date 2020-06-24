#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx


class EdgeMinibatchIterator(object):
    """This is minibatch iterator iterates over batchs of sampled edges or
    random pairs of co-occurring edges for graph snapshots. Adapted from minibatch.EdgeMinibatchIterator.
    graphs -- a list of networkx graphs, each element representing a graph snapshot.
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, graphs_with_views, id2idx,
                 placeholders, val_edges, context_pairs=None, batch_size=100, max_degree=25,
                 n2v_retrain=False, fixed_n2v=False,
                 **kwargs):
        self.graphs_with_views = graphs_with_views
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        nodes = set()
        for graph_views in self.graphs_with_views:
            for graph_view in graph_views:
                for node in graph_view.nodes():
                    nodes.add(node)
        self.nodes = list(nodes)
        self.nodes = np.random.permutation(self.nodes)
        self.adjs, self.degs = self.construct_adj()
        self.test_adjs = self.construct_test_adj()
        if context_pairs is None:
            edges = self.graphs_with_views[-1].edges()
        else:
            edges = context_pairs
        self.train_edges = self.edges = np.random.permutation(edges)
        if not n2v_retrain:
            # self.train_edges = self._remove_isolated(self.train_edges)
            # self.val_edges = [e for g in self.graphs for e in g.edges() if g[e[0]][e[1]]['train_removed']]
            self.val_edges = val_edges
        else:
            if fixed_n2v:
                self.train_edges = self.val_edges = self._n2v_prune(self.edges)
            else:
                self.train_edges = self.val_edges = self.edges

        # print(len(set([n for graph_views in self.graphs_with_views for graph_view in graph_views for n in graph_view.nodes() if not graph_view.node[n]['test'] and not graph_view.node[n]['val']])), 'train nodes')
        # print(len(set([n for graph_views in self.graphs_with_views for graph_view in graph_views for n in graph_view.nodes() if graph_view.node[n]['test'] and graph_view.node[n]['val']])), 'test nodes')
        self.val_set_size = len(self.val_edges)

    def _remove_isolated(self, edge_list):

        new_edge_list = []
        missing = 0
        check_edge_num = 0
        for n1, n2 in edge_list:
            check_edge_num = check_edge_num + 1
            if check_edge_num % 100 == 0:
                print("_remove_isolated check edge num: {}".format(check_edge_num))
            if not n1 in self.nodes or not n2 in self.nodes:
                missing += 1
                continue
            else:
                new_edge_list.append((n1,n2))
        print("Unexpected missing:", missing)
        return new_edge_list

    def construct_adj(self):
        adjs = []
        degs = []
        for graph_views in self.graphs_with_views:
            temp_adjs = []
            temp_degs = []
            for graph_view in graph_views:
                adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
                deg = np.zeros((len(self.id2idx),))

                for nodeid in graph_view.nodes():
                    if graph_view.node[nodeid]['test'] or graph_view.node[nodeid]['val']:
                        continue
                    neighbors = np.array([self.id2idx[neighbor]
                        for neighbor in graph_view.neighbors(nodeid)
                        if (not graph_view[nodeid][neighbor]['train_removed'])])
                    deg[self.id2idx[nodeid]] = len(neighbors)
                    if len(neighbors) == 0:
                        continue
                    if len(neighbors) > self.max_degree:
                        neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                    elif len(neighbors) < self.max_degree:
                        neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                    adj[self.id2idx[nodeid], :] = neighbors
                temp_adjs.append(adj)
                temp_degs.append(deg)

            adjs.append(temp_adjs)
            degs.append(temp_degs)

        return adjs, degs

    def construct_test_adj(self):
        adjs = []
        for graph_views in self.graphs_with_views:
            temp_adjs = []
            for graph_view in graph_views:
                adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
                for nodeid in graph_view.nodes():
                    neighbors = np.array([self.id2idx[neighbor]
                        for neighbor in graph_view.neighbors(nodeid)])
                    if len(neighbors) == 0:
                        continue
                    if len(neighbors) > self.max_degree:
                        neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                    elif len(neighbors) < self.max_degree:
                        neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                    adj[self.id2idx[nodeid], :] = neighbors
                temp_adjs.append(adj)
            adjs.append(temp_adjs)

        return adjs

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size,
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0


