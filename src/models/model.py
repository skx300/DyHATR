#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .tf_utils import GeneralizedModel
from .prediction_layers import BipartiteEdgePredLayer
from .temporal_layers import GRULearnerLayer,LSTMLearnerLayer
from .attention_layers import EdgeAttentionLayer,AttentionAggregatorVectorized,TemporalAttentionLayer

flags = tf.app.flags
FLAGS = flags.FLAGS


class DyHATR(GeneralizedModel):

	def __init__(self, placeholders, features, adjs, degs, structural_layer_infos, concat=True, aggregator_type="gat",
				 model_size="small", identity_dim=0, temporal_learner="", **kwargs):
		"""
		:param placeholders: Stanford TensorFlow placeholder object.
		:param features: Numpy array with node features.
						NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
		:param adjs: a list of adj
		:param degs: a list of node degree
		:param layer_infos: List of SAGEInfo namedtuples that describe the parameters of all
				   the recursive layers. See SAGEInfo definition above.
		:param concat:
		:param model_size:
		:param identity_dim:
		:param kwargs:
		"""
		super(DyHATR, self).__init__(**kwargs)
		if aggregator_type == "gat":
			self.structural_aggregator_cls = AttentionAggregatorVectorized
		else:
			raise Exception("Unknown aggregator: ", self.structural_aggregator_cls)

		self.aggregator_type = aggregator_type

		# get info from placeholders...
		self.inputs1 = placeholders["batch1"]
		self.inputs2 = placeholders["batch2"]
		self.model_size = model_size
		self.adjs = adjs
		if identity_dim > 0:
			self.embeds = tf.compat.v1.get_variable("node_embeddings", [adjs[0][0].get_shape().as_list()[0], identity_dim])
		else:
			self.embeds = None
		if features is None:
			if identity_dim == 0:
				raise Exception("Must have a positive value for identity feature dimension if no input features given.")
			self.features = self.embeds
		else:
			self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
			if not self.embeds is None:
				self.features = tf.concat([self.embeds, self.features], axis=1)
		self.degs = degs
		self.concat = concat
		print("shape of features", self.features.shape)
		print(self.features[0:1,:])
		self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
		self.dims.extend([structural_layer_infos[i].output_dim for i in range(len(structural_layer_infos))])
		self.batch_size = placeholders["batch_size"]
		self.placeholders = placeholders
		self.structural_layer_infos = structural_layer_infos
		self.num_graph_snapshots = len(adjs)
		self.num_graph_views = len(adjs[0])
		self.temporal_learner = temporal_learner
		self.num_heads_hat = FLAGS.num_heads_hat
		self.num_heads_tat = FLAGS.num_heads_tat

		if FLAGS.optimizer == 'Adam':
			self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
		elif FLAGS.optimizer == 'SGD':
			self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
		elif FLAGS.optimizer == 'Adade':
			self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
		elif FLAGS.optimizer == 'RSMP':
			self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
		elif FLAGS.optimizer == 'Momentum':
			self.optimizer == tf.compat.v1.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate)

		self.build()

	def edge_specific_sample(self, inputs, layer_infos, batch_size=None):
		"""
		Sample neighbors to be the supportive fields for each edge specific subgraph of each snapshot.

		"""
		if batch_size is None:
			batch_size = self.batch_size

		# size of convolution support at each layer per node
		support_size = 1
		support_sizes = [support_size]
		for k in range(len(layer_infos)):
			t = len(layer_infos) - k - 1
			support_size *= layer_infos[t].num_samples
			support_sizes.append(support_size)

		num_snapshots = len(layer_infos[0].neigh_sampler)
		for layer_info in layer_infos:
			assert num_snapshots == len(layer_info.neigh_sampler)

		samples_layer_snapshots_graphviews = []
		for k in range(len(layer_infos) + 1):
			samples = []
			t = len(layer_infos) - k
			for i in range(self.num_graph_snapshots):
				temp_samples = []
				for j in range(self.num_graph_views):
					if k == 0:
						temp_samples.append(inputs)
					else:
						# for gat, there is no need to sample neighbors. instead, it will use all neighbors.
						# node = tf.nn.embedding_lookup(self.adjs[i], samples_layer_graphviews[k-1][i])
						node = tf.nn.embedding_lookup(self.adjs[i][j], samples_layer_snapshots_graphviews[k - 1][i][j])
						# samples.append(tf.reshape(node, [support_sizes[k] * batch_size, ]))
						temp_samples.append(tf.reshape(node, [support_sizes[k] * batch_size, ]))
				samples.append(temp_samples)
			samples_layer_snapshots_graphviews.append(samples)

		return samples_layer_snapshots_graphviews, support_sizes

	def edge_specific_aggregate(self, samples_layer_snapshots_graphviews, input_features, dims, num_samples,
								support_sizes, batch_size=None,
								aggregators=None, name=None, concat=False, model_size="small"):
		"""
		Aggregate nodes for specific type edge.

		"""
		if batch_size is None:
			batch_size = self.batch_size

		# list of tensors [layer_size, num of snapshots, num of graph views]
		hidden_total = []
		for samples_layer in samples_layer_snapshots_graphviews:
			snapshot_hidden = []
			for samples_snapshot in samples_layer:
				hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples_snapshot]
				snapshot_hidden.append(hidden)
			hidden_total.append(snapshot_hidden)

		# use existing aggregators or create new.
		# for each graph view, we have different aggregators.
		new_agg = aggregators is None
		if new_agg:
			aggregators = []
			for layer in range(len(num_samples)):
				dim_mult = 2 if concat and (layer != 0) else 1
				layer_aggegators = []
				for graph_view_num in range(self.num_graph_views):
					# for each graph view, we create a new aggregator.
					# aggregator at current layer
					if layer == len(num_samples) - 1:
						aggregator = self.structural_aggregator_cls(dim_mult * dims[layer], dims[layer + 1],
																	act=lambda x: x,num_heads=self.num_heads_hat,
																	dropout=self.placeholders['dropout'],
																	name=name, concat=concat, model_size=model_size)
					else:
						aggregator = self.structural_aggregator_cls(dim_mult * dims[layer], dims[layer + 1],
																	num_heads=self.num_heads_hat,
																	dropout=self.placeholders['dropout'],
																	name=name, concat=concat, model_size=model_size)
					layer_aggegators.append(aggregator)
				aggregators.append(layer_aggegators)

		# do aggregation, the aggregators are shared by different graph views.
		# A list of tensors, each tensor has shape [batch_size, num_snapshots, num_graph_views, dim]
		hidden = []
		for layer_hidden in hidden_total:
			temp_hidden = []
			for snapshot_hidden in layer_hidden:
				temp_hidden.append(tf.concat([tf.expand_dims(ele, axis=1) for ele in snapshot_hidden], axis=1))
			hidden.append(tf.concat([tf.expand_dims(ele, axis=1) for ele in temp_hidden], axis=1))

		hidden_final = []
		for layer in range(len(num_samples)):
			# hidden representation at current layer for all support nodes that are various hops away
			next_hidden = []
			# as layer increases, the number of support nodes needed decreases
			for hop in range(len(num_samples) - layer):
				dim_mult = 2 if concat and (layer != 0) else 1
				temp_hidden = []
				neigh_dims = [batch_size * support_sizes[hop],num_samples[len(num_samples)-hop-1],
							  self.num_graph_snapshots, dim_mult * dims[layer]]
				for graph_view_num in range(self.num_graph_views):
					h = aggregators[layer][graph_view_num]((hidden[hop][:, :, graph_view_num, :],
										tf.reshape(hidden[hop + 1][:, :, graph_view_num, :], neigh_dims)))
					if self.num_graph_snapshots == 1:
						h = tf.expand_dims(h, -1)
						h = tf.transpose(h, perm=[0,2,1])
					temp_hidden.append(h)
				next_hidden.append(tf.stack(temp_hidden, axis=2))
			hidden = next_hidden
		return hidden[0], aggregators

	def edge_aggregate(self, inputs, input_dim, edge_aggregators=None, name=None):
		"""
		Aggregate node embedding from different edge-type embeddings.

		"""
		# use existing edge aggregators or create new.
		new_agg = edge_aggregators is None
		if new_agg:
			edge_aggregators = []
			aggregator = EdgeAttentionLayer(input_dim=input_dim,atten_vec_size=128,attn_drop=self.placeholders['dropout'])
			edge_aggregators.append(aggregator)

		# do aggregation
		edge_specific_inputs = inputs
		for aggregator in edge_aggregators:
			outputs = aggregator(edge_specific_inputs)
			edge_specific_inputs = outputs

		return outputs, edge_aggregators

	def temporal_aggregate(self, inputs, input_dim, n_heads, num_time_steps, temporal_aggregators=None, name=None):
		"""
		Aggregate node embedding from each snapshot graph.

		"""
		# use existing temporal aggregators or create new.
		# print("###:", inputs)
		print("Multi-head Temporal Attention:",n_heads)
		new_agg = temporal_aggregators is None
		if new_agg:
			temporal_aggregators = []
			aggregator = TemporalAttentionLayer(input_dim=input_dim,
												n_heads=n_heads,
												num_time_steps=num_time_steps,
												attn_drop=self.placeholders['temporal_dropout'])
			temporal_aggregators.append(aggregator)

		# do aggregation
		temporal_inputs = inputs
		for aggregator in temporal_aggregators:
			# print(aggregator)
			outputs = aggregator(temporal_inputs)
			# print(outputs)
			temporal_inputs = outputs

		return outputs, temporal_aggregators

	def gru_propogation(self, inputs, input_dim, num_time_steps, gru_learners=None, name=None):

		new_agg = gru_learners is None
		if new_agg:
			gru_learners = []
			learner = GRULearnerLayer(input_dim=input_dim, num_time_steps=num_time_steps)
			gru_learners.append(learner)

		gru_inputs = inputs
		for learner in gru_learners:
			gru_outputs = learner(gru_inputs)
			gru_inputs = gru_outputs

		return gru_outputs, gru_learners

	def lstm_propogation(self, inputs, input_dim, num_time_steps, lstm_learners=None, name=None):

		new_agg = lstm_learners is None
		if new_agg:
			lstm_learners = []
			learner = LSTMLearnerLayer(input_dim=input_dim, num_time_steps=num_time_steps)
			lstm_learners.append(learner)

		lstm_inputs = inputs
		for learner in lstm_learners:
			lstm_outputs = learner(lstm_inputs)
			lstm_inputs = lstm_outputs

		return lstm_outputs, lstm_learners


	def _build(self):
		labels = tf.reshape(
			tf.cast(self.placeholders['batch2'], dtype=tf.int64),
			[self.batch_size, 1])

		self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
			true_classes=labels,
			num_true=1,
			num_sampled=FLAGS.neg_sample_size,
			unique=False,
			range_max=len(sum(self.degs[-1])),
			distortion=0.75,
			unigrams=sum(self.degs[-1]).tolist()))

		# Step 1: edge specific node aggregation
		samples1_edge_specific, support_sizes1 = self.edge_specific_sample(self.inputs1, self.structural_layer_infos)
		samples2_edge_specific, support_sizes2 = self.edge_specific_sample(self.inputs2, self.structural_layer_infos)
		neg_samples_edge_specific, neg_support_sizes = self.edge_specific_sample(self.neg_samples, self.structural_layer_infos, FLAGS.neg_sample_size)
		num_samples = [layer_info.num_samples for layer_info in self.structural_layer_infos]

		self.outputs1_edge_specific, self.edge_specific_aggregators = \
			self.edge_specific_aggregate(samples1_edge_specific,
									  [self.features],
									  self.dims,
									  num_samples,
									  support_sizes1,
									  concat=self.concat,
									  model_size=self.model_size)
		self.outputs2_edge_specific, _ = \
			self.edge_specific_aggregate(samples2_edge_specific, [self.features], self.dims,
									  num_samples, support_sizes2,
									  aggregators=self.edge_specific_aggregators,
									  concat=self.concat,
									  model_size=self.model_size)

		self.neg_outputs_edge_specific, _ = \
			self.edge_specific_aggregate(neg_samples_edge_specific, [self.features],
										 self.dims, num_samples,
										 neg_support_sizes,
										 batch_size=FLAGS.neg_sample_size,
										 aggregators=self.edge_specific_aggregators,
										 concat=self.concat,
										 model_size=self.model_size)

		# Step 2: edge aggregation-aggregate node embedding from different edge-type embeddings.
		print("### Start edge aggregation...")
		dim_mult = 2 if self.concat else 1
		self.outputs1_snapshots, self.edge_aggregators = \
			self.edge_aggregate(inputs=self.outputs1_edge_specific,
								input_dim=self.dims[-1] * dim_mult)

		self.outputs2_snapshots, _ = \
			self.edge_aggregate(inputs=self.outputs2_edge_specific,
								input_dim=self.dims[-1] * dim_mult,
								edge_aggregators=self.edge_aggregators)

		self.neg_outputs_snapshots, _ = \
			self.edge_aggregate(inputs=self.neg_outputs_edge_specific,
								input_dim=self.dims[-1] * dim_mult,
								edge_aggregators=self.edge_aggregators)

		# self.outputs1_snapshots = tf.nn.l2_normalize(self.outputs1_snapshots,dim=1,epsilon=1e-12)
		# self.outputs2_snapshots = tf.nn.l2_normalize(self.outputs2_snapshots,dim=1,epsilon=1e-12)
		# self.neg_outputs_snapshots = tf.nn.l2_normalize(self.neg_outputs_snapshots,dim=1,epsilon=1e-12)

		# self.outputs1 = tf.reshape(self.outputs1_snapshots, [-1,10*32])
		# self.outputs2 = tf.reshape(self.outputs2_snapshots, [-1,10*32])
		# self.neg_outputs = tf.reshape(self.neg_outputs_snapshots,[2,10*32])

		# self.outputs1 = self.outputs1_snapshots[:,-1,:]
		# self.outputs2 = self.outputs2_snapshots[:,-1,:]
		# self.neg_outputs = self.neg_outputs_snapshots[:,-1,:]

		# Step 3: temporal attentive RNN learning
		if self.temporal_learner == "GRU":
			print("### Start GRU propogation...")
			dim_mult = 2 if self.concat else 1
			self.outputs1_temporal, self.gru_learner = \
				self.gru_propogation(inputs=self.outputs1_snapshots,input_dim=self.dims[-1]*dim_mult,
					num_time_steps=self.num_graph_snapshots)

			self.outputs2_temporal, _ = \
				self.gru_propogation(inputs=self.outputs2_snapshots,input_dim=self.dims[-1]*dim_mult,
					num_time_steps=self.num_graph_snapshots,gru_learners=self.gru_learner)

			self.neg_outputs_tmeporal, _ = \
				self.gru_propogation(inputs=self.neg_outputs_snapshots,input_dim=self.dims[-1]*dim_mult,
					num_time_steps=self.num_graph_snapshots,gru_learners=self.gru_learner)
			self.rnn_learner = self.gru_learner
		elif self.temporal_learner == "LSTM":
			print("### Start LSTM propogation...")
			dim_mult = 2 if self.concat else 1
			self.outputs1_temporal, self.lstm_learner = \
				self.lstm_propogation(inputs=self.outputs1_snapshots,input_dim=self.dims[-1]*dim_mult,
					num_time_steps=self.num_graph_snapshots)

			self.outputs2_temporal, _ = \
				self.lstm_propogation(inputs=self.outputs2_snapshots,input_dim=self.dims[-1]*dim_mult,
					num_time_steps=self.num_graph_snapshots,lstm_learners=self.lstm_learner)

			self.neg_outputs_tmeporal, _ = \
				self.lstm_propogation(inputs=self.neg_outputs_snapshots,input_dim=self.dims[-1]*dim_mult,
					num_time_steps=self.num_graph_snapshots,lstm_learners=self.lstm_learner)
			self.rnn_learner = self.lstm_learner

		# self.outputs1 = tf.reshape(self.outputs1_temporal, [-1,10*32])
		# self.outputs2 = tf.reshape(self.outputs2_temporal, [-1,10*32])
		# self.neg_outputs = tf.reshape(self.neg_outputs_tmeporal,[5,10*32])

		# self.outputs1 = self.outputs1_temporal[:,-1,:]
		# self.outputs2 = self.outputs2_temporal[:,-1,:]
		# self.neg_outputs = self.neg_outputs_tmeporal[:,-1,:]

		# self.outputs1_temporal = self.outputs1_snapshots
		# self.outputs2_temporal = self.outputs2_snapshots
		# self.neg_outputs_tmeporal = self.neg_outputs_snapshots

		# self.outputs1_temporal = tf.nn.l2_normalize(self.outputs1_temporal,dim=1,epsilon=1e-12)
		# self.outputs2_temporal = tf.nn.l2_normalize(self.outputs2_temporal,dim=1,epsilon=1e-12)
		# self.neg_outputs_tmeporal = tf.nn.l2_normalize(self.neg_outputs_tmeporal,dim=1,epsilon=1e-12)


		### temporal aggregation
		print("### Start temporal aggregation...")
		dim_mult = 2 if self.concat else 1
		self.outputs1, self.temporal_aggregators = \
			 self.temporal_aggregate(inputs=self.outputs1_temporal, #self.outputs1_snapshots,
									 input_dim=self.dims[-1] * dim_mult,
									 n_heads=self.num_heads_tat,
									 num_time_steps=self.num_graph_snapshots)
		self.outputs2, _ = \
			 self.temporal_aggregate(inputs=self.outputs2_temporal, #self.outputs2_snapshots,
									 input_dim=self.dims[-1] * dim_mult,
									 n_heads=self.num_heads_tat,
									 num_time_steps=self.num_graph_snapshots,
									 temporal_aggregators=self.temporal_aggregators)
		self.neg_outputs, _ = \
			 self.temporal_aggregate(inputs=self.neg_outputs_tmeporal, #self.neg_outputs_snapshots,
									 input_dim=self.dims[-1] * dim_mult,
									 n_heads=self.num_heads_tat,
									 num_time_steps=self.num_graph_snapshots,
									 temporal_aggregators=self.temporal_aggregators)


		self.link_pred_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
				dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid,
				bilinear_weights=False, name='edge_predict')

		# print(self.outputs1.get_shape(), self.outputs2.get_shape(), self.neg_outputs.get_shape())

		self.outputs1 = tf.nn.l2_normalize(self.outputs1,dim=1,epsilon=1e-12)
		self.outputs2 = tf.nn.l2_normalize(self.outputs2,dim=1,epsilon=1e-12)
		self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs,dim=1,epsilon=1e-12)

		# self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
		# self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
		# self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)

	def build(self):
		self._build()

		# TF graph management
		self._loss()
		self._accuracy()
		self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
		grads_and_vars = self.optimizer.compute_gradients(self.loss)
		clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
				for grad, var in grads_and_vars]
		self.grad, _ = clipped_grads_and_vars[0]
		self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

	def _loss(self):
		for aggregators in self.edge_specific_aggregators:
			for aggregator in aggregators:
				for var in aggregator.vars.values():
					self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

		for aggregator in self.edge_aggregators:
			for var in aggregator.vars.values():
				self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

		for aggregator in self.rnn_learner:
			for var in aggregator.vars.values():
				self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

		for aggregator in self.temporal_aggregators:
			 for var in aggregator.vars.values():
				 self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

		self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs)
		tf.compat.v1.summary.scalar('loss', self.loss)

	def _accuracy(self):
		# shape: [batch_size]
		aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
		# shape : [batch_size x num_neg_samples]
		self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
		self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
		_aff = tf.expand_dims(aff, axis=1)
		self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
		size = tf.shape(self.aff_all)[1]
		_, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
		_, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
		self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
		tf.compat.v1.summary.scalar('mrr', self.mrr)


