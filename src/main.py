#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from utils.data_helper import load_data,load_train_test
from models.tf_utils import *
from models.model import DyHATR
from models.minibatch import EdgeMinibatchIterator
from evals.validation import cross_validation

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
flags.DEFINE_string('model', 'DyHATR', 'model names. See README for possible values.')
flags.DEFINE_string('optimizer', 'Adam', 'which optimizer to choose.')
flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', 'EComm', 'name of the object file that stores the training data. must be specified.')
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.2, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('temporal_dropout', 0.0, 'temporal dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 30, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 15, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 32, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 32, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 5, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 32, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_integer('num_heads_hat', 4, 'number of heads for HAT.')
flags.DEFINE_integer('num_heads_tat', 4, 'number of heads for TAT.')
flags.DEFINE_string('base_log_dir', '../logs', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 50, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 128, "how many nodes per validation sample.")
flags.DEFINE_integer('print_every', 10, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 1000, "Maximum total number of iterations")
flags.DEFINE_string('temporal_learner', 'LSTM', 'Which temporal learner to choose.')
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


#  main procedure
print("### loading training data...")
train_data = load_data(FLAGS.train_prefix)

graphs_with_views = train_data[0] #list,(10,4)
features = train_data[1] #None
id_map = train_data[2] #dict,35981
val_edges = train_data[4] #list,(399,2)
context_pairs = train_data[3] if FLAGS.random_context else None
# pad with dummy zero vector
if not features is None:
	features = np.vstack([features, np.zeros((features.shape[1],))])
# print(features) #None

print("### Initializing minibatch iterator...")
placeholders = construct_placeholders()
minibatch = EdgeMinibatchIterator(graphs_with_views, id_map, placeholders, val_edges, 
		batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree, context_pairs=context_pairs)

adjs_info_ph = []
for snapshot_adjs in minibatch.adjs:
	adjs_info_snapshot_ph = [tf.compat.v1.placeholder(tf.int32, shape=adj.shape) for adj in snapshot_adjs]
	adjs_info_ph.append(adjs_info_snapshot_ph)

adjs_info = []
for snapshot_adjs_info_ph in adjs_info_ph:
	adjs_info_snapshot = [tf.Variable(adj_info_ph, trainable=False) for adj_info_ph in snapshot_adjs_info_ph]
	adjs_info.append(adjs_info_snapshot)

print("### Initializing model...")
samplers = []
for adj_info in adjs_info:
	samplers.append(None)
structural_layer_infos = [SAGEInfo("node", samplers, minibatch.adjs[0][0].shape[-1], FLAGS.dim_1)]
model = DyHATR(placeholders, features, adjs_info, minibatch.degs, 
					structural_layer_infos=structural_layer_infos,
					aggregator_type="gat",
					model_size=FLAGS.model_size,
					identity_dim=FLAGS.identity_dim,
					temporal_learner=FLAGS.temporal_learner,
					concat=False,logging=True)

config = tf.compat.v1.ConfigProto(log_device_placement=FLAGS.log_device_placement)
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# Initialize session
sess = tf.compat.v1.Session(config=config)
merged = tf.compat.v1.summary.merge_all()
summary_writer = tf.compat.v1.summary.FileWriter(log_dir(), sess.graph)
run_options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.compat.v1.RunMetadata()

# Init variables
num_graph_snapshots = len(minibatch.adjs)
num_graph_views = len(minibatch.adjs[0])
feed_val = {}
for i in range(num_graph_snapshots):
	for j in range(num_graph_views):
		feed_val[adjs_info_ph[i][j]] = minibatch.adjs[i][j]

sess.run(tf.compat.v1.global_variables_initializer(), feed_dict=feed_val)

print("### All trainable vars: ")
for var in tf.compat.v1.trainable_variables():
	print("{}".format(var))

# Train model
train_shadow_mrr = None
shadow_mrr = None
total_steps = 0
avg_time = 0.0
epoch_val_costs = []

train_adjs_info = [tf.compat.v1.assign(adjs_info[t][i], minibatch.adjs[t][i]) 
					for t in range(num_graph_snapshots) for i in range(num_graph_views)]
val_adjs_info = [tf.compat.v1.assign(adjs_info[t][i], minibatch.test_adjs[t][i]) 
					for t in range(num_graph_snapshots) for i in range(num_graph_views)]

for epoch in range(FLAGS.epochs):
	minibatch.shuffle()
	iter = 0
	print('Epoch: %04d' % (epoch + 1))
	epoch_val_costs.append(0)
	min_val_loss = 100
	while not minibatch.end():
		# Construct feed dictionary
		feed_dict = minibatch.next_minibatch_feed_dict()
		feed_dict.update({placeholders['dropout']: FLAGS.dropout})
		feed_dict.update({placeholders['temporal_dropout']: FLAGS.temporal_dropout})

		t = time.time()
		# Training step
		outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all, 
								model.mrr, model.outputs1_edge_specific], feed_dict=feed_dict,
								options=run_options, run_metadata=run_metadata)
		train_cost = outs[2]
		train_mrr = outs[5]
		if train_shadow_mrr is None:
			train_shadow_mrr = train_mrr
		else:
			train_shadow_mrr -= (1 - 0.99) * (train_shadow_mrr - train_mrr)

		if iter % FLAGS.validate_iter == 0:
			# Validation
			sess.run([val_adj_info.op for val_adj_info in val_adjs_info])
			val_cost, ranks, val_mrr, duration = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size)
			sess.run([train_adj_info.op for train_adj_info in train_adjs_info])
			epoch_val_costs[-1] += val_cost

		if shadow_mrr is None:
			shadow_mrr = val_mrr
		else:
			shadow_mrr -= (1 - 0.99) * (shadow_mrr - val_mrr)

		if total_steps % FLAGS.print_every == 0:
			summary_writer.add_summary(outs[0], total_steps)
		# Print results
		avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
		if total_steps % FLAGS.print_every == 0:
			print("Iter:", '%04d' % iter,
					"train_loss=", "{:.5f}".format(train_cost),
					"train_mrr=", "{:.5f}".format(train_mrr),
					"train_mrr_ema=", "{:.5f}".format(train_shadow_mrr),  # exponential moving average
					"val_loss=", "{:.5f}".format(val_cost),
					"val_mrr=", "{:.5f}".format(val_mrr),
					"val_mrr_ema=", "{:.5f}".format(shadow_mrr),  # exponential moving average
					"time=", "{:.5f}".format(avg_time))
		iter += 1
		total_steps += 1
			
		if total_steps > FLAGS.max_total_steps:
			break

	if total_steps > FLAGS.max_total_steps:
		break

print("### Optimization Finished...")
# Create the Timeline object, and write it to a json
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open(log_dir() + 'timeline.json', 'w') as f:
	f.write(ctf)

print("### Cross-Validation...")
sess.run([val_adj_info.op for val_adj_info in val_adjs_info])
embed, nodes = save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir())

node_embeddings = dict()
for i in range(len(nodes)):
	node_embeddings[nodes[i]] = embed[i]

avg_auroc,std_auroc,avg_auprc,std_auprc = cross_validation(node_embeddings, score_type="cosine", n_trials=5)
print("### Average (over trials):  AUROC: {:.4f}({:.4f}), AUPRC: {:.4f}({:.4f})".format(avg_auroc,std_auroc,avg_auprc,std_auprc))
