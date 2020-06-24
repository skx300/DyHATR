#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.python.training import moving_averages

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GeneralizedModel(Model):

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        
    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)


SAGEInfo = namedtuple("SAGEInfo",
	['layer_name', # name of the layer (to get feature embedding etc.)
	 'neigh_sampler', # callable neigh_sampler constructor
	 'num_samples',
	 'output_dim' # the output (i.e., hidden) dimension
	])

def construct_placeholders():
	# Define placeholders
	placeholders = {
		'batch1': tf.compat.v1.placeholder(tf.int32, shape=(None), name='batch1'),
		'batch2': tf.compat.v1.placeholder(tf.int32, shape=(None), name='batch2'),
		# negative samples for all nodes in the batch
		'neg_samples': tf.compat.v1.placeholder(tf.int32, shape=(None,), name='neg_sample_size'),
		'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
		'temporal_dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='temporal_dropout'),
		'batch_size': tf.compat.v1.placeholder(tf.int32, name='batch_size'),
	}

	return placeholders

def log_dir():
	log_dir = FLAGS.base_log_dir + "/unsup-" + FLAGS.train_prefix
	log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(model=FLAGS.model,
		model_size=FLAGS.model_size, lr=FLAGS.learning_rate)

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	return log_dir

def evaluate(sess, model, minibatch_iter, size=None):
	t_test = time.time()
	feed_dict_val = minibatch_iter.val_feed_dict(size)
	outs_val = sess.run([model.loss, model.ranks, model.mrr],
						feed_dict=feed_dict_val)
	return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
	val_embeddings = []
	finished = False
	seen = set([])
	nodes = []
	iter_num = 0
	name = "val"
	while not finished:
		feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
		iter_num += 1
		outs_val = sess.run([model.loss, model.mrr, model.outputs1],
							feed_dict=feed_dict_val)
		# ONLY SAVE FOR embeds1 because of planetoid
		for i, edge in enumerate(edges):
			if not edge[0] in seen:
				val_embeddings.append(outs_val[-1][i, :])
				nodes.append(edge[0])
				seen.add(edge[0])
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	val_embeddings = np.vstack(val_embeddings)
	np.save(out_dir + name + mod + ".npy", val_embeddings)
	with open(out_dir + name + mod + ".txt", "w") as fp:
		fp.write("\n".join(map(str, nodes)))
	return val_embeddings, nodes

def batch_normalization_layer(inputs, isTrain=True):
	
	EPSILON = 0.01
	CHANNEL = inputs.shape[2]
	# print(inputs.shape) #(?,6,32)
	MEANDECAY = 0.99

	ave_mean = tf.Variable(tf.zeros(shape = [CHANNEL]), trainable = False)
	ave_var = tf.Variable(tf.zeros(shape = [CHANNEL]), trainable = False)
	mean, var = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=False)

	update_mean_op = moving_averages.assign_moving_average(ave_mean, mean, MEANDECAY)
	update_var_op = moving_averages.assign_moving_average(ave_var, var, MEANDECAY)

	tf.add_to_collection("update_op", update_mean_op)
	tf.add_to_collection("update_op", update_var_op)

	scale = tf.Variable(tf.constant(1.0, shape=mean.shape))
	offset = tf.Variable(tf.constant(0.0, shape=mean.shape))

	if isTrain:
		inputs = tf.nn.batch_normalization(inputs, mean = mean, variance = var, offset = offset, scale = scale, variance_epsilon = EPSILON)
	else:
		inputs = tf.nn.batch_normalization(inputs, mean = ave_mean, variance = ave_var, offset = offset, scale = scale, variance_epsilon = EPSILON)

	return inputs