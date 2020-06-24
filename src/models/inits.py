#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gcn
# which is under an identical MIT license as GraphSAGE

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    # init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    init_range = np.sqrt(3.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    # initial = tf.random_normal(shape, mean=0.0, stddev=np.sqrt(2.0/(shape[0]+shape[1])), dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def gru_init(shape, name=None):
    initial = tf.truncated_normal(shape=shape, mean=-0.1, stddev=0.1, dtype=tf.float32)
    # initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.01, dtype=tf.float32)
    return tf.Variable(initial, name=name)
    # return tf.get_variable(shape=shape, initializer=tf.orthogonal_initializer(),name=name)

def gru_zeros(shape, name=None):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

    