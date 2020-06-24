#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.training import moving_averages

from .layers import Layer
from .inits import glorot, zeros, uniform, gru_init, gru_zeros


class GRULearnerLayer(Layer):

	def __init__(self, input_dim, num_time_steps, act=tf.sigmoid, name=None, **kwargs):
		super(GRULearnerLayer, self).__init__(**kwargs)

		self.input_dim = input_dim
		self.num_time_steps = num_time_steps
		self.act = act

		if name is not None:
			name = '/' + name
		else:
			name = ''

		with tf.variable_scope(self.name + name + '_vars'):
			self.vars['gru_weights_zx'] = glorot([input_dim, input_dim], name='gru_weights_zx')
			self.vars['gru_weights_zh'] = glorot([input_dim, input_dim], name='gru_weights_zh')
			self.vars['gru_weights_rx'] = glorot([input_dim, input_dim], name='gru_weights_rx')
			self.vars['gru_weights_rh'] = glorot([input_dim, input_dim], name='gru_weights_rh')
			self.vars['gru_weights_hx'] = glorot([input_dim, input_dim], name='gru_weights_hx')
			self.vars['gru_weights_hh'] = glorot([input_dim, input_dim], name='gru_weights_hh')

			self.vars['gru_bias_z'] = zeros([input_dim], name='gru_bias_z')
			self.vars['gru_bias_r'] = zeros([input_dim], name='gru_bias_r')
			self.vars['gru_bias_h'] = zeros([input_dim], name='gru_bias_h')

		if self.logging:
			self._log_vars()

	def _call(self, inputss):
		# inputss: [sample,6,32] neg:[5,6,32]
		graphs = tf.transpose(inputss, perm=[1,2,0]) #(6,32,sample)
		state_h_t = tf.zeros_like(graphs[0,:,:]) #(32,sample)
		outputs = []

		for idx in range(0, self.num_time_steps):
			# print("### GRU ",idx+1)
			inputs = graphs[idx,:,:] #(32,sample)

			# mul_feats = tf.concat([inputs, state_h_t], 0) #[64,sample]

			gate_z_ = tf.matmul(self.vars['gru_weights_zx'], inputs) #[32,sample]
			gate_z_ += tf.matmul(self.vars['gru_weights_zh'], state_h_t) #[32,sample]
			gate_z_ = tf.transpose(gate_z_)+self.vars['gru_bias_z']
			gate_z = tf.transpose(self.act(gate_z_))

			gate_r_ = tf.matmul(self.vars['gru_weights_rx'], inputs) #[32,sample]
			gate_r_ += tf.matmul(self.vars['gru_weights_rh'], state_h_t) #[32,sample]
			gate_r_ = tf.transpose(gate_r_)+self.vars['gru_bias_r']
			gate_r = tf.transpose(self.act(gate_r_))

			part_a = tf.multiply(gate_r, state_h_t)
			# state_h_ = tf.matmul(self.vars['gru_weights_h'], tf.concat([inputs, part_a], 0))
			state_h_ = tf.matmul(self.vars['gru_weights_hx'], inputs)
			state_h_ += tf.matmul(self.vars['gru_weights_hh'], part_a)
			state_h_ = tf.transpose(state_h_)+self.vars['gru_bias_h']
			state_h_ = tf.transpose(tf.nn.tanh(state_h_))

			state_h = tf.multiply(1.0-gate_z, state_h_t)
			state_h += tf.multiply(gate_z, state_h_) #[32, sample]

			state_h_t = state_h
			outputs.append(state_h)

		outputs = tf.transpose(outputs, perm=[2,0,1])
		return outputs


class LSTMLearnerLayer(Layer):

	def __init__(self, input_dim, num_time_steps, act=tf.sigmoid, name=None, **kwargs):
		super(LSTMLearnerLayer, self).__init__(**kwargs)

		self.input_dim = input_dim
		self.num_time_steps = num_time_steps
		self.act = act

		if name is not None:
			name = '/' + name
		else:
			name = ''

		with tf.variable_scope(self.name + name + '_vars'):
			self.vars['lstm_weights_gx'] = glorot([input_dim, input_dim], name='lstm_weights_gx')
			self.vars['lstm_weights_gh'] = glorot([input_dim, input_dim], name='lstm_weights_gh')
			self.vars['lstm_weights_ix'] = glorot([input_dim, input_dim], name='lstm_weights_ix')
			self.vars['lstm_weights_ih'] = glorot([input_dim, input_dim], name='lstm_weights_ih')
			self.vars['lstm_weights_fx'] = glorot([input_dim, input_dim], name='lstm_weights_fx')
			self.vars['lstm_weights_fh'] = glorot([input_dim, input_dim], name='lstm_weights_fh')
			self.vars['lstm_weights_ox'] = glorot([input_dim, input_dim], name='lstm_weights_ox')
			self.vars['lstm_weights_oh'] = glorot([input_dim, input_dim], name='lstm_weights_oh')

			self.vars['lstm_bias_g'] = zeros([input_dim], name='lstm_bias_g')
			self.vars['lstm_bias_i'] = zeros([input_dim], name='lstm_bias_i')
			self.vars['lstm_bias_f'] = zeros([input_dim], name='lstm_bias_f')
			self.vars['lstm_bias_o'] = zeros([input_dim], name='lstm_bias_o')

		if self.logging:
			self._log_vars()

	def _call(self, inputss):
		# inputss: [sample,6,32] neg:[5,6,32]
		graphs = tf.transpose(inputss, perm=[1,2,0]) #(6,32,sample)
		state_h_t = tf.zeros_like(graphs[0,:,:]) #(32,sample)
		state_s_t = tf.zeros_like(graphs[0,:,:]) #(32,sample)
		outputs = []

		for idx in range(0, self.num_time_steps):
			inputs = graphs[idx,:,:] #(32,sample)

			gate_g_ = tf.matmul(self.vars['lstm_weights_gx'], inputs) #[32,sample]
			gate_g_ += tf.matmul(self.vars['lstm_weights_gh'], state_h_t) 
			gate_g_ = tf.transpose(gate_g_)+self.vars['lstm_bias_g']
			gate_g = tf.transpose(tf.nn.tanh(gate_g_))

			gate_i_ = tf.matmul(self.vars['lstm_weights_ix'], inputs) #[32,sample]
			gate_i_ += tf.matmul(self.vars['lstm_weights_ih'], state_h_t)
			gate_i_ = tf.transpose(gate_i_)+self.vars['lstm_bias_i']
			gate_i = tf.transpose(tf.sigmoid(gate_i_))

			gate_f_ = tf.matmul(self.vars['lstm_weights_fx'], inputs) #[32,sample]
			gate_f_ += tf.matmul(self.vars['lstm_weights_fh'], state_h_t) 
			gate_f_ = tf.transpose(gate_f_)+self.vars['lstm_bias_f']
			gate_f = tf.transpose(tf.sigmoid(gate_f_))

			gate_o_ = tf.matmul(self.vars['lstm_weights_ox'], inputs) #[32,sample]
			gate_o_ += tf.matmul(self.vars['lstm_weights_oh'], state_h_t)
			gate_o_ = tf.transpose(gate_o_)+self.vars['lstm_bias_o']
			gate_o = tf.transpose(tf.sigmoid(gate_o_))

			state_s = tf.multiply(gate_g, gate_i) + tf.multiply(state_s_t, gate_f)
			state_h = tf.multiply(state_s, gate_o)

			state_h_t = state_h
			state_s_t = state_s
			outputs.append(state_h)

		outputs = tf.transpose(outputs, perm=[2,0,1])
		return outputs
