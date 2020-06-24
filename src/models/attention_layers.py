#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from .inits import glorot, zeros
from .layers import Layer


class AttentionHead(Layer):
    """
    One attention head over 1st-order neighbors
    Adapted from https://github.com/PetarV-/GAT
    """

    def __init__(self, input_dim, attend_head_units, neigh_input_dim=None,
                 dropout=0, bias=False, act=tf.nn.leaky_relu,
                 name=None, concat=False, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['before_attend_weights'] = glorot([input_dim, attend_head_units], name='before_attend_weights')
            self.vars['attend_weights_self'] = glorot([attend_head_units, 1], name='attend_weights_self')
            self.vars['attend_weights_neigh'] = glorot([attend_head_units, 1], name='attend_weights_neigh')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = attend_head_units
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        """
        The shape of self_vecs is unknown.
        Since tf.layers.conv1d cannot be applied to unknown shape tensor,
        here we use dense layer instead to implement self-attention.
        :param inputs:
        :return:
        """
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [batch_size * num_neighbors, neigh_input_dim]
        neigh_vecs_reshaped = tf.reshape(neigh_vecs, (batch_size * num_neighbors, self.neigh_input_dim))

        # [batch_size * num_neighbors, output_dim]
        neigh_vecs_transformed = tf.matmul(neigh_vecs_reshaped, self.vars['before_attend_weights'])
        # [batch_size, output_dim]
        self_vecs_transformed = tf.matmul(self_vecs, self.vars['before_attend_weights'])

        # [batch_size * num_neighbors, 1]
        neigh_f1 = tf.matmul(neigh_vecs_transformed, self.vars['attend_weights_neigh'])
        # [batch_size, 1]
        self_f1 = tf.matmul(self_vecs_transformed, self.vars['attend_weights_self'])

        # [batch_size, num_neighbors]
        # TODO: CHECK if the self vector in the neighbor vectors
        logits = self_f1 + tf.reshape(neigh_f1, [batch_size, num_neighbors])
        # [batch_size, num_neighbors]
        attend_coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))

        # [batch_size, num_neighbors, self.output_dim]
        neigh_vecs_h = tf.reshape(neigh_vecs_transformed, [batch_size, num_neighbors, self.output_dim])

        # calculate the final output as the linear combination of attend coefficients and features
        # [batch_size, 1, self.output_dim] =
        # [batch_size, 1, self.output_dim] * [batch_size, num_neighbors, self.output_dim]
        output = tf.matmul(tf.expand_dims(attend_coefs, axis=1), neigh_vecs_h)
        # [batch_size, self.output_dim]
        output = tf.reshape(output, [-1, self.output_dim])

        return self.act(output)


class TemporalAttentionLayer(Layer):
    """
    Temporal attention layer with google's multi-head attention.
    """

    def __init__(self, input_dim, n_heads, num_time_steps, attn_drop, residual=False, bias=True,
                 use_position_embedding=True, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

        self.bias = bias
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.attn_drop = attn_drop
        self.attn_wts_means = []
        self.attn_wts_vars = []
        self.residual = residual
        self.input_dim = input_dim

        xavier_init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(self.name + '_vars'):
            if use_position_embedding:
                self.vars['position_embeddings'] = tf.get_variable('position_embeddings', dtype=tf.float32,
                                                                   shape=[self.num_time_steps, input_dim], initializer=xavier_init)  # [T, F]

            self.vars['Q_embedding_weights'] = tf.get_variable('Q_embedding_weights', dtype=tf.float32,
                                                               shape=[input_dim, input_dim], initializer=xavier_init)  # [F, F]
            self.vars['K_embedding_weights'] = tf.get_variable('K_embedding_weights', dtype=tf.float32,
                                                               shape=[input_dim, input_dim], initializer=xavier_init)  # [F, F]
            self.vars['V_embedding_weights'] = tf.get_variable('V_embedding_weights', dtype=tf.float32,
                                                               shape=[input_dim, input_dim], initializer=xavier_init)  # [F, F]
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # inputs_reshaped = tf.concat([tf.expand_dims(ele, axis=1) for ele in inputs], axis=1)
        inputs_reshaped = inputs # for vectorization

        # 1: Add position embeddings to input
        position_inputs = tf.tile(tf.expand_dims(tf.range(self.num_time_steps), 0), [tf.shape(inputs_reshaped)[0], 1])
        position_embeddings = tf.nn.embedding_lookup(self.vars['position_embeddings'], position_inputs)
        inputs_temporal = inputs_reshaped + position_embeddings

        # 2: Query, Key based multi-head self attention.
        q = tf.tensordot(inputs_temporal, self.vars['Q_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        k = tf.tensordot(inputs_temporal, self.vars['K_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        v = tf.tensordot(inputs_temporal, self.vars['V_embedding_weights'], axes=[[2], [0]])  # [N, T, F]

        # 3: Split, concat and scale.
        q_ = tf.concat(tf.split(q, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        k_ = tf.concat(tf.split(k, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        v_ = tf.concat(tf.split(v, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]

        outputs = tf.matmul(q_, tf.transpose(k_, [0, 2, 1]))  # [hN, T, T]
        outputs = outputs / (self.input_dim ** 0.5)

        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = tf.ones_like(outputs[0, :, :])  # [T, T]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense() # [T, T] for tensorflow140
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # [hN, T, T]
        padding = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), padding, outputs)  # [h*N, T, T]
        outputs = tf.nn.softmax(outputs)  # Masked attention.
        self.attn_wts_all = outputs

        # 5: Dropout on attention weights.
        outputs = tf.layers.dropout(outputs, rate=self.attn_drop)
        outputs = tf.matmul(outputs, v_)  # [hN, T, C/h]

        split_outputs = tf.split(outputs, self.n_heads, axis=0)
        outputs = tf.concat(split_outputs, axis=-1)
        # only return the last snapshot embeddings
        final_outputs = tf.squeeze(tf.slice(outputs, [0, tf.shape(outputs)[1] - 1, 0], [-1, -1, -1]))

        return final_outputs


class EdgeAttentionLayer(Layer):
    """
    edge attention layer. A FC layer to calculate attention weights.
    The final output is linear combination of attention weights and inputs.
    """
    def __init__(self, input_dim, atten_vec_size, attn_drop, residual=False,
                 bias=True, act=tf.tanh, name=None,
                 **kwargs):
        super(EdgeAttentionLayer, self).__init__(**kwargs)

        self.bias = bias
        self.act = act
        self.attn_drop = attn_drop
        self.residual = residual
        self.input_dim = input_dim
        self.atten_vec_size = atten_vec_size

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['nonlinear_weights'] = glorot([input_dim, atten_vec_size], name='nonlinear_weights')
            self.vars['attention_vector'] = glorot([atten_vec_size, 1], name='attention_vector')

            if self.bias:
                self.vars['nonlinear_bias'] = zeros([self.atten_vec_size], name='nonlinear_bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        multi_val = tf.tensordot(inputs, self.vars['nonlinear_weights'], axes=1)
        if self.bias:
            multi_val = multi_val + self.vars['nonlinear_bias']

        multi_val = self.act(multi_val)
        e = tf.tensordot(multi_val, self.vars['attention_vector'], axes=1)

        alphas = tf.nn.softmax(e)

        # outputs = tf.reduce_sum(inputs * alphas, 1)
        outputs = tf.reduce_sum(inputs * alphas, -2)

        return outputs


class AttentionAggregatorVectorized(Layer):
    """
    Multi-head Attention which Aggregates via attention over 1st-order neighbors
    Every head outputs vector with output_dim, then concatenates all head's outputs,
    therefore it needs another FC to transform the concatenated vector to output_dim.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0, bias=False, act=tf.nn.relu,
                 name=None, concat=False, num_heads=8, **kwargs):
        super(AttentionAggregatorVectorized, self).__init__(**kwargs)

        attend_head_units = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.act = act

        print("Hierarchical attention aggregator using {} heads.".format(self.num_heads))

        if name is not None:
            name = '/' + name
        else:
            name = ''

        self.attention_heads = []
        for _ in range(self.num_heads):
            self.attention_heads.append(AttentionHeadVectorized(input_dim=input_dim, attend_head_units=output_dim,
                                                      neigh_input_dim=neigh_input_dim, dropout=dropout,
                                                      bias=bias, act=tf.nn.leaky_relu))

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([attend_head_units * self.num_heads, output_dim], name='weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

    def _call(self, inputs):
        attention_results = []
        for i in range(self.num_heads):
            attention_results.append(self.attention_heads[i](inputs))

        h = tf.concat(attention_results, axis=-1)

        # make the final output is output_dim
        # output = tf.matmul(h, self.vars['weights'])
        output = tf.tensordot(h, self.vars['weights'], axes=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class AttentionHeadVectorized(Layer):
    """
    One attention head over 1st-order neighbors
    Adapted from https://github.com/PetarV-/GAT
    """

    def __init__(self, input_dim, attend_head_units, neigh_input_dim=None,
                 dropout=0, bias=False, act=tf.nn.leaky_relu,
                 name=None, concat=False, **kwargs):
        super(AttentionHeadVectorized, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['before_attend_weights'] = glorot([input_dim, attend_head_units], name='before_attend_weights')
            self.vars['attend_weights_self'] = glorot([attend_head_units, 1], name='attend_weights_self')
            self.vars['attend_weights_neigh'] = glorot([attend_head_units, 1], name='attend_weights_neigh')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = attend_head_units
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        """
        The shape of self_vecs is unknown.
        Since tf.layers.conv1d cannot be applied to unknown shape tensor,
        here we use dense layer instead to implement self-attention.
        :param inputs:
        :return:
        """
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        num_neighbors = dims[1]
        num_graphs = dims[2]
        # [batch_size * num_neighbors * num_graphs, neigh_input_dim]
        neigh_vecs_reshaped = tf.reshape(neigh_vecs, (batch_size * num_neighbors * num_graphs, self.neigh_input_dim))

        # [batch_size * num_neighbors * num_graphs, output_dim]
        neigh_vecs_transformed = tf.matmul(neigh_vecs_reshaped, self.vars['before_attend_weights'])
        # neigh_vecs_transformed = tf.tensordot(neigh_vecs_reshaped, self.vars['before_attend_weights'], axes=1)
        # [batch_size, output_dim]
        # self_vecs_transformed = tf.matmul(self_vecs, self.vars['before_attend_weights'])
        # [batch_size, num_graphs, output_dim]
        self_vecs_transformed = tf.tensordot(self_vecs, self.vars['before_attend_weights'], axes=1)

        # [batch_size * num_neighbors * num_graphs, 1]
        neigh_f1 = tf.matmul(neigh_vecs_transformed, self.vars['attend_weights_neigh'])
        # [batch_size, 1]
        # self_f1 = tf.matmul(self_vecs_transformed, self.vars['attend_weights_self'])
        # [batch_size, num_graphs, 1]
        self_f1 = tf.tensordot(self_vecs_transformed, self.vars['attend_weights_self'], axes=1)

        # [batch_size, num_neighbors]
        # TODO: CHECK if the self vector in the neighbor vectors
        # logits = self_f1 + tf.reshape(neigh_f1, [batch_size, num_neighbors, num_graphs])
        # [batch_size, num_neighbors， num_graphs]
        logits = tf.reshape(self_f1, [batch_size, 1, num_graphs]) + \
                 tf.reshape(neigh_f1, [batch_size, num_neighbors, num_graphs])
        # [batch_size, num_neighbors]
        attend_coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))

        # [batch_size, num_neighbors, self.output_dim]
        # neigh_vecs_h = tf.reshape(neigh_vecs_transformed, [batch_size, num_neighbors, self.output_dim])
        neigh_vecs_h = tf.reshape(neigh_vecs_transformed, [batch_size, num_neighbors, num_graphs, self.output_dim])

        # calculate the final output as the linear combination of attend coefficients and features
        # [batch_size, 1, self.output_dim] =
        # [batch_size, 1, self.output_dim] * [batch_size, num_neighbors, self.output_dim]
        # output = tf.matmul(tf.expand_dims(attend_coefs, axis=1), neigh_vecs_h)
        output = tf.matmul(tf.expand_dims(tf.transpose(attend_coefs, perm=[0, 2, 1]), axis=2),
                           tf.transpose(neigh_vecs_h, perm=[0, 2, 1, 3]))
        # [batch_size, self.output_dim]
        # output = tf.reshape(output, [-1, self.output_dim])
        output = tf.squeeze(output)

        return self.act(output)

