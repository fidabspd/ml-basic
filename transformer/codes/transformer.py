import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class MultiHeadAttentionLayer(Layer):
    
    def __init__(self, hidden_dim, n_heads, dropout_ratio, dtype=tf.float32):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hidden_dim % n_heads == 0, f'hidden_dim must be multiple of n_heads.'

        self.dtype = dtype
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim//n_heads

        # in_shape: [batch_size, seq_len, hidden_dim]
        # out_shape: [batch_size, seq_len, hidden_dim]
        self.fc_q = Dense(hidden_dim)
        self.fc_k = Dense(hidden_dim)
        self.fc_v = Dense(hidden_dim)

        self.fc_o = Dense(hidden_dim)

        self.dropout = Dropout(dropout_ratio)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.n_heads, self.head_dim))
        # [batch_size, seq_len, n_heads, head_dim]
        splits = tf.transpose(inputs, perm=[0, 2, 1, 3])
        return splits  # [batch_size, n_heads, seq_len, head_dim] -> n_heads를 앞으로

    def scaled_dot_product_attention(self, query, key, value, mask):
        key_t = tf.transpose(key, perm=[0, 1, 3, 2])
        energy = tf.matmul(query, key_t) / tf.sqrt(tf.cast(self.head_dim, self.dtype))
        if mask is not None:
            energy = tf.where(mask==0, -1e10, energy)
        attention = tf.nn.softmax(energy, axis=-1)  # axis=-1 은 key의 문장 위치
        x = tf.matmul(self.dropout(attention), value)
        return x, attention  # attention 시각화에 쓸 수 있음

    def call(self, inputs):
        query, key, value, mask = \
            inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        x, attention = scaled_dot_product_attention(query, key, value, mask)
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [batch_size, seq_len, n_heads, head_dim]
        x = tf.reshape(x, shape=(batch_size, -1, self.hidden_dim))  # [batch_size, seq_len, hidden_dim]

        outputs = self.fc_o(x)
        
        return outputs, attention
    