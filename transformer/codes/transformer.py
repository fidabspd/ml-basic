import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class MultiHeadAttentionLayer(Layer):
    
    def __init__(self, hidden_dim, n_heads, dropout_ratio, dtype=tf.float32, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)

        assert hidden_dim % n_heads == 0, f'hidden_dim must be multiple of n_heads.'
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim//n_heads
        self.scale = tf.sqrt(tf.cast(self.head_dim, dtype))

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
        energy = tf.matmul(query, key_t) / self.scale
        if mask is not None:
            energy = tf.where(mask==0, -1e10, energy)
        attention = tf.nn.softmax(energy, axis=-1)  # axis=-1 은 key의 문장 위치
        x = tf.matmul(self.dropout(attention), value)
        return x, attention  # attention 시각화에 쓸 수 있음

    def call(self, query, key, value, mask=None):

        batch_size = tf.shape(query)[0]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        x, attention = self.scaled_dot_product_attention(query, key, value, mask)
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [batch_size, seq_len, n_heads, head_dim]
        x = tf.reshape(x, shape=(batch_size, -1, self.hidden_dim))  # [batch_size, seq_len, hidden_dim]

        outputs = self.fc_o(x)
        
        return outputs, attention


class PositionwiseFeedforwardLayer(Layer):

    def __init__(self, pf_dim, hidden_dim, dropout_ratio, **kwargs):
        super(PositionwiseFeedforwardLayer, self).__init__(**kwargs)
        self.fc_0 = Dense(pf_dim, activation='relu')
        self.fc_1 = Dense(hidden_dim)
        self.dropout = Dropout(dropout_ratio)

    def call(self, inputs):
        x = self.fc_0(inputs)
        x = self.dropout(x)
        outputs = self.fc_1(x)
        return outputs


class EncoderLayer(Layer):

    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, epsilon=1e-5, dtype=tf.float32, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio

        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, dtype)
        self.self_attn_norm = LayerNormalization(epsilon=epsilon)
        self.pos_feedforward = PositionwiseFeedforwardLayer(pf_dim, hidden_dim, dropout_ratio)
        self.pos_ff_norm = LayerNormalization(epsilon=epsilon)

        self.dropout = Dropout(dropout_ratio)

    def call(self, inputs, mask=None):
        attn_outputs, _ = self.self_attention(inputs, inputs, inputs, mask)
        attn_outputs = self.dropout(attn_outputs)
        attn_outputs = self.self_attn_norm(inputs+attn_outputs)  # residual connection

        ff_outputs = self.pos_feedforward(attn_outputs)
        ff_outputs = self.dropout(ff_outputs)
        ff_outputs = self.pos_ff_norm(attn_outputs+ff_outputs)  # residual connection

        return ff_outputs


class DecoderLayer(Layer):

    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, epsilon=1e-5, dtype=tf.float32, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, dtype)
        self.self_attn_norm = LayerNormalization(epsilon=epsilon)
        self.encd_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, dtype)
        self.encd_attn_norm = LayerNormalization(epsilon=epsilon)
        self.pos_feedforward = PositionwiseFeedforwardLayer(pf_dim, hidden_dim, dropout_ratio)
        self.pos_ff_norm = LayerNormalization(epsilon=epsilon)

        self.dropout = Dropout(dropout_ratio)

    def call(self, target, encd, target_mask, encd_mask):
        self_attn_outputs, _ = self.self_attention(target, target, target, target_mask)
        self_attn_outputs = self.dropout(self_attn_outputs)
        self_attn_outputs = self.self_attn_norm(target+self_attn_outputs)

        encd_attn_outputs, attention = self.encd_attention(target, encd, encd, encd_mask)
        encd_attn_outputs = self.dropout(encd_attn_outputs)
        encd_attn_outputs = self.encd_attn_norm(self_attn_outputs+encd_attn_outputs)

        outputs = self.pos_feedforward(encd_attn_outputs)
        outputs = self.dropout(outputs)
        outputs = self.pos_ff_norm(outputs)

        return outputs, attention


class Encoder(Model):

    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim,
                 dropout_ratio, max_seq_len=100, dtype=tf.float32, **kwargs):
        # input_dim = len(vocab)
        super(Encoder, self).__init__(**kwargs)
        self.scale = tf.sqrt(tf.cast(hidden_dim, dtype))

        self.tok_emb = Embedding(input_dim, hidden_dim)
        self.pos_emb = Embedding(max_seq_len, hidden_dim)

        self.encd_stk = [
            EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, dtype=dtype)
            for _ in range(n_layers)
        ]

        self.dropout = Dropout(dropout_ratio)

    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        pos = tf.tile(tf.expand_dims(tf.range(seq_len), axis=0), (batch_size, 1))

        emb = self.tok_emb(x) * self.scale + self.pos_emb(pos)
        outputs = self.dropout(emb)

        for layer in self.encd_stk:
            outputs = layer(outputs, mask)

        return outputs


class Decoder(Model):
    
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim,
                 dropout_ratio, max_seq_len=100, dtype=tf.float32, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.scale = tf.sqrt(tf.cast(hidden_dim, dtype))

        self.tok_emb = Embedding(output_dim, hidden_dim)
        self.pos_emb = Embedding(max_seq_len, hidden_dim)

        self.decd_stk = [
            DecoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, dtype=dtype)
            for _ in range(n_layers)
        ]

        self.fc_out = Dense(output_dim)

        self.dropout = Dropout(dropout_ratio)

    def call(self, target, encd, target_mask, encd_mask):
        batch_size = tf.shape(target)[0]
        seq_len = tf.shape(target)[1]

        pos = tf.tile(tf.expand_dims(tf.range(seq_len), axis=0), (batch_size, 1))

        emb = self.tok_emb(target) * self.scale + self.pos_emb(pos)
        outputs = self.dropout(emb)

        for layer in self.decd_stk:
            outputs, attention = layer(target, encd, target_mask, encd_mask)

        outputs = self.fc_out(outputs)

        return outputs, attention
        