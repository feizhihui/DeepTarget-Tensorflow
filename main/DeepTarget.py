# encoding=utf-8

import tensorflow as tf
from LSTMAutoencoder import LSTMAutoencoder
from tensorflow.contrib import layers


class DeepTarget(object):
    def __init__(self, step_num=40, hidden_num=30, learning_rate=0.01, autoencoder=False):
        # placeholder
        self.labels = tf.placeholder(tf.int32, [None])
        self.input_seq1 = tf.placeholder(tf.int32, [None, step_num])
        input_seq1_1hot = tf.one_hot(self.input_seq1, depth=4, dtype=tf.float32)

        input_seq1_emb = layers.fully_connected(tf.reshape(input_seq1_1hot, [-1, 4]), 4,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                activation_fn=None)
        input_seq1_emb = tf.reshape(input_seq1_emb, [-1, step_num, 4])

        self.input_seq2 = tf.placeholder(tf.int32, [None, step_num])
        input_seq2_1hot = tf.one_hot(self.input_seq2, depth=4, dtype=tf.float32)
        input_seq2_emb = layers.fully_connected(tf.reshape(input_seq2_1hot, [-1, 4]), 4,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                activation_fn=None)
        input_seq2_emb = tf.reshape(input_seq2_emb, [-1, step_num, 4])

        # static_rnn input format is a list
        p_inputs_seq1 = [tf.squeeze(t, [1]) for t in tf.split(input_seq1_emb, step_num, 1)]
        p_inputs_seq2 = [tf.squeeze(t, [1]) for t in tf.split(input_seq2_emb, step_num, 1)]

        if autoencoder==True:
            with tf.name_scope("autoencoder"):
                mi_cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
                m_cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
                with tf.variable_scope('miRNA_autoencoder'):
                    self.mi_ae = LSTMAutoencoder(step_num, hidden_num, p_inputs_seq1, cell=mi_cell,
                                                 decode_without_input=True)
                with tf.variable_scope('mRNA_autoencoder'):
                    self.m_ae = LSTMAutoencoder(step_num, hidden_num, p_inputs_seq2, cell=m_cell, decode_without_input=True)

                m_decoder = tf.stack(self.m_ae.z_codes, 1)

                mi_decoder = tf.stack(self.mi_ae.z_codes, 1)

                m_decoder = tf.reshape(m_decoder, [-1, step_num, 1, hidden_num])
                mi_decoder = tf.reshape(mi_decoder, [-1, step_num, 1, hidden_num])
                h_concate = tf.concat([m_decoder, mi_decoder], axis=2)
                h_concate = tf.reshape(h_concate, [-1, step_num, 2 * hidden_num])  # hidden_num
        else:
            m_decoder = tf.reshape(input_seq1_1hot, [-1, step_num, 1, 4])
            mi_decoder = tf.reshape(input_seq2_1hot, [-1, step_num, 1, 4])
            h_concate = tf.concat([m_decoder, mi_decoder], axis=2)
            h_concate = tf.reshape(h_concate, [-1, step_num, 2 * 4])  # hidden_num


        # cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
        cell = tf.contrib.rnn.GRUCell(hidden_num)
        outputs, state = tf.nn.dynamic_rnn(cell, h_concate, dtype=tf.float32)
        output = outputs[:, -1, :]

        weights = tf.Variable(tf.truncated_normal([hidden_num, 2], stddev=0.1))
        biase = tf.Variable(tf.truncated_normal([2], stddev=0.1))
        logits = tf.matmul(output, weights) + biase
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        self.scores = tf.nn.softmax(logits)
        self.mean_loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.mean_loss)
        self.predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.labels), tf.float32))


if __name__ == '__main__':
    print(DeepTarget())
