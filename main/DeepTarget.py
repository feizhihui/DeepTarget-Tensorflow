# encoding=utf-8

import tensorflow as tf
from LSTMAutoencoder import LSTMAutoencoder


class DeepTarget(object):
    def __init__(self, step_num=40, hidden_num=30, learning_rate=0.01):
        # placeholder list
        self.input_seq1 = tf.placeholder(tf.int32, [None, step_num])
        input_seq1_1hot = tf.one_hot(self.input_seq1, depth=3, dtype=tf.float32)
        self.input_seq2 = tf.placeholder(tf.int32, [None, step_num])
        input_seq2_1hot = tf.one_hot(self.input_seq2, depth=3, dtype=tf.float32)

        self.labels = tf.placeholder(tf.int32, [None])
        p_inputs_seq1 = [tf.squeeze(t, [1]) for t in tf.split(input_seq1_1hot, step_num, 1)]
        p_inputs_seq2 = [tf.squeeze(t, [1]) for t in tf.split(input_seq2_1hot, step_num, 1)]

        mi_cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
        m_cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
        with tf.variable_scope('miRNA_autoencoder'):
            self.mi_ae = LSTMAutoencoder(step_num, hidden_num, p_inputs_seq1, cell=mi_cell, decode_without_input=True)
        with tf.variable_scope('mRNA_autoencoder'):
            self.m_ae = LSTMAutoencoder(step_num, hidden_num, p_inputs_seq2, cell=m_cell, decode_without_input=True)

        m_decoder = self.m_ae.z_codes[-1]
        mi_decoder = self.mi_ae.z_codes[-1]  # batch_size*hidden_num

        m_decoder = tf.reshape(m_decoder, [-1, 1])
        mi_decoder = tf.reshape(mi_decoder, [-1, 1])
        h_concate = tf.concat([m_decoder, mi_decoder], axis=1)
        h_concate = tf.reshape(h_concate, [-1, hidden_num, 2])

        cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
        outputs, state = tf.nn.dynamic_rnn(cell, h_concate, dtype=tf.float32)
        output = outputs[:, -1, :]

        weights = tf.Variable(tf.truncated_normal([hidden_num, 2], stddev=0.1))
        biase = tf.Variable(tf.truncated_normal([2], stddev=0.1))
        logits = tf.matmul(output, weights) + biase
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        self.mean_loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.mean_loss)
        self.predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.labels), tf.float32))


if __name__ == '__main__':
    print(DeepTarget())
