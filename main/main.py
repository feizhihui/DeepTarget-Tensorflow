# Basic libraries
import numpy as np
import tensorflow as tf
import data_input

tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from DeepTarget import DeepTarget

batch_num = 256
hidden_num = 30
step_num = 40
learning_rate = 0.01
display_num = 100

model = DeepTarget(step_num, hidden_num, learning_rate)
master = data_input.DataMaster()
mi_ae = model.mi_ae
m_ae = model.m_ae

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('begin to training miRNA Autoencoder', len(master.dataset))
    for i, index in enumerate(range(0, len(master.dataset), batch_num)):
        x = master.dataset[index:index + batch_num]
        seq1 = master.transpose(x)[0]
        loss_val, _ = sess.run([mi_ae.loss, mi_ae.train], {model.input_seq1: seq1})
        if i % display_num == 0:
            print("iter %d:" % (i + 1), loss_val)
            break

    print('begin to training mRNA Autoencoder', len(master.dataset))
    for i, index in enumerate(range(0, len(master.dataset), batch_num)):
        x = master.dataset[index:index + batch_num]
        seq2 = master.transpose(x)[1]
        loss_val, _ = sess.run([m_ae.loss, m_ae.train], {model.input_seq2: seq2})
        if i % display_num == 0:
            print("iter %d:" % (i + 1), loss_val)

    print('begin to training RNN-Model', len(master.train_data))
    for i, index in enumerate(range(0, len(master.train_data), batch_num)):
        x = master.dataset[index:index + batch_num]
        seq1, seq2, ys = master.transpose(x)
        accuracy, loss_val, _ = sess.run(
            [model.accuracy, model.mean_loss, model.optimizer],
            {model.input_seq1: seq1, model.input_seq2: seq2,
             model.labels: ys})
        if i % display_num == 0:
            print("iter %d:" % (i + 1), loss_val, "accuracy %.3f:" % accuracy)

    print('begin to testing ALL-Model', len(master.test_data))
    accuracy = 0
    for i, index in enumerate(range(0, len(master.test_data), batch_num)):
        x = master.test_data[index:index + batch_num]
        seq1, seq2, ys = master.transpose(x)
        accuracy += sess.run(model.accuracy, {model.input_seq1: seq1, model.input_seq2: seq2, model.labels: ys})
    print("final accuracy %.3f" % (accuracy / (i + 1)))
