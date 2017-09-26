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
miRNA_auto_epoch = 1  # 13
mRNA_auto_epoch = 1  # 13
Pred_rnn_epoch = 1  # 7

model = DeepTarget(step_num, hidden_num, learning_rate)
master = data_input.DataMaster()
mi_ae = model.mi_ae  # miRNA Autoencoder
m_ae = model.m_ae  # mRNA Autoencoder

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('begin to training miRNA Autoencoder, sample number:', len(master.train_data))
    for j in range(miRNA_auto_epoch):
        print('miRNA_auto_epoch:%d/%d' % (j + 1, miRNA_auto_epoch))
        for i, index in enumerate(range(0, len(master.train_data), batch_num)):
            x = master.train_data[index:index + batch_num]
            seq1 = master.transpose(x)[0]
            loss_val, _ = sess.run([mi_ae.loss, mi_ae.train], {model.input_seq1: seq1})
            if i % display_num == 0:
                print("iter %d:" % (i + 1), loss_val)

    print('begin to training mRNA Autoencoder, sample number:', len(master.train_data))
    for j in range(mRNA_auto_epoch):
        print('mRNA_auto_epoch:%d/%d' % (j + 1, mRNA_auto_epoch))
        for i, index in enumerate(range(0, len(master.train_data), batch_num)):
            x = master.train_data[index:index + batch_num]
            seq2 = master.transpose(x)[1]
            loss_val, _ = sess.run([m_ae.loss, m_ae.train], {model.input_seq2: seq2})
            if i % display_num == 0:
                print("iter %d:" % (i + 1), loss_val)

    print('begin to training RNN-Model, training dataset:', len(master.train_data))
    for j in range(Pred_rnn_epoch):
        print('Pred_rnn_epoch:%d/%d' % (j + 1, Pred_rnn_epoch))
        for i, index in enumerate(range(0, len(master.train_data), batch_num)):
            x = master.train_data[index:index + batch_num]
            seq1, seq2, ys = master.transpose(x)
            accuracy, loss_val, _ = sess.run(
                [model.accuracy, model.mean_loss, model.optimizer],
                {model.input_seq1: seq1, model.input_seq2: seq2,
                 model.labels: ys})
            if i % display_num == 0:
                print("iter %d:" % (i + 1), loss_val, "accuracy %.3f:" % accuracy)

    # begin to save the model!
    filename = 'log/model/deepTargetModel'
    print("Saving model to %s..." % (filename))
    new_saver = tf.train.Saver()  # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, filename)
    print("Saved.")
