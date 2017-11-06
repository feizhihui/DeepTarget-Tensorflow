# Basic libraries
import numpy as np
import tensorflow as tf
import data_input
import os
import glob
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"


tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from DeepTarget import DeepTarget

batch_num = 256
hidden_num = 30
step_num = 40
learning_rate = 0.01
display_num = 100
miRNA_auto_epoch = 40 # 13
mRNA_auto_epoch = 40  # 13
Pred_rnn_epoch = 20

# delete previous pretrain model
if os.path.exists("log/model/checkpoint"):
    os.remove("log/model/checkpoint")
for filename in glob.glob("log/model/deepTargetModel.*"):
    os.remove(filename)

model = DeepTarget(step_num, hidden_num, learning_rate, True)
master = data_input.DataMaster()
mi_ae = model.mi_ae  # miRNA Autoencoder
m_ae = model.m_ae  # mRNA Autoencoder

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    miRNAseq = master.get_miRNA()
    print('begin to training miRNA Autoencoder, sample number:', len(miRNAseq))
    for j in range(miRNA_auto_epoch):
        print('miRNA_auto_epoch:%d/%d' % (j + 1, miRNA_auto_epoch))
        random.shuffle(miRNAseq)
        for i, index in enumerate(range(0, len(miRNAseq), batch_num)):
            x = miRNAseq[index:index + batch_num]
            seq1 = master.transpose_seq(x)
            loss_val, _ = sess.run([mi_ae.loss, mi_ae.train], {model.input_seq1: seq1})
            if i % display_num == 0:
                print(seq1)
                print("iter %d:" % (i + 1), loss_val)

    mRNAseq = master.get_mRNA()
    print('begin to training mRNA Autoencoder, sample number:', len(mRNAseq))
    for j in range(mRNA_auto_epoch):
        print('mRNA_auto_epoch:%d/%d' % (j + 1, mRNA_auto_epoch))
        random.shuffle(mRNAseq)
        for i, index in enumerate(range(0, len(mRNAseq), batch_num)):
            x = mRNAseq[index:index + batch_num]
            seq2 = master.transpose_seq(x)
            loss_val, _ = sess.run([m_ae.loss, m_ae.train], {model.input_seq2: seq2})
            if i % display_num == 0:
                print(seq2)
                print("iter %d:" % (i + 1), loss_val)

    print('begin to training RNN-Model, training dataset:', len(master.train_data))
    for j in range(Pred_rnn_epoch):
        print('Pred_rnn_epoch:%d/%d' % (j + 1, Pred_rnn_epoch))
        random.shuffle(master.train_data)
        for i, index in enumerate(range(0, len(master.train_data), batch_num)):
            x = master.train_data[index:index + batch_num]
            seq1, seq2, ys = master.transpose(x)
            accuracy, loss_val, _ = sess.run(
                [model.accuracy, model.mean_loss, model.optimizer],
                {model.input_seq1: seq1, model.input_seq2: seq2,
                 model.labels: ys})
            if i % display_num == 0:
                # print(seq1)
                # print(seq2)
                print("iter %d:" % (i + 1), loss_val, "accuracy %.3f:" % accuracy)

    # begin to save the model!
    filename = 'log/model/deepTargetModel'
    print("Saving model to %s..." % (filename))
    new_saver = tf.train.Saver()  # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, filename)
    print("Saved.")
