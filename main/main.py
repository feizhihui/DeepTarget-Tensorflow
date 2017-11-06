# Basic libraries
import numpy as np
import tensorflow as tf
import data_input
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from DeepTarget import DeepTarget

batch_num = 256
hidden_num = 30
step_num = 40
learning_rate = 0.01
display_num = 100
miRNA_auto_epoch = 40
mRNA_auto_epoch = 40
Pred_rnn_epoch = 20

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
                    print("iter %d:" % (i + 1), loss_val, "accuracy %.3f:" % accuracy)

        print('begin to testing ALL-Model', len(master.test_data))
        result_score = []
        result_pred = []
        for i, index in enumerate(range(0, len(master.test_data), batch_num)):
            x = master.test_data[index:index + batch_num]
            seq1, seq2 = master.transpose_nolabel(x)
            scores, predictions = sess.run([model.scores, model.predictions],
                                           {model.input_seq1: seq1, model.input_seq2: seq2})
            result_pred += list(predictions)
            result_score += list(scores)
        print("result_score=", len(result_score))
        with open(datapath + '/result_n%d.txt' % (datalabel + 1), 'w') as result, open(testpath, 'r') as file:
            for row, line in enumerate(file.readlines()):
                result.write(line.strip() + " " + "%.8f" % result_score[row][1] + "\n")


for i in range(10):
    datapath = "datasets/deeptarget%d" % (i + 1)
    for j in range(10):
        trainpath = datapath + "/trainset%d.txt" % (j + 1)
        testpath = datapath + "/testset%d.txt" % (j + 1)
        train_and_test(trainpath, testpath, datapath, j)
