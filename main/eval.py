# encoding=utf-8

import tensorflow as tf
import data_input
from DeepTarget import DeepTarget

batch_num = 256
hidden_num = 30
step_num = 40
learning_rate = 0.01
display_num = 100


model = DeepTarget(step_num, hidden_num, learning_rate)
master = data_input.DataMaster()
mi_ae = model.mi_ae  # miRNA Autoencoder
m_ae = model.m_ae  # mRNA Autoencoder

saver = tf.train.Saver()
filename = 'log/model/deepTargetModel'
with tf.Session() as sess:
    saver.restore(sess, filename)

    print('begin to testing ALL-Model', len(master.test_data))
    accuracy = 0
    result_score = []
    result_label = []
    result_pred = []
    tp = 0
    for i, index in enumerate(range(0, len(master.test_data), batch_num)):
        x = master.test_data[index:index + batch_num]
        seq1, seq2 = master.transpose_nolabel(x)
        scores, predictions = sess.run([model.scores, model.predictions],
                                       {model.input_seq1: seq1, model.input_seq2: seq2})
        result_pred += list(predictions)
        result_score += list(scores)
    print(result_score)
    with open('result.txt', 'w') as result, open('test_data.txt', 'r') as file:
        for row, line in enumerate(file.readlines()):
            result.write(line.strip() + " " + "%.3f" % result_score[row][1] + "\n")
