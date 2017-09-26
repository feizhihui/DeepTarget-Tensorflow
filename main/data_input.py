# encoding=utf-8
import random
import numpy as np

training_split_rate = 0.90
step_num = 40

code = {'A': 0, 'G': 1, 'U': 2, 'C': 3}


class DataMaster(object):
    def __init__(self):
        self.train_data = []
        with open('train_data.txt', 'r') as file:
            for line in file.readlines():
                field = line.split()
                miRNA = field[2]
                mRNA = field[3]
                label = int(field[4])
                self.train_data.append((miRNA, mRNA, label))
        random.shuffle(self.train_data)
        self.test_data = []
        with open('test_data.txt', 'r') as file:
            for line in file.readlines():
                field = line.split()
                miRNA = field[2]
                mRNA = field[3]
                self.test_data.append((miRNA, mRNA))

    def transpose(self, xs):
        seq1s = []
        seq2s = []
        ys = []
        batch_size = len(xs)
        for x in xs:
            (miRNA, mRNA, y) = x
            seq1 = np.zeros((batch_size, step_num), dtype=np.int32)
            seq2 = np.zeros((batch_size, step_num), dtype=np.int32)
            padding1 = padding2 = 0
            if len(miRNA) < step_num:
                padding1 = step_num - len(miRNA)
            if len(mRNA) < step_num:
                padding2 = step_num - len(mRNA)
            for i, c in enumerate(miRNA):
                seq1[padding1 + i] = code[c]
            for i, c in enumerate(mRNA):
                seq2[padding2 + i] = code[c]
            seq1s.append(seq1)
            seq2s.append(seq2)
            ys.append(y)
        return np.array(seq1, dtype=np.int32), np.array(seq2, dtype=np.int32), ys

    def transpose_nolabel(self, xs):
        seq1s = []
        seq2s = []
        batch_size = len(xs)
        for x in xs:
            (miRNA, mRNA) = x
            seq1 = np.zeros((batch_size, step_num), dtype=np.int32)
            seq2 = np.zeros((batch_size, step_num), dtype=np.int32)
            padding1 = padding2 = 0
            if len(miRNA) < step_num:
                padding1 = step_num - len(miRNA)
            if len(mRNA) < step_num:
                padding2 = step_num - len(mRNA)
            for i, c in enumerate(miRNA):
                seq1[padding1 + i] = code[c]
            for i, c in enumerate(mRNA):
                seq2[padding2 + i] = code[c]
            seq1s.append(seq1)
            seq2s.append(seq2)
        return np.array(seq1, dtype=np.int32), np.array(seq2, dtype=np.int32)
