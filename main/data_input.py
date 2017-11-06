# encoding=utf-8
import random
import numpy as np

training_split_rate = 0.90
step_num = 40

code = {'A': 0, 'G': 1, 'U': 2, 'C': 3}


class DataMaster(object):
    def __init__(self, trainpath, testpath):
        self.train_data = []
        with open(trainpath, 'r') as file:
            for line in file.readlines():
                field = line.split()
                miRNA = field[2]
                mRNA = field[3]
                label = int(field[4])
                self.train_data.append((miRNA, mRNA, label))
        random.shuffle(self.train_data)
        self.test_data = []
        with open(testpath, 'r') as file:
            for line in file.readlines():
                field = line.split()
                miRNA = field[2]
                mRNA = field[3]
                self.test_data.append((miRNA, mRNA))

    def transpose(self, xs):
        seq1s = []
        seq2s = []
        ys = []
        for x in xs:
            (miRNA, mRNA, y) = x
            seq1 = np.ones((step_num), dtype=np.int32) * (-1)
            seq2 = np.ones((step_num), dtype=np.int32) * (-1)
            for i, c in enumerate(miRNA):
                seq1[i] = code[c]
            for i, c in enumerate(mRNA):
                seq2[i] = code[c]
            seq1s.append(seq1[::-1])
            seq2s.append(seq2[::-1])
            ys.append(y)
        return np.array(seq1s, dtype=np.int32), np.array(seq2s, dtype=np.int32), ys

    def transpose_nolabel(self, xs):
        seq1s = []
        seq2s = []
        for x in xs:
            (miRNA, mRNA) = x
            seq1 = np.ones((step_num), dtype=np.int32) * (-1)
            seq2 = np.ones((step_num), dtype=np.int32) * (-1)
            for i, c in enumerate(miRNA):
                seq1[i] = code[c]
            for i, c in enumerate(mRNA):
                seq2[i] = code[c]
            seq1s.append(seq1[::-1])
            seq2s.append(seq2[::-1])
        return np.array(seq1s, dtype=np.int32), np.array(seq2s, dtype=np.int32)

    def get_miRNA(self):
        miRNAs = set()
        for sample in self.train_data:
            miRNAs.add(sample[0])
        return list(miRNAs)

    def get_mRNA(self):
        mRNAs = set()
        for sample in self.train_data:
            mRNAs.add(sample[1])
        return list(mRNAs)

    def transpose_seq(self, seq_list):
        seq1s = []
        for seq in seq_list:
            seq1 = np.ones((step_num), dtype=np.int32) * (-1)
            for i, c in enumerate(seq):
                seq1[i] = code[c]
            seq1s.append(seq1[::-1])
        return np.array(seq1s, dtype=np.int32)
