# encoding=utf-8
import random
import numpy as np

training_split_rate = 0.90
step_num = 40

code = {'A': 0, 'G': 1, 'U': 2, 'C': 3}


class DataMaster(object):
    def __init__(self):
        dataset = []
        with open('result_OK.txt', 'r') as file:
            for line in file.readlines():
                field = line.split()
                miRNA = field[2]
                mRNA = field[3]
                # label=int(label[4])
                label = random.randint(0, 1)
                dataset.append((miRNA, mRNA, label))

        random.shuffle(dataset)

        self.dataset = dataset
        separat = int(len(dataset) * training_split_rate)
        self.train_data = dataset[:separat]
        self.test_data = dataset[separat:]

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
