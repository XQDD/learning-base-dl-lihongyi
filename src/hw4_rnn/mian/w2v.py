# w2v.py
# 這個 block 是用來訓練 word to vector 的 word embedding
# 注意！這個 block 在訓練 word to vector 時是用 cpu，可能要花到 10 分鐘以上
import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec

from src.hw4_rnn.mian.utils import *


def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    # 输入分好词的文本，输出一个稠密向量表示每一个词
    # 词向量的重要意义在于将自然语言转换成了计算机能够理解的向量
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model


if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data(path_prefix + "/" + 'training_label.txt')
    train_x_no_label = load_training_data(path_prefix + "/" + 'training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data(path_prefix + "/" + 'testing_data.txt')

    # model = train_word2vec(train_x + train_x_no_label + test_x)
    model = train_word2vec(train_x + test_x)

    print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save(os.path.join(path_prefix, 'w2v_all.model'))
