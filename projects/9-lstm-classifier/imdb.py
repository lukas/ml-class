import numpy as np
import os

sep = os.path.sep

def load_imdb():
    X_train = []
    y_train = []

    path = os.path.join('aclImdb', 'train', 'pos', '')
    X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
    y_train.extend([1 for _ in range(12500)])

    path = os.path.join('aclImdb', 'train', 'neg', '')
    X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
    y_train.extend([0 for _ in range(12500)])

    X_test = []
    y_test = []

    path = os.path.join('aclImdb', 'test', 'pos', '')
    X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
    y_test.extend([1 for _ in range(12500)])

    path = os.path.join('aclImdb', 'test', 'neg', '')
    X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
    y_test.extend([0 for _ in range(12500)])

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    return (X_train, y_train), (X_test, y_test)