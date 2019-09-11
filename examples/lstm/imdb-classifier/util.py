import os
import subprocess
import numpy as np


def load_imdb():
    if not os.path.exists("./aclImdb"):
        print("Downloading imdb dataset...")
        subprocess.check_output(
            "curl -SL http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz | tar xz", shell=True)

    if not os.path.exists('./aclImdb/cache.npz'):
        X_train = []
        y_train = []

        path = './aclImdb/train/pos/'
        X_train.extend([open(path + f).read()
                        for f in os.listdir(path) if f.endswith('.txt')])
        y_train.extend([1 for _ in range(12500)])

        path = './aclImdb/train/neg/'
        X_train.extend([open(path + f).read()
                        for f in os.listdir(path) if f.endswith('.txt')])
        y_train.extend([0 for _ in range(12500)])

        X_test = []
        y_test = []

        path = './aclImdb/test/pos/'
        X_test.extend([open(path + f).read()
                       for f in os.listdir(path) if f.endswith('.txt')])
        y_test.extend([1 for _ in range(12500)])

        path = './aclImdb/test/neg/'
        X_test.extend([open(path + f).read()
                       for f in os.listdir(path) if f.endswith('.txt')])
        y_test.extend([0 for _ in range(12500)])

        np.savez('./aclImdb/cache.npz', X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test)

    cached = np.load('./aclImdb/cache.npz')
    return (cached['X_train'], cached['y_train']), (cached['X_test'], cached['y_test'])


if __name__ == '__main__':
    load_imdb()
