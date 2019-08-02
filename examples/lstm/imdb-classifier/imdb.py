import os
import subprocess


def load_imdb():
    X_train = []
    y_train = []
    if not os.path.exists('aclImdb'):
        print("Downloading IMDB")
        subprocess.check_output(
            "curl -SL http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz | tar xz", shell=True)

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

    return (X_train, y_train), (X_test, y_test)
