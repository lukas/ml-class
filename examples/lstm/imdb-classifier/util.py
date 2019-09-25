import os
import subprocess
import wandb
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb

word_to_id = imdb.get_word_index()
word_to_id = {k: (v+3) for k, v in word_to_id.items()}
id_to_word = {value: key for key, value in word_to_id.items()}
id_to_word[0] = ""  # Padding
id_to_word[1] = ""  # Start token
id_to_word[2] = "�"  # Unknown
id_to_word[3] = ""  # End token


def decode(word):
    return ' '.join(id_to_word[id] for id in word if id > 0)


class TextLogger(tf.keras.callbacks.Callback):
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out

    def on_epoch_end(self, logs, epoch):
        out = self.model.predict(self.inp)
        data = [[decode(self.inp[i]), o, self.out[i]]
                for i, o in enumerate(out)]
        wandb.log({"text": wandb.Table(rows=data)}, commit=False)


def load_imdb():
    if not os.path.exists("./aclImdb"):
        print("Downloading imdb dataset...")
        subprocess.check_output(
            "curl -SL http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz | tar xz", shell=True)

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

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    load_imdb()
