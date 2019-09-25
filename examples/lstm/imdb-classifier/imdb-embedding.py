import wandb
import tensorflow as tf
import numpy as np
import subprocess
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
from tensorflow.keras.datasets import imdb
import util
import os

# set parameters:
wandb.init()
config = wandb.config
config.vocab_size = 1000
config.maxlen = 300
config.batch_size = 64
config.embedding_dims = 100
config.filters = 250
config.kernel_size = 3
config.hidden_dims = 100
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config.vocab_size)

if not os.path.exists("glove.6B.100d.txt"):
    print("Downloading glove embeddings...")
    subprocess.check_output(
        "curl -OL http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip", shell=True)

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

embeddings_index = dict()

f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((config.vocab_size, config.embedding_dims))
for index in range(config.vocab_size):
    word = util.id_to_word[index]
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# overide LSTM & GRU
if 'GPU' in str(device_lib.list_local_devices()):
    print("Using CUDA for RNN layers")
    LSTM = CuDNNLSTM
    GRU = CuDNNGRU

# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(config.vocab_size, 100, input_length=config.maxlen,
                                    weights=[embedding_matrix], trainable=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(config.hidden_dims))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[util.TextLogger(X_test[:20], y_test[:20]), wandb.keras.WandbCallback(save_model=False)])
