from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, CuDNNLSTM, GRU, CuDNNGRU
from keras.layers import Conv1D, Flatten
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import imdb
import numpy as np
import subprocess
from keras.preprocessing import text
from tensorflow.python.client import device_lib
import os

wandb.init()
config = wandb.config

# set parameters:
config.vocab_size = 1000
config.maxlen = 300
config.batch_size = 32
config.embedding_dims = 50
config.filters = 250
config.kernel_size = 3
config.hidden_dims = 100
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

if not os.path.exists("glove.6B.100d.txt"):
    print("Downloading glove embeddings...")
    subprocess.check_output(
        "curl -OL http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip", shell=True)

print("Tokenizing input...")
tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

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

embedding_matrix = np.zeros((config.vocab_size, 100))
for word, index in tokenizer.word_index.items():
    if index > config.vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# overide LSTM & GRU
if 'GPU' in str(device_lib.list_local_devices()):
    print("Using CUDA for RNN layers")
    LSTM = CuDNNLSTM
    GRU = CuDNNGRU

# create model
model = Sequential()
model.add(Embedding(config.vocab_size, 100, input_length=config.maxlen,
                    weights=[embedding_matrix], trainable=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(config.hidden_dims))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback(input_type="time")])
