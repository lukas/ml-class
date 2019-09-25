import wandb
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
from tensorflow.keras.datasets import imdb
from util import TextLogger

# set parameters:
wandb.init()
config = wandb.config
config.vocab_size = 1000
config.maxlen = 1000
config.batch_size = 64
config.embedding_dims = 50
config.filters = 50
config.kernel_size = 3
config.hidden_dims = 100
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config.vocab_size)

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)
print(X_train.shape)
print("After pre-processing", X_train[0])


# overide LSTM & GRU
if 'GPU' in str(device_lib.list_local_devices()):
    print("Using CUDA for RNN layers")
    LSTM = CuDNNLSTM
    GRU = CuDNNGRU


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(config.vocab_size,
                                    config.embedding_dims,
                                    input_length=config.maxlen))
model.add(tf.keras.layers.Conv1D(config.filters,
                                 config.kernel_size,
                                 padding='valid',
                                 activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(config.hidden_dims, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[TextLogger(X_test[:20], y_test[:20]),
                                                       wandb.keras.WandbCallback(save_model=False)])
