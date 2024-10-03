from keras.api.preprocessing import sequence
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Activation
from keras.api.layers import Embedding, LSTM, Bidirectional
from keras.api.layers import Conv1D, Flatten
import wandb
from wandb.integration.keras import WandbCallback
import imdb
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

wandb.init()
config = wandb.config

# set parameters:
config.vocab_size = 1000
config.maxlen = 300
config.batch_size = 32
config.embedding_dims = 50
config.filters = 10
config.kernel_size = 3
config.hidden_dims = 10
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

tokenizer = Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen))
model.add(LSTM(config.hidden_dims, activation="sigmoid"))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])