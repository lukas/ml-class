# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config

# load the dataset but only keep the top n words, zero the rest
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config.num_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=config.max_length)
X_test = sequence.pad_sequences(X_test, maxlen=config.max_length)
# create the model
#embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(config.num_words, config.embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(32,3,activation='relu'))
model.add(MaxPooling1D(2))
model.add(LSTM(config.lstm_output_size))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size,
    callbacks=[WandbKerasCallback()],
    validation_data=(X_test, y_test))
