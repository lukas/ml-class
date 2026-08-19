# The goal is to classify amazon reviews of video games
#
# To download the reviews run "bash download-amazon.sh"
# Or get the file from http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz
# and gunzip it in this directory.
#
# This simple model is 85% accurate right now.
# Can you modify it to get over 88% val accuracy?
from keras.preprocessing import sequence
from keras.preprocessing import text
import amazon
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing import text
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config

(train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels) = amazon.load_amazon_smaller()
(X_train, y_train), (X_test, y_test) = (train_summary, train_labels), (test_summary, test_labels)

# Summaries are short (a headline, not a paragraph), so a maxlen of 1000
# was mostly wasted padding. 40 tokens comfortably covers nearly all
# review summaries and lets the model focus on real signal.
config.vocab_size = 5000
config.maxlen = 40
config.embedding_dims = 50
config.filters = 32
config.kernel_size = 3
config.hidden_dims = 250
config.epochs = 10

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

model = Sequential()
# This is the piece the original model was missing: Embedding turns each
# integer token id into a dense learnable vector, so the network can
# actually find patterns in the text instead of treating token ids as
# raw numeric magnitudes.
model.add(Embedding(config.vocab_size, config.embedding_dims, input_length=config.maxlen))
model.add(Dropout(0.3))
model.add(Conv1D(config.filters, config.kernel_size, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(config.hidden_dims, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=config.epochs, validation_data=(X_test, y_test),
    callbacks=[WandbCallback()])
