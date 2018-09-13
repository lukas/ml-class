import amazon
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten
from keras.preprocessing import text
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config
config.vocab_size = 1000

(train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels) = amazon.load_amazon()

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(train_review_text)
X_train = tokenizer.texts_to_matrix(train_review_text)
X_test = tokenizer.texts_to_matrix(test_review_text)

# Build the model
model = Sequential()
model.add(Dense(1, activation='softmax', input_shape=(config.vocab_size,)))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, train_labels, epochs=10, validation_data=(X_test, test_labels),
    callbacks=[WandbCallback()])
