import imdb
import numpy as np
from keras.preprocessing import text
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import wandb
from wandb.keras import WandbCallback
from sklearn.linear_model import LogisticRegression

wandb.init()
config = wandb.config
config.vocab_size = 1000

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# create model
model=Sequential()
model.add(Dense(2, activation="softmax", input_shape=(1000,)))
model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(save_model=False)])
