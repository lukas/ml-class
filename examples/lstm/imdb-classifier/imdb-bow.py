import util
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import text
import wandb

wandb.init()
config = wandb.config
config.vocab_size = 1000

(X_train, y_train), (X_test, y_test) = util.load_imdb()

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2, activation="softmax", input_shape=(1000,)))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[wandb.keras.WandbCallback(save_model=False)])
