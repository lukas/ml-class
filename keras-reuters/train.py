import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config

config.max_words = 1000
config.batch_size = 32
config.epochs = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=config.max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

# Vectorizing sequence data...
tokenizer = Tokenizer(num_words=config.max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# One hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

model = Sequential()
model.add(Dense(num_classes, input_shape=(config.max_words,), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    validation_data=(x_test, y_test), callbacks=[WandbCallback()])

