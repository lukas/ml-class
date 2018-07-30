from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, Flatten
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config

# set parameters:
config.max_features = 5000
config.maxlen = 400
config.batch_size = 32
config.embedding_dims = 50
config.filters = 250
config.kernel_size = 3
config.hidden_dims = 250
config.epochs = 2

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=config.max_features)

x_train = sequence.pad_sequences(x_train, maxlen=config.maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=config.maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(config.max_features,
                    config.embedding_dims,
                    input_length=config.maxlen))
model.add(Dropout(0.2))
model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(config.hidden_dims, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(x_test, y_test))
