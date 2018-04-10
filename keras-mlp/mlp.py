import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.callbacks import Callback
import json

from wandb.wandb_keras import WandbKerasCallback
import wandb

run = wandb.init()
config = run.config

config.hidden_nodes = 100

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]

y_test = np_utils.to_categorical(y_test)

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Dense(config.hidden_nodes, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=config.optimizer,
                    metrics=['accuracy'])

class Images(Callback):
      def on_epoch_end(self, epoch, logs):
#            indices = np.random.randint(self.validation_data[0].shape[0], size=8)
            test_data = self.validation_data[0][:10]
            val_data = self.validation_data[1][:10]

            test_data = X_test[:10]
            val_data = y_test[:10]
            print(val_data)

            pred_data = self.model.predict(test_data)
            run.history.row.update({
                  "examples": [
                        wandb.Image(test_data[i], caption=str(val_data[i])+str(np.argmax(val_data[i]))) for i in range(8)
                        ]
            })



# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
        callbacks=[Images(), WandbKerasCallback()], epochs=config.epochs)
