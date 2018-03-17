from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model, Sequential

from keras.datasets import mnist
import numpy as np
import wandb
from wandb.wandb_keras import WandbKerasCallback


run = wandb.init()
config = run.config

config.encoding_dim = 32
config.epochs = 1
config.batch_size = 32

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# this is the size of our encoded representations
encoding_dim = 32  

# this is our input placeholder
model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add( Dense(encoding_dim, activation='relu'))
model.add( Dense(784, activation='sigmoid'))
model.add( Reshape((28,28)))
model.compile(optimizer='adam', loss='binary_crossentropy')



model.fit(x_train, x_train,
                epochs=config.epochs,
                batch_size=config.batch_size,
                validation_data=(x_test, x_test), callbacks=[WandbKerasCallback()])


model.save("auto.h5")



