from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, Add
from keras.utils import np_utils
from wandb.keras import WandbCallback
import wandb
import os

run = wandb.init()
config = run.config
config.first_layer_convs = 32
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
config.dropout = 0.2
config.dense_layer_size = 128
config.img_width = 28
config.img_height = 28
config.epochs = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# reshape input data
X_train = X_train.reshape(
    X_train.shape[0], config.img_width, config.img_height, 1)
X_test = X_test.reshape(
    X_test.shape[0], config.img_width, config.img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
labels = [str(i) for i in range(10)]

# build model
input = Input(shape=(28, 28, 1))
input_copy = input
conv_out = Conv2D(32,
                  (config.first_layer_conv_width, config.first_layer_conv_height),
                  activation='relu', padding='same')(input)
res_input = Conv2D(32, (1, 1), activation='relu', padding='same')(input)
add_out = Add()([conv_out, res_input])

max_pool_out = MaxPooling2D(pool_size=(2, 2))(conv_out)
flatten_out = Flatten()(max_pool_out)
dense1_out = Dense(config.dense_layer_size, activation='relu')(flatten_out)
dense2_out = Dense(num_classes, activation='softmax')(dense1_out)

model = Model(input, dense2_out)


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=config.epochs,
          callbacks=[WandbCallback(data_type="image", save_model=False)])
