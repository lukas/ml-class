from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from wandb.keras import WandbCallback
import wandb

run = wandb.init()
config = run.config
config.img_width = 28
config.img_height = 28
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
config.dense_layer_size = 100
config.epochs = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

#reshape input data
X_train = X_train.reshape(X_train.shape[0], config.img_width, config.img_height, 1)
X_test = X_test.reshape(X_test.shape[0], config.img_width, config.img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
labels=range(10)

# build model
model = Sequential()
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(28, 28,1),
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(config.dense_layer_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test),
        epochs=config.epochs,
        callbacks=[WandbCallback(data_type="image")])
