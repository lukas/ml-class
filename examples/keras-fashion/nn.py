import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Reshape, Conv2D, MaxPooling2D
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.epochs = 100
config.lr = 0.01
config.layers = 3
config.dropout = 0.4
config.hidden_layer_1_size = 128

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.
X_test = X_test / 255.

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
          "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# create model
model = Sequential()
#model.add(Reshape((img_width, img_height, 1), input_shape=(img_width,img_height)))
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dropout(config.dropout))
model.add(Dense(config.hidden_layer_1_size, activation='relu'))
model.add(Dropout(config.dropout))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=labels)])
