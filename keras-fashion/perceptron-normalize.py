import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
import wandb
from wandb.wandb_keras import WandbKerasCallback

# logging code
run = wandb.init()
config = run.config
config.epochs = 10
config.normalize = True

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# normalize data
X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbKerasCallback()])

# Output some predictions

from PIL import Image
from PIL import ImageDraw 
import numpy as np

model_output = model.predict(X_test)

labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

for i in range(10):
    prediction = np.argmax(model_output[i])
    img = Image.fromarray(X_test[i])
    img = img.resize((280, 280), Image.ANTIALIAS)

    draw = ImageDraw.Draw(img)
    draw.text((10, 10),labels[prediction],(255))
    img.save(str(i)+".jpg")