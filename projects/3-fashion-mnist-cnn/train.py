# This is a very simple fashion classifier
# It uses a data set called "fashion mnist", a set of small b&w images of apparel
# 
# This training script classifies with around 87% accuracy currently.
#
# Can you get the validation accuracy (val_acc) above 90% with a CNN?
#
# The most common error looks something like:
# ValueError: Error when checking input: expected conv2d_input to have 4 dimensions, 
#    but got array with shape (60000, 28, 28)
#
# This means you need to reshape your b&w image input from (28, 28) to (28, 28, 1)
# You can use numpy.reshape or the keras Reshape layer.
# There is example code in the examples/keras-cnn directory.
#
# Getting the CNN to work better than a multi-layer perceptron on this dataset
# may take some experimentation. 


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, Dropout
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init(project="fashion")

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
          "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


X_train = X_train / 255.
X_test = X_test / 255.

num_classes = y_train.shape[1]

# create model
model = Sequential()
model.add(Flatten(input_shape=(img_width, img_height, 1)))
model.add(Dense(num_classes, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=labels)])
