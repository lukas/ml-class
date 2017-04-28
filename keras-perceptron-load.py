from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils


(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = load_model("perceptron.h5")

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

eval = model.evaluate(X_test, y_test)
print(eval)
