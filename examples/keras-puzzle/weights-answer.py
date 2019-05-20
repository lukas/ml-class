from tensorflow.python import pywrap_tensorflow
import h5py
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils
from keras.losses import categorical_crossentropy


def log_softmax(w):
    assert len(w.shape) == 1
    max_weight = np.max(w, axis=0)
    rightHandSize = np.log(np.sum(np.exp(w - max_weight), axis=0))
    return w - (max_weight + rightHandSize)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



def load_weights_from_tensorflow(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    weights = reader.get_tensor('Variable')
    return weights

def load_biases_from_tensorflow(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    bias = reader.get_tensor('Variable_1')
    return bias

def load_weights_from_keras_perceptron(filename):
    f = h5py.File(filename)
    bias = f['model_weights']['dense_1']['dense_1']['bias:0'][()]
    weights = f['model_weights']['dense_1']['dense_1']['kernel:0'][()]
    return weights, bias


def load_weights_from_keras_two_layer(filename):
    f = h5py.File(filename)
    bias1 = (f['model_weights']['dense_2']['dense_2']["bias:0"][()])
    weights1 = (f['model_weights']['dense_2']['dense_2']["kernel:0"][()])
    bias0 = (f['model_weights']['dense_1']['dense_1']["bias:0"][()])
    weights0 = (f['model_weights']['dense_1']['dense_1']["kernel:0"][()])

    return weights0, bias0, weights1, bias1

weight0, bias0 = load_weights_from_keras_perceptron('perceptron.h5')
#weights0, bias0, weights1, bias1 = load_weights_from_keras_two_layer('two-layer.h5')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

img_width=28
img_height=28

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = load_model("perceptron.h5")
print(model.predict(X_test[:1]))

x= X_test[:1].flat
output = np.zeros(10)
for i in range(10):
    out = 0
    for j in range(len(x)):
        out = out + weight0[j,i] * x[j]
    print(out)
    output[i] = out

print
softmax(output)
