from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()



model = load_model("perceptron.h5")

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

pred = model.predict(X_test)

digit = X_test[1]

str = ""
for i in range(digit.shape[0]):
    for j in range(digit.shape[1]):
        if digit[i][j] == 0:
            str += " "
        elif digit[i][j] < 128:
            str += "."
        else:
            str += "X"
    str += "\n"

print(str)
print(pred[1])
