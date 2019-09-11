from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = [x.flatten() for x in X_train]
y_train = to_categorical(y_train)

perceptron = Perceptron()

scores = cross_val_score(perceptron, X_train, y_train, cv=10)
print(scores)
print(scores.mean())
