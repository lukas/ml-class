from sklearn import datasets

digits = datasets.load_digits()

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Perceptron

perceptron = Perceptron()

scores = cross_val_score(perceptron, digits.data, digits.target, cv=10)
print(scores)
print(scores.mean())

