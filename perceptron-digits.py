from sklearn import datasets

digits = datasets.load_digits()

from sklearn import cross_validation
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

scores = cross_validation.cross_val_score(perceptron, digits.data, digits.target, cv=10)
print scores
print scores.mean()
