from sklearn import datasets

digits = datasets.load_digits()

from sklearn import model_selection
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

scores = model_selection.cross_val_score(perceptron, digits.data, digits.target, cv=10)
print scores
print scores.mean()
