from sklearn import datasets

digits = datasets.load_digits()

from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation

nb = MultinomialNB()

scores = cross_validation.cross_val_score(nb, digits.data, digits.target, cv=10)
print scores
print scores.mean()
