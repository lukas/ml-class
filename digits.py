from sklearn import datasets

digits = datasets.load_digits()

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

nb = MultinomialNB()

scores = cross_val_score(nb, digits.data, digits.target, cv=10)
print(scores)
print(scores.mean())

