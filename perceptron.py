import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(fixed_text)

counts = count_vect.transform(fixed_text)

from sklearn.linear_model import Perceptron

perceptron = Perceptron()

from sklearn import cross_validation

scores = cross_validation.cross_val_score(perceptron, counts, fixed_target, cv=10)
print scores
print scores.mean()
