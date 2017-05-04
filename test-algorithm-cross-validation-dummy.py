import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(fixed_text)

counts = count_vect.transform(fixed_text)

from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

nb = DummyClassifier(strategy='most_frequent')

from sklearn.model_selection import cross_val_score

scores = cross_val_score(nb, counts, fixed_target, cv=10)
print(scores)
print(scores.mean())


nb.fit(counts, fixed_target)
print(nb.predict(count_vect.transform(["love I my iphone!!!"])))
