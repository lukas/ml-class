import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

p = Pipeline(steps=[('counts', CountVectorizer(ngram_range=(1, 2))),
                ('feature_selection', SelectKBest(chi2, k=10000)),
                ('multinomialnb', MultinomialNB())])

p.fit(fixed_text, fixed_target)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(p, fixed_text, fixed_target, cv=10)
print(scores)
print(scores.mean())

