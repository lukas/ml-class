import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class NumBangExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def num_bang(self, str):
        """Helper code to compute number of exclamation points"""
        return str.count('!')

    def transform(self, X, y=None):
        counts = []
        for x in X:
            counts.append([self.num_bang(x)])    # the [] are important!
        return counts

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion

p = Pipeline(steps=[('feats', FeatureUnion([
                            ('counts_preserve_case', CountVectorizer(lowercase=False)),
                            ('counts_lower_case', CountVectorizer(lowercase=True)),
                            ('numbang', NumBangExtractor())
                            ])),
                ('multinomialnb', MultinomialNB())])

from sklearn.model_selection import cross_val_score

scores = cross_val_score(p, fixed_text, fixed_target, cv=10)
print(scores)
print(scores.mean())
