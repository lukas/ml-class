import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class NumBangExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe outputs average word length"""

    def __init__(self):
        pass

    def num_bang(self, str):
        """Helper code to compute number of exclamation points"""
        return str.count('!')

    def transform(self, inp, y=None):
        out = np.array(map(self.num_bang, inp))
        return out.reshape(-1,1)


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
                            ('numbang', NumBangExtractor()),
                            ('counts', CountVectorizer())
                            ])),
                ('multinomialnb', MultinomialNB())])

p.fit(fixed_text, fixed_target)
print(p.predict(["I love my iphone!"]))
