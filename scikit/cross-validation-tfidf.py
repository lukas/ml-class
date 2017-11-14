import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from wandblog import log
import wandb
run = wandb.init()
config = run.config

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


clf = SGDClassifier()

from sklearn.model_selection import cross_val_score, cross_val_predict

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(max_iter=1000)),
 ])

scores = cross_val_score(text_clf, fixed_text, fixed_target)
print(scores)
print(scores.mean())

predictions = cross_val_predict(text_clf, fixed_text, fixed_target)
log(run, fixed_text, fixed_target, predictions)
