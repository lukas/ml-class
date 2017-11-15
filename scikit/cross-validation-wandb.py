import pandas as pd
import numpy as np
import wandb
from wandblog import log

run = wandb.init()
config = run.config   # for tracking model inputs
summary = run.summary # for tracking model outputs

config.lowercase = False
config.alpha = 1.0

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(lowercase = config.lowercase)
count_vect.fit(fixed_text)
counts = count_vect.transform(fixed_text)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha = config.alpha)

from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(nb, counts, fixed_target, cv=20)

print(scores)
print(scores.mean())

predictions = cross_val_predict(nb, counts, fixed_target)
log(run, fixed_text, fixed_target, predictions)
