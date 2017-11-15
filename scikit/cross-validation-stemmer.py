from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

from wandblog import log
import wandb
run = wandb.init()
config = run.config

stemmer = SnowballStemmer("english")
def tokenizer(doc):
    stemmed_words = [stemmer.stem(word) for word in word_tokenize(doc)]
    return stemmed_words

import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

count_vect = CountVectorizer(tokenizer=tokenizer)
count_vect.fit(fixed_text)

counts = count_vect.transform(fixed_text)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

from sklearn.model_selection import cross_val_score, cross_val_predict

scores = cross_val_score(nb, counts, fixed_target)
print(scores)
print(scores.mean())

predictions = cross_val_predict(nb, counts, fixed_target)
log(run, fixed_text, fixed_target, predictions)
