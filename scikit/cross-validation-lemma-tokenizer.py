from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from wandblog import log
import wandb
run = wandb.init()
config = run.config

wnl = WordNetLemmatizer()

def lemma_tokenizer():
    return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vect = CountVectorizer(tokenizer=LemmaTokenizer())

import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

count_vect = CountVectorizer(tokenizer=tokenize_1)
count_vect.fit(fixed_text)

counts = count_vect.transform(lemma_tokenizer)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(nb, counts, fixed_target, cv=10)
print(scores)
print(scores.mean())


predictions = cross_val_predict(text_clf, fixed_text, fixed_target)
log(run, fixed_text, fixed_target, predictions)
