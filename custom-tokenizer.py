
import pandas as pd
import numpy as np
import re, string

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

def tokenize_0(text):
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    return re.compile(r'\W+').split(text) # split at non words

def tokenize_1(text):
    return text.split()

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(tokenizer=tokenize_0)
count_vect.fit(fixed_text)

counts = count_vect.transform(fixed_text)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(nb, counts, fixed_target, cv=10)
print(scores)
print(scores.mean())
