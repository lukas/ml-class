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

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

prop_train = 0.7
n_train = int(np.ceil(fixed_target.shape[0] * prop_train))
n_test = fixed_target.shape[0] - n_train
print('training on {} examples ({:.1%})'.format(n_train, prop_train))
print('testing on {} examples'.format(n_test)

nb.fit(counts[:n_train], fixed_target[:n_train])

predictions = nb.predict(counts[n_train:])
print(sum(predictions == fixed_target[n_train:]))
