# second attempt at feature extraction

import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

# what did we do here?
fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(fixed_text)

# print the number of words in the vocabulary
print(len(count_vect.vocabulary_))
