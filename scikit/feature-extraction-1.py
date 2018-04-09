# First attempt at feature extraction
# Leads to an error, can you tell why?

import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

from sklearn.feature_extraction.text import CountVectorizer

count_vect=CountVectorizer()
count_vect.fit(text[:10])
