import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(lowercase=True)
count_vect.fit(fixed_text)

# turns the text into a sparse matrix
counts = count_vect.transform(fixed_text)

my_counts = count_vect.transform(["love that iphone!", "HATE that iphone"])
print(my_counts)
