import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']
product = df['emotion_in_tweet_is_directed_at']


print(target[0:5])
print(text[0:5])
print(product.value_counts())
print(target.value_counts())

print(text[9])

