import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
#print(df)
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']
product = df['emotion_in_tweet_is_directed_at']

#print(target[0])
#print(text.str.contains("iphone"))
print(target.value_counts())
#print(text[0])
#print(text[target == "I can't tell"])
