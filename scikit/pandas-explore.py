# Take a look at the data using pandas

import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']
product = df['emotion_in_tweet_is_directed_at']

# For more ideas check out
#    https://pandas.pydata.org/pandas-docs/stable/10min.html
#print(text.head())

# Prints the string where the target contains the word Positive
#print(text[target.str.contains("Negative")])

# Other ideas
#print(target[0])
#print(text.str.contains("iphone"))
print(target.value_counts())
#print(text[0])
