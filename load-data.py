# Quick example of loading our data into variables

import pandas as pd
import numpy as np

# Puts tweets into a data frame
df = pd.read_csv('tweets.csv')

# Selects columns from our data frame
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

print(len(target))
