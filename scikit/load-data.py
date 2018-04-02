# Quick example of loading our data into variables

import pandas as pd
import numpy as np

# Puts tweets into a data frame
df = pd.read_csv('tweets.csv')

# Selects the first column from our data frame
target = df['is_there_an_emotion_directed_at_a_brand_or_product']

# Selects the third column from our data frame
text = df['tweet_text']

print(len(text))
